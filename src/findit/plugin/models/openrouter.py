"""OpenRouter wrapper that disables reasoning on every call (by default).
Also rearranges user-message content into interleaved form when the prompt text
carries `<image N>` placeholders. Upstream OpenAICompatible builds the content
array text-first-then-images, so without this hook the model wouldn't receive
each image at its placeholder position.

Encodes large images as JPEG (>= 2048 px on the max side) — Anthropic caps each
image at 5 MB and OpenRouter at 30 MB. A 4096x3072 natural-image PNG is ~14 MB
while the same JPEG is ~1 MB; PNGs were silently failing upstream with
`choices=None`. Smaller images (refcoco, pascal, etc.) stay on PNG so existing
runs don't change format mid-benchmark.

Optional model_args (all off by default):
    reasoning=default        send no `reasoning` field (provider default effort)
                             instead of `reasoning.enabled=false`
    max_output_tokens=N      override lmms-eval's max_tokens (capped at 4096
                             upstream); reasoning tokens count against it
    drop_temperature=true    don't send `temperature` (reasoning models ignore
                             or reject it, and only some providers honour it)
    journal=PATH             append every successful response to a JSONL file
                             and replay it on restart, keyed by payload hash, so
                             a killed job never pays twice for the same request
    replay_max_tokens=N[;M]  also replay journal entries recorded under these
                             earlier max_tokens values (if they stopped normally
                             within the current cap), so lowering the cap mid-run
                             doesn't re-buy finished calls
    max_image_patches=N      shrink every image the way OpenAI's `high` detail does
    max_image_side=M         (GPT-5.4's default): fit M px per side, then at most N
                             32x32 patches, e.g. 2500 / 2048 to match GPT-5.4
    dry_run=true             never touch the network: log payload stats to the
                             journal and return an empty-box answer

Usage:
    --model openrouter --model_args model=anthropic/claude-4.5-sonnet
"""

import base64
import hashlib
import io
import json
import math
import os
import threading
import time

from PIL import Image
from lmms_eval.models.simple.openai import OpenAICompatible
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64_with_size_limit
from openai.types.chat import ChatCompletion

from ...evaluate import API_MAX_SIDE
from ._interleave import interleave_chat_messages

_LARGE_IMAGE_MIN_SIDE = 2048
# Pre-resize target for very large images (> this on max side) before JPEG encoding.
# Matches the scene_max_size=4096 used in hr_insdet scripts. The scorer applies the same
# cap (findit.evaluate.API_MAX_SIDE) when it maps pixel answers back to the original image.
# Open-source models (Qwen2.5-VL) process 8K images at ~4116x3080 effective resolution;
# this pre-resize targets the same scale (4096/8192 = 0.5).
_LARGE_IMAGE_TARGET_SIDE = API_MAX_SIDE
_JPEG_MAGIC = b"\xff\xd8\xff"
_SDK_MAX_RETRIES = 6       # openai-sdk exponential backoff on 429/5xx, per lmms-eval attempt
_SDK_TIMEOUT_S = 900       # reasoning calls on multi-image prompts can take minutes


def _openai_high_dims(w, h, max_patches, max_side, patch=32):
    """OpenAI's documented `high` image sizing: fit max_side, then shrink until
    ceil(w/32)*ceil(h/32) <= max_patches, flooring the dimensions."""
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = int(w * s), int(h * s)
    if math.ceil(w / patch) * math.ceil(h / patch) <= max_patches:
        return w, h
    s = math.sqrt(patch * patch * max_patches / (w * h))
    s *= min(math.floor(w * s / patch) / (w * s / patch), math.floor(h * s / patch) / (h * s / patch))
    return int(w * s), int(h * s)


def _as_bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes")


def _payload_key(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _image_stats(messages):
    """Per-image (mime, bytes, w, h) of the data-URLs in a payload, for dry runs."""
    stats = []
    for msg in messages:
        for c in msg.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "image_url":
                url = c["image_url"]["url"]
                head, b64 = url.split(",", 1)
                raw = base64.b64decode(b64)
                with Image.open(io.BytesIO(raw)) as im:
                    stats.append({"mime": head[5:].split(";")[0], "bytes": len(raw), "w": im.size[0], "h": im.size[1]})
    return stats


class _Journal:
    """Append-only JSONL of successful responses keyed by payload hash."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.cache = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn final line from a killed job
                    if rec.get("ok"):
                        self.cache[rec["key"]] = rec["response"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fh = open(path, "a")

    def get(self, key):
        return self.cache.get(key)

    def put(self, rec):
        line = json.dumps(rec, default=str) + "\n"
        with self.lock:
            self.fh.write(line)
            self.fh.flush()
            os.fsync(self.fh.fileno())
            if rec.get("ok"):
                self.cache[rec["key"]] = rec["response"]


class OpenRouterNoThink(OpenAICompatible):
    def __init__(self, base_url=None, api_key=None, reasoning="off", max_output_tokens=None,
                 drop_temperature=False, journal=None, replay_max_tokens=None,
                 max_image_patches=None, max_image_side=None, dry_run=False, **kwargs):
        dry_run = _as_bool(dry_run)
        super().__init__(
            base_url=base_url or "https://openrouter.ai/api/v1",
            # never hand the real key to a dry run
            api_key="dry-run-no-key" if dry_run else (api_key or os.getenv("OPENROUTER_API_KEY")),
            **kwargs,
        )
        if reasoning not in ("off", "default"):
            raise ValueError(f"reasoning must be 'off' or 'default', got {reasoning!r}")
        drop_temperature = _as_bool(drop_temperature)
        max_output_tokens = int(max_output_tokens) if max_output_tokens else None
        replay_max_tokens = [int(x) for x in str(replay_max_tokens).split(";")] if replay_max_tokens else []
        self.image_budget = (int(max_image_patches), int(max_image_side)) if max_image_patches else None
        self.journal = _Journal(journal) if journal else None
        if dry_run and self.journal is None:
            raise ValueError("dry_run=true needs journal=PATH to record payload stats")

        self.client = self.client.with_options(max_retries=_SDK_MAX_RETRIES, timeout=_SDK_TIMEOUT_S)
        orig_create = self.client.chat.completions.create

        def create_no_think(**payload):
            payload["messages"] = interleave_chat_messages(payload["messages"])
            # Upstream hardcodes data:image/png; sniff actual encoding from the
            # base64 payload and rewrite the prefix when our encode_image used JPEG.
            for msg in payload["messages"]:
                if not isinstance(msg.get("content"), list):
                    continue
                for c in msg["content"]:
                    if isinstance(c, dict) and c.get("type") == "image_url":
                        url = c.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/png;base64,"):
                            b64 = url[len("data:image/png;base64,"):]
                            try:
                                head = base64.b64decode(b64[:8] + "=" * (-len(b64[:8]) % 4))
                            except Exception:
                                head = b""
                            if head.startswith(_JPEG_MAGIC):
                                c["image_url"]["url"] = "data:image/jpeg;base64," + b64
            extra = dict(payload.get("extra_body") or {})
            if reasoning == "off":
                extra["reasoning"] = {"enabled": False}
            extra["usage"] = {"include": True}
            payload["extra_body"] = extra
            if max_output_tokens:
                payload["max_tokens"] = max_output_tokens
            if drop_temperature:
                payload.pop("temperature", None)

            if self.journal is None:
                return orig_create(**payload)

            key = _payload_key(payload)
            texts = [c.get("text", "") for m in payload["messages"] for c in (m.get("content") or []) if isinstance(c, dict) and c.get("type") == "text"]
            n_images = sum(1 for m in payload["messages"] for c in (m.get("content") or []) if isinstance(c, dict) and c.get("type") == "image_url")
            cached = self.journal.get(key)
            if cached is None:
                # answers journaled under an earlier max_tokens are still valid if they
                # finished normally within the current cap
                for mt in replay_max_tokens:
                    old = self.journal.get(_payload_key({**payload, "max_tokens": mt}))
                    if (old and old["choices"][0].get("finish_reason") == "stop"
                            and (old.get("usage") or {}).get("completion_tokens", 1 << 30) <= payload.get("max_tokens", 1 << 30)):
                        cached = old
                        break
            if cached is not None:
                return ChatCompletion.model_validate(cached)
            if dry_run:
                response = {
                    "id": "dry-run", "object": "chat.completion", "created": int(time.time()), "model": payload["model"],
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "[]"}}],
                }
                self.journal.put({"key": key, "ok": False, "dry_run": True, "ts": time.time(), "model": payload["model"],
                                  "text": texts, "images": _image_stats(payload["messages"]),
                                  "request": {k: v for k, v in payload.items() if k != "messages"}})
                return ChatCompletion.model_validate(response)

            started = time.time()
            response = orig_create(**payload)
            dump = response.model_dump()
            choice = (dump.get("choices") or [{}])[0]
            ok = bool(choice.get("message", {}).get("content"))  # empty/length-truncated answers are retried on rerun
            self.journal.put({"key": key, "ok": ok, "ts": started, "latency_s": time.time() - started,
                              "model": payload["model"], "served_model": dump.get("model"), "provider": dump.get("provider"),
                              "finish_reason": choice.get("finish_reason"), "usage": dump.get("usage"),
                              "n_images": n_images, "text_head": " | ".join(texts)[:300], "response": dump})
            return response

        self.client.chat.completions.create = create_no_think

    def encode_image(self, image):
        if isinstance(image, str):
            with Image.open(image) as loaded:
                img = loaded.convert("RGB")
        else:
            img = image
        if self.image_budget:
            dims = _openai_high_dims(*img.size, *self.image_budget)
            if dims != img.size:
                img = img.resize(dims, Image.Resampling.LANCZOS)
        use_jpeg = max(img.size) >= _LARGE_IMAGE_MIN_SIDE
        if use_jpeg and max(img.size) > _LARGE_IMAGE_TARGET_SIDE:
            scale = _LARGE_IMAGE_TARGET_SIDE / max(img.size)
            img = img.resize(
                (int(img.size[0] * scale), int(img.size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
        return encode_image_to_base64_with_size_limit(
            img,
            max_size_bytes=self.max_size_in_mb * 1024 * 1024,
            image_format="JPEG" if use_jpeg else "PNG",
            convert_rgb=False,
            quality=95 if use_jpeg else None,
            copy_if_pil=False,
            resize_factor=0.75,
            min_side=100,
            resample=Image.Resampling.LANCZOS,
        )
