"""OpenRouter wrapper that disables reasoning on every call.
Also rearranges user-message content into interleaved form when the prompt text
carries `<image N>` placeholders. Upstream OpenAICompatible builds the content
array text-first-then-images, so without this hook the model wouldn't receive
each image at its placeholder position.

Encodes large images as JPEG (>= 2048 px on the max side) — Anthropic caps each
image at 5 MB and OpenRouter at 30 MB. A 4096x3072 natural-image PNG is ~14 MB
while the same JPEG is ~1 MB; PNGs were silently failing upstream with
`choices=None`. Smaller images (refcoco, pascal, etc.) stay on PNG so existing
runs don't change format mid-benchmark.

Usage:
    --model openrouter --model_args model=anthropic/claude-4.5-sonnet
"""

import base64
import os

from PIL import Image
from lmms_eval.models.simple.openai import OpenAICompatible
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64_with_size_limit

from ._interleave import interleave_chat_messages

_LARGE_IMAGE_MIN_SIDE = 2048
# Pre-resize target for very large images (>= this on max side) before JPEG encoding.
# Matches the scene_max_size=4096 used in hr_insdet scripts so the scorer's bbox
# rescaling assumption is consistent with what the model actually receives.
# Open-source models (Qwen2.5-VL) process 8K images at ~4116x3080 effective resolution;
# this pre-resize targets the same scale (4096/8192 = 0.5).
_LARGE_IMAGE_TARGET_SIDE = 4096
_JPEG_MAGIC = b"\xff\xd8\xff"


class OpenRouterNoThink(OpenAICompatible):
    def __init__(self, base_url=None, api_key=None, **kwargs):
        super().__init__(
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            **kwargs,
        )
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
            extra["reasoning"] = {"enabled": False}
            payload["extra_body"] = extra
            return orig_create(**payload)

        self.client.chat.completions.create = create_no_think

    def encode_image(self, image):
        if isinstance(image, str):
            with Image.open(image) as loaded:
                img = loaded.convert("RGB")
        else:
            img = image
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
