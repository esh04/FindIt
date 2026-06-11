"""lmms-eval hooks: doc_to_visual, doc_to_text, process_results, aggregators."""
import re

import numpy as np
from PIL import Image

from .evaluate import (
    compute_iou_matrix,
    extract_bboxes,
    extract_json_bboxes,
    extract_video_json_bboxes,
    extract_video_text_bboxes,
    hungarian_algorithm,
)
from .promptloader import get_prompt

_IMAGE_PLACEHOLDER_RE = re.compile(r"<image (\d+)>")


def _rescale_flags(mode):
    return mode == "no_rescale", mode == "unit_rescale", mode == "smart_resize"


# outputs a list of PIL.Image objects
def doc_to_visual(doc, kwargs):
    tf = kwargs["task_family"]
    scene_max = kwargs.get("scene_max_size")

    def _maybe_resize(img):
        if not scene_max:
            return img
        w, h = img.size
        if max(w, h) <= scene_max:
            return img
        scale = scene_max / max(w, h)
        return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    if tf in ("objdet_video", "insdet_video"):
        paths = doc["frame_paths"]
        n = kwargs["max_frames"]
        if n < len(paths):
            idx = np.linspace(0, len(paths) - 1, n, dtype=int)
            paths = [paths[i] for i in idx]
        frames = [_maybe_resize(Image.open(p).convert("RGB")) for p in paths]
        if tf == "insdet_video":
            shots = [Image.open(p).convert("RGB") for p in doc["support_shots"]]
            return shots + frames
        return frames

    if tf == "insdet":
        shots = [Image.open(p).convert("RGB") for p in doc["support_shots"]]
        return shots + [_maybe_resize(Image.open(doc["image_path"]).convert("RGB"))]

    if "image" in doc and doc["image"] is not None:
        return [_maybe_resize(doc["image"])]
    return [_maybe_resize(Image.open(doc["image_path"]).convert("RGB"))]


# outputs the prompt text
def doc_to_text(doc, kwargs):
    task = kwargs["task_family"]
    is_video = task.endswith("_video")
    num_frames = kwargs["max_frames"] if is_video else 1
    if num_frames == 1:
        task = task.removesuffix("_video")

    template = get_prompt(
        task=task,
        multilabel=kwargs["multi_label"],
        fmt=kwargs["fmt"],
        bbox_repr=kwargs["bbox_repr"],
        json_structure=kwargs.get("json_structure", "per_box"),
        json_key=kwargs.get("json_key", "bbox"),
        center_exp=kwargs["center_exp"],
        negative_examples=kwargs["negative_examples"],
    )
    text = template.format(label=doc["label"], num_frames=num_frames)
    if kwargs["with_caption"] and doc.get("caption"):
        scope = "This image shows" if num_frames == 1 else "These video frames show"
        text = f'{scope}: "{doc["caption"]}". {text}'

    tf = kwargs["task_family"]
    if tf in ("insdet", "insdet_video"):
        k = len(doc["support_shots"])
        if tf == "insdet_video":
            n = min(kwargs["max_frames"], len(doc["frame_paths"]))
        else:
            n = 1
        ref_block = "\n".join(f"<image {j + 1}>" for j in range(k))
        query_block = "\n".join(f"<image {k + f + 1}>" for f in range(n))
        text = (
            f"Here are reference images of an object:\n{ref_block}\n"
            f"Locate all instances of this object in:\n{query_block}\n\n"
            f"{text}"
        )

    if kwargs.get("image_position") == "last":
        text = f"{text}\n<image 1>"
    return text


# Chat-style hook: build interleaved [user] message content from doc_to_text
# (which may contain `<image N>` placeholders) and doc_to_visual. Used by chat
# wrappers (e.g. internvl_hf) that build messages via task.doc_to_messages.
# When no placeholders exist, falls back to images-first-then-text (the lmms-
# eval auto-bridge default), so non-insdet tasks behave identically.
def doc_to_messages(doc, kwargs):
    text = doc_to_text(doc, kwargs)
    visuals = doc_to_visual(doc, kwargs)

    if not _IMAGE_PLACEHOLDER_RE.search(text):
        content = [{"type": "image", "url": v} for v in visuals]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    parts = _IMAGE_PLACEHOLDER_RE.split(text)
    content = []
    if parts[0].strip():
        content.append({"type": "text", "text": parts[0]})
    for i in range(1, len(parts), 2):
        idx = int(parts[i]) - 1
        if 0 <= idx < len(visuals):
            content.append({"type": "image", "url": visuals[idx]})
        if i + 1 < len(parts) and parts[i + 1].strip():
            content.append({"type": "text", "text": parts[i + 1]})
    return [{"role": "user", "content": content}]


def _extract(response, doc, kwargs):
    no_rescale, unit_rescale, smart_resize = _rescale_flags(kwargs["rescale_mode"])
    tf = kwargs["task_family"]
    fmt = kwargs["fmt"]
    bbox_repr = kwargs["bbox_repr"]
    multi_label = kwargs["multi_label"]
    json_key = kwargs.get("json_key", "bbox")
    json_structure = kwargs.get("json_structure", "per_box")
    scene_max_size = kwargs.get("scene_max_size")
    h, w = doc["height"], doc["width"]

    if tf.endswith("_video"):
        n = kwargs["max_frames"]
        if fmt == "json":
            return extract_video_json_bboxes(response, h, w, bbox_repr, n,
                                             json_key=json_key, json_structure=json_structure,
                                             multilabel=multi_label,
                                             no_rescale=no_rescale, unit_rescale=unit_rescale,
                                             smart_resize=smart_resize, scene_max_size=scene_max_size)
        return extract_video_text_bboxes(response, h, w, bbox_repr, n,
                                         json_key=json_key, multilabel=multi_label,
                                         no_rescale=no_rescale, unit_rescale=unit_rescale,
                                         smart_resize=smart_resize, scene_max_size=scene_max_size)
    if fmt == "json":
        return extract_json_bboxes(response, h, w, bbox_repr,
                                   json_key=json_key, json_structure=json_structure,
                                   multilabel=multi_label,
                                   no_rescale=no_rescale, unit_rescale=unit_rescale,
                                   smart_resize=smart_resize, scene_max_size=scene_max_size)
    return extract_bboxes(response, h, w, bbox_repr, multilabel=multi_label,
                          no_rescale=no_rescale, unit_rescale=unit_rescale,
                          smart_resize=smart_resize, scene_max_size=scene_max_size)


def _match(gt, pred, multi_label):
    if multi_label:
        iou_m = compute_iou_matrix([b for _, b in gt], [b for _, b in pred])
    else:
        iou_m = compute_iou_matrix(gt, pred)
    ious, _, label_matched, _ = hungarian_algorithm(iou_m, gt, pred, multilabel=multi_label)
    return ious, label_matched


def _normalize_multilabel(raw):
    # Arrow stores {'label': s, 'bbox': [...]} dicts; evaluate.py expects (label, bbox) tuples.
    return [(item["label"], list(item["bbox"])) if isinstance(item, dict) else item for item in raw]


def process_results(doc, result, kwargs):
    response = result[0] if result else ""
    parsed, format_ok = _extract(response, doc, kwargs)
    multi_label = kwargs["multi_label"]

    if kwargs["task_family"].endswith("_video"):
        gt_frames = doc["bboxes"]
        n = kwargs["max_frames"]
        if n < len(gt_frames):
            idx = np.linspace(0, len(gt_frames) - 1, n, dtype=int)
            gt_frames = [gt_frames[i] for i in idx]
        if multi_label:
            gt_frames = [_normalize_multilabel(f) for f in gt_frames]
        tp = n_pred = n_gt = 0
        ious_flat = []
        for gt, pred in zip(gt_frames, parsed):
            ious, label_matched = _match(gt, pred, multi_label)
            tp += sum(1 for iou, lm in zip(ious, label_matched) if iou >= 0.5 and lm)
            ious_flat.extend(ious)
            n_pred += len(pred)
            n_gt += len(gt)
    else:
        gt = doc["bboxes"]
        if multi_label:
            gt = _normalize_multilabel(gt)
        ious, label_matched = _match(gt, parsed, multi_label)
        tp = sum(1 for iou, lm in zip(ious, label_matched) if iou >= 0.5 and lm)
        ious_flat = list(ious)
        n_pred, n_gt = len(parsed), len(gt)

    return {
        "format_adherence": 1.0 if format_ok else 0.0,
        "mean_iou":         ious_flat,
        "precision_at_05":  (tp, n_pred),
        "recall_at_05":     (tp, n_gt),
        "f1_at_05":         (tp, n_pred, n_gt),
        "raw_response":  response,
        "parsed_bboxes": parsed,
        "gt_bboxes":     doc["bboxes"],
        "filename":      doc["filename"],
        "width":         doc["width"],
        "height":        doc["height"],
    }


def agg_format_adherence(results):
    return (sum(results) / len(results)) * 100 if results else 0.0


def agg_mean_iou(results):
    xs = [x for s in results for x in s]
    return (sum(xs) / len(xs)) * 100 if xs else 0.0


def agg_precision_at_05(results):
    tp = sum(r[0] for r in results)
    tot = sum(r[1] for r in results)
    return (tp / tot) * 100 if tot else 0.0


def agg_recall_at_05(results):
    tp = sum(r[0] for r in results)
    tot = sum(r[1] for r in results)
    return (tp / tot) * 100 if tot else 0.0


def agg_f1_at_05(results):
    tp     = sum(r[0] for r in results)
    n_pred = sum(r[1] for r in results)
    n_gt   = sum(r[2] for r in results)
    p = (tp / n_pred) * 100 if n_pred else 0.0
    r = (tp / n_gt)   * 100 if n_gt   else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0
