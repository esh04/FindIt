import ast
import math
import re
from torchvision.ops import box_iou
import torch
from scipy.optimize import linear_sum_assignment
import json
import string
import numpy as np

from .axes import BBOX_COORDS

# Qwen2.5-VL answers in pixels of its processor's resized input. Defaults: max_pixels=12.8 MP runs;
# min_pixels is the lmms-eval Qwen2.5-VL default. Tasks can override both (qwen_max_pixels /
# qwen_min_pixels, see axes.model_task_kwargs).
_QWEN_MAX_PIXELS = 12_845_056
_QWEN_MIN_PIXELS = 256 * 28 * 28
_QWEN_FACTOR = 28
# qwen-vl-utils `fetch_image` limits for an image that carries no per-image min/max_pixels, which is
# how the lmms-eval chat interface passes images: the image is rounded to multiples of 28 here first,
# and the processor then applies its own min/max_pixels (two resizes, `qwen_two_step`).
_QWEN_UTILS_MIN_PIXELS = 4 * 28 * 28
_QWEN_UTILS_MAX_PIXELS = 16384 * 28 * 28
# The OpenRouter plugin (plugin/models/openrouter.py) sends any image whose longer side exceeds this
# at this size (int() rounding), so pixel-space answers of API models live in that frame.
API_MAX_SIDE = 4096


def norm_string(s):
    # Drop punctuation before trimming, so ', <car>' (next label on the same line) becomes 'car'.
    return " ".join(s.lower().translate(str.maketrans("", "", string.punctuation)).split())


NUM = r'\d+(?:\.\d+)?'

# list of four coordinates
BBOX_STD = re.compile(
    rf'\[\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*\]'
)

# standard multilabel text format: "label": [x_min, y_min, x_max, y_max]
BBOX_STD_WITH_LABEL = re.compile(
     rf'\s*([^:\[\]]+?)\s*:\s*(\[\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*\])'
)

# multilabel text format for all/all-labelled: "label": [x1,y1,x2,y2,x3,y3,x4,y4]
BBOX_STD_WITH_LABEL_EIGHT = re.compile(
     rf'\s*([^:\[\]]+?)\s*:\s*(\[\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*\])'
)

# list of eight
BBOX_LIST_OF_EIGHT = re.compile(
    r'\[\s*'
    rf'{NUM}\s*,\s*'
    rf'{NUM}\s*,\s*'
    rf'{NUM}\s*,\s*'
    rf'{NUM}\s*,\s*'
    rf'{NUM}\s*,\s*'
    rf'{NUM}\s*,\s*'
    rf'{NUM}\s*,\s*'
    rf'{NUM}\s*'
    r'\]'
)

# free form text
BBOX_PAREN = re.compile(
    rf'\(\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*\)'
)

# free form multilabel text: "label": (x_min, y_min, x_max, y_max)
BBOX_PAREN_WITH_LABEL = re.compile(
     rf'\s*([^:\[\]()]+?)\s*:\s*(\(\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*\))'
)

BBOX_TWO_PAIRS_WITH_TEXT = re.compile(
    r'\(\s*'
    rf'({NUM})\s*,\s*'
    rf'({NUM})\s*'
    r'\)'
    r'.*?'
    r'\(\s*'
    rf'({NUM})\s*,\s*'
    rf'({NUM})\s*'
    r'\)',
    re.DOTALL
)

# Empty-list marker: the prescribed "no objects" response when negative_examples
# prompt is on. Counts as format-adhered (empty result) even with no bbox hits.
BBOX_EMPTY_LIST = re.compile(r'\[\s*\]')


# ── Video per-frame extraction ────────────────────────────────────────────────

FRAME_BBOX_RE = re.compile(
    r'(?:frame\s*)?<?(\d+)>?\s*:\s*\[\s*'
    rf'({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})'
    r'\s*\]',
    re.IGNORECASE,
)


def _scale_and_convert_bbox(coords, scale_x, scale_y, bbox_repr):
    x1, y1, x2, y2 = coords
    if bbox_repr == 'cxcywh':
        cx, cy, w, h = x1, y1, x2, y2
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
    elif bbox_repr == 'yxyx':
        y1, x1, y2, x2 = coords
    elif bbox_repr == 'xywh':
        x_min, y_min, w, h = coords
        x1, y1, x2, y2 = x_min, y_min, x_min + w, y_min + h
    elif bbox_repr == 'yxhw':
        y_min, x_min, h, w = coords
        x1, y1, x2, y2 = x_min, y_min, x_min + w, y_min + h
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def _smart_resize_dims(width, height, max_pixels=_QWEN_MAX_PIXELS, min_pixels=_QWEN_MIN_PIXELS, factor=_QWEN_FACTOR):
    """qwen-vl-utils smart_resize: dims rounded to multiples of `factor`,
    fitted into [`min_pixels`, `max_pixels`] while preserving aspect ratio.
    Images below `min_pixels` (e.g. most 500x375 Pascal images) are upscaled."""
    w = max(factor, round(width / factor) * factor)
    h = max(factor, round(height / factor) * factor)
    if w * h > max_pixels:
        beta = math.sqrt(width * height / max_pixels)
        w = max(factor, math.floor(width / beta / factor) * factor)
        h = max(factor, math.floor(height / beta / factor) * factor)
    elif w * h < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        w = math.ceil(width * beta / factor) * factor
        h = math.ceil(height * beta / factor) * factor
    return w, h


def _claude_resized_dims(width, height, max_edge, max_tokens):
    """Anthropic's reference `resized_size`: the largest aspect-preserving size
    with both edges (padded to 28 px) <= max_edge and ceil(w/28)*ceil(h/28)
    <= max_tokens. Claude returns pixel coords in this resized frame.
    https://platform.claude.com/docs/en/build-with-claude/vision-coordinates"""
    def fits(w, h):
        return (math.ceil(w / 28) * 28 <= max_edge and math.ceil(h / 28) * 28 <= max_edge
                and math.ceil(w / 28) * math.ceil(h / 28) <= max_tokens)

    if fits(width, height):
        return width, height
    if height > width:
        rh, rw = _claude_resized_dims(height, width, max_edge, max_tokens)
        return rw, rh
    aspect = width / height
    lo, hi = 1, width  # lo always fits; hi never fits
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect), 1)


def _openai_high_dims(width, height, max_patches, max_side, patch=32):
    """OpenAI's documented `high`-detail sizing (GPT-5.4 default): fit max_side,
    then shrink until ceil(w/32)*ceil(h/32) <= max_patches, flooring the dims.
    https://developers.openai.com/api/docs/guides/images-vision"""
    w, h = width, height
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = int(w * s), int(h * s)
    if math.ceil(w / patch) * math.ceil(h / patch) <= max_patches:
        return w, h
    s = math.sqrt(patch * patch * max_patches / (w * h))
    s *= min(math.floor(w * s / patch) / (w * s / patch), math.floor(h * s / patch) / (h * s / patch))
    return int(w * s), int(h * s)


def _get_scale(width, height, no_rescale=False, unit_rescale=False, smart_resize=False, scene_max_size=None):
    # When the scene was downscaled before sending to the model (doc_to_visual), the model's
    # pixel coords live in the resized space — scale back to original pixels.
    sent_w, sent_h = width, height
    if scene_max_size and max(width, height) > scene_max_size:
        scale = scene_max_size / max(width, height)
        sent_w = round(width * scale)
        sent_h = round(height * scale)
    if no_rescale:
        # API models: the plugin caps the longer side at API_MAX_SIDE before sending ...
        if max(sent_w, sent_h) > API_MAX_SIDE:
            scale = API_MAX_SIDE / max(sent_w, sent_h)
            sent_w, sent_h = int(sent_w * scale), int(sent_h * scale)
        # ... and no_rescale may carry the provider's resize rule, which is applied to the
        # sent image before the model sees it (and answers in it).
        if isinstance(no_rescale, tuple):
            kind, a, b = no_rescale
            resize = _openai_high_dims if kind == "openai" else _claude_resized_dims
            sent_w, sent_h = resize(sent_w, sent_h, a, b)
        return float(width) / sent_w, float(height) / sent_h
    if unit_rescale:
        return float(width), float(height)
    if smart_resize:
        # smart_resize is True (defaults) or (max_pixels, min_pixels, two_step) from the task kwargs.
        max_pixels, min_pixels, two_step = (smart_resize if isinstance(smart_resize, tuple)
                                            else (_QWEN_MAX_PIXELS, _QWEN_MIN_PIXELS, False))
        if two_step:
            sent_w, sent_h = _smart_resize_dims(sent_w, sent_h, max_pixels=_QWEN_UTILS_MAX_PIXELS,
                                                min_pixels=_QWEN_UTILS_MIN_PIXELS)
        rw, rh = _smart_resize_dims(sent_w, sent_h, max_pixels=max_pixels, min_pixels=min_pixels)
        return float(width) / rw, float(height) / rh
    return float(width) / 1000.0, float(height) / 1000.0


def extract_video_json_bboxes(text, height, width, bbox_repr, num_frames, json_key='bbox', json_structure='per_box', multilabel=False, no_rescale=False, unit_rescale=False, smart_resize=False, scene_max_size=None):
    if num_frames == 1:
        boxes, format_ok = extract_json_bboxes(
            text, height, width, bbox_repr, json_key=json_key, json_structure=json_structure, multilabel=multilabel,
            no_rescale=no_rescale, unit_rescale=unit_rescale, smart_resize=smart_resize, scene_max_size=scene_max_size,
        )
        return [boxes], format_ok

    if multilabel:
        raise NotImplementedError(
            "Multi-label JSON extraction for num_frames > 1 is not implemented; "
            "no current task uses this combination."
        )

    FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    match = FENCE_RE.search(text)
    if match:
        text = match.group(1)

    # GLM-4.6V either emits valid JSON first and then appends <|begin_of_box|>...repetitions
    # (truncate at the first box token so json.loads sees only the leading JSON), or starts
    # with the box blocks (read the first block that parses, as extract_json_bboxes does).
    box_start = text.find("<|begin_of_box|>")
    if box_start != -1:
        block_text = None
        if not text[:box_start].strip():
            BOX_TOKEN_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)
            for block in BOX_TOKEN_RE.findall(text):
                try:
                    json.loads(block)
                except Exception:
                    continue
                block_text = block
                break
        text = block_text if block_text is not None else text[:box_start]

    per_frame = [[] for _ in range(num_frames)]
    if not text:
        return per_frame, False

    try:
        entries = json.loads(text)
    except Exception:
        return per_frame, False

    if not isinstance(entries, list):
        return per_frame, False

    scale_x, scale_y = _get_scale(width, height, no_rescale, unit_rescale, smart_resize, scene_max_size)
    format_ok = True

    # Cast before taking the min: models may emit "frame_idx": "0" (string), and
    # "0" == 0 is False, which would misread a 0-indexed answer as 1-indexed.
    frame_indices = []
    for entry in entries:
        if isinstance(entry, dict) and "frame_idx" in entry:
            try:
                frame_indices.append(int(entry["frame_idx"]))
            except (TypeError, ValueError):
                pass
    if not frame_indices:
        return per_frame, False
    # check all frame indices are present either 0-num_frames-1 or 1-num_frames
    frame_min = min(frame_indices)

    zero_indexed = False
    if frame_min == 0:
        zero_indexed = True

    for entry in entries:
        if not isinstance(entry, dict):
            format_ok = False
            continue

        try:
            frame_idx = int(entry['frame_idx'])
            bbox = entry[json_key]
            coords = [float(v) for v in bbox]
        except Exception as e:
            print(f"Error: {e} entry: {entry}")
            format_ok = False
            continue

        if len(coords) != 4:
            format_ok = False
            continue
        if not zero_indexed:
            frame_idx = frame_idx - 1
        if frame_idx < 0 or frame_idx >= num_frames:
            format_ok = False
            continue
        per_frame[frame_idx].append(
            _scale_and_convert_bbox(coords, scale_x, scale_y, bbox_repr)
        )

    return per_frame, format_ok


def extract_video_text_bboxes(text, height, width, bbox_repr, num_frames, json_key='bbox', multilabel=False, no_rescale=False, unit_rescale=False, smart_resize=False, scene_max_size=None):
    if num_frames == 1:
        boxes, format_ok = extract_bboxes(
            text, height, width, bbox_repr, multilabel=multilabel,
            no_rescale=no_rescale, unit_rescale=unit_rescale, smart_resize=smart_resize, scene_max_size=scene_max_size,
        )
        return [boxes], format_ok

    if multilabel:
        raise NotImplementedError(
            "Multi-label text extraction for num_frames > 1 is not implemented; "
            "no current task uses this combination."
        )

    per_frame = [[] for _ in range(num_frames)]
    if not text:
        return per_frame, False

    # GLM-4.6V wraps output in <|begin_of_box|>...<|end_of_box|> and repeats the
    # block verbatim 3-4x. Extract only the first block to avoid triple-counting.
    BOX_TOKEN_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)
    m = BOX_TOKEN_RE.search(text)
    if m:
        text = m.group(1)

    scale_x, scale_y = _get_scale(width, height, no_rescale, unit_rescale, smart_resize, scene_max_size)
    format_ok = True

    entries = FRAME_BBOX_RE.findall(text)
    if not entries:
        return per_frame, False

    zero_indexed = False
    frame_min = min(int(entry[0]) for entry in entries)
    if frame_min == 0:
        zero_indexed = True

    for entry in entries:
        frame_idx = int(entry[0])
        if not zero_indexed:
            frame_idx = frame_idx - 1
        if frame_idx < 0 or frame_idx >= num_frames:
            format_ok = False
            continue
        coords = [float(entry[i]) for i in range(1, 5)]
        per_frame[frame_idx].append(
            _scale_and_convert_bbox(coords, scale_x, scale_y, bbox_repr)
        )
    return per_frame, format_ok

# Images

def extract_bboxes(text, height, width, bbox_repr, multilabel=False, no_rescale=False, unit_rescale=False, smart_resize=False, scene_max_size=None):

    if not text:
        return [], False

    # GLM-4.6V wraps output in <|begin_of_box|>...<|end_of_box|> and repeats the
    # block verbatim 3-4x. Extract only the first block to avoid triple-counting.
    BOX_TOKEN_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)
    m = BOX_TOKEN_RE.search(text)
    if m:
        text = m.group(1)

    regex = BBOX_STD
    if bbox_repr in ('all', 'all-labelled'):
        regex = BBOX_LIST_OF_EIGHT

    scale_x, scale_y = _get_scale(width, height, no_rescale, unit_rescale, smart_resize, scene_max_size)

    if multilabel:
        regex = BBOX_STD_WITH_LABEL_EIGHT if bbox_repr in ('all', 'all-labelled') else BBOX_STD_WITH_LABEL
    hits = regex.findall(text)

    if bbox_repr == 'unconstrained' and not multilabel:
        hits.extend(BBOX_TWO_PAIRS_WITH_TEXT.findall(text))
        hits.extend(BBOX_PAREN.findall(text))
    if bbox_repr == 'unconstrained' and multilabel:
        hits.extend(BBOX_PAREN_WITH_LABEL.findall(text))

    format_adhered = bool(hits) or bool(BBOX_EMPTY_LIST.search(text))
    out = []
    incomplete_bbox = 0

    for h in hits:
        if multilabel:
            # extract label and bbox
            label, bbox = h
            h = bbox

        # for the two-pair with text case
        if isinstance(h, tuple):
            h = f"[{', '.join(h)}]"

        parsed = ast.literal_eval(h)

        if isinstance(parsed, tuple):
            parsed = list(parsed)

        if bbox_repr == 'cxcywh':
            cx, cy, w, h = parsed
            x1 = cx - (w / 2.0)
            y1 = cy - (h / 2.0)
            x2 = cx + (w / 2.0)
            y2 = cy + (h / 2.0)
            parsed = [x1, y1, x2, y2]

        elif bbox_repr == 'xywh':
            x_min, y_min, w, h = parsed
            parsed = [x_min, y_min, x_min + w, y_min + h]

        elif bbox_repr == 'yxyx':
            y_min, x_min, y_max, x_max = parsed
            parsed = [x_min, y_min, x_max, y_max]

        elif bbox_repr == 'yxhw':
            y_min, x_min, h_, w_ = parsed
            parsed = [x_min, y_min, x_min + w_, y_min + h_]

        elif bbox_repr in ('all', 'all-labelled'):
            xs, ys = parsed[0::2], parsed[1::2]
            if (len(set(xs)) != 2 or len(set(ys)) != 2
                    or len(set(zip(xs, ys))) != 4):
                parsed = [0.0, 0.0, 0.0, 0.0]
            else:
                parsed = [min(xs), min(ys), max(xs), max(ys)]

        # rescale from 1000 x 1000 to image size
        parsed[0] = parsed[0] * scale_x
        parsed[1] = parsed[1] * scale_y
        parsed[2] = parsed[2] * scale_x
        parsed[3] = parsed[3] * scale_y

        if len(parsed) == 4:
            if multilabel:
                out.append((str(label).strip(), parsed))
            else:
                out.append(parsed)
        else:
            incomplete_bbox += 1

    if incomplete_bbox > 0:
        print(f"Found {len(hits)} bboxes, {incomplete_bbox} incomplete.")
        format_adhered = False
    return out, format_adhered


def extract_json_bboxes(text, height, width, bbox_repr, json_key='bbox', json_structure='per_box', multilabel=False, no_rescale=False, unit_rescale=False, smart_resize=False, scene_max_size=None):
    FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    match = FENCE_RE.search(text)
    if match:
        text = match.group(1)

    # GLM-4.6V wraps structured answers in <|begin_of_box|>...<|end_of_box|> markers.
    BOX_TOKEN_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)
    for block in BOX_TOKEN_RE.findall(text):
        try:
            json.loads(block)
        except Exception:
            continue
        text = block
        break

    if not text:
        return [], False
    try:
        bboxes = json.loads(text)
    except Exception as e:
        print("JSON parsing error:", e, text)
        return [], False
    if not isinstance(bboxes, list):
        return [], False
    scale_x, scale_y = _get_scale(width, height, no_rescale, unit_rescale, smart_resize, scene_max_size)

    class_as_key = (multilabel and json_key == 'class_name')
    decomposed = (json_structure == 'decomposed')

    out = []
    skipped = False
    for bbox in bboxes:
        if not isinstance(bbox, dict):
            print(type(bbox), bbox)
            skipped = True
            continue

        if decomposed:
            keys = BBOX_COORDS[bbox_repr]
            try:
                parsed = [bbox[k] for k in keys]
            except KeyError as e:
                print(f"decomposed key {e} not found in {bbox}")
                skipped = True
                continue
        elif class_as_key:
            # Multilabel class-as-key: dict has one entry, key is the class label.
            if len(bbox) != 1:
                print("class-as-key dict must have exactly one entry:", bbox)
                skipped = True
                continue
            label_key, parsed = next(iter(bbox.items()))
            bbox = {"label": label_key, "_coords": parsed}
        else:
            if json_key not in bbox:
                print(json_key, "not found in", bbox)
                skipped = True
                continue
            parsed = bbox[json_key]

        expected_n = 8 if bbox_repr in ('all', 'all-labelled') else 4
        if not isinstance(parsed, (list, tuple)) or len(parsed) != expected_n:
            print(json_key, f"has wrong coord shape (expected list of {expected_n}):", parsed)
            skipped = True
            continue

        if bbox_repr in ('all', 'all-labelled'):
            xs, ys = parsed[0::2], parsed[1::2]
            if (len(set(xs)) != 2 or len(set(ys)) != 2
                    or len(set(zip(xs, ys))) != 4):
                parsed = [0.0, 0.0, 0.0, 0.0]
            else:
                parsed = [min(xs), min(ys), max(xs), max(ys)]
        elif bbox_repr == 'cxcywh':
            cx, cy, w, h = parsed
            parsed = [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]
        elif bbox_repr == 'xywh':
            x_min, y_min, w, h = parsed
            parsed = [x_min, y_min, x_min + w, y_min + h]
        elif bbox_repr == 'yxyx':
            y_min, x_min, y_max, x_max = parsed
            parsed = [x_min, y_min, x_max, y_max]
        elif bbox_repr == 'yxhw':
            y_min, x_min, h, w = parsed
            parsed = [x_min, y_min, x_min + w, y_min + h]

        try:
            parsed[0] = float(parsed[0]) * scale_x
            parsed[1] = float(parsed[1]) * scale_y
            parsed[2] = float(parsed[2]) * scale_x
            parsed[3] = float(parsed[3]) * scale_y
        except Exception as e:
            print("Error: ", e, "parsed:", parsed, "output:", bbox)
            skipped = True
            continue

        if multilabel:
            label = bbox.get("label", "")
            out.append((str(label).strip(), parsed))
        else:
            out.append(parsed)

    return out, not skipped


def compute_iou_matrix(gt_boxes, pred_boxes):
    if len(pred_boxes) == 0:
        return torch.zeros((0, len(gt_boxes)))
    if len(gt_boxes) == 0:
        return torch.zeros((len(pred_boxes), 0))
    gt = torch.tensor(gt_boxes)
    pred = torch.tensor(pred_boxes)
    # A GT box with swapped corners (RoboTools doc 143) gets IoU 0 with every
    # prediction, except 0/0 = NaN when the areas cancel exactly; keep that at 0.
    return torch.nan_to_num(box_iou(pred, gt), nan=0.0, posinf=0.0, neginf=0.0)


def hungarian_algorithm(iou_matrix, gt_boxes, pred_boxes, multilabel=False):
    _, G = iou_matrix.shape

    assert iou_matrix.ndim == 2

    if len(pred_boxes) == 0:
        return [0.0] * G, [0.0] * G, [0.0] * G, [False] * G

    cost_matrix = 1 - iou_matrix.numpy()
    label_matrix = None
    if multilabel:
        # pred_boxes look like [(bbox, label), ...]

        label_matrix = np.array([
            [1.0 if norm_string(pred_label) == norm_string(gt_label) else 0.0 for gt_label, _ in gt_boxes]
            for pred_label, _ in pred_boxes
        ])
        cost_matrix = cost_matrix - label_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    ious = [0.0] * G
    label_matched = [0.0] * G if multilabel else [1.0] * G
    assigned = [False] * G
    for r, c in zip(row_ind, col_ind):
        ious[c] = iou_matrix[r, c].item()
        if multilabel:
            label_matched[c] = label_matrix[r, c]
        assigned[c] = True
    assert len(ious) == G and len(label_matched) == G and len(assigned) == G

    return ious, cost_matrix.tolist(), label_matched, assigned
