"""Axis constants, dataset config, model rescale modes, task-names.

Implements the three benchmark axes from FindIt (paper Section 3):
    - task and data (`DATASET_CONFIG`)
    - bounding-box representation (`BBOX_COORDS` and the literals enumerated
      in `generate_tasks.ALL_BBOX`)
    - output format (text vs JSON; covered in `generate_tasks.ALL_JK` /
      `ALL_JS`).
"""

TASK_PREFIX = "FindIt"

BBOX_COORDS = {
    "xyxy":   ["xmin", "ymin", "xmax", "ymax"],
    "xywh":   ["x", "y", "width", "height"],
    "yxyx":   ["ymin", "xmin", "ymax", "xmax"],
    "yxhw":   ["y", "x", "height", "width"],
    "cxcywh": ["cx", "cy", "width", "height"],
}

DATASET_CONFIG = {
    "pascal":             {"task_family": "objdet",       "negative_examples": False, "supports": {"multi_label"}},
    "openimages":         {"task_family": "objdet",       "negative_examples": False, "supports": {"multi_label"}},
    "refcoco_test":       {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "refcoco_testA":      {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "refcoco_testB":      {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "refcoco_plus_testA": {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "refcoco_plus_testB": {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "refcoco_g_test":     {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "refl4":              {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "phrasecut":          {"task_family": "refexp_all",   "negative_examples": False, "supports": set()},
    "svg_relations":      {"task_family": "refexp",       "negative_examples": False, "supports": set()},
    "d3":                 {"task_family": "refexp_all",   "negative_examples": True,  "supports": set()},
    "flickr30k":          {"task_family": "refexp_all",   "negative_examples": False, "supports": {"with_caption"}},
    "hr_insdet_easy":     {"task_family": "insdet",       "negative_examples": False, "supports": set()},
    "hr_insdet_hard":     {"task_family": "insdet",       "negative_examples": False, "supports": set()},
    "iground":            {"task_family": "objdet_video", "negative_examples": False, "supports": {"multi_label", "with_caption"}},
    "robotools":          {"task_family": "insdet_video", "negative_examples": False, "supports": set()},
}

# coordinate-space assumption used to map model outputs back to pixel space.
MODEL_RESCALE = {
    "qwen2_5_vl":                   "smart_resize",
    "qwen3_vl":                     "standard",
    "qwen3_5":                      "standard",
    "qwen3_5_nothink":              "standard",
    "internvl_hf":                  "standard",
    "glm4v_nothink":                "standard",
    "gemma4":                       "standard",
    "openai/gpt-5.4":               "openai_high",    # pixel coords of OpenAI's resized image (default `high` detail: 2048 px, 2500 patches)
    "google/gemini-2.5-flash":      "standard",
    "anthropic/claude-4.5-sonnet":  "claude_standard", # pixel coords of Anthropic's resized image (1568 px / 1568 tokens)
    "openai/gpt-6-astra":           "no_rescale",     # pixel coords of the sent image; canary 2026-09-25
    "anthropic/claude-opus-5.5":    "claude_highres", # pixel coords of Anthropic's resized image (2576 px / 4784 tokens); canary 2026-09-25
}

# Qwen2.5-VL answers in pixels of its processor input, so the scorer must know the pixel budget and
# the lmms-eval interface the run used. The paper ran it (a) through the default *chat* interface with
# the lmms-eval default budget on object detection, referring expressions and iGround video, where
# qwen-vl-utils first rounds the image and the processor then applies min/max_pixels (two resizes),
# and (b) with `--force_simple` and max_pixels=12845056 on instance detection (HR-InsDet, RoboTools),
# where the image is resized once. Override with generate_tasks --qwen-max-pixels / --qwen-interface.
QWEN25_FRAME = {
    "chat":   {"qwen_max_pixels": 1_605_632, "qwen_min_pixels": 200_704, "qwen_two_step": True},
    "simple": {"qwen_max_pixels": 12_845_056, "qwen_min_pixels": 200_704, "qwen_two_step": False},
}

# How the answer to the unconstrained prompt (no format requested) is parsed: the model's own default
# scheme, keyed by (prompt output format, multi_label). Missing entries use the plain unconstrained
# parser (text: four numbers read as xyxy; JSON: xyxy boxes under the task's JSON key).
_JSON = lambda repr_, key, structure="per_box": {"parse_fmt": "json", "parse_bbox_repr": repr_,
                                                 "parse_json_key": key, "parse_json_structure": structure}
_ALL4 = lambda d: {("text", False): d, ("text", True): d, ("json", False): d, ("json", True): d}
UNCONSTRAINED_PARSE = {
    "qwen2_5_vl":                  {("text", False): _JSON("xyxy", "bbox_2d"), ("text", True): _JSON("xyxy", "bbox_2d")},
    "qwen3_vl":                    {("text", True): _JSON("xyxy", "bbox_2d")},
    "qwen3_5":                     {("text", True): _JSON("xyxy", "bbox_2d")},
    "qwen3_5_nothink":             {("text", False): _JSON("xyxy", "bbox_2d"), ("text", True): _JSON("xyxy", "bbox_2d")},
    "gemma4":                      _ALL4(_JSON("yxyx", "box_2d")),
    "google/gemini-2.5-flash":     _ALL4(_JSON("yxyx", "box_2d")),
    "anthropic/claude-4.5-sonnet": {("json", False): _JSON("xyxy", "decomposed", "decomposed"),
                                    ("json", True): _JSON("xyxy", "bbox")},
}


def model_task_kwargs(model, task_family, fmt, bbox_repr, multi_label, qwen_interface=None,
                      qwen_max_pixels=None, qwen_min_pixels=None):
    """Extra scorer kwargs a model needs in its task YAMLs (emitted by generate_tasks): the
    Qwen2.5-VL input frame and the unconstrained-answer parser. Defaults reproduce the paper."""
    out = {}
    if MODEL_RESCALE.get(model) == "smart_resize":
        interface = qwen_interface or ("simple" if task_family in ("insdet", "insdet_video") else "chat")
        out.update(QWEN25_FRAME[interface])
        if qwen_max_pixels:
            out["qwen_max_pixels"] = int(qwen_max_pixels)
        if qwen_min_pixels:
            out["qwen_min_pixels"] = int(qwen_min_pixels)
    if bbox_repr == "unconstrained":
        out.update(UNCONSTRAINED_PARSE.get(model, {}).get((fmt, bool(multi_label)), {}))
    return out

def build_task_name(*, dataset, task_family, fmt, bbox_repr,
                    json_structure=None, json_key=None,
                    multi_label=False, with_caption=False,
                    max_frames=None, center_exp="none"):
    parts = [TASK_PREFIX, dataset, task_family]
    if max_frames is not None:
        parts.append(f"f{max_frames}")
    parts += [fmt, bbox_repr]
    if fmt == "json":
        parts += [json_key, json_structure]
    if multi_label:
        parts.append("multilabel")
    if with_caption:
        parts.append("with_caption")
    if center_exp != "none":
        parts.append(f"cx_{center_exp}")
    return "_".join(parts)

def parse_task_name(name):
    tokens = name[len(TASK_PREFIX) + 1:].split("_")
    out = {
        "dataset": None, "task_family": None,
        "fmt": None, "bbox_repr": None,
        "json_structure": None, "json_key": None,
        "multi_label": False, "with_caption": False,
        "center_exp": "none", "max_frames": None,
    }

    if len(tokens) >= 2 and tokens[-2] == "cx" and tokens[-1] in ("definitions", "formula"):
        out["center_exp"] = tokens[-1]
        tokens = tokens[:-2]
    if tokens[-2:] == ["with", "caption"]:
        out["with_caption"] = True
        tokens = tokens[:-2]
    if tokens[-1:] == ["multilabel"]:
        out["multi_label"] = True
        tokens = tokens[:-1]

    # dataset name may span multiple underscore tokens (e.g. "refcoco_plus_testA");
    # task_family also spans (e.g. "objdet_video"). Walk longest prefix first.
    for i in range(len(tokens), 0, -1):
        candidate = "_".join(tokens[:i])
        if candidate in DATASET_CONFIG:
            tf = DATASET_CONFIG[candidate]["task_family"]
            tf_tokens = tf.split("_")
            if tokens[i:i + len(tf_tokens)] == tf_tokens:
                out["dataset"] = candidate
                out["task_family"] = tf
                tokens = tokens[i + len(tf_tokens):]
                break
    else:
        raise ValueError(f"no dataset prefix matches in {tokens}")

    if tokens and tokens[0].startswith("f") and tokens[0][1:].isdigit():
        out["max_frames"] = int(tokens[0][1:])
        tokens = tokens[1:]

    out["fmt"], out["bbox_repr"] = tokens[0], tokens[1]
    tokens = tokens[2:]

    if out["fmt"] == "json":
        if tokens[-2:] == ["per", "box"]:
            out["json_structure"] = "per_box"
            tokens = tokens[:-2]
        else:
            out["json_structure"] = tokens[-1]  # "nested"
            tokens = tokens[:-1]
        out["json_key"] = "_".join(tokens)

    return out
