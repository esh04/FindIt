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
    "openai/gpt-5.4":               "no_rescale",
    "google/gemini-2.5-flash":      "standard",
    "anthropic/claude-4.5-sonnet":  "no_rescale",
}

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
