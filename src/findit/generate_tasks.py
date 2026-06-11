"""Generate lmms-eval variant YAMLs + a group YAML on demand.

One variant YAML per cell of the FindIt grid (paper Section 3): task family x
dataset x bbox representation x output format x JSON key x JSON structure (and
frame count for video). The two-stage format search described in Section 4 is
driven by sweeping ALL_BBOX in stage 1 and ALL_JK in stage 2; concrete model
sweeps are wired up by the caller via `--bbox-repr all` / `--json-key all`.
"""
import argparse
import shutil
from pathlib import Path

import yaml

from .axes import DATASET_CONFIG, MODEL_RESCALE, build_task_name

HERE = Path(__file__).resolve().parent
TEMPLATE_SRC = HERE / "task_templates" / "_default_template.yaml"
BUILT_DIR = HERE / "built_datasets"

ALL_BBOX = ["unconstrained", "xyxy", "xywh", "yxyx", "yxhw", "cxcywh", "all", "all-labelled"]
ALL_JK = ["bbox", "bbox_2d", "coordinates", "bounding_box", "class_name"]
ALL_JS = ["per_box", "nested"]


def _built_path(dataset, tf, multi_label):
    name = f"{dataset}_{tf}" + ("_multilabel" if multi_label else "")
    return str(BUILT_DIR / name)


def enumerate_variants(dataset, fmts, bbox_reprs, json_keys, json_structures,
                       multi_label, with_caption, max_frames, center_exps,
                       scene_max_size=None):
    cfg = DATASET_CONFIG[dataset]
    tf = cfg["task_family"]
    frame_options = max_frames if tf.endswith("_video") else [None]

    for fmt in fmts:
        for br in bbox_reprs:
            for nf in frame_options:
                for ce in center_exps:
                    if ce != "none" and br != "cxcywh":
                        continue
                    base = dict(
                        dataset=dataset, task_family=tf, fmt=fmt, bbox_repr=br,
                        multi_label=multi_label, with_caption=with_caption,
                        max_frames=nf, center_exp=ce,
                        negative_examples=cfg["negative_examples"],
                        scene_max_size=scene_max_size,
                    )
                    if fmt == "text":
                        yield {**base, "json_key": None, "json_structure": None}
                    else:
                        for jk in json_keys:
                            if jk == "class_name" and not multi_label:
                                continue
                            for js in json_structures:
                                yield {**base, "json_key": jk, "json_structure": js}


def _max_new_tokens(task_family, max_frames):
    frames = max_frames if (task_family.endswith("_video") and max_frames) else 1
    return max(768, 192 * frames)


def emit_variant(variant, out_dir, rescale_mode):
    name = build_task_name(**{k: variant[k] for k in (
        "dataset", "task_family", "fmt", "bbox_repr", "json_structure", "json_key",
        "multi_label", "with_caption", "max_frames", "center_exp",
    )})
    kwargs = {
        "dataset_name":      variant["dataset"],
        "task_family":       variant["task_family"],
        "fmt":               variant["fmt"],
        "bbox_repr":         variant["bbox_repr"],
        "multi_label":       variant["multi_label"],
        "with_caption":      variant["with_caption"],
        "negative_examples": variant["negative_examples"],
        "center_exp":        variant["center_exp"],
        "rescale_mode":      rescale_mode,
    }
    if variant["fmt"] == "json":
        kwargs["json_key"] = variant["json_key"]
        kwargs["json_structure"] = variant["json_structure"]
    if variant["max_frames"] is not None:
        kwargs["max_frames"] = variant["max_frames"]
    if variant.get("scene_max_size") is not None:
        kwargs["scene_max_size"] = variant["scene_max_size"]

    data = {
        "include": "_default_template.yaml",
        "task": name,
        "dataset_path": _built_path(variant["dataset"], variant["task_family"], variant["multi_label"]),
        "generation_kwargs": {
            "max_new_tokens": _max_new_tokens(variant["task_family"], variant["max_frames"]),
            "temperature": 0,
            "do_sample": False,
        },
        "lmms_eval_specific_kwargs": {"default": kwargs},
    }
    (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return name


def _commasep(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--fmt", default="text,json")
    p.add_argument("--bbox-repr", default="xyxy")
    p.add_argument("--json-key", default="bbox")
    p.add_argument("--json-structure", default="per_box")
    p.add_argument("--max-frames", default=None)
    p.add_argument("--multi-label", action="store_true")
    p.add_argument("--with-caption", action="store_true")
    p.add_argument("--include", default="")
    p.add_argument("--scene-max-size", type=int, default=None,
                   help="Resize the query/scene image (and video frames) to this max-side "
                        "before passing to the model. Support shots are not resized.")
    p.add_argument("--group-name", default="FindIt_stage")
    p.add_argument("--out", default=str(HERE / "tasks"))
    p.add_argument("--append", action="store_true")
    args = p.parse_args()

    rescale_mode = MODEL_RESCALE[args.model]
    fmts = _commasep(args.fmt)
    bbox_reprs = ALL_BBOX if args.bbox_repr == "all" else _commasep(args.bbox_repr)
    json_keys = ALL_JK if args.json_key == "all" else _commasep(args.json_key)
    json_structures = ALL_JS if args.json_structure == "all" else _commasep(args.json_structure)
    max_frames = [int(x) for x in _commasep(args.max_frames)] if args.max_frames else [1]
    center_exps = (
        ["none", "definitions", "formula"]
        if "center-exp" in args.include else ["none"]
    )

    out_dir = Path(args.out)
    if not args.append:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE_SRC, out_dir / "_default_template.yaml")

    task_names = []
    for dataset in _commasep(args.dataset):
        for v in enumerate_variants(dataset, fmts, bbox_reprs, json_keys, json_structures,
                                    args.multi_label, args.with_caption, max_frames, center_exps,
                                    scene_max_size=args.scene_max_size):
            task_names.append(emit_variant(v, out_dir, rescale_mode))

    group_path = out_dir / f"{args.group_name}.yaml"
    if args.append and group_path.exists():
        existing = yaml.safe_load(group_path.read_text()).get("task", [])
        merged = list(dict.fromkeys(existing + task_names))
    else:
        merged = task_names
    group_path.write_text(yaml.safe_dump({"group": args.group_name, "task": merged}, sort_keys=False))

    print(f"emitted {len(task_names)} variants; group '{args.group_name}' has {len(merged)} tasks → {out_dir}")


if __name__ == "__main__":
    main()
