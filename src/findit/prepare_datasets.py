"""Build on-disk Arrow datasets from the frozen Parquet subsets."""
import argparse
import os
import shutil

import yaml
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

from . import dataset as _ds

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
BUILT_DIR = os.path.join(HERE, "built_datasets")
DATASETS_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "datasets.yaml")
DUMMY_PROMPT = "{label}"

_PIL_FEATURES = Features({
    "filename": Value("string"),
    "label":    Value("string"),
    "image":    Image(),
    "bboxes":   Sequence(Sequence(Value("float32"))),
    "height":   Value("int32"),
    "width":    Value("int32"),
})


def save_dataset(ds, name):
    out = os.path.join(BUILT_DIR, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(BUILT_DIR, exist_ok=True)
    DatasetDict({"test": ds}).save_to_disk(out)
    print(f"  saved: {out} ({len(ds)} rows)")


BUILDERS = {}
def register(name):
    def _wrap(fn):
        BUILDERS[name] = fn
        return fn
    return _wrap


def _build_simple(loader_fn, image_dir, out_name, multi_label=False):
    kw = dict(image_dir=image_dir, prompt=DUMMY_PROMPT)
    if multi_label:
        kw["multi_label"] = True
    prompts, bboxes, filenames, labels, heights, widths = loader_fn(**kw)
    image_paths = [p[1] for p in prompts]

    rows = []
    for fn, lbl, ip, bb, h, w in zip(filenames, labels, image_paths, bboxes, heights, widths):
        entry = {"filename": fn, "label": lbl, "image_path": ip,
                 "height": int(h), "width": int(w)}
        if multi_label:
            entry["bboxes"] = [{"label": l_, "bbox": [float(x) for x in b_]} for (l_, b_) in bb]
        else:
            entry["bboxes"] = [[float(x) for x in b] for b in bb]
        rows.append(entry)

    save_dataset(Dataset.from_list(rows), out_name)


def _save_pil(prompts, bboxes, filenames, labels, heights, widths, out_name):
    pil_images = [p[1] for p in prompts]
    rows = [
        {"filename": fn, "label": lbl, "image": img,
         "bboxes": [[float(x) for x in b] for b in bb],
         "height": int(h), "width": int(w)}
        for fn, lbl, img, bb, h, w in zip(filenames, labels, pil_images, bboxes, heights, widths)
    ]
    save_dataset(Dataset.from_list(rows, features=_PIL_FEATURES), out_name)


@register("pascal")
def _pascal(cfg): _build_simple(_ds.load_pascal, cfg["pascal"]["images"], "pascal_objdet")

@register("pascal_multilabel")
def _pascal_ml(cfg): _build_simple(_ds.load_pascal, cfg["pascal"]["images"], "pascal_objdet_multilabel", multi_label=True)

@register("openimages")
def _openimages(cfg): _build_simple(_ds.load_open_images, cfg["openimages"]["images"], "openimages_objdet")

@register("openimages_multilabel")
def _openimages_ml(cfg): _build_simple(_ds.load_open_images, cfg["openimages"]["images"], "openimages_objdet_multilabel", multi_label=True)

@register("d3")
def _d3(cfg): _build_simple(_ds.load_d3, cfg["d3"]["images"], "d3_refexp_all")

@register("phrasecut")
def _phrasecut(cfg): _build_simple(_ds.load_phrasecut, cfg["phrasecut"]["images"], "phrasecut_refexp_all")

@register("svg_relations")
def _svg_relations(cfg): _build_simple(_ds.load_svg_relations, cfg["svg_relations"]["images"], "svg_relations_refexp")


@register("flickr30k")
def _flickr30k(cfg):
    prompts, bboxes, filenames, labels, heights, widths = _ds.load_flickr30k(
        image_dir=cfg["flickr30k"]["images"], prompt=DUMMY_PROMPT,
    )
    image_paths = [p[1] for p in prompts]
    captions = _ds._read("flickr30k")["caption"].tolist()
    rows = [
        {"filename": fn, "label": lbl, "image_path": ip,
         "bboxes": [[float(x) for x in b] for b in bb],
         "caption": cap, "height": int(h), "width": int(w)}
        for fn, lbl, ip, bb, h, w, cap in zip(filenames, labels, image_paths, bboxes, heights, widths, captions)
    ]
    save_dataset(Dataset.from_list(rows), "flickr30k_refexp_all")


def _build_refcoco(hf_path, split, out_name):
    _save_pil(*_ds.load_refcoco(hf_path=hf_path, prompt=DUMMY_PROMPT, split=split), out_name)


@register("refcoco_test")
def _rc_t(cfg):  _build_refcoco(cfg["refcoco"]["hf"], "test",  "refcoco_test_refexp")
@register("refcoco_testA")
def _rc_tA(cfg): _build_refcoco(cfg["refcoco"]["hf"], "testA", "refcoco_testA_refexp")
@register("refcoco_testB")
def _rc_tB(cfg): _build_refcoco(cfg["refcoco"]["hf"], "testB", "refcoco_testB_refexp")
@register("refcoco_plus_testA")
def _rcp_tA(cfg): _build_refcoco(cfg["refcoco_plus"]["hf"], "testA", "refcoco_plus_testA_refexp")
@register("refcoco_plus_testB")
def _rcp_tB(cfg): _build_refcoco(cfg["refcoco_plus"]["hf"], "testB", "refcoco_plus_testB_refexp")
@register("refcoco_g_test")
def _rcg_t(cfg):  _build_refcoco(cfg["refcoco_g"]["hf"], "test", "refcoco_g_test_refexp")


@register("refl4")
def _refl4(cfg):
    _save_pil(*_ds.load_refl4(hf_path=cfg["refl4"]["hf"], prompt=DUMMY_PROMPT), "refl4_refexp")


def _build_hr_insdet(cfg, difficulty, out_name):
    scenes_dir = os.path.join(cfg["hr_insdet"]["scenes"], difficulty)
    prompts, bboxes, filenames, labels, heights, widths = _ds.load_hr_insdet(
        scenes_dir=scenes_dir, support_dir=cfg["hr_insdet"]["support"],
        prompt=DUMMY_PROMPT, difficulty=difficulty,
    )
    rows = [
        {"filename": fn, "label": lbl, "image_path": p[2],
         "support_shots": list(p[1]),
         "bboxes": [[float(x) for x in b] for b in bb],
         "height": int(h), "width": int(w)}
        for fn, lbl, p, bb, h, w in zip(filenames, labels, prompts, bboxes, heights, widths)
    ]
    save_dataset(Dataset.from_list(rows), out_name)


@register("hr_insdet_easy")
def _hr_e(cfg): _build_hr_insdet(cfg, "easy", "hr_insdet_easy_insdet")
@register("hr_insdet_hard")
def _hr_h(cfg): _build_hr_insdet(cfg, "hard", "hr_insdet_hard_insdet")


@register("robotools")
def _robotools(cfg):
    prompts, bboxes, filenames, labels, heights, widths = _ds.load_robotools_video(
        prompt="{num_frames}",
        support_dir=cfg["robotools"]["support"],
        frames_dir=cfg["robotools"]["frames"],
        max_frames=64,
    )
    rows = [
        {"filename": fn, "label": lbl,
         "support_shots": list(p[1]),
         "frame_paths": list(p[2]),
         "frame_indices": [int(x) for x in p[3]],
         "total_scene_frames": int(p[4]),
         "bboxes": [[[float(x) for x in b] for b in frame] for frame in bb],
         "height": int(h), "width": int(w)}
        for fn, lbl, p, bb, h, w in zip(filenames, labels, prompts, bboxes, heights, widths)
    ]
    save_dataset(Dataset.from_list(rows), "robotools_insdet_video")


def _build_iground(cfg, multi_label, out_name):
    prompts, bboxes, filenames, labels, heights, widths = _ds.load_iground(
        prompt="{label} {num_frames}",
        frames_dir=cfg["iground"]["frames"],
        num_sample_frames=8,
        multi_label=multi_label,
        with_caption=True,
    )
    captions = _ds._read("iground_multilabel" if multi_label else "iground")["caption"].tolist()

    rows = []
    for fn, lbl, p, bb, h, w, cap in zip(filenames, labels, prompts, bboxes, heights, widths, captions):
        entry = {
            "filename": fn, "label": lbl,
            "frame_paths": list(p[1]), "caption": cap,
            "height": int(h), "width": int(w),
        }
        if multi_label:
            entry["bboxes"] = [
                [{"label": l_, "bbox": [float(x) for x in b_]} for (l_, b_) in frame]
                for frame in bb
            ]
        else:
            entry["bboxes"] = [[[float(x) for x in b] for b in frame] for frame in bb]
        rows.append(entry)
    save_dataset(Dataset.from_list(rows), out_name)




@register("iground")
def _iground(cfg): _build_iground(cfg, False, "iground_objdet_video")
@register("iground_multilabel")
def _iground_ml(cfg): _build_iground(cfg, True, "iground_objdet_video_multilabel")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated dataset names (default: all)")
    args = p.parse_args()

    with open(DATASETS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    names = args.only.split(",") if args.only else list(BUILDERS.keys())
    for name in names:
        print(f"building {name} ...")
        BUILDERS[name](cfg)


if __name__ == "__main__":
    main()
