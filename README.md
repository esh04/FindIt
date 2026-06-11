# [FindIt: A Format-Informed Visual Detection Benchmark for Generalist Multimodal LLMs](https://esh04.github.io/FindIt/)

_[Eshika Khandelwal](https://esh04.github.io/)<sup>1</sup>, [Jingjing Pan](https://www.linkedin.com/in/jingjing-pan)<sup>2</sup>, [Mingfang Zhang](https://mf-zhang.github.io/)<sup>2</sup>, [Quan Kong](https://fusionk.github.io/quankong/)<sup>2</sup>, [Lorenzo Garattoni](https://www.linkedin.com/in/lorenzo-garattoni/)<sup>3</sup>, and [Hilde Kuehne](https://hildekuehne.github.io/)<sup>1</sup>_

<sup>1</sup> Tübingen AI Center, University of Tübingen,
<sup>2</sup> Woven by Toyota, Inc., Tokyo, Japan,
<sup>3</sup> Toyota Motor Europe, Brussels, Belgium.

[![arXiv](https://img.shields.io/badge/arXiv-2606.04282-b31b1b)](https://arxiv.org/abs/2606.04282) [![Project](https://img.shields.io/badge/Project-Website-blue)](https://esh04.github.io/FindIt/) [![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/esh04/FindIt)

Code release for *FindIt: A Format-Informed Visual Detection Benchmark for Generalist Multimodal LLMs*.

FindIt evaluates the promptable bounding-box localization ability of
generalist MLLMs across four task families:

1. **Object detection** (single- and multi-label) — Pascal VOC, OpenImages V7,
   iGround.
2. **Referring expression detection** — RefCOCO, RefCOCO+, RefCOCO-g, RefL4,
   D3, PhraseCut, Flickr30k Entities, Synthetic Visual Genome.
3. **Instance detection** (visual support image instead of a text query) —
   HR-InsDet (easy and hard) and RoboTools.
4. **Video detection** (multi-frame extension) — iGround as multi-frame object
   detection, RoboTools as multi-frame instance detection.

For every (model, dataset) cell, the benchmark sweeps two further axes
(Section 3.2 / 3.3 of the paper):

- **Bounding-box representation** — `xyxy`, `xywh`, `yxyx`, `yxhw`, `cxcywh`,
  `all`, `all-labelled`, plus `unconstrained` which omits any format
  specification from the prompt.
- **Output format** — plain text vs JSON. JSON variants additionally sweep
  the dictionary key (`bbox`, `bbox_2d`, `coordinates`, `bounding_box`, and
  `class_name` for multi-label queries). All paper results use the per-box
  structure (one JSON object per detected instance). A nested variant that
  collects all boxes under a single list key is also available via
  `--json-structure nested` but did not work well in practice.

The evaluation reports each model at the `(bbox_repr, output_format,
json_key, json_structure)` combination that maximizes its average F1@0.5,
chosen via the multi-stage format search described in Section 4 of the paper.

## Metrics

Predictions are matched to ground truth by Hungarian assignment with cost
`1 - IoU` (single-label) or `1 - IoU - 1[label matches]` (multi-label, so
label agreement dominates IoU). On the matched pairs we report:

- **F1@0.5** — a prediction counts as true positive when its IoU with the
  matched ground-truth box is `>= 0.5`; multi-label queries additionally
  require the predicted and ground-truth labels to agree.
- **mIoU** — mean IoU averaged over all ground-truth boxes, where an
  unmatched ground-truth box contributes an IoU of 0.
- **Format Adherence (FA)** — fraction of responses that parse as the
  prompted format. Non-parseable responses are scored as an empty prediction
  list, which lowers precision/recall.

No post-processing is applied to model outputs; we score what the model emits directly.

## Repository layout

    configs/                   # YAML configs (edit paths before use)
        datasets.yaml          # per-dataset image-folder paths
    src/findit/                # Python package
        axes.py                # task-name builder, dataset registry, rescale modes
        promptloader.py        # prompts for every (task, fmt, bbox_repr, json_key)
        dataset.py             # parquet -> in-memory loaders, one per dataset
        prepare_datasets.py    # build the on-disk Arrow datasets lmms-eval reads
        generate_tasks.py      # emit per-variant lmms-eval task YAMLs
        evaluate.py            # bbox extraction, Hungarian matching, metrics
        doc_utils.py           # lmms-eval hooks (doc_to_visual / process_results)
        task_templates/        # shared lmms-eval template
        plugin/                # custom model wrappers
            models/
                gemma4.py            # Gemma 4 with positional <image N> binding
                glm4v_nothink.py     # GLM-4.6V with thinking disabled
                qwen3_5_nothink.py   # Qwen3.5-VL with thinking disabled
                openrouter.py        # OpenRouter-routed proprietary models
                _interleave.py       # shared helper for <image N> placeholders
    subsets/                   # 1,000-query parquets (one per dataset variant)
    data/README.md             # where to obtain the source images
    LICENSE
    pyproject.toml

## Installation

    pip install -e .

This installs `findit` and all dependencies including
[`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Datasets

`subsets/*.parquet` ships the 1,000-query subsets used for every result in
the paper. You still need the source images. Edit
`configs/datasets.yaml` so each `images:` / `scenes:` / `support:` /
`frames:` value points to your local copy. See `data/README.md` for download
URLs.

After paths are set, build the on-disk Arrow datasets that lmms-eval consumes:

    python -m findit.prepare_datasets
    # or only a subset:
    python -m findit.prepare_datasets --only pascal,refcoco_test,hr_insdet_easy

Built datasets land under `src/findit/built_datasets/` (gitignored).

RefCOCO/+/g and RefL4 are pulled directly from HuggingFace at build time, so
no local image folder is required for those.

## Running the benchmark

`findit.generate_tasks` emits one `lmms-eval` task YAML per variant of the
benchmark grid (task family x bbox representation x output format x JSON
key x JSON structure x frame count). Example: object detection across three
datasets (emits 6 task YAMLs — 3 datasets × 2 output formats):

    python -m findit.generate_tasks \
        --model qwen3_vl \
        --dataset pascal,openimages,iground \
        --fmt text,json \
        --bbox-repr xyxy \
        --json-key bbox_2d \
        --json-structure per_box \
        --out src/findit/tasks/qwen3_vl \
        --group-name FindIt_objdet_xyxy

Then run lmms-eval with the same model and matching task directory:

    lmms-eval \
        --include_path src/findit/tasks/qwen3_vl \
        --tasks FindIt_objdet_xyxy \
        --model qwen3_vl \
        --output_path ./Outputs/qwen3_vl_objdet_xyxy


### Image resolution (`max_pixels`)

`lmms-eval` Qwen models cap input images at `max_pixels=1605632`
(~1.6 MP). For the high-resolution images like HR-InsDet
(up to 50 MP) and RoboTools (~2 MP) — the paper raises this cap to
`max_pixels=12845056` (12.8 MP), which is the intended default for any
large-image run. On HR-InsDet you must additionally pass
`interleave_visuals=True` so the `<image N>` support/scene placeholders bind:

    lmms-eval \
        --include_path src/findit/tasks/qwen3_vl \
        --tasks FindIt_hr_insdet_hard_xyxy \
        --model qwen3_vl \
        --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_pixels=12845056,interleave_visuals=True \
        --output_path ./Outputs/qwen3_vl_hr_insdet_hard

`max_pixels` is only an upper cap: when `width*height` is already below it the image is untouched.

> **Qwen2.5-VL.** `smart_resize` scoring inverts the resize assuming
> `max_pixels=12845056`. On images above ~1.6 MP, use the same value in
> `--model_args` at inference, or boxes will be mis-scored.

To reproduce the multi-stage format search from Section 4 (50 queries for
open-source models, 20 for proprietary models):

    # Stage 1: sweep bbox representations on Pascal at the default JSON key.
    python -m findit.generate_tasks \
        --model qwen3_vl --dataset pascal \
        --fmt text,json --bbox-repr all --json-key bbox_2d \
        --out src/findit/tasks/qwen3_vl \
        --group-name FindIt_stage1
    lmms-eval \
        --include_path src/findit/tasks/qwen3_vl \
        --tasks FindIt_stage1 \
        --model qwen3_vl \
        --output_path ./Outputs/qwen3_vl_stage1 \
        --limit 50

    # Stage 2: pin the winning bbox representation from Stage 1 (e.g. xyxy) and sweep JSON keys.
    python -m findit.generate_tasks \
        --model qwen3_vl --dataset pascal --append \
        --fmt json --bbox-repr xyxy --json-key all \
        --out src/findit/tasks/qwen3_vl \
        --group-name FindIt_stage2
    lmms-eval \
        --include_path src/findit/tasks/qwen3_vl \
        --tasks FindIt_stage2 \
        --model qwen3_vl \
        --output_path ./Outputs/qwen3_vl_stage2 \
        --limit 50

## Models

The paper evaluates six open-source models — Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL
(both with and without reasoning), InternVL3, Gemma 4, GLM-4.6V — and
three proprietary models routed through OpenRouter — GPT-5.4, Claude
Sonnet 4.5, Gemini 2.5 Flash. Identifiers and rescale modes are declared in
`src/findit/axes.py::MODEL_RESCALE`. Custom `lmms-eval` model wrappers live
under `src/findit/plugin/models/`:

| Wrapper | Purpose |
|---|---|
| `gemma4.Gemma4` | Reuses `lmms-eval`'s Gemma3 loop with the Gemma4 HF class; patches `apply_chat_template` so `<image N>` placeholders bind positionally. |
| `glm4v_nothink.GLM4VNoThink` | GLM-4.6V with `enable_thinking=False`. |
| `qwen3_5_nothink.Qwen3_5NoThink` | Qwen3.5-VL run without the `<think>...</think>` block. |
| `openrouter.OpenRouterNoThink` | OpenRouter-routed proprietary models (Claude / GPT / Gemini); disables reasoning, JPEG-encodes images >= 2048 px on the longest side, and pre-resizes scenes > 4096 px to keep payloads under provider size caps. |

Coordinate-space assumptions per model are recorded in `MODEL_RESCALE` and
matched in `evaluate._get_scale`:

- `standard` — model returns coordinates in a 1000x1000 normalized grid.
- `no_rescale` — model returns pixel coordinates of the input image.
- `unit_rescale` — model returns coordinates in `[0, 1]`.
- `smart_resize` — Qwen2.5-VL: pixel coordinates of the smart-resized image
  (`max_pixels=12.8M`, `factor=28`); we invert the resize to recover original
  pixel space before scoring.

GLM-4.6V wraps every box in `<|begin_of_box|>...<|end_of_box|>` and repeats
the block; `evaluate.extract_*_bboxes` reads only the first block, matching
the paper's protocol (Section 4, "Output parsing").

## Outputs

`lmms-eval` writes its run logs and per-sample predictions to whatever path
you pass via `--output_path`. The five aggregated metrics
(`format_adherence`, `mean_iou`, `precision_at_05`, `recall_at_05`,
`f1_at_05`) are produced by the aggregators in
`findit.doc_utils` and reported by `lmms-eval` as the run summary.

## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{khandelwal2026findit,
  title   = {FindIt: A Format-Informed Visual Detection Benchmark for Generalist Multimodal {LLMs}},
  author  = {Khandelwal, Eshika and Pan, Jingjing and Zhang, Mingfang
             and Kong, Quan and Garattoni, Lorenzo and Kuehne, Hilde},
  journal = {arXiv},
  year    = {2026}
}
```

If you run into any issues setting up or running the benchmark, feel free to open a GitHub issue or reach out via [email](mailto:eshikak0412@gmail.com).

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See
[LICENSE](LICENSE).
