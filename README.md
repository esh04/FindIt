# [FindIt: A Format-Informed Visual Detection Benchmark for Generalist Multimodal LLMs](https://esh04.github.io/FindIt/)

*[Eshika Khandelwal](https://esh04.github.io/)1, [Jingjing Pan](https://www.linkedin.com/in/jingjing-pan)2, [Mingfang Zhang](https://mf-zhang.github.io/)2, [Quan Kong](https://fusionk.github.io/quankong/)2, [Lorenzo Garattoni](https://www.linkedin.com/in/lorenzo-garattoni/)3, and [Hilde Kuehne](https://hildekuehne.github.io/)1*

1 Tübingen AI Center, University of Tübingen,
2 Woven by Toyota, Inc., Tokyo, Japan,
3 Toyota Motor Europe, Brussels, Belgium.

![arXiv](https://img.shields.io/badge/arXiv-2606.04282-b31b1b) ![Project](https://img.shields.io/badge/Project-Website-blue) ![GitHub](https://img.shields.io/badge/Code-GitHub-black)

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

The evaluation reports each model at the `(bbox_repr, output_format, json_key, json_structure)` combination that maximizes its average F1@0.5,
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
- **Format Adherence (FA)** — fraction of responses that parse completely in
the prompted format. A malformed box or JSON entry makes a response
non-adherent, but its well-formed boxes are still scored; a response without
any parsable box counts as an empty prediction list, which lowers recall.

No post-processing is applied to model outputs; we score what the model emits directly.

## Repository layout

```
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
```



## Installation

```
pip install -e .
```

This installs `findit` and all dependencies including
`[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)`.

### Reproducibility

The paper results were produced with the stack below; match it for exact
reproduction:

```
Python 3.11.15
torch 2.11.0 (CUDA 13.0)   torchvision 0.26.0
transformers 5.6.0   qwen-vl-utils 0.0.14   lmms-eval 0.7.1
numpy 2.4.4   scipy 1.17.1   Pillow 12.2.0   decord 0.6.0
```



## Datasets

`subsets/*.parquet` ships the 1,000-query subsets used for every result in
the paper. You still need the source images. Edit
`configs/datasets.yaml` so each `images:` / `scenes:` / `support:` /
`frames:` value points to your local copy. See `data/README.md` for download
URLs.

After paths are set, build the on-disk Arrow datasets that lmms-eval consumes:

```
python -m findit.prepare_datasets
# or only a subset:
python -m findit.prepare_datasets --only pascal,refcoco_test,hr_insdet_easy
```

Built datasets land under `src/findit/built_datasets/` (gitignored).

RefCOCO/+/g and RefL4 are pulled directly from HuggingFace at build time, so
no local image folder is required for those.

## Running the benchmark

`findit.generate_tasks` emits one `lmms-eval` task YAML per variant of the
benchmark grid (task family x bbox representation x output format x JSON
key x JSON structure x frame count). Example: object detection across three
datasets (emits 6 task YAMLs — 3 datasets × 2 output formats):

```
python -m findit.generate_tasks \
    --model qwen3_vl \
    --dataset pascal,openimages,iground \
    --fmt text,json \
    --bbox-repr xyxy \
    --json-key bbox_2d \
    --json-structure per_box \
    --out src/findit/tasks/qwen3_vl \
    --group-name FindIt_objdet_xyxy
```

Then run lmms-eval with the same model and matching task directory:

```
lmms-eval \
    --include_path src/findit/tasks/qwen3_vl \
    --tasks FindIt_objdet_xyxy \
    --model qwen3_vl \
    --output_path ./Outputs/qwen3_vl_objdet_xyxy
```



### Image resolution (`max_pixels`)

`lmms-eval` Qwen models cap input images at `max_pixels=1605632`
(~1.6 MP). For the high-resolution images like HR-InsDet
(up to 50 MP) and RoboTools (~2 MP) — the paper raises this cap to
`max_pixels=12845056` (12.8 MP), which is the intended default for any
large-image run. On instance-detection tasks you must additionally pass
`interleave_visuals=True` so the `<image N>` support/scene placeholders bind,
**and** `--force_simple` so the flag actually takes effect (see the note
below):

```
lmms-eval \
    --include_path src/findit/tasks/qwen3_vl \
    --tasks FindIt_hr_insdet_hard_xyxy \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_pixels=12845056,interleave_visuals=True \
    --force_simple \
    --output_path ./Outputs/qwen3_vl_hr_insdet_hard   # --force_simple required for Qwen, see below
```

> `--force_simple` **is required for Qwen instance detection.** Recent
> `lmms-eval` (>= 0.7) routes `qwen2_5_vl` / `qwen3_vl` to the *chat* model
> interface by default. On that interface `interleave_visuals` is **silently
> ignored** (the message is built images-first and `<image N>` is left as
> literal text) **and** per-image `max_pixels` is dropped by the `ChatMessages`
> protocol.
> `--force_simple` forces the *simple* interface, where both `interleave_visuals`
> and `max_pixels` are used. It is a no-op for models without a simple class
> (e.g. `internvl_hf`, which interleaves through a separate plugin path) and for
> Qwen3.5-VL (simple-only already).

`max_pixels` is only an upper cap: when `width*height` is already below it the image is untouched.

> **Qwen2.5-VL** answers in pixels of its processor input, so the scorer has
> to invert exactly the resize the run used. `generate_tasks` writes it into
> each task (`qwen_max_pixels`, `qwen_min_pixels`, `qwen_two_step`). The
> defaults reproduce the paper: instance detection (HR-InsDet, RoboTools)
> assumes `--force_simple` with `max_pixels=12845056` (one resize), all other
> tasks assume the default *chat* interface with the `lmms-eval` defaults
> `max_pixels=1605632` / `min_pixels=200704`, where `qwen-vl-utils` first
> rounds the image to multiples of 28 and the processor then resizes it again.
> If you run differently, pass the same settings to `generate_tasks`
> (`--qwen-interface chat|simple`, `--qwen-max-pixels N`, `--qwen-min-pixels N`),
> or boxes will be mis-scored.

To reproduce the multi-stage format search from Section 4 (50 queries for
open-source models, 20 for proprietary models):

```
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
```

## Reproducing the paper's numbers

`python -m findit.generate_tasks --model <name>` writes everything the scorer
needs into the task YAMLs: the model's coordinate frame (`axes.MODEL_RESCALE`),
the Qwen2.5-VL input frame (see above) and, for the unconstrained prompt, the
model's own answer scheme (`axes.UNCONSTRAINED_PARSE`, e.g. JSON `yxyx` boxes
under `box_2d` for Gemini 2.5 Flash and Gemma-4). The paper runs used the
defaults except for the settings below.

| Model (`--model`) | Tasks | Additional settings |
|---|---|---|
| `qwen2_5_vl`, `qwen3_vl` | HR-InsDet, RoboTools | `max_pixels=12845056,interleave_visuals=True`, `--force_simple` |
| `qwen3_5`, `qwen3_5_nothink` | HR-InsDet, RoboTools | `max_pixels=12845056,interleave_visuals=True` |
| `internvl_hf` | HR-InsDet, RoboTools | `max_patches=64` |
| `gemma4` | RoboTools (2 and 8 frames) | `interleave_visuals=True` |
| `openrouter` (all models) | HR-InsDet | `generate_tasks --scene-max-size 4096` |
| `openrouter` (GPT-6 Astra, Opus 5.5) | all | `reasoning=default,max_output_tokens=8000` |
| `openrouter` (GPT-6 Astra) | HR-InsDet | `generate_tasks --scene-max-size 1824` (instead of 4096), `max_image_patches=2500,max_image_side=2048` |

The proprietary models answer in pixels of the image the provider feeds to the
model; the scorer maps these answers back through the provider's documented
resize (GPT-5.4: OpenAI's default `high` detail; Claude: Anthropic's resize
tiers) and the plugin's 4096-px cap for larger images. The format-search probes
(Stage 1 and 2) use `--limit 50` for open-source and `--limit 20` for
proprietary models.

## Models

The paper evaluates six open-source models — Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL
(both with and without reasoning), InternVL3, Gemma 4, GLM-4.6V — and five
proprietary models routed through OpenRouter — GPT-5.4, Claude Sonnet 4.5,
Gemini 2.5 Flash, GPT-6 Astra, Claude Opus 5.5. Identifiers and rescale modes
are declared in `src/findit/axes.py::MODEL_RESCALE`. Custom `lmms-eval` model
wrappers live under `src/findit/plugin/models/`:

| Wrapper | Purpose |
|---|---|
| `gemma4.Gemma4` | Reuses `lmms-eval`'s Gemma3 loop with the Gemma4 HF class; patches `apply_chat_template` so `<image N>` placeholders bind positionally. |
| `glm4v_nothink.GLM4VNoThink` | GLM-4.6V with `enable_thinking=False`. |
| `qwen3_5_nothink.Qwen3_5NoThink` | Qwen3.5-VL run without the `<think>...</think>` block. |
| `openrouter.OpenRouterNoThink` | OpenRouter-routed proprietary models; disables reasoning unless `reasoning=default`, JPEG-encodes images >= 2048 px on the longest side, and pre-resizes images > 4096 px to keep payloads under provider size caps. |

Coordinate-space assumptions per model are recorded in `MODEL_RESCALE` and
matched in `evaluate._get_scale`:

- `standard` — model returns coordinates in a 1000x1000 normalized grid.
- `no_rescale` — model returns pixel coordinates of the image it receives.
- `openai_high`, `claude_standard`, `claude_highres` — pixel coordinates of the
  image after the provider's documented resize (OpenAI `high` detail;
  Anthropic's standard and high-resolution tiers).
- `unit_rescale` — model returns coordinates in `[0, 1]`.
- `smart_resize` — Qwen2.5-VL: pixel coordinates of its processor input; the
  pixel budget and interface are set per task (see Image resolution above).

All pixel-space answers are mapped back to the original image before scoring.
GLM-4.6V wraps every box in `<|begin_of_box|>...<|end_of_box|>` and repeats
the block; `evaluate.extract_*_bboxes` reads only the first block (in JSON
mode, the first block that parses), matching the paper's protocol (Section 4,
"Output parsing").

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

If you run into any issues setting up or running the benchmark, feel free to open a GitHub issue.

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See
[LICENSE](LICENSE).
