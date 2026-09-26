# Data

The `subsets/` directory at the repository root ships pre-frozen 1,000-query
parquets — one per dataset variant in Table 1 of the paper. These are
sufficient to reproduce the reported numbers; the parquets list filenames,
labels, and ground-truth boxes only, so you also need the source images.

`configs/datasets.yaml` is a flat mapping from dataset name to local image
folder. Edit it once after cloning. Below is where each dataset is canonically
distributed; we do not redistribute any image data.

| Dataset            | Source                                                                 | What goes in `configs/datasets.yaml` |
|--------------------|------------------------------------------------------------------------|--------------------------------------|
| Pascal VOC 2007    | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/                        | `pascal.images` -> `VOCdevkit/VOC2007/JPEGImages` |
| OpenImages V7      | https://storage.googleapis.com/openimages/web/index.html               | `openimages.images` -> the test split |
| RefCOCO / +/g      | HuggingFace: `lmms-lab/RefCOCO`, `RefCOCOplus`, `RefCOCOg`             | already wired by `hf:` keys; no local path needed |
| RefL4              | HuggingFace: `JierunChen/Ref-L4`                                       | wired by `hf:` key |
| D3 (D-Cube)        | https://github.com/shikras/d-cube                                      | `d3.images` -> `d-cube/d3_images` |
| Flickr30K Entities | https://bryanplummer.com/Flickr30kEntities/                            | `flickr30k.images` -> `flickr30k-images` |
| PhraseCut          | https://github.com/ChenyunWu/PhraseCutDataset                          | `phrasecut.images` -> `VGPhraseCut_v0/images` |
| Synthetic VG (SVG) | https://github.com/jamespark3922/synthetic-visual-genome (built on GQA) | `svg_relations.images` -> the GQA images |
| HR-InsDet          | https://github.com/insdet/instance-detection                           | `hr_insdet.scenes` and `hr_insdet.support` |
| RoboTools          | https://github.com/Jaraxxus-Me/VoxDet                                  | `robotools.frames` and `robotools.support` |
| iGround            | https://github.com/EvanKazakos/iground                                 | `iground.frames` |

After paths are set, build the on-disk Arrow datasets that lmms-eval consumes:

    python -m findit.prepare_datasets

This reads each parquet under `subsets/`, joins it with the configured image
folder, and writes one Arrow dataset per variant under
`src/findit/built_datasets/`.

