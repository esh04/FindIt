import os

import numpy as np
import pyarrow.parquet as pq

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))
SUBSETS_DIR = os.path.join(_REPO_ROOT, 'subsets')


def set_subsets_dir(path):
    global SUBSETS_DIR
    SUBSETS_DIR = path

REFCOCO_HF_TO_PARQUET = {
    'lmms-lab/RefCOCO': 'refcoco',
    'lmms-lab/RefCOCOplus': 'refcoco_plus',
    'lmms-lab/RefCOCOg': 'refcoco_g',
}


def _read(name):
    path = os.path.join(SUBSETS_DIR, f'{name}.parquet')
    return pq.read_table(path).to_pandas()


def _box(b):
    return [float(x) for x in b]


def _boxes(col):
    return [[_box(b) for b in row] for row in col]


def _labeled_boxes(col):
    return [[(item['label'], _box(item['bbox'])) for item in row] for row in col]


def _per_frame_boxes(col):
    return [[[_box(b) for b in frame] for frame in row] for row in col]


def _per_frame_labeled_boxes(col):
    return [
        [[(item['label'], _box(item['bbox'])) for item in frame] for frame in row]
        for row in col
    ]


def _common(df):
    return (
        df['label'].tolist(),
        df['filename'].tolist(),
        [int(h) for h in df['height']],
        [int(w) for w in df['width']],
    )


def _subsample_indices(n_src, n_target):
    if n_target >= n_src:
        return list(range(n_src))
    return np.linspace(0, n_src - 1, n_target, dtype=int).tolist()


def _load_dataset(name, image_dir, prompt, filename_suffix=''):
    df = _read(name)
    labels, filenames, heights, widths = _common(df)
    bboxes = _boxes(df['bboxes'])
    prompts = [
        (prompt.format(label=lbl), os.path.join(image_dir, fn + filename_suffix))
        for lbl, fn in zip(labels, filenames)
    ]
    return prompts, bboxes, filenames, labels, heights, widths


def _load_dataset_multilabel(name, image_dir, prompt, filename_suffix=''):
    df = _read(name)
    labels, filenames, heights, widths = _common(df)
    bboxes = _labeled_boxes(df['bboxes'])
    prompts = [
        (prompt.format(label=lbl), os.path.join(image_dir, fn + filename_suffix))
        for lbl, fn in zip(labels, filenames)
    ]
    return prompts, bboxes, filenames, labels, heights, widths


def load_pascal(image_dir, prompt, multi_label=False):
    if multi_label:
        return _load_dataset_multilabel('pascal_multilabel', image_dir, prompt)
    return _load_dataset('pascal', image_dir, prompt)


def load_open_images(image_dir, prompt, multi_label=False):
    # filenames stored without extension
    if multi_label:
        return _load_dataset_multilabel('openimages_multilabel', image_dir, prompt, '.jpg')
    return _load_dataset('openimages', image_dir, prompt, '.jpg')


def load_d3(image_dir, prompt):
    return _load_dataset('d3', image_dir, prompt)


def load_flickr30k(image_dir, prompt, with_caption=False):
    df = _read('flickr30k')
    labels, filenames, heights, widths = _common(df)
    bboxes = _boxes(df['bboxes'])
    captions = df['caption'].tolist() if with_caption else None
    prompts = []
    for i, (lbl, fn) in enumerate(zip(labels, filenames)):
        text = prompt.format(label=lbl)
        if with_caption:
            text = f'This image shows: "{captions[i]}". {text}'
        prompts.append((text, os.path.join(image_dir, fn)))
    return prompts, bboxes, filenames, labels, heights, widths


def load_phrasecut(image_dir, prompt):
    return _load_dataset('phrasecut', image_dir, prompt)


def load_svg_relations(image_dir, prompt):
    return _load_dataset('svg_relations', image_dir, prompt)


def _refcoco_image_map(hf_path, split):
    from datasets import load_dataset
    ds = load_dataset(hf_path)[split]
    m = {}
    for row in ds:
        fn = row['file_name']
        if fn not in m:
            m[fn] = row['image']
    return m


def load_refcoco(hf_path, prompt, split='testA'):
    base_name = REFCOCO_HF_TO_PARQUET[hf_path]
    df = _read(f'{base_name}_{split}')
    labels, filenames, heights, widths = _common(df)
    bboxes = _boxes(df['bboxes'])
    img_map = _refcoco_image_map(hf_path, split)
    prompts = [
        (prompt.format(label=lbl), img_map[fn])
        for lbl, fn in zip(labels, filenames)
    ]
    return prompts, bboxes, filenames, labels, heights, widths


def _refl4_image_map(hf_path):
    from ref_l4 import RefL4Dataset
    ds = RefL4Dataset(hf_path, split='test')
    m = {}
    for img, data in ds:
        fn = data['file_name']
        if fn not in m:
            m[fn] = img
    return m


def load_refl4(hf_path, prompt):
    df = _read('refl4')
    labels, filenames, heights, widths = _common(df)
    bboxes = _boxes(df['bboxes'])
    img_map = _refl4_image_map(hf_path)
    prompts = [
        (prompt.format(label=lbl), img_map[fn])
        for lbl, fn in zip(labels, filenames)
    ]
    return prompts, bboxes, filenames, labels, heights, widths


def load_hr_insdet(scenes_dir, support_dir, prompt, difficulty='hard'):
    df = _read(f'hr_insdet_{difficulty}')
    labels, filenames, heights, widths = _common(df)
    bboxes = _boxes(df['bboxes'])
    support_shots = [
        [os.path.join(support_dir, s) for s in row] for row in df['support_shots']
    ]
    prompts = [
        (prompt, shots, os.path.join(scenes_dir, fn))
        for shots, fn in zip(support_shots, filenames)
    ]
    return prompts, bboxes, filenames, labels, heights, widths


def load_robotools_video(prompt, support_dir, frames_dir, max_frames):
    df = _read('robotools')
    labels, filenames, heights, widths = _common(df)
    support_shots = [
        [os.path.join(support_dir, s) for s in row] for row in df['support_shots']
    ]
    frame_paths_all = [
        [os.path.join(frames_dir, f) for f in row] for row in df['frame_paths']
    ]
    frame_indices_all = [[int(x) for x in f] for f in df['frame_indices']]
    total_scene_frames = [int(x) for x in df['total_scene_frames']]
    bboxes_all = _per_frame_boxes(df['bboxes'])

    prompts, bboxes = [], []
    for i in range(len(df)):
        fp = frame_paths_all[i]
        fi = frame_indices_all[i]
        bb = bboxes_all[i]
        if max_frames < len(fp):
            idx = _subsample_indices(len(fp), max_frames)
            fp = [fp[k] for k in idx]
            fi = [fi[k] for k in idx]
            bb = [bb[k] for k in idx]
        prompts.append((
            prompt.format(num_frames=len(fp)),
            support_shots[i], fp, fi, total_scene_frames[i],
        ))
        bboxes.append(bb)
    return prompts, bboxes, filenames, labels, heights, widths


def load_iground(prompt, frames_dir, num_sample_frames, multi_label=False, with_caption=False):
    name = 'iground_multilabel' if multi_label else 'iground'
    df = _read(name)
    labels, filenames, heights, widths = _common(df)
    captions = df['caption'].tolist() if with_caption else None
    frame_paths_all = [
        [os.path.join(frames_dir, f) for f in row] for row in df['frame_paths']
    ]
    bboxes_all = (
        _per_frame_labeled_boxes(df['bboxes']) if multi_label
        else _per_frame_boxes(df['bboxes'])
    )

    prompts, bboxes = [], []
    for i in range(len(df)):
        fp = frame_paths_all[i]
        bb = bboxes_all[i]
        if num_sample_frames < len(fp):
            idx = _subsample_indices(len(fp), num_sample_frames)
            fp = [fp[k] for k in idx]
            bb = [bb[k] for k in idx]
        text = prompt.format(label=labels[i], num_frames=len(fp))
        if with_caption:
            scope = 'This image shows' if len(fp) == 1 else 'These video frames show'
            text = f'{scope}: "{captions[i]}". {text}'
        prompts.append((text, fp))
        bboxes.append(bb)
    return prompts, bboxes, filenames, labels, heights, widths
