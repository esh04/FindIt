from .axes import BBOX_COORDS

# bbox_repr → syntax string shown to the model (None = no syntax given)
_BBOX_SYNTAX = {
    'unconstrained': None,
    'xyxy':    '[x_min, y_min, x_max, y_max]',
    'xywh':    '[x_min, y_min, bw, bh]',
    'yxyx':    '[y_min, x_min, y_max, x_max]',
    'yxhw':    '[y_min, x_min, bh, bw]',
    'all':         '[x1, y1, x2, y2, x3, y3, x4, y4]',
    'all-labelled': '[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]',
    'cxcywh':  '[cx, cy, bw, bh]',
}

# 'class_name' is a valid json_key for multilabel tasks — the dict key IS the class label.
_JSON_KEY_PLURAL = {
    'bbox':         'bboxes',
    'bbox_2d':      'bboxes_2d',
    'coordinates':  'coordinates',
    'bounding_box': 'bounding_boxes',
}

_CENTER_DEFINITIONS = (
    "Use the following definitions: cx and cy are the coordinates of the box "
    "center; bw is the box width; bh is the box height."
)
_CENTER_FORMULA = (
    "Use these formulas to convert from corner-format [x_min, y_min, x_max, y_max]: "
    "cx = (x_min + x_max) / 2, cy = (y_min + y_max) / 2, "
    "bw = x_max - x_min, bh = y_max - y_min."
)


def get_prompt(task, multilabel, fmt, bbox_repr,
               json_structure='per_box', json_key='bbox',
               center_exp='none', negative_examples=False):
    video = task.endswith('_video')

    if fmt == 'text':
        clause = _build_text_clause(bbox_repr, video, multilabel, negative_examples)
    else:
        clause = _build_json_clause(bbox_repr, video, multilabel,
                                    json_structure, json_key, negative_examples)

    if center_exp == 'definitions':
        clause = f"{clause} {_CENTER_DEFINITIONS}"
    elif center_exp == 'formula':
        clause = f"{clause} {_CENTER_FORMULA}"

    opener = _opener(task, multilabel)
    return f"{opener} {clause}" if opener else clause


def _opener(task, multilabel):
    if task.startswith('insdet'):
        return ""  # insdet scaffolding is prepended in doc_utils.doc_to_text
    video = task.endswith('_video')
    scope = "the video frames" if video else "the image"
    base = task.removesuffix('_video') if video else task
    if base == 'objdet':
        if multilabel:
            return f"Locate all instances of the following labels in {scope}: {{label}}."
        return f"Locate all instances of '{{label}}' in {scope}."
    if base == 'refexp':
        return f"Locate the object that matches the description '{{label}}' in {scope}."
    # refexp_all
    return f"Locate every object that matches the description '{{label}}' in {scope}."


def _build_text_clause(bbox_repr, video, multilabel, negative_examples):
    syntax = _BBOX_SYNTAX[bbox_repr]
    lead = "Return only the bounding box coordinates"
    if video:
        lead += " for each of the {num_frames} frames"

    if syntax is None and multilabel:
        tail = "with the corresponding labels and nothing else."
    elif syntax is None:
        tail = "and nothing else."
    elif multilabel and video:
        tail = f"in the format:\n<frame_idx>: <label>: {syntax}\nDo not include any other text or comments."
    elif multilabel:
        tail = f"with the corresponding labels in the format: <label>: {syntax} and nothing else."
    elif video:
        tail = f"in the format:\n<frame_idx>: {syntax}\nDo not include any other text or comments."
    else:
        tail = f"in the format {syntax} and nothing else."

    clause = f"{lead} {tail}"
    if negative_examples:
        clause += " If no matching objects are found, return []."
    return clause


def _build_json_clause(bbox_repr, video, multilabel, json_structure, json_key, negative_examples):
    syntax = _BBOX_SYNTAX[bbox_repr]

    # 'unconstrained' JSON: no schema.
    if syntax is None:
        lead = "Return only the bounding box coordinates"
        if multilabel:
            lead += " with the corresponding labels"
        if video:
            lead += " for each of the {num_frames} frames"
        clause = f"{lead} in a JSON format and nothing else."
        if negative_examples:
            clause += " If no matching objects are found, return []."
        return clause

    if json_structure == 'decomposed':
        return _build_decomposed_clause(bbox_repr, video, multilabel, negative_examples)

    schema = _json_schema_example(syntax, json_structure, json_key, video, multilabel)
    lead = "Return the bounding box coordinates"
    if video:
        lead += " for each of the {num_frames} frames"
    lead += " in a JSON format:"
    clause = f"{lead} {schema}\n"

    class_key = (multilabel and json_key == 'class_name')
    plural = _JSON_KEY_PLURAL.get(json_key)

    if json_structure == 'per_box':
        if class_key:
            clause += "Use one object per instance, keyed by the instance's class name. "
        elif multilabel:
            clause += (f'Repeat the object for each instance, with its class as "label" '
                       f'and its coordinates under "{json_key}". ')
        else:
            clause += f'Repeat the "{json_key}" key for each instance. '
    else:  # nested
        if class_key:
            clause += ("Use one object per class, keyed by the class name, whose value "
                       "is the list of all boxes of that class. ")
        elif multilabel:
            clause += (f'Use one object per class, with its name as "label" and the list '
                       f'of its boxes under "{plural}". ')
        else:
            clause += f'List all boxes inside the single "{plural}" list. '

    clause += "Do not include any other text or comments."
    if negative_examples:
        clause += " If no matching objects are found, return []."
    return clause


def _build_decomposed_clause(bbox_repr, video, multilabel, negative_examples):
    keys = BBOX_COORDS[bbox_repr]
    fields = []
    if video:
        fields.append('"frame_idx": <frame_idx>')
    if multilabel:
        fields.append('"label": "<label>"')
    fields += [f'"{k}": <{k}>' for k in keys]
    entry = "{{" + ", ".join(fields) + "}}"
    schema = f"[{entry}, ...]"

    lead = "Return the bounding box coordinates"
    if video:
        lead += " for each of the {num_frames} frames"
    lead += " in a JSON format:"
    clause = (f"{lead} {schema}\n"
              "Repeat the object for each instance, with each coordinate under its own key. "
              "Do not include any other text or comments.")
    if negative_examples:
        clause += " If no matching objects are found, return []."
    return clause


def _json_schema_example(syntax, json_structure, json_key, video, multilabel):
    class_key = (multilabel and json_key == 'class_name')

    if class_key:
        display_key = "<class_name>"
    elif json_structure == 'nested':
        display_key = _JSON_KEY_PLURAL[json_key]
    else:
        display_key = json_key

    inner = f"[{syntax}, {syntax}, ...]" if json_structure == 'nested' else syntax

    fields = []
    if video:
        fields.append('"frame_idx": <frame_idx>')
    if multilabel and not class_key:
        fields.append('"label": "<label>"')
    fields.append(f'"{display_key}": {inner}')
    # Double braces so the JSON entry survives a later str.format(label=..., num_frames=...)
    entry = "{{" + ", ".join(fields) + "}}"

    if json_structure == 'per_box' or video or multilabel:
        return f"[{entry}, ...]"
    return f"[{entry}]"
