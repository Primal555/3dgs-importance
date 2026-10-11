"""Uncorrected RGB/color-difference diagnostics; never an extra training loss."""
import json
import math
from pathlib import Path


def checked_region(box):
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or
            any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
            not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError('ROI must be normalized [x0,y0,x1,y1] with positive area within [0,1]')
    return box


def load_regions(path):
    """JSON: scene -> camera image_name -> predeclared region label -> box."""
    if not path:
        return {}
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(record, dict):
        raise ValueError('ROI file must map scenes to views to named boxes')
    for scene, views in record.items():
        if not isinstance(views, dict):
            raise ValueError(f'{scene}: ROI views must be an object')
        for view, regions in views.items():
            if not isinstance(regions, dict):
                raise ValueError(f'{scene}/{view}: ROI labels must be an object')
            for box in regions.values():
                checked_region(box)
    return record


def appearance_metrics(image, target, regions=None):
    # Import lazily: color_ablation also imports the validation panel helper.
    from .color_ablation import color_metrics
    if image.shape != target.shape or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('matching RGB [3,H,W] images required')
    result = color_metrics(image, target)
    if regions:
        h, w = image.shape[-2:]
        result['regions'] = {}
        for name, box in regions.items():
            x0, y0, x1, y1 = checked_region(box)
            a, b, c, d = math.floor(x0*w), math.floor(y0*h), math.ceil(x1*w), math.ceil(y1*h)
            observed, reference = image[:, b:d, a:c], target[:, b:d, a:c]
            result['regions'][name] = {**color_metrics(observed, reference),
                'rgb_mse': float((observed-reference).square().mean()), 'pixels': (d-b)*(c-a)}
    return result
