"""COCO RLE masks as stored in VistaQA files and predictions: {"size": [height, width], "counts": <ASCII string>}.

Used by the pipeline (encode), the dev set builders, the COCO dev scorer and the renderer.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def _coco(rle: dict) -> dict:
    """The form pycocotools takes: counts as bytes."""
    return {"size": rle["size"], "counts": rle["counts"].encode()}


def encode(mask) -> dict:
    """RLE of a mask given as an HxW array or a PIL image; nonzero pixels are foreground."""
    m = np.asfortranarray((np.asarray(mask) > 0).astype(np.uint8))
    rle = mask_utils.encode(m)
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": rle["counts"].decode("ascii")}


def from_segmentation(seg, height: int, width: int) -> dict:
    """RLE of a COCO segmentation: polygons (those with fewer than 3 points are dropped), uncompressed or compressed RLE."""
    if isinstance(seg, list):
        polys = [p for p in seg if len(p) >= 6]
        rle = mask_utils.merge(mask_utils.frPyObjects(polys, height, width))
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}
    return {"size": [height, width], "counts": rle["counts"].decode("ascii")}


def decode(rle: dict) -> np.ndarray:
    """Boolean HxW mask."""
    return mask_utils.decode(_coco(rle)).astype(bool)


def box(rle: dict) -> list[float]:
    """Bounding box [x_min, y_min, x_max, y_max]."""
    x, y, w, h = mask_utils.toBbox(_coco(rle)).tolist()
    return [x, y, x + w, y + h]


def area(rle: dict) -> int:
    return int(mask_utils.area(_coco(rle)))


def iou(a: list[dict], b: list[dict], crowd: bool = False) -> np.ndarray:
    """len(a) x len(b) IoU matrix; with crowd, the intersection is divided by the area of the mask from a."""
    return mask_utils.iou([_coco(m) for m in a], [_coco(m) for m in b], [int(crowd)] * len(b))
