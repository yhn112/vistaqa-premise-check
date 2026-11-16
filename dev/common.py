"""Code shared by the dev set builders: the sample record, wording helpers and the split writer.

A change here changes every set built with it (open, SA-Co, math, science, AV, robotics); rebuild and compare them
(dev/README.md). Masks are handled by vistaqa/rle.py, which all builders use.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from vistaqa import rle

HOLDOUT_FRACTION = 0.3
TASK_TYPES = ["counting", "attribute", "identification", "spatial", "comparison", "OCR"]
IRREGULAR_PLURALS = {
    "person": "people",
    "mouse": "mice",
    "computer mouse": "computer mice",
    "sheep": "sheep",
    "scissors": "pairs of scissors",
    "knife": "knives",
    "shelf": "shelves",
}
OPPOSITE = {"left": "right", "right": "left", "above": "below", "below": "above"}
REL_PHRASE = {"left": "to the left of", "right": "to the right of", "above": "above", "below": "below"}


@dataclass
class Item:
    generator: str
    task_type: str
    question: str
    answer: str
    masks: list[dict[str, Any]]
    image_key: tuple[Any, ...]
    image_src: str
    width: int
    height: int
    domain: str
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def hallucination(self) -> int:
        return int(not self.masks)


def plural(name: str) -> str:
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith(("s", "sh", "ch", "x")):
        return name + "es"
    if name.endswith("y") and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def be(k: int) -> str:
    return "is" if k == 1 else "are"


def xywh(box: list[float]) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def side_of(a: list[float], b: list[float], width: int, height: int) -> str | None:
    """Image-plane relation of box a to box b when they are clearly separated, else None."""
    gx, gy = 0.02 * width, 0.02 * height
    if a[2] + gx < b[0]:
        return "left"
    if b[2] + gx < a[0]:
        return "right"
    if a[3] + gy < b[1]:
        return "above"
    if b[3] + gy < a[1]:
        return "below"
    return None


def split_and_write(
    by_gen: dict[str, list[Item]],
    targets: dict[str, int],
    rng: random.Random,
    out: Path,
    read: Callable[[Item], bytes],
    seed: int,
) -> None:
    """Split every generator's samples into dev/holdout, write both splits and <out>/summary.json."""
    splits: dict[str, list[Item]] = {"dev": [], "holdout": []}
    for name in targets:
        items = by_gen.get(name, [])
        if len(items) < targets[name]:
            print(f"warning: {name} has {len(items)} of {targets[name]} samples")
        rng.shuffle(items)
        k = round(HOLDOUT_FRACTION * len(items))
        splits["holdout"] += items[:k]
        splits["dev"] += items[k:]
    summary = {}
    for split, items in splits.items():
        rng.shuffle(items)
        write_split(items, out / split, read)
        summary[split] = {
            "n": len(items),
            "hallucination": sum(it.hallucination for it in items),
            "task_type": dict(Counter(it.task_type for it in items)),
            "generator": dict(Counter(it.generator for it in items)),
        }
        print(split, json.dumps(summary[split]))
    (out / "summary.json").write_text(json.dumps({"seed": seed, "targets": targets, **summary}, indent=2))


def write_split(items: list[Item], out_dir: Path, read: Callable[[Item], bytes]) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; remove it to rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, it in enumerate(items, start=1):
        dst = out_dir / f"{idx}.jpg"
        dst.write_bytes(read(it))
        with Image.open(dst) as img:
            if img.size != (it.width, it.height):
                raise ValueError(f"{it.source}: image size {img.size} != annotation size {(it.width, it.height)}")
        record = {
            "image": {"image_id": idx, "width": it.width, "height": it.height, "file_name": dst.name},
            "question": it.question,
            "answer": it.answer,
            "task_type": it.task_type,
            "task_domain": it.domain,
            "num_instances": len(it.masks),
            "hallucination": it.hallucination,
            "annotations": [
                {"id": j, "segmentation": m, "bbox": xywh(rle.box(m)), "area": rle.area(m)}
                for j, m in enumerate(it.masks, start=1)
            ],
            "generator": it.generator,
            "source": it.source,
        }
        (out_dir / f"{idx}.json").write_text(json.dumps(record))
