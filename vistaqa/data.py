"""Load a directory in VistaQA layout (<id>.json + image).

load_samples returns only the fields an inference system may see: image metadata and the question.
load_records returns the full ground truth, for evaluation and dev tools only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Sample:
    image_id: int
    image_path: Path
    width: int
    height: int
    question: str


def _image_path(data_dir: Path, meta_path: Path, file_name: str) -> Path:
    path = data_dir / file_name
    if path.exists():
        return path
    # VistaQA 1079.json names 1079.png, but the dataset ships 1079.jpg.
    candidates = [p for p in data_dir.glob(f"{meta_path.stem}.*") if p.suffix != ".json"]
    if len(candidates) != 1:
        raise FileNotFoundError(f"{meta_path}: image {file_name} not found, candidates {candidates}")
    return candidates[0]


def load_image(sample: Sample) -> Image.Image:
    """RGB image at the metadata resolution, which is the resolution of the GT masks.

    VistaQA 138 and 496 ship downscaled image files (same aspect ratio) with masks at the metadata size.
    """
    with Image.open(sample.image_path) as img:
        img = img.convert("RGB")
    if img.size != (sample.width, sample.height):
        aspect_error = abs(img.width / img.height - sample.width / sample.height)
        if aspect_error > 0.01:
            raise ValueError(f"{sample.image_path}: size {img.size} vs metadata {(sample.width, sample.height)}")
        img = img.resize((sample.width, sample.height), Image.Resampling.LANCZOS)
    return img


def load_samples(data_dir: Path) -> list[Sample]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir}: expected a dataset directory")
    samples = []
    for meta_path in sorted(data_dir.glob("*.json"), key=lambda p: int(p.stem)):
        record = json.loads(meta_path.read_text())
        image = record["image"]
        samples.append(
            Sample(
                image_id=int(image["image_id"]),
                image_path=_image_path(data_dir, meta_path, image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
                question=record["question"],
            )
        )
    if not samples:
        raise ValueError(f"{data_dir}: no sample JSON files found")
    return samples


def load_records(data_dir: Path) -> dict[int, dict]:
    """Full sample records (answers, masks and all other fields) by image id, in id order.

    Ground truth: for evaluation and dev tools only, never for the pipeline.
    """
    records = {}
    for meta_path in data_dir.glob("*.json"):
        record = json.loads(meta_path.read_text())
        records[int(record["image"]["image_id"])] = record
    return dict(sorted(records.items()))
