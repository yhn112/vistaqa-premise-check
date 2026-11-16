"""Build a robotics development set in VistaQA layout on DROID frames and RoboRefIt images (never VistaQA).

VistaQA's robotics domain uses DROID images with SAM 3 plus manual masks (paper, Sec. 3.2).
- DROID (dev/robotics_reviewed.json): middle frames of external-camera recordings (dev/fetch_droid_frames.py),
  questions of all six task types and hallucination written by Claude subagents on SAM 3 text proposals
  (dev/droid_candidates.py, dev/review.py). The chosen SAM 3 masks are the GT as they are; no mask is drawn or refined,
  so Sm on these samples is optimistic.
- RoboRefIt testB (dev/roborefit_reviewed.json): cluttered tabletop photographs (640x480) with human target masks,
  questions on those masks; `source.sam3_iou` is the IoU of the best SAM 3 text proposal with each human mask,
  a measure of how far SAM 3 masks are from human ones (the bias of the DROID part).
Evidence convention and dev/holdout split as in dev/build_open_dev.py; sources and limits in dev/README.md.

Usage: uv run python -m dev.build_robotics_dev --src data/open_dev/src --out data/robotics_dev
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from dev.common import Item, split_and_write
from vistaqa import rle

DROID = Path(__file__).with_name("robotics_reviewed.json")
DROID_FRAMES = Path(__file__).with_name("droid_frames.json")
ROBOREFIT = Path(__file__).with_name("roborefit_reviewed.json")
ROBOREFIT_IN_VISTAQA = {"0000004.png"}  # overlap check 2026-11-13, correlation 0.455 (degenerate transform, excluded)


def roborefit_mask(src: Path, sample_id: str) -> dict:
    """A RoboRefIt target mask: any nonzero channel is foreground, as in the released loader."""
    m = np.asarray(Image.open(src / "masks" / f"{sample_id}.png"))
    return rle.encode(m.any(axis=2) if m.ndim == 3 else m > 0)


def jpeg(path: Path) -> bytes:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="directory with droid_frames/ and roborefit/testB/")
    parser.add_argument("--out", type=Path, required=True, help="writes <out>/dev and <out>/holdout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    frames = {f["uuid"]: f for f in json.loads(DROID_FRAMES.read_text())}
    by_gen: dict[str, list[Item]] = {
        "droid_reviewed": [],
        "droid_absent": [],
        "roborefit_reviewed": [],
        "roborefit_absent": [],
    }
    images: dict[str, bytes] = {}
    for e in json.loads(DROID.read_text()) if DROID.exists() else []:
        f = frames[e["uuid"]]
        path = args.src / "droid_frames" / f"{e['uuid']}.jpg"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != f["sha256"] or "vistaqa_corr" in f:
            raise ValueError(f"{path}: sha256 differs from {DROID_FRAMES} or the frame matches a VistaQA image")
        images[f"droid/{e['uuid']}"] = data
        gen = "droid_absent" if e["hallucination"] else "droid_reviewed"
        by_gen[gen].append(
            Item(
                gen,
                e["task_type"],
                e["question"],
                e["answer"],
                [ev["rle"] for ev in e["evidence"]],
                ("droid", e["uuid"]),
                f"droid/{e['uuid']}",
                e["width"],
                e["height"],
                "robotics",
                {
                    "dataset": "droid raw 1.0.1",
                    "uuid": e["uuid"],
                    "lab": f["lab"],
                    "sheet": e["sheet"],
                    "prompts": [ev.get("prompt") for ev in e["evidence"]],
                    "sam3_scores": [ev.get("score") for ev in e["evidence"]],
                },
            )
        )
    rr = args.src / "roborefit" / "testB"
    for e in json.loads(ROBOREFIT.read_text()) if ROBOREFIT.exists() else []:
        if e["image"] in ROBOREFIT_IN_VISTAQA:
            raise ValueError(f"{e['image']} matches a VistaQA image")
        images[f"roborefit/{e['image']}"] = jpeg(rr / "images" / e["image"])
        gen = "roborefit_absent" if e["hallucination"] else "roborefit_reviewed"
        by_gen[gen].append(
            Item(
                gen,
                e["task_type"],
                e["question"],
                e["answer"],
                [roborefit_mask(rr, ev["ref"]) for ev in e["evidence"]],
                ("roborefit", e["image"]),
                f"roborefit/{e['image']}",
                e["width"],
                e["height"],
                "robotics",
                {
                    "dataset": "roborefit testB",
                    "image": e["image"],
                    "sheet": e["sheet"],
                    "samples": [ev["ref"] for ev in e["evidence"]],
                    "sam3_iou": e.get("sam3_iou"),
                },
            )
        )
    targets = {g: len(items) for g, items in by_gen.items()}
    split_and_write(by_gen, targets, rng, args.out, lambda it: images[it.image_src], args.seed)


if __name__ == "__main__":
    main()
