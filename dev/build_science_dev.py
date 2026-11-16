"""Build the science dev set from AI2D diagrams (never from VistaQA).

AI2D (Kembhavi et al., ECCV 2016): 4903 school science diagrams whose pictures ("blobs"), texts and arrows
were outlined by crowd workers and linked in a parse graph.
Questions refer to a picture by its printed label; evidence is the picture, not the label
(the convention of the VistaQA paper's science figures, dev/README.md).
Only pictures with exactly one label that labels nothing else are used.

Usage: uv run python -m dev.build_science_dev --src data/open_dev/src --out data/science_dev
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dev.common import REL_PHRASE, TASK_TYPES, Item, side_of, split_and_write
from vistaqa import rle

TARGETS = {
    "sci_ocr_extreme": 30,
    "sci_spatial_relation": 33,
    "sci_nearest": 9,
    "sci_arrow_target": 35,
    "sci_count_sources": 14,
    "sci_compare_size": 25,
    "sci_absent_label": 52,
}
IN_VISTAQA = Path(__file__).with_name("ai2d_in_vistaqa.json")  # AI2D diagrams that are VistaQA images
# Reviewed questions on labelled parts with SAM 3 masks (dev/science_part_candidates.py, dev/review.py); their diagrams
# lie outside the templated generators' pool, and they come last in the split, so earlier samples do not change.
PARTS = Path(__file__).with_name("science_parts.json")
MIN_SIDE = 400  # px, short side of the diagram
MIN_AREA = 0.005  # of the image, per picture
LABEL = re.compile(r"[A-Za-z][A-Za-z' -]{1,28}[A-Za-z]")
BACKGROUND_TOLERANCE = 24  # per colour channel, of 255


def tighten(image: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """The drawing inside a loose crowd-drawn polygon: the polygon without the background that reaches in from its border.

    Returns the tight mask, or None when there is none: the polygon lies on no uniform background
    (a photograph, a patterned panel) or under half of it is left (also a dark part of a drawing on a dark
    background, which is lost); and whether the polygon holds several separate drawings (a group),
    for which one mask and one label would be wrong.
    Gaps in outlines up to about 1% of the image side are closed first, so that line drawings keep their interior.
    """
    kernel = np.ones((5, 5), np.uint8)
    ring = cv2.dilate(polygon.astype(np.uint8), kernel).astype(bool) & ~polygon
    if not ring.any():
        return None, False
    background = np.median(image[ring], axis=0)
    near = np.abs(image.astype(np.int16) - background).max(axis=2) <= BACKGROUND_TOLERANCE
    if near[ring].mean() < 0.8:
        return None, False
    k = max(3, round(0.012 * max(image.shape[:2])) | 1)
    ink = cv2.morphologyEx(
        (~near & polygon).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ).astype(bool)
    _, regions = cv2.connectedComponents((~ink).astype(np.uint8), connectivity=4)
    outside = np.isin(regions, np.unique(regions[~polygon]))
    tight = polygon & ~outside
    n, parts, stats, _ = cv2.connectedComponentsWithStats(tight.astype(np.uint8), connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    if not len(sizes) or sizes.sum() < 0.5 * polygon.sum():
        return None, False
    if (sizes >= 0.15 * sizes.sum()).sum() > 1:
        return None, True
    return np.isin(parts, 1 + np.flatnonzero(sizes >= 0.02 * sizes.sum())), False


@dataclass
class Picture:
    id: str
    label: str | None
    mask: dict
    box: list[float]
    area: int
    outline_area: int  # of the crowd-drawn polygon
    tight: bool  # the mask follows the drawing (see tighten); only such pictures are evidence

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


class Diagram:
    def __init__(self, src: Path, name: str, category: str):
        self.name, self.category = name, category
        self.path = src / "images" / name
        ann = json.loads((src / "annotations" / f"{name}.json").read_text())
        with Image.open(self.path) as img:
            self.w, self.h = img.size
        self.pictures: dict[str, Picture] = {}
        self.arrows = []
        self.complete = False
        if min(self.w, self.h) < MIN_SIDE:
            return
        pixels = np.asarray(Image.open(io.BytesIO(self.image())))
        rels = list(ann["relationships"].values())
        # Only a label of the whole picture names it. A label of a part or a region of the picture
        # (intraObjectLinkage, intraObjectRegionLabel) has no mask of its own in AI2D and is never asked about.
        labels_of: dict[str, list[str]] = {}
        blobs_of: dict[str, list[str]] = {}
        for r in rels:
            if r["category"] == "intraObjectLabel":
                labels_of.setdefault(r["destination"], []).append(r["origin"])
                blobs_of.setdefault(r["origin"], []).append(r["destination"])
        boxes = {
            b["id"]: [
                min(x for x, _ in b["polygon"]),
                min(y for _, y in b["polygon"]),
                max(x for x, _ in b["polygon"]),
                max(y for _, y in b["polygon"]),
            ]
            for b in ann["blobs"].values()
            if b["polygon"]
        }
        self.texts = [t["value"].strip() for t in ann["text"].values()]
        values = [t.lower() for t in self.texts]
        for b in ann["blobs"].values():
            flat = [float(v) for p in b["polygon"] for v in p]
            if len(flat) < 6:
                continue
            mask = rle.from_segmentation([flat], self.h, self.w)
            if not rle.area(mask):
                continue
            outline_area = rle.area(mask)
            tight, group = tighten(pixels, rle.decode(mask))
            if tight is not None:
                mask = rle.encode(tight)
            label = None
            texts = labels_of.get(b["id"], [])
            if not group and len(texts) == 1 and blobs_of[texts[0]] == [b["id"]]:
                value = " ".join(ann["text"][texts[0]]["value"].split())
                (x0, y0), (x1, y1) = ann["text"][texts[0]]["rectangle"]
                text_box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                own = box_gap(text_box, boxes[b["id"]])
                # The label may belong to a group of which AI2D links one picture: no other picture may be
                # about as close to it as its own.
                shared = any(
                    box_gap(text_box, box) <= max(1.5 * own, 0.02 * max(self.w, self.h))
                    for other, box in boxes.items()
                    if other != b["id"]
                )
                # No other text of the diagram may contain the label ("Condensation (clouds form)").
                if (
                    LABEL.fullmatch(value)
                    and len(value.split()) <= 4
                    and not shared
                    and sum(value.lower() in v for v in values) == 1
                ):
                    label = value
            self.pictures[b["id"]] = Picture(
                b["id"], label, mask, rle.box(mask), rle.area(mask), outline_area, tight is not None
            )
        # Arrows between two things; an end that is no picture (a text node) is kept as None.
        self.arrows = [
            (self.pictures.get(r["origin"]), self.pictures.get(r["destination"]), r["hasDirectionality"])
            for r in rels
            if r["category"] == "interObjectLinkage"
        ]
        # Arrow questions need a complete graph: every outlined arrow is a directed link between two pictures
        # and has exactly one head, which lies nearer to the link's destination than to its origin.
        heads: dict[str, list[list[float]]] = {}
        for r in rels:
            if r["category"] == "arrowHeadTail":
                (x0, y0), (x1, y1) = ann["arrowHeads"][r["destination"]]["rectangle"]
                heads.setdefault(r["origin"], []).append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])

        def points_to_destination(r: dict) -> bool:
            head = heads.get(r["connector"], [])
            return (
                len(head) == 1
                and r["origin"] in boxes
                and r["destination"] in boxes
                and box_gap(head[0], boxes[r["destination"]]) < box_gap(head[0], boxes[r["origin"]])
            )

        self.complete = (
            len(self.arrows) == len(ann["arrows"])
            and all(o and t and directed for o, t, directed in self.arrows)
            and all(points_to_destination(r) for r in rels if r["category"] == "interObjectLinkage")
        )

    def labelled(self) -> list[Picture]:
        """Labelled pictures that are large enough to ask about."""
        return [p for p in self.pictures.values() if p.label and p.area >= MIN_AREA * self.w * self.h]

    def evidence(self) -> list[Picture]:
        """Labelled pictures whose mask can be the answer's evidence."""
        return [p for p in self.labelled() if p.tight]

    def arrows_at(self, p: Picture) -> list[tuple[Picture | None, Picture | None, bool]]:
        return [a for a in self.arrows if p in (a[0], a[1])]

    def sources(self, p: Picture) -> list[Picture] | None:
        """Pictures with an arrow to p; None if an arrow at p is undirected, starts at something else
        or starts at a picture without a tight mask."""
        out = []
        for origin, dest, directed in self.arrows_at(p):
            if not directed or origin is None or dest is None or not origin.tight:
                return None
            if dest is p and origin is not p:
                out.append(origin)
        return out if len({id(o) for o in out}) == len(out) else None

    def image(self) -> bytes:
        with Image.open(self.path) as file:
            img = file.convert("RGBA")
            flat = Image.new("RGB", img.size, "white")
            flat.paste(img, mask=img.getchannel("A"))
        buf = io.BytesIO()
        flat.save(buf, "JPEG", quality=95)
        return buf.getvalue()


def box_gap(a: list[float], b: list[float]) -> float:
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def box_area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def ref(p: Picture) -> str:
    return f'the picture labelled "{p.label}"'


# Each generator returns (task_type, question, answer, pictures) or None when the diagram does not support it.


def gen_ocr_extreme(rng, d: Diagram):
    """The label of the outermost picture on one side: centre and outer edge beyond every other picture's
    by 5% of the image side."""
    opts = []
    for p in d.evidence():
        others = [q for q in d.pictures.values() if q is not p]
        if len(others) < 2:
            continue
        (x, y), (x0, y0, x1, y1), gx, gy = p.center, p.box, 0.05 * d.w, 0.05 * d.h
        wide, tall = x1 - x0 > 0.4 * d.w, y1 - y0 > 0.4 * d.h  # a picture spanning the image is on no side
        for word, ok in (
            ("leftmost", not wide and all(x + gx < q.center[0] and x0 + gx < q.box[0] for q in others)),
            ("rightmost", not wide and all(x - gx > q.center[0] and x1 - gx > q.box[2] for q in others)),
            ("topmost", not tall and all(y + gy < q.center[1] and y0 + gy < q.box[1] for q in others)),
            ("bottommost", not tall and all(y - gy > q.center[1] and y1 - gy > q.box[3] for q in others)),
        ):
            if ok:
                opts.append((word, p))
    if not opts:
        return None
    word, p = rng.choice(opts)
    return "OCR", f"What is the label of the {word} picture in the diagram?", p.label, [p]


def gen_spatial_relation(rng, d: Diagram):
    opts = [
        (a, b, s) for a in d.evidence() for b in d.labelled() if a is not b and (s := side_of(a.box, b.box, d.w, d.h))
    ]
    if not opts:
        return None
    a, b, s = rng.choice(opts)
    q = (
        f"Is {ref(a)} to the left or to the right of {ref(b)}?"
        if s in ("left", "right")
        else f"Is {ref(a)} above or below {ref(b)}?"
    )
    return "spatial", q, REL_PHRASE[s].removesuffix(" of"), [a]


def gen_nearest(rng, d: Diagram):
    """The picture nearest to another by centre distance and by the gap between their boxes,
    each at most half of that to any other picture, labelled or not."""
    lab, everything = d.labelled(), list(d.pictures.values())
    opts = []
    for a in lab:
        others = [q for q in everything if q is not a]
        if len(others) < 2:
            continue
        # The default binds this iteration's picture; both measures are used only within it.
        for measure in (lambda q, a=a: math.dist(a.center, q.center), lambda q, a=a: box_gap(a.box, q.box)):
            first, second = sorted(others, key=measure)[:2]
            if first not in d.evidence() or measure(first) > 0.5 * measure(second):
                break
        else:
            if sorted(others, key=lambda q: math.dist(a.center, q.center))[0] is first:
                opts.append((a, first))
    if not opts:
        return None
    a, near = rng.choice(opts)
    return "spatial", f"Which picture is closest to {ref(a)}?", near.label, [near]


def gen_arrow_target(rng, d: Diagram):
    """The picture that the only arrow leaving a picture points to."""
    opts = []
    for p in d.labelled() if d.complete else []:
        out = [a for a in d.arrows_at(p) if a[0] is p]
        if (
            len(out) == 1
            and out[0][2]
            and out[0][1] in d.evidence()
            and out[0][1] is not p
            and all(a[2] for a in d.arrows_at(p))
        ):
            opts.append((p, out[0][1]))
    if not opts:
        return None
    p, target = rng.choice(opts)
    return "identification", f"What does the arrow that starts at {ref(p)} point to?", target.label, [target]


def gen_count_sources(rng, d: Diagram):
    if not d.complete:
        return None
    opts = [(p, src) for p in d.labelled() if (src := d.sources(p)) and len(src) <= 8]
    several = [o for o in opts if len(o[1]) >= 2]
    if not several and (not opts or rng.random() > 0.15):  # most diagrams are cycles, where every count is 1
        return None
    p, src = rng.choice(several or opts)
    return "counting", f"How many pictures have an arrow pointing to {ref(p)}?", str(len(src)), src


def gen_compare_size(rng, d: Diagram):
    opts = [
        (a, b)
        for a in d.evidence()
        for b in d.labelled()
        if a.outline_area >= 2.5 * b.outline_area
        and (not b.tight or a.area >= 2.5 * b.area)
        and box_area(a.box) >= 1.5 * box_area(b.box)
    ]  # larger by extent as well as by area
    if not opts:
        return None
    a, b = rng.choice(opts)
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    return (
        "comparison",
        f'Which picture is drawn larger, the one labelled "{x.label}" or the one labelled "{y.label}"?',
        a.label,
        [a],
    )


def gen_absent_label(rng, d: Diagram, task: str, vocabulary: dict[str, list[str]]):
    """A question about a label of the same diagram category that shares no word with any text of this diagram."""
    lab = d.labelled()
    if len(lab) < 2 or any(not p.label for p in d.pictures.values()):  # an unlabelled picture may show the thing
        return None
    words = {w for t in d.texts for w in re.findall(r"[a-z]+", t.lower())}
    opts = [v for v in vocabulary[d.category] if not words & set(re.findall(r"[a-z]+", v.lower()))]
    if not opts:
        return None
    absent, other = rng.choice(opts), rng.choice(lab)
    a = f'the picture labelled "{absent}"'
    question = {
        "counting": f"How many pictures have an arrow pointing to {a}?",
        "attribute": f"What is the main color of {a}?",
        "identification": f"What does the arrow that starts at {a} point to?",
        "spatial": f"Is {a} to the left or to the right of {ref(other)}?",
        "comparison": f'Which picture is drawn larger, the one labelled "{absent}" or the one labelled "{other.label}"?',
        "OCR": f"What text is printed inside {a}?",
    }[task]
    return task, question, f'there is no picture labelled "{absent}" in the diagram.', []


GENERATORS = {
    "sci_ocr_extreme": gen_ocr_extreme,
    "sci_spatial_relation": gen_spatial_relation,
    "sci_nearest": gen_nearest,
    "sci_arrow_target": gen_arrow_target,
    "sci_count_sources": gen_count_sources,
    "sci_compare_size": gen_compare_size,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="directory with ai2d/ai2d (the unzipped ai2d-all.zip)")
    parser.add_argument("--out", type=Path, required=True, help="writes <out>/dev and <out>/holdout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    src = args.src / "ai2d" / "ai2d"
    categories = json.loads((src / "categories.json").read_text())
    excluded = set(json.loads(IN_VISTAQA.read_text()))
    diagrams = []
    for name in sorted(set(categories) - excluded, key=lambda n: int(n.split(".")[0])):
        d = Diagram(src, name, categories[name])
        if min(d.w, d.h) >= MIN_SIDE and len(d.labelled()) >= 2:
            diagrams.append(d)
    vocabulary: dict[str, list[str]] = {}
    for d in diagrams:
        labels = (p.label for p in d.labelled())
        vocabulary.setdefault(d.category, []).extend(labels)  # type: ignore[arg-type]  # labelled() ones have labels
    rng.shuffle(diagrams)
    print(f"{len(diagrams)} diagrams with at least two labelled pictures")

    used: set[str] = set()
    by_gen: dict[str, list[Item]] = {}
    # Scarcest first: few diagrams have a verified arrow graph, fewer a picture with several tight sources.
    first = ["sci_count_sources", "sci_arrow_target", "sci_nearest"]
    order = first + [g for g in TARGETS if g not in first]
    for name in order:
        items: list[Item] = []
        for d in diagrams:
            if len(items) == TARGETS[name]:
                break
            if d.name in used:
                continue
            if name == "sci_absent_label":
                out = gen_absent_label(rng, d, TASK_TYPES[len(items) % len(TASK_TYPES)], vocabulary)
            else:
                out = GENERATORS[name](rng, d)
            if out is None:
                continue
            task, question, answer, pictures = out
            used.add(d.name)
            items.append(
                Item(
                    name,
                    task,
                    question,
                    answer,
                    [p.mask for p in pictures],
                    ("ai2d", d.name),
                    d.name,
                    d.w,
                    d.h,
                    "science",
                    {
                        "dataset": "ai2d",
                        "file_name": d.name,
                        "category": d.category,
                        "pictures": [p.id for p in pictures],
                    },
                )
            )
        by_gen[name] = items
    by_name = {d.name: d for d in diagrams}
    targets = dict(TARGETS)
    for gen in ("sci_part_reviewed", "sci_part_absent"):
        by_gen[gen] = []
    for e in json.loads(PARTS.read_text()) if PARTS.exists() else []:
        name = e["file_name"]
        if name in by_name or name in excluded:
            raise ValueError(f"{name} is in the templated pool or matches a VistaQA image")
        by_name[name] = Diagram(src, name, categories[name])
        gen = "sci_part_absent" if e["hallucination"] else "sci_part_reviewed"
        by_gen[gen].append(
            Item(
                gen,
                e["task_type"],
                e["question"],
                e["answer"],
                [ev["rle"] for ev in e["evidence"]],
                ("ai2d", name),
                name,
                e["width"],
                e["height"],
                "science",
                {
                    "dataset": "ai2d",
                    "file_name": name,
                    "category": categories[name],
                    "sheet": e["sheet"],
                    "prompts": [ev.get("prompt") for ev in e["evidence"]],
                    "sam3_scores": [ev.get("score") for ev in e["evidence"]],
                },
            )
        )
    for gen in ("sci_part_reviewed", "sci_part_absent"):
        if by_gen[gen]:
            targets[gen] = len(by_gen[gen])
    split_and_write(by_gen, targets, rng, args.out, lambda it: by_name[it.image_src].image(), args.seed)


if __name__ == "__main__":
    main()
