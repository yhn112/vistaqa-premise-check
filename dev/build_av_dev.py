"""Build an AV development set in VistaQA layout on nuImages key frames (never VistaQA).

VistaQA's AV domain uses nuScenes camera images with SAM 3 plus manual masks (paper, Sec. 3.2). nuImages is the
image-level sibling of nuScenes (same vehicles, cities and six cameras, 1600x900 px) with human instance masks for
foreground objects (vehicles, cycles, pedestrians, cones, barriers, ...); its tables are read as plain JSON.
Signs, lights, road markings and buildings have no masks, so no question is about them.
- Templated generators where the wording is simple: counts of a category or of people sitting or lying down,
  cyclists and unridden bicycles; "more X or Y"; absent categories phrased as several task types.
  Motion attributes (moving, parked, stopped) are never used: a single frame does not show them.
- Reviewed questions (dev/av_reviewed.json, written by dev/review.py from dev/av_candidates.py sheets):
  spatial, attribute, identification, comparison and OCR on single objects, and hallucination (`av_absent`).
Evidence convention and dev/holdout split as in dev/build_open_dev.py; sources and limits in dev/README.md.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from dev.common import TASK_TYPES, Item, be, plural, split_and_write

REVIEWED = Path(__file__).with_name("av_reviewed.json")
# Frames whose VistaQA overlap check (dev/README.md) found a VistaQA image at correlation >= 0.4: file name -> correlation.
IN_VISTAQA = Path(__file__).with_name("nuimages_in_vistaqa.json")
# Templated samples rejected on the visual check, by sample_data token.
REJECTED = {"9b4335372ff443b99e7780d36413d41e": "cars counted are reflections in shop windows"}
TARGETS = {"av_count": 25, "av_compare_count": 10, "av_absent_object": 22}
# px; a category with a smaller instance in the frame is not counted: scenes with distant instances are where
# nuImages may leave some unannotated.
MIN_AREA = 400
PEOPLE = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.wheelchair",
}
NOUN = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.construction": "construction vehicle",
    "vehicle.trailer": "trailer",
    "vehicle.emergency.police": "police car",
    "vehicle.emergency.ambulance": "ambulance",
    "movable_object.trafficcone": "traffic cone",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "debris",
    "movable_object.pushable_pullable": "cart",
    "static_object.bicycle_rack": "bicycle rack",
    "human.pedestrian.stroller": "stroller",
    "animal": "animal",
}
NOUN |= dict.fromkeys(PEOPLE, "person")
# Nouns counted and compared; a car is not counted next to things a model may call a car or a truck.
COUNTED = ["car", "person", "traffic cone", "bus", "motorcycle", "bicycle"]
CONFUSABLE = {
    "car": {"truck", "police car", "ambulance", "construction vehicle", "trailer"},
    "bus": {"truck"},
    "motorcycle": {"bicycle"},
    "bicycle": {"motorcycle", "bicycle rack"},
}
COUNT_BINS = [(1, 1, 6), (2, 3, 8), (4, 7, 7), (8, 12, 4)]  # (low, high, quota) of av_count answers
# Absent things plausible in any street scene; each needs none of its confusable partners either
# (scooter riders, human.pedestrian.personal_mobility, could pass for a motorcycle or a bicycle).
ABSENT = {
    "bus": {"truck"},
    "motorcycle": {"bicycle", "scooter"},
    "bicycle": {"motorcycle", "bicycle rack", "scooter"},
    "construction vehicle": {"truck"},
    "police car": set(),
    "ambulance": {"truck"},
    "stroller": set(),
}
ABSENT_PHRASES = {
    "counting": "how many people are standing next to {a}?",
    "attribute": "what color is the {n}?",
    "identification": "what is parked behind the {n}?",
    "spatial": "what is to the left of the {n}?",
    "OCR": "what is written on the {n}?",
}


@dataclass
class Obj:
    token: str
    category: str
    attributes: list[str]
    mask: dict | None  # RLE in vistaqa/rle.py form, None when nuImages has no mask
    box: list[float]

    @property
    def noun(self) -> str:
        return NOUN.get(self.category, "")


@dataclass
class Frame:
    token: str
    filename: str
    width: int
    height: int
    camera: str
    location: str
    objects: list[Obj] = field(default_factory=list)

    def of(self, noun: str) -> list[Obj]:
        return [o for o in self.objects if o.noun == noun]


def decode_mask(m: dict | None) -> dict | None:
    """nuImages stores the COCO compressed counts base64-encoded."""
    return None if m is None else {"size": m["size"], "counts": base64.b64decode(m["counts"]).decode("ascii")}


def load(src: Path, version: str) -> dict[str, Frame]:
    """Key frames by sample_data token, with their objects (ego vehicle and surfaces are surface_ann, not here)."""
    table = lambda name: json.loads((src / version / f"{name}.json").read_text())  # noqa: E731
    cats = {c["token"]: c["name"] for c in table("category")}
    attrs = {a["token"]: a["name"] for a in table("attribute")}
    logs = {lg["token"]: lg["location"] for lg in table("log")}
    sample_log = {s["token"]: s["log_token"] for s in table("sample")}
    frames = {
        sd["token"]: Frame(
            sd["token"],
            sd["filename"],
            sd["width"],
            sd["height"],
            sd["filename"].split("/")[1],
            logs[sample_log[sd["sample_token"]]],
        )
        for sd in table("sample_data")
        if sd["is_key_frame"]
    }
    for o in table("object_ann"):
        if o["sample_data_token"] in frames:
            frames[o["sample_data_token"]].objects.append(
                Obj(
                    o["token"],
                    cats[o["category_token"]],
                    [attrs[t] for t in o["attribute_tokens"]],
                    decode_mask(o["mask"]),
                    [float(v) for v in o["bbox"]],
                )
            )
    return frames


@cache
def vistaqa_matches() -> set[str]:
    return set(json.loads(IN_VISTAQA.read_text()))


def in_vistaqa(frame: Frame) -> bool:
    """Skipped after shuffling, so that a newly found match changes no other frame's selection."""
    return Path(frame.filename).name in vistaqa_matches()


def article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def countable(frame: Frame, objs: list[Obj], noun: str) -> bool:
    """Every instance has a mask of at least MIN_AREA px and no confusable thing is in the frame."""
    from vistaqa import rle

    if not objs or any(o.mask is None or rle.area(o.mask) < MIN_AREA for o in objs):
        return False
    return not any(frame.of(other) for other in CONFUSABLE.get(noun, ()))


def count_options(frame: Frame) -> list[tuple[str, list[Obj], str | None]]:
    """(plural phrase, instances, fixed question or None) for every countable thing in the frame, 1-12 instances."""
    riders = [o for o in frame.objects if "cycle.with_rider" in o.attributes]
    opts: list[tuple[str, list[Obj], str | None]] = []
    for noun in COUNTED:
        objs = frame.of(noun)
        if noun == "person" and riders:  # cyclists are annotated as cycles, not people
            continue
        if 1 <= len(objs) <= 12 and countable(frame, objs, noun):
            opts.append(("people" if noun == "person" else plural(noun), objs, None))
    if not riders and countable(frame, frame.of("person"), "person"):
        sitting = [o for o in frame.of("person") if "pedestrian.sitting_lying_down" in o.attributes]
        if 1 <= len(sitting) <= 12:
            opts.append(
                ("people sitting or lying down", sitting, "how many people are sitting or lying down in the image?")
            )
    bikes = frame.of("bicycle")
    if bikes and countable(frame, bikes, "bicycle") and len(bikes) <= 12:
        ridden = [o for o in bikes if "cycle.with_rider" in o.attributes]
        unridden = [o for o in bikes if "cycle.without_rider" in o.attributes]
        if len(ridden) + len(unridden) == len(bikes):
            if ridden:
                opts.append(("cyclists", ridden, None))
            if unridden:
                opts.append(("bicycles without a rider", unridden, "how many bicycles have no rider in the image?"))
    return opts


def item(gen: str, task: str, frame: Frame, question: str, answer: str, objs: list[Obj], **source) -> Item:
    return Item(
        gen,
        task,
        question,
        answer,
        [o.mask for o in objs],  # type: ignore[misc]  # countable() admits only objects with a mask
        ("nuimages", frame.token),
        frame.filename,
        frame.width,
        frame.height,
        "AV",
        {
            "dataset": "nuimages v1.0-val",
            "sample_data": frame.token,
            "camera": frame.camera,
            "location": frame.location,
            "objects": [o.token for o in objs],
            **source,
        },
    )


def gen_count(rng, frame: Frame, quota: dict[tuple[int, int], int]):
    opts = [
        (o, b)
        for o in count_options(frame)
        if (b := next(((lo, hi) for lo, hi, _ in COUNT_BINS if lo <= len(o[1]) <= hi), None)) and quota[b] > 0
    ]
    if not opts:
        return None
    (phrase, objs, fixed), b = rng.choice(opts)
    quota[b] -= 1
    k = len(objs)
    q = fixed or rng.choice(
        ["how many {} are there in the image?", "how many {} can you see?", "how many {} are visible in this picture?"]
    ).format(phrase)
    return "counting", q, f"there {be(k)} {k} {phrase if k > 1 else singular(phrase)} in the image.", objs


def singular(phrase: str) -> str:
    return {
        "people": "person",
        "people sitting or lying down": "person sitting or lying down",
        "cyclists": "cyclist",
        "bicycles without a rider": "bicycle without a rider",
    }.get(phrase, next((n for n in COUNTED if plural(n) == phrase), phrase))


def gen_compare_count(rng, frame: Frame):
    opts = [(p, objs) for p, objs, _ in count_options(frame) if p in {plural(n) for n in COUNTED} | {"people"}]
    pairs = [(a, b) for i, a in enumerate(opts) for b in opts[i + 1 :] if abs(len(a[1]) - len(b[1])) >= 2]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    if rng.random() < 0.5:
        a, b = b, a
    more, less = (a, b) if len(a[1]) > len(b[1]) else (b, a)
    return (
        "comparison",
        f"are there more {a[0]} or {b[0]} in the image?",
        f"there are more {more[0]} than {less[0]}: {len(more[1])} and {len(less[1])}.",
        more[1],
    )


def gen_absent(rng, frame: Frame, task: str):
    present = {o.noun for o in frame.objects}
    if any(o.category == "human.pedestrian.personal_mobility" for o in frame.objects):
        present.add("scooter")
    opts = [
        n
        for n, partners in ABSENT.items()
        if n not in present and not partners & present and not (n == "stroller" and "person" not in present)
    ]
    if task == "counting" and "person" not in present:
        opts = []  # "how many people are standing next to a bus?" needs people, or it has a second false premise
    if not opts:
        return None
    n = rng.choice(opts)
    return task, ABSENT_PHRASES[task].format(a=article(n), n=n), f"there is no {n} in the image.", []


def gen_reviewed(frames: dict[str, Frame], entries: list[dict]) -> tuple[list[Item], list[Item]]:
    grounded: list[Item] = []
    absent: list[Item] = []
    for e in entries:
        frame = frames[e["sample_data"]]
        by_token = {o.token: o for o in frame.objects}
        objs = [by_token[ev["ref"]] for ev in e["evidence"]]
        gen, out = ("av_absent", absent) if e["hallucination"] else ("av_reviewed", grounded)
        out.append(item(gen, e["task_type"], frame, e["question"], e["answer"], objs, sheet=e["sheet"]))
    return grounded, absent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="nuImages root with <version>/ tables and samples/")
    parser.add_argument("--version", default="v1.0-val")
    parser.add_argument("--reviewed", type=Path, default=REVIEWED)
    parser.add_argument("--out", type=Path, required=True, help="writes <out>/dev and <out>/holdout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    frames = load(args.src, args.version)
    entries = json.loads(args.reviewed.read_text()) if args.reviewed.exists() else []
    reviewed, reviewed_absent = gen_reviewed(frames, entries)
    used = {e["sample_data"] for e in entries}
    tokens = sorted(frames)
    rng.shuffle(tokens)
    by_gen: dict[str, list[Item]] = defaultdict(list)
    by_gen["av_reviewed"], by_gen["av_absent"] = reviewed, reviewed_absent
    tasks = [t for t in TASK_TYPES if t in ABSENT_PHRASES]
    quota = {(lo, hi): k for lo, hi, k in COUNT_BINS}
    for name, n in TARGETS.items():
        for token in tokens:
            if len(by_gen[name]) == n:
                break
            if token in used or token in REJECTED or in_vistaqa(frames[token]):
                continue
            frame = frames[token]
            if name == "av_absent_object":
                out = gen_absent(rng, frame, tasks[len(by_gen[name]) % len(tasks)])
            elif name == "av_count":
                out = gen_count(rng, frame, quota)
            else:
                out = gen_compare_count(rng, frame)
            if out is None:
                continue
            task, question, answer, objs = out
            by_gen[name].append(item(name, task, frame, question, answer, objs))
            used.add(token)
    targets = {"av_reviewed": len(reviewed), "av_absent": len(reviewed_absent), **TARGETS}
    print("count answers:", sorted(Counter(len(it.masks) for it in by_gen["av_count"]).items()))
    split_and_write(by_gen, targets, rng, args.out, lambda it: (args.src / it.image_src).read_bytes(), args.seed)


if __name__ == "__main__":
    main()
