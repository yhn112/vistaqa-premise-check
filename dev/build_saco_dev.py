"""Build a development set in VistaQA layout on large SA-1B photographs (never VistaQA).

SA-1B images are large (1500-5239 px) and are the source of VistaQA's indoor/outdoor domains.
Most samples come from SA-Co/Gold "SA-1B captioner NPs". Non-photographic images (stamps, drawings) are skipped
and task_domain is indoor/outdoor, both from dev/saco_scene_labels.json (dev/label_saco_scenes.py plus visual review).
Only image-phrase pairs on which all three SA-Co/Gold annotators agree are used: negative for all three,
or the same number of instances with matched masks (IoU >= 0.5 per instance and >= 0.7 on average against
annotator a). GT masks are annotator a's; `source.human_sm` is the mean matched IoU of annotators b and c
against them, a human reference for Sm. Templated questions are built from simple, concrete phrases only
(at most 4 words, no prepositions or conjunctions, no vague heads such as "attire", no left/right).
Identification questions (dev/saco_identification.json) and OCR questions on SA-Text words in SA-1B shard
sa_000000 (dev/satext_ocr.json) were written during visual review, attribute, comparison and hallucination questions
on agreed instances (dev/saco_reviewed.json) through dev/review.py. Evidence convention and dev/holdout split as in
dev/build_open_dev.py; sources and review procedure in dev/README.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from dev.common import OPPOSITE, REL_PHRASE, TASK_TYPES, Item, be, plural, side_of, split_and_write
from vistaqa import rle

IDENTIFICATION = Path(__file__).with_name("saco_identification.json")
OCR = Path(__file__).with_name("satext_ocr.json")
ABSENT_INFO = Path(__file__).with_name("saco_absent_info.json")
REVIEWED = Path(__file__).with_name("saco_reviewed.json")
# About a sixth of the set per task type, hallucination samples counted by the task type they are phrased as
# (VistaQA's types are nearly equal); hallucination quotas keep about 27% (VistaQA: 27.1%).
# Reviewed identification, attribute, comparison and absent-information samples are all used;
# OCR takes SATEXT_OCR of the reviewed SA-Text lines, chosen with a separate random.Random(seed).
SATEXT_OCR = 36
TARGETS = {
    "saco_count": 50,
    "saco_compare_count": 15,
    "saco_spatial_relation": 35,
    "saco_color": 15,
    "saco_absent_attribute": 25,
    "saco_absent_object": 20,
    "saco_absent_relation": 15,
    "saco_absent_reference": 10,
}
COUNT_BINS = [(1, 1, 10), (2, 3, 15), (4, 7, 15), (8, 99, 10)]
# Phrases with these words are skipped: relations, vague or abstract heads, ordinals, indefinite pronouns,
# and left/right (person's or image side?).
STOP = re.compile(
    r"\b(of|and|or|with|in|on|that|which|near|from|for|at|to|by|image|picture|background|item|object"
    r"|thing|area|part|side|left|right|attire|outfit|clothing|clothes|stuff|scene|view|decoration"
    r"|design|pattern|structure|element|piece|setup|equipment|accessory|accessories|gear|apparel"
    r"|garment|material|celebration|event|gathering|display|setting|environment|atmosphere|scenery"
    r"|landscape|activity|crowd|individual|architecture|group|unit|section|portion|region|feature"
    r"|detail|arrangement|collection|variety|array|series|first|second|third|other|another|last|next"
    r"|main|same|different|someone|something|somebody|anyone)s?\b"
)
STUFF = {
    "wall",
    "floor",
    "ground",
    "sky",
    "grass",
    "road",
    "street",
    "water",
    "hair",
    "sand",
    "snow",
    "ceiling",
    "field",
    "sidewalk",
    "pavement",
    "dirt",
    "shadow",
    "sunlight",
    "foliage",
    "vegetation",
    "mud",
    "smoke",
    "fog",
    "sea",
    "ocean",
    "track",
    "surface",
    "asphalt",
    "concrete",
    "carpet",
    "lawn",
    "soil",
    "land",
    "terrain",
    "racetrack",
    "green",
    "tarmac",
}
PEOPLE = {
    "man",
    "woman",
    "person",
    "people",
    "men",
    "women",
    "child",
    "children",
    "boy",
    "girl",
    "player",
    "officer",
    "firefighter",
    "worker",
    "runner",
    "cyclist",
    "golfer",
    "referee",
    "performer",
    "stylist",
    "hairstylist",
    "model",
    "dancer",
    "soldier",
    "spectator",
    "student",
    "lady",
    "guy",
}
# Heads too generic or partial to anchor an absent-reference question ("is the tail to the left of the plane?").
VAGUE_PRESENT = {
    "vehicle",
    "vehicles",
    "tail",
    "logo",
    "updo",
    "updos",
    "sign",
    "headpiece",
    "top",
    "bottom",
    "front",
    "back",
    "edge",
    "corner",
    "light",
    "lights",
    "line",
    "lines",
    "mark",
    "text",
    "writing",
    "letters",
    "face",
    "faces",
    "grille",
    "grilles",
    "river",
    "rivers",
    "caddie",
    "artist",
    "wheel",
    "wheels",
    "window",
    "windows",
    "roof",
    "hand",
    "hands",
    "head",
    "arm",
    "leg",
    "cuff",
    "collar",
    "sleeve",
    "wing",
    "wings",
    "component",
    "components",
    "entrance",
    "door",
    "doorway",
    "foot",
    "feet",
    "backdrop",
    "walkway",
    "path",
    "star",
    "advertisement",
}
COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "white",
    "black",
    "brown",
    "grey",
    "gray",
    "silver",
    "gold",
    "beige",
    "tan",
}
PLURAL_FORMS = {
    "man": "men",
    "woman": "women",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "leaf": "leaves",
    "person": "people",
    "knife": "knives",
    "shelf": "shelves",
    "mouse": "mice",
}
ALREADY_PLURAL = set(PLURAL_FORMS.values()) | {"police", "cattle"}


@dataclass
class Phrase:
    core: str  # noun phrase without a leading article
    masks: list[dict]  # annotator a's masks; empty for agreed negatives
    human_iou: float | None = None

    @property
    def head(self) -> str:
        return self.core.split()[-1]

    @property
    def stuff(self) -> bool:
        return self.head in STUFF or self.head.removesuffix("s") in STUFF

    @property
    def boxes(self) -> list[list[float]]:
        return [rle.box(m) for m in self.masks]


def is_plural(word: str) -> bool:
    return word in ALREADY_PLURAL or (word.endswith("s") and not word.endswith(("ss", "us", "is")))


def plural_np(core: str) -> str:
    *mods, head = core.split()
    if not is_plural(head):
        head = PLURAL_FORMS.get(head, plural(head))
    return " ".join(mods + [head])


def no(core: str) -> str:
    """'is no tie' / 'are no jeans' for an absence answer."""
    return f"{'are' if is_plural(core.split()[-1]) else 'is'} no {core}"


def noun_np(core: str, k: int) -> str:
    return f"{k} {core if k == 1 else plural_np(core)}"


def matched_ious(a: list[dict], b: list[dict]) -> np.ndarray:
    iou = rle.iou(a, b)
    rows, cols = linear_sum_assignment(-iou)
    matched: np.ndarray = iou[rows, cols]
    return matched


def load(gold_dir: Path, scenes: dict[str, dict]) -> dict[str, tuple[dict, dict[str, Phrase]]]:
    """Per photographic image file: image info and the agreed simple phrases keyed by core."""
    ann = {}
    for r in "abc":
        d = json.loads((gold_dir / f"gold_sa1b_merged_{r}_release_test.json").read_text())
        per = defaultdict(list)
        for a in d["annotations"]:
            per[a["image_id"]].append(a)
        ann[r] = {(i["file_name"], i["text_input"]): (i, per.get(i["id"], [])) for i in d["images"]}
    images: dict[str, tuple[dict, dict[str, Phrase]]] = {}
    duplicates: set[tuple[str, str]] = set()
    for key, (info, _anns_a) in ann["a"].items():
        file_name, text = key
        if scenes[file_name]["label"] == "other":
            continue
        if not re.fullmatch(r"[a-z][a-z -]*", text) or len(text.split()) > 4 or STOP.search(text):
            continue
        if not info["is_instance_exhaustive"] or any(x["iscrowd"] for r in "abc" for x in ann[r][key][1]):
            continue
        counts = {len(ann[r][key][1]) for r in "abc"}
        if len(counts) != 1:
            continue
        h, w = info["height"], info["width"]
        masks = {r: [{"size": [h, w], "counts": x["segmentation"]["counts"]} for x in ann[r][key][1]] for r in "abc"}
        phrase = Phrase(re.sub(r"^(a|an|the|some)\s+", "", text), masks["a"])
        if masks["a"]:
            ious = [matched_ious(masks["a"], masks[r]) for r in "bc"]
            if min(x.min() for x in ious) < 0.5 or min(x.mean() for x in ious) < 0.7:
                continue
            phrase.human_iou = float(np.mean(np.concatenate(ious)))
        entry = images.setdefault(file_name, (info, {}))
        if phrase.core in entry[1]:
            duplicates.add((file_name, phrase.core))
        entry[1][phrase.core] = phrase
    for file_name, core in duplicates:  # "a woman" and "the woman" may disagree; drop both
        del images[file_name][1][core]
    return images


class Saco:
    def __init__(self, images: dict[str, tuple[dict, dict[str, Phrase]]], scenes: dict[str, dict]):
        self.images = images
        self.scenes = scenes

    def pos(self, f: str) -> list[Phrase]:
        return [p for p in self.images[f][1].values() if p.masks]

    def neg(self, f: str) -> list[Phrase]:
        return [p for p in self.images[f][1].values() if not p.masks]

    def unique(self, f: str) -> list[Phrase]:
        return [p for p in self.pos(f) if len(p.masks) == 1 and not is_plural(p.head) and not p.stuff]

    def item(self, gen, task, f, question, answer, evidence: list[Phrase], **source) -> Item:
        info = self.images[f][0]
        masks = [m for p in evidence for m in p.masks]
        n = sum(len(p.masks) for p in evidence)
        # evidence phrases carry human_iou
        human = sum(p.human_iou * len(p.masks) for p in evidence) / n if n else None  # type: ignore[operator, misc]
        return Item(
            gen,
            task,
            question,
            answer,
            masks,
            ("sa1b", f),
            f"sa1b/{f}",
            info["width"],
            info["height"],
            self.scenes[f]["label"],
            {
                "dataset": "sa-co/gold sa1b",
                "file_name": f,
                "phrases": [p.core for p in evidence],
                "human_sm": human,
                **source,
            },
        )


def disjoint(p: Phrase, q: Phrase) -> bool:
    """Different things: no shared head and no instance mostly inside another (a tie on a person)."""
    if p.head == q.head or p.core in q.core or q.core in p.core:
        return False
    a, b = p.masks, q.masks
    inside = max(float(np.max(rle.iou(a, b, crowd=True))), float(np.max(rle.iou(b, a, crowd=True))))
    return inside < 0.3


def gen_count(rng, saco: Saco, ids, used, n):
    quota = {(lo, hi): k for lo, hi, k in COUNT_BINS}
    out = []
    for f in ids:
        if f in used:
            continue
        options = []
        for p in saco.pos(f):
            k = len(p.masks)
            b = next(((lo, hi) for lo, hi, _ in COUNT_BINS if lo <= k <= hi), None)
            if b and quota[b] > 0 and not p.stuff and not (k == 1 and is_plural(p.head)):
                options.append((p, b))
        if not options:
            continue
        p, b = rng.choice(options)
        quota[b] -= 1
        k = len(p.masks)
        q = rng.choice(
            [
                "how many {} are there in the image?",
                "how many {} can you see?",
                "how many {} are visible in this picture?",
            ]
        ).format(plural_np(p.core))
        out.append(saco.item("saco_count", "counting", f, q, f"there {be(k)} {noun_np(p.core, k)} in the image.", [p]))
        used.add(f)
        if len(out) == n:
            break
    return out


def gen_compare_count(rng, saco: Saco, ids, used, n):
    out = []
    for f in ids:
        if f in used:
            continue
        cands = [p for p in saco.pos(f) if len(p.masks) <= 15 and not p.stuff]
        pairs = [
            (p, q)
            for i, p in enumerate(cands)
            for q in cands[i + 1 :]
            if len(p.masks) != len(q.masks) and disjoint(p, q)
        ]
        if not pairs:
            continue
        p, q = rng.choice(pairs)
        if rng.random() < 0.5:
            p, q = q, p
        more, less = (p, q) if len(p.masks) > len(q.masks) else (q, p)
        question = f"are there more {plural_np(p.core)} or {plural_np(q.core)} in the image?"
        answer = (
            f"there are more {plural_np(more.core)} than {plural_np(less.core)}: "
            f"{noun_np(more.core, len(more.masks))} and {noun_np(less.core, len(less.masks))}."
        )
        out.append(saco.item("saco_compare_count", "comparison", f, question, answer, [more]))
        used.add(f)
        if len(out) == n:
            break
    return out


def separated_pairs(saco: Saco, f: str) -> list[tuple[Phrase, Phrase, str]]:
    info = saco.images[f][0]
    uniq = saco.unique(f)
    return [
        (p, q, s)
        for p in uniq
        for q in uniq
        if p is not q and p.head != q.head and (s := side_of(p.boxes[0], q.boxes[0], info["width"], info["height"]))
    ]


def gen_spatial_relation(rng, saco: Saco, ids, used, n):
    out = []
    for f in ids:
        if f in used or not (pairs := separated_pairs(saco, f)):
            continue
        p, q, s = rng.choice(pairs)
        if s in ("left", "right"):
            question = f"is the {p.core} to the left or to the right of the {q.core} in the image?"
        else:
            question = f"is the {p.core} above or below the {q.core} in the image?"
        out.append(
            saco.item(
                "saco_spatial_relation",
                "spatial",
                f,
                question,
                f"the {p.core} is {REL_PHRASE[s]} the {q.core}.",
                [p],
                reference=q.core,
            )
        )
        used.add(f)
        if len(out) == n:
            break
    return out


def gen_absent_relation(rng, saco: Saco, ids, used, n):
    out = []
    for f in ids:
        if f in used or not (pairs := separated_pairs(saco, f)):
            continue
        p, q, s = rng.choice(pairs)
        wrong = REL_PHRASE[OPPOSITE[s]]
        question = rng.choice(
            ["what color is the {a} {r} the {b}?", "describe the {a} {r} the {b}.", "how big is the {a} {r} the {b}?"]
        ).format(a=p.core, r=wrong, b=q.core)
        out.append(
            saco.item(
                "saco_absent_relation",
                "spatial",
                f,
                question,
                f"there is no {p.core} {wrong} the {q.core} in the image.",
                [],
                phrases=[p.core, q.core],
            )
        )
        used.add(f)
        if len(out) == n:
            break
    return out


def gen_color(rng, saco: Saco, ids, used, n):
    out = []
    for f in ids:
        if f in used:
            continue
        uniq = {p.core: p for p in saco.unique(f)}
        options = []
        for core, p in uniq.items():
            words = core.split()
            head = uniq.get(words[-1])
            if len(words) == 2 and words[0] in COLORS and head and matched_ious(p.masks, head.masks)[0] >= 0.8:
                options.append((words[0], head))
        if not options:
            continue
        color, head = rng.choice(options)
        out.append(
            saco.item(
                "saco_color",
                "attribute",
                f,
                f"what color is the {head.core}?",
                f"the {head.core} is {color}.",
                [head],
                color_phrase=f"{color} {head.core}",
            )
        )
        used.add(f)
        if len(out) == n:
            break
    return out


def absent_question(task: str, core: str) -> str:
    if task == "counting":
        return f"how many {plural_np(core)} are there in the image?"
    if task == "attribute":
        if core.split()[-1].removesuffix("s") in PEOPLE or core.split()[-1] in PEOPLE:
            return f"what is the {core} wearing?"
        return f"what is the {core} made of?" if set(core.split()) & COLORS else f"what color is the {core}?"
    if task == "identification":
        return f"what is next to the {core}?"
    if task == "OCR":
        if core.split()[-1].removesuffix("s") in PEOPLE or core.split()[-1] in PEOPLE:
            return f"what is written on the {core}'s shirt?"
        return f"what is written on the {core}?"
    return f"what is to the left of the {core}?"


def gen_absent(rng, saco: Saco, ids, used, n, attribute: bool):
    """Agreed-negative phrases; attribute=True: the head noun is present in the image under another phrase."""
    gen = "saco_absent_attribute" if attribute else "saco_absent_object"
    tasks = [t for t in TASK_TYPES if t in ("counting", "attribute", "identification", "spatial", "OCR")]
    out: list[Item] = []
    for f in ids:
        if f in used:
            continue
        heads = {p.head for p in saco.pos(f)}
        if attribute:
            options = [p for p in saco.neg(f) if len(p.core.split()) >= 2 and p.head in heads]
        else:
            options = [p for p in saco.neg(f) if p.head not in heads]
        task = tasks[len(out) % len(tasks)]
        if task == "OCR":  # nothing is written on hair, walls or the sky
            options = [p for p in options if not p.stuff]
        if not options:
            continue
        p = rng.choice(options)
        out.append(
            saco.item(
                gen, task, f, absent_question(task, p.core), f"there {no(p.core)} in the image.", [], phrases=[p.core]
            )
        )
        used.add(f)
        if len(out) == n:
            break
    return out


def gen_absent_reference(rng, saco: Saco, ids, used, n):
    """A present unique object related to an agreed-negative one: the absent reference makes the question unanswerable."""
    tasks = ["spatial", "counting", "identification", "comparison"]
    out: list[Item] = []
    for f in ids:
        if f in used:
            continue
        # The absent reference shares no word with any present phrase, so that it is not a variant of a present thing
        # ("passenger plane" next to an "airplane"), and the present object is a whole thing named specifically.
        present_words = {w for p in saco.pos(f) for w in p.core.split()}
        refs = [
            q
            for q in saco.neg(f)
            if not q.stuff
            and q.head not in VAGUE_PRESENT
            and not any(w in pw or pw in w for w in q.core.split() for pw in present_words)
        ]
        present = [p for p in saco.unique(f) if p.head not in VAGUE_PRESENT and p.head not in PEOPLE]
        if not refs or not present:
            continue
        task = tasks[len(out) % len(tasks)]
        if task == "counting" and not any(p.head in PEOPLE or p.head.removesuffix("s") in PEOPLE for p in saco.pos(f)):
            continue  # "how many people ..." needs people in the image, or it has a second false premise
        p, q = rng.choice(present), rng.choice(refs)
        question = {
            "spatial": f"is the {p.core} to the left or to the right of the {q.core}?",
            "counting": f"how many people are standing next to the {q.core}?",
            "identification": f"what is between the {p.core} and the {q.core}?",
            "comparison": f"which is bigger, the {p.core} or the {q.core}?",
        }[task]
        out.append(
            saco.item(
                "saco_absent_reference",
                task,
                f,
                question,
                f"there {no(q.core)} in the image.",
                [],
                phrases=[p.core, q.core],
            )
        )
        used.add(f)
        if len(out) == n:
            break
    return out


def gen_absent_info(saco: Saco, entries: list[dict], used) -> list[Item]:
    """Reviewed questions about information that a present object does not carry; evidence is empty."""
    out = []
    for e in entries:
        f = e["file_name"]
        out.append(
            saco.item(
                "saco_absent_info",
                e["task_type"],
                f,
                e["question"],
                e["answer"],
                [],
                phrases=[e["phrase"]],
                kind=e["kind"],
            )
        )
        used.add(f)
    return out


def gen_reviewed(saco: Saco, entries: list[dict], used) -> tuple[list[Item], list[Item]]:
    """Reviewed attribute and comparison questions (dev/review.py) and the hallucination ones, as two generators.

    Evidence refs name agreed instances ({"phrase", "instance"}); each phrase contributes only the chosen instances.
    """
    grounded: list[Item] = []
    absent: list[Item] = []
    for e in entries:
        f = e["file_name"]
        chosen: dict[str, list[int]] = {}
        for ev in e["evidence"]:
            chosen.setdefault(ev["ref"]["phrase"], []).append(ev["ref"]["instance"])
        phrases = [
            Phrase(core, [saco.images[f][1][core].masks[i] for i in idx], saco.images[f][1][core].human_iou)
            for core, idx in chosen.items()
        ]
        gen, out = ("saco_reviewed_absent", absent) if e["hallucination"] else ("saco_reviewed", grounded)
        out.append(saco.item(gen, e["task_type"], f, e["question"], e["answer"], phrases, sheet=e["sheet"]))
        used.add(f)
    return grounded, absent


def gen_identify(saco: Saco, entries: list[dict], used) -> list[Item]:
    """Reviewed questions; evidence is every instance of the agreed phrase."""
    out = []
    for e in entries:
        f = e["file_name"]
        p = saco.images[f][1][e["phrase"]]
        out.append(saco.item("saco_identify", "identification", f, e["question"], e["answer"], [p]))
        used.add(f)
    return out


def gen_satext_ocr(entries: list[dict], used) -> list[Item]:
    """Reviewed questions on SA-Text words; evidence is the convex hull of the answer's words."""
    out = []
    for e in entries:
        f = e["file_name"]
        mask = rle.from_segmentation([[c for point in e["polygon"] for c in point]], e["height"], e["width"])
        out.append(
            Item(
                "satext_ocr",
                "OCR",
                e["question"],
                e["answer"],
                [mask],
                ("sa1b", f),
                f"shard:{f}",
                e["width"],
                e["height"],
                e["domain"],
                {
                    "dataset": "sa-text on sa-1b sa_000000",
                    "file_name": f,
                    "words": e["words"],
                    "satext_transcription": e["transcription"],
                },
            )
        )
        used.add(f)
    return out


def shard_images(path: Path, names: set[str]) -> dict[str, bytes]:
    """Image bytes from the gzip-compressed SA-1B shard in one streaming pass."""
    out = {}
    with tarfile.open(path, "r|gz") as archive:
        for m in archive:
            if m.isfile() and Path(m.name).name in names:
                out[Path(m.name).name] = archive.extractfile(m).read()  # type: ignore[union-attr]  # a file member
                if len(out) == len(names):
                    break
    if missing := names - set(out):
        raise FileNotFoundError(f"{path}: missing {sorted(missing)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src", type=Path, required=True, help="directory with saco_gold/, saco_gold_images/ and sa1b/sa_000000.tar.gz"
    )
    parser.add_argument("--out", type=Path, required=True, help="writes <out>/dev and <out>/holdout")
    parser.add_argument("--scenes", type=Path, default=Path(__file__).with_name("saco_scene_labels.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    scenes = json.loads(args.scenes.read_text())
    saco = Saco(load(args.src / "saco_gold", scenes), scenes)
    ids = sorted(saco.images)
    rng.shuffle(ids)
    used: set[str] = set()
    identification = json.loads(IDENTIFICATION.read_text())
    ocr = random.Random(args.seed).sample(json.loads(OCR.read_text()), SATEXT_OCR)
    absent_info = json.loads(ABSENT_INFO.read_text())
    reviewed, reviewed_absent = gen_reviewed(saco, json.loads(REVIEWED.read_text()), used)
    targets = {
        "saco_identify": len(identification),
        "satext_ocr": len(ocr),
        "saco_absent_info": len(absent_info),
        "saco_reviewed": len(reviewed),
        "saco_reviewed_absent": len(reviewed_absent),
        **TARGETS,
    }
    by_gen = {
        "saco_identify": gen_identify(saco, identification, used),
        "satext_ocr": gen_satext_ocr(ocr, used),
        "saco_absent_info": gen_absent_info(saco, absent_info, used),
        "saco_reviewed": reviewed,
        "saco_reviewed_absent": reviewed_absent,
        "saco_color": gen_color(rng, saco, ids, used, TARGETS["saco_color"]),
        "saco_absent_attribute": gen_absent(rng, saco, ids, used, TARGETS["saco_absent_attribute"], attribute=True),
        "saco_count": gen_count(rng, saco, ids, used, TARGETS["saco_count"]),
        "saco_compare_count": gen_compare_count(rng, saco, ids, used, TARGETS["saco_compare_count"]),
        "saco_spatial_relation": gen_spatial_relation(rng, saco, ids, used, TARGETS["saco_spatial_relation"]),
        "saco_absent_relation": gen_absent_relation(rng, saco, ids, used, TARGETS["saco_absent_relation"]),
        "saco_absent_object": gen_absent(rng, saco, ids, used, TARGETS["saco_absent_object"], attribute=False),
        "saco_absent_reference": gen_absent_reference(rng, saco, ids, used, TARGETS["saco_absent_reference"]),
    }
    archive = tarfile.open(args.src / "saco_gold_images" / "sa_co_gold.tar")  # noqa: SIM115 - read until exit
    shard = shard_images(args.src / "sa1b" / "sa_000000.tar.gz", {e["file_name"] for e in ocr})

    def read(it: Item) -> bytes:
        if it.image_src.startswith("shard:"):
            return shard[it.image_src.removeprefix("shard:")]
        return archive.extractfile(it.image_src).read()  # type: ignore[union-attr]  # images are file members

    split_and_write(by_gen, targets, rng, args.out, read, args.seed)


if __name__ == "__main__":
    main()
