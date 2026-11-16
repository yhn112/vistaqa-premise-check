"""Evidence plans, immutable mask versions, deterministic checks and verifier images.

Coordinates always refer to the metadata-resolution image, in pixels. Anchors are
reasoning context; only explicitly selected target candidates may be exported.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, model_validator

from vistaqa import rle

VERSION = 1  # deterministic checks and model-facing rendering

Box = Annotated[list[float], Field(min_length=4, max_length=4)]
Point = Annotated[list[float], Field(min_length=2, max_length=2)]
SamSearchText = Annotated[str, Field(description=(
    "Short visual object noun phrase, usually 1-6 words: 'car', 'white dining chair', 'small coloured flag'. "
    "SAM3 accepts 32 tokens including special tokens. No instructions, coordinates, ordinals, "
    "spatial relations or exclusions. Empty text is allowed only with positive box prompts."
))]


class ObjectRef(BaseModel):
    id: str
    description: str
    box: Box | None = None
    search_text: str = Field(description="Short visual noun phrase, without spatial relations or ordinals.")


class EvidencePlan(BaseModel):
    provisional_answer: str
    answerable: bool
    task: Literal["instance_count", "spatial", "comparison", "ocr", "other"]
    targets: list[ObjectRef]
    anchors: list[ObjectRef]
    expected_mask_count: int = Field(ge=0)
    search_texts: list[str]

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [o.id for o in self.targets + self.anchors]
        if len(ids) != len(set(ids)):
            raise ValueError("Object IDs must be unique across targets and anchors")
        return self


class MaskCandidate(BaseModel):
    id: str
    number: int = 0
    target_id: str | None
    role: Literal["target", "anchor", "proposal"]
    description: str
    box: Box
    mask: dict
    source: Literal["direct", "proposal", "relocate", "refine"]
    entry: str
    parent_id: str | None = None
    score: float | None = None
    positive_points: list[Point] = Field(default_factory=list)
    negative_points: list[Point] = Field(default_factory=list)


class Accept(BaseModel):
    action: Literal["accept"]
    mask_id: str


class Reject(BaseModel):
    action: Literal["reject"]
    mask_id: str
    reason: str


class Propose(BaseModel):
    action: Literal["propose"]
    search_text: SamSearchText
    positive_boxes: list[Box]
    negative_boxes: list[Box]


class Relocate(BaseModel):
    action: Literal["relocate"]
    target_id: str
    description: str
    search_text: SamSearchText
    new_box: Box


class Refine(BaseModel):
    action: Literal["refine_mask"]
    mask_id: str
    positive_points: list[Point]
    negative_points: list[Point]


class SearchMissing(BaseModel):
    action: Literal["search_missing"]
    search_text: SamSearchText


class Reconsider(BaseModel):
    action: Literal["reconsider_answer"]
    answer: str


Action = Annotated[Accept | Reject | Propose | Relocate | Refine | SearchMissing | Reconsider,
                   Field(discriminator="action")]


class Verification(BaseModel):
    actions: list[Action]
    complete: bool = Field(description="All required instances and premises were checked, not just mask boundaries.")


class SelectionItem(BaseModel):
    mask_id: str
    target_id: str
    description: str


class Selection(BaseModel):
    targets: list[SelectionItem]
    answer: str
    complete: bool


class MissingSearch(BaseModel):
    status: Literal["present", "absent", "missing_information", "uncertain"]
    answer: str
    targets: list[ObjectRef]
    search_texts: list[SamSearchText]


class FinalAnswer(BaseModel):
    answer: str
    selected_mask_ids: list[str]
    status: Literal["resolved", "confirmed_absent", "unresolved"]


class Presupposition(BaseModel):
    id: str
    kind: Literal["object", "attribute", "relation", "text_or_information"]
    claim: str = Field(description="Short declarative statement about the image that the question takes for granted.")


class Presuppositions(BaseModel):
    claims: list[Presupposition]

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [c.id for c in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("Claim IDs must be unique")
        return self


class ClaimVerdict(BaseModel):
    id: str
    verdict: Literal["holds", "fails", "unclear"]
    reason: str = Field(description="The visual observation behind the verdict.")


class PremiseCheck(BaseModel):
    verdicts: list[ClaimVerdict]
    absence_answer: str = Field(description="Short answer stating what is absent; empty unless a claim fails.")


def valid_box(box: list[float], width: int, height: int) -> bool:
    return (all(math.isfinite(x) for x in box) and
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def mask_checks(candidates: list[MaskCandidate], selected: list[str], expected: int,
                width: int, height: int) -> tuple[list[dict], set[str]]:
    """Hard failures exclude a candidate; geometric heuristics only raise flags."""
    flags, invalid, decoded = [], set(), {}
    for c in candidates:
        try:
            if c.mask["size"] != [height, width]:
                raise ValueError("wrong resolution")
            m = rle.decode(c.mask)
            if m.shape != (height, width) or not m.any():
                raise ValueError("empty or malformed mask")
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            flags.append({"problem": "invalid_mask", "mask_id": c.id, "detail": str(exc)})
            invalid.add(c.id)
            continue
        decoded[c.id] = m
        area = int(m.sum())
        fraction = area / (width * height)
        if fraction < 0.00001 or fraction > 0.9:
            flags.append({"problem": "unusual_area", "mask_id": c.id, "fraction": fraction})
        if valid_box(c.box, width, height):
            x1, y1, x2, y2 = c.box
            inside = int(m[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)].sum()) / area
            if inside < 0.5:
                flags.append({"problem": "outside_box", "mask_id": c.id, "inside_fraction": inside})
    active = [i for i in selected if i in decoded]
    if len(active) != expected:
        flags.append({"problem": "mask_count", "expected": expected, "actual": len(active)})
    for j, a in enumerate(active):
        for b in active[j + 1:]:
            iou = np.logical_and(decoded[a], decoded[b]).sum() / np.logical_or(decoded[a], decoded[b]).sum()
            if iou >= 0.9:
                flags.append({"problem": "possible_duplicate", "mask_ids": [a, b], "iou": float(iou)})
    return flags, invalid


def overlay(image: Image.Image, candidates: list[MaskCandidate]) -> Image.Image:
    """Render only inference artifacts; never use GT in model-facing images."""
    out = np.asarray(image, dtype=np.float32).copy()
    colors = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
    for n, c in enumerate(candidates):
        m = rle.decode(c.mask)
        out[m] = out[m] * 0.65 + np.asarray(colors[n % len(colors)]) * 0.35
    result = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=max(16, round(max(image.size) / 85)))
    for n, c in enumerate(candidates):
        x, y, _, _ = rle.box(c.mask)
        draw.text((x, y), str(c.number), font=font, fill=colors[n % len(colors)], stroke_width=2, stroke_fill="white")
    return result


def review_images(image: Image.Image, candidates: list[MaskCandidate]) -> list[Image.Image]:
    """Numbered panels, alternative versions and enlarged small-object context."""
    images = []
    by_id = {c.id: c for c in candidates}
    for c in candidates:
        if c.parent_id not in by_id:
            continue
        parent = by_id[c.parent_id]
        boxes = [rle.box(v.mask) for v in (parent, c)]
        box = (max(0, int(min(b[0] for b in boxes)) - 16), max(0, int(min(b[1] for b in boxes)) - 16),
               min(image.width, math.ceil(max(b[2] for b in boxes)) + 16),
               min(image.height, math.ceil(max(b[3] for b in boxes)) + 16))
        panels = [overlay(image, [v]).crop(box) for v in (parent, c)]
        scale = min(1.0, 1024 / max(panels[0].size))
        size = (max(1, round(panels[0].width * scale)), max(1, round(panels[0].height * scale)))
        pair = Image.new("RGB", (2 * size[0], size[1]), "white")
        for i, panel in enumerate(panels):
            pair.paste(panel.resize(size, Image.Resampling.LANCZOS), (i * size[0], 0))
        images.append(pair)
    for start in range(0, len(candidates), 16):
        page = candidates[start:start + 16]
        images.append(overlay(image, page))
        # Individual crops also make overlapping versions and small instances readable.
        for c in page:
            x1, y1, x2, y2 = rle.box(c.mask)
            if max(x2 - x1, y2 - y1) >= 128 and c.parent_id is None:
                continue
            pad = max(16, 0.2 * max(x2 - x1, y2 - y1))
            region = overlay(image, [c]).crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                                              min(image.width, math.ceil(x2 + pad)),
                                              min(image.height, math.ceil(y2 + pad))))
            scale = 1024 / max(region.size)
            images.append(region.resize((max(1, round(region.width * scale)), max(1, round(region.height * scale))),
                                        Image.Resampling.LANCZOS))
    return images
