"""Build a synthetic math development set in VistaQA layout (never VistaQA).

VistaQA's math domain (paper, Sec. 3.1-3.2): images rendered by custom scripts for "geometric interpretation and
diagram-based reasoning", masks generated programmatically with the images, the same six task types and
hallucination share as the other domains. The generator is not published, so this is our own design:
- shape figures: 8-30 non-overlapping triangles, squares, pentagons, hexagons, circles and stars in named colours,
  filled or outlined, some with a letter or number inside, on white or graph paper;
- geometry figures: labelled points, segments, written segment lengths proportional to the drawn lengths, a circle;
- after the figure types shown in the paper (dev/README.md): function plots with a legend and lettered regions
  between two curves, pie charts with a value table, and dice with pips;
- grids of coloured cells holding numbers, letters or symbols, marked angles with written degree measures
  (fans of rays and triangles), and bar charts.
Canvases are 1200-2400 px. One question per image; answers and evidence come from the scene description and masks
are rendered from the same geometry without antialiasing, so they cover the drawn pixels exactly.
Evidence convention after the paper's figure examples (dev/README.md): the objects asked about, not reference
objects; for comparison only the answer; for OCR the object that the text names or labels (shape, segment, curve,
pie section, cell, bar, angle sector), not the text. Split into dev/holdout by generator.
"""

from __future__ import annotations

import argparse
import io
import math
import random
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from dev.common import TASK_TYPES, Item, plural, split_and_write
from vistaqa import rle

TARGETS = {
    "math_count_shape": 10,
    "math_count_color": 8,
    "math_count_segments": 6,
    "math_color": 8,
    "math_style": 5,
    "math_sides": 5,
    "math_identify_label": 8,
    "math_identify_largest": 6,
    "math_spatial_relation": 8,
    "math_spatial_count": 6,
    "math_nearest": 6,
    "math_point_in_circle": 6,
    "math_compare_size": 7,
    "math_compare_count": 5,
    "math_longer_segment": 6,
    "math_ocr_label": 10,
    "math_side_length": 6,
    "math_plot_region": 8,
    "math_plot_mislabelled": 6,
    "math_plot_compare": 5,
    "math_plot_color": 4,
    "math_plot_legend": 4,
    "math_plot_identify": 5,
    "math_pie_inconsistent": 6,
    "math_pie_top": 5,
    "math_pie_compare": 5,
    "math_pie_percent": 5,
    "math_pie_color": 3,
    "math_pie_count": 4,
    "math_dice_pips": 7,
    "math_dice_count_value": 5,
    "math_dice_compare": 5,
    "math_dice_color": 5,
    "math_dice_max": 5,
    # hallucination (VistaQA: 27.1%)
    "math_absent_shape": 10,
    "math_absent_attribute": 9,
    "math_absent_label": 6,
    "math_absent_segment": 6,
    "math_absent_reference": 6,
    "math_absent_shape_text": 4,
    "math_absent_length": 4,
    "math_absent_curve": 8,
    "math_no_mislabel": 4,
    "math_absent_category": 8,
    "math_absent_die": 8,
    # grids, marked angles and bar charts (added 2026-11-13), 20 grounded and 2 hallucination samples each
    "math_grid_count": 4,
    "math_grid_color": 3,
    "math_grid_identify": 3,
    "math_grid_neighbor": 4,
    "math_grid_compare": 3,
    "math_grid_ocr": 3,
    "math_angle_measure": 3,
    "math_angle_compare": 3,
    "math_angle_largest": 3,
    "math_angle_count": 3,
    "math_angle_type": 3,
    "math_angle_unknown": 2,
    "math_angle_between": 3,
    "math_bar_value": 4,
    "math_bar_count": 3,
    "math_bar_color": 3,
    "math_bar_identify": 3,
    "math_bar_spatial": 4,
    "math_bar_compare": 3,
    "math_absent_grid": 2,
    "math_absent_angle": 2,
    "math_absent_bar": 2,
}
COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 90, 220),
    "green": (40, 160, 60),
    "yellow": (240, 200, 30),
    "orange": (245, 140, 30),
    "purple": (140, 60, 180),
    "pink": (240, 120, 180),
    "brown": (140, 90, 50),
    "gray": (128, 128, 128),
    "black": (20, 20, 20),
}
# Colours a reader may call by each other's name: an absent colour must not have a present neighbour.
NEAR_COLORS = {
    "red": {"orange", "pink", "brown"},
    "orange": {"red", "yellow", "brown"},
    "yellow": {"orange"},
    "pink": {"red", "purple"},
    "purple": {"pink", "blue"},
    "blue": {"purple"},
    "brown": {"red", "orange"},
    "gray": {"black"},
    "black": {"gray"},
    "green": set(),
}
SIDES = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}
KINDS = ["triangle", "square", "pentagon", "hexagon", "circle", "star"]
ABSENT_KINDS = ["octagon", "heart", "arrow", "cross", "crescent", "trapezoid"]
LETTERS = [c for c in string.ascii_uppercase if c not in "IOQ"]  # I/O/Q read like 1/0
SIZES = [(1600, 1200), (1200, 1600), (2000, 1500), (1500, 2000), (1800, 1800), (2400, 1600), (1600, 2400)]
WIDE_SIZES = [(1600, 1200), (2000, 1500), (1800, 1800), (2400, 1600)]  # plots and pie charts with a side panel
FUNCS: dict[str, Callable[[float], float]] = {
    "y = sin(x)": math.sin,
    "y = cos(x)": math.cos,
    "y = -sin(x)": lambda x: -math.sin(x),
    "y = -cos(x)": lambda x: -math.cos(x),
    "y = sin(2x)": lambda x: math.sin(2 * x),
    "y = cos(2x)": lambda x: math.cos(2 * x),
    "y = 0.5 sin(x)": lambda x: 0.5 * math.sin(x),
    "y = 0.5 cos(x)": lambda x: 0.5 * math.cos(x),
    "y = 2 sin(x)": lambda x: 2 * math.sin(x),
    "y = 2 cos(x)": lambda x: 2 * math.cos(x),
}
ABSENT_FUNCS = ["y = tan(x)", "y = x^2", "y = e^x", "y = ln(x)", "y = sin(3x)", "y = cos(3x)"]
CURVE_COLORS = ["red", "blue", "green", "orange", "purple", "black"]
PIE_THEMES = {
    "Metal production share": ["Silver", "Nickel", "Iron", "Zinc", "Copper", "Gold", "Lead", "Tin"],
    "Fruit sales share": ["Apples", "Pears", "Plums", "Grapes", "Lemons", "Mangoes", "Cherries", "Peaches"],
    "Transport mode share": ["Bus", "Train", "Car", "Bicycle", "Tram", "Ferry", "Walking"],
}
PIE_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray"]
DIE_RGB = COLORS | {"white": (255, 255, 255)}
DIE_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]
PIPS = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
    6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
}
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six"]
GRID_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "gray"]
SYMBOLS = ["circle", "triangle", "square", "diamond", "star", "heart"]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
ANGLE_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink"]
BAR_THEMES = {
    "Members of school clubs": ["Chess", "Drama", "Choir", "Art", "Robotics", "Debate", "Hiking", "Garden"],
    "Birds counted in the park": ["Ducks", "Geese", "Swans", "Owls", "Crows", "Robins", "Herons", "Finches"],
    "Books sold by genre": ["Mystery", "Fantasy", "Poetry", "History", "Travel", "Comics", "Science", "Cooking"],
    "Rainfall by city (mm)": ["Oslo", "Lima", "Cairo", "Perth", "Quito", "Dakar", "Hanoi", "Riga"],
}


def article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def font(size: float) -> ImageFont.FreeTypeFont:
    # With a size and FreeType available (it is: Pillow wheels bundle it), Pillow returns a FreeTypeFont.
    return ImageFont.load_default(size=max(12, round(size)))  # type: ignore[return-value]


def jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def contrast(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """White on dark colours, black on light ones."""
    r, g, b = rgb
    return (255, 255, 255) if 0.299 * r + 0.587 * g + 0.114 * b < 150 else (0, 0, 0)


@dataclass
class Shape:
    kind: str
    color: str
    filled: bool
    cx: float
    cy: float
    r: float
    angle: float
    label: str | None = None
    stroke: int = 5

    def points(self) -> list[tuple[float, float]]:
        if self.kind == "star":
            return [
                (
                    self.cx + (self.r if i % 2 == 0 else 0.45 * self.r) * math.cos(self.angle + i * math.pi / 5),
                    self.cy + (self.r if i % 2 == 0 else 0.45 * self.r) * math.sin(self.angle + i * math.pi / 5),
                )
                for i in range(10)
            ]
        n = SIDES[self.kind]
        return [
            (
                self.cx + self.r * math.cos(self.angle + 2 * math.pi * i / n),
                self.cy + self.r * math.sin(self.angle + 2 * math.pi * i / n),
            )
            for i in range(n)
        ]

    def draw(self, d: ImageDraw.ImageDraw, color, mask: bool = False) -> None:
        fill = color if (self.filled or mask) else None
        if self.kind == "circle":
            d.ellipse(
                [self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r],
                fill=fill,
                outline=color,
                width=self.stroke,
            )
        else:
            d.polygon(self.points(), fill=fill, outline=color, width=self.stroke)

    @property
    def box(self) -> tuple[float, float, float, float]:
        if self.kind == "circle":
            return self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r
        xs, ys = zip(*self.points(), strict=True)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def inradius(self) -> float:
        return self.r if self.kind == "circle" else self.r * math.cos(math.pi / SIDES[self.kind])

    def area(self) -> float:
        if self.kind == "circle":
            return math.pi * self.r**2
        pts = self.points()
        return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=True))) / 2


class ShapeScene:
    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(SIZES)
        self.grid = rng.random() < 0.4
        scale = math.sqrt(self.w * self.h) / 1800
        kinds = rng.sample(KINDS, rng.randint(3, 5))
        colors = rng.sample(list(COLORS), rng.randint(3, 6))
        n = rng.randint(8, 30)
        self.shapes: list[Shape] = []
        for _ in range(20000):
            if len(self.shapes) == n:
                break
            r = rng.uniform(30, 160) * scale
            cx, cy = rng.uniform(r + 15, self.w - r - 15), rng.uniform(r + 15, self.h - r - 15)
            if all(math.hypot(cx - s.cx, cy - s.cy) > r + s.r + 20 * scale for s in self.shapes):
                kind = rng.choice(kinds)
                angle = rng.uniform(-0.35, 0.35) - math.pi / 4 if kind == "square" else rng.uniform(0, 2 * math.pi)
                self.shapes.append(
                    Shape(
                        kind, rng.choice(colors), rng.random() < 0.7, cx, cy, r, angle, stroke=max(3, round(5 * scale))
                    )
                )
        letters = iter(rng.sample(LETTERS, len(LETTERS)))
        numbers = iter(rng.sample(range(1, 100), 99))
        for s in self.shapes:
            if s.kind != "star" and s.inradius >= 20 and rng.random() < 0.5:
                s.label = next(letters) if rng.random() < 0.6 else str(next(numbers))

    def label_font(self, s: Shape) -> ImageFont.FreeTypeFont:
        return font(0.8 * s.inradius)

    def text_color(self, s: Shape):
        return contrast(COLORS[s.color]) if s.filled else COLORS["black"]

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        if self.grid:
            step = max(30, self.w // 40)
            for x in range(0, self.w, step):
                d.line([(x, 0), (x, self.h)], fill=(225, 225, 225), width=1)
            for y in range(0, self.h, step):
                d.line([(0, y), (self.w, y)], fill=(225, 225, 225), width=1)
        for s in self.shapes:
            s.draw(d, COLORS[s.color])
            if s.label:
                d.text((s.cx, s.cy), s.label, fill=self.text_color(s), font=self.label_font(s), anchor="mm")
        return jpeg(img)

    def mask(self, s: Shape) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        s.draw(ImageDraw.Draw(m), 255, mask=True)
        return rle.encode(m)

    def of(self, kind: str, color: str | None = None) -> list[Shape]:
        return [s for s in self.shapes if s.kind == kind and (color is None or s.color == color)]

    def describe(self, s: Shape, use_label: bool = True) -> str | None:
        """A phrase that picks out exactly this shape, or None."""
        if len(self.of(s.kind)) == 1:
            return f"the {s.kind}"
        if len(self.of(s.kind, s.color)) == 1:
            return f"the {s.color} {s.kind}"
        if use_label and s.label:
            return f"the {s.kind} labelled {s.label}"
        return None


@dataclass
class Segment:
    a: int
    b: int
    label: str | None = None
    label_xy: tuple[float, float] = (0, 0)


class GeometryScene:
    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(SIZES)
        s = math.sqrt(self.w * self.h) / 1800
        self.dot, self.width, self.font = 8 * s, max(3, round(5 * s)), font(34 * s)
        k = rng.randint(5, 9)
        self.pts: list[tuple[float, float]] = []
        for _ in range(5000):
            if len(self.pts) == k:
                break
            p = (rng.uniform(120 * s, self.w - 120 * s), rng.uniform(120 * s, self.h - 120 * s))
            if all(math.dist(p, q) > 220 * s for q in self.pts):
                self.pts.append(p)
        self.names = [chr(ord("A") + i) for i in range(len(self.pts))]
        unit = rng.uniform(15, 60) * s  # pixels per length unit
        pairs = [(i, j) for i in range(len(self.pts)) for j in range(i + 1, len(self.pts))]
        rng.shuffle(pairs)
        self.segments: list[Segment] = []
        for i, j in pairs[: rng.randint(len(self.pts) - 1, min(len(pairs), 2 * len(self.pts)))]:
            # no other point on or next to the segment
            if all(self.dist_to_segment(self.pts[m], i, j) >= 40 * s for m in range(len(self.pts)) if m not in (i, j)):
                self.segments.append(Segment(i, j))
        # A length label must be unambiguous: clear of other labels and dots, and at most half as far
        # from its own segment as from any other segment.
        boxes = [self.point_label_box(i) for i in range(len(self.pts))]
        boxes += [(x - 2 * self.dot, y - 2 * self.dot, x + 2 * self.dot, y + 2 * self.dot) for x, y in self.pts]
        for seg in self.segments:
            if rng.random() > 0.75:
                continue
            (x0, y0), (x1, y1) = self.pts[seg.a], self.pts[seg.b]
            length = math.dist(self.pts[seg.a], self.pts[seg.b])
            nx, ny = (y0 - y1) / length, (x1 - x0) / length
            text = str(max(1, round(length / unit)))
            for t in (0.5, 0.35, 0.65):
                for side in (1, -1):
                    xy = (x0 + t * (x1 - x0) + side * 32 * s * nx, y0 + t * (y1 - y0) + side * 32 * s * ny)
                    box = self.text_box(xy, text)
                    own = self.dist_to_segment(xy, seg.a, seg.b)
                    other = min(
                        (self.dist_to_segment(xy, g.a, g.b) for g in self.segments if g is not seg), default=math.inf
                    )
                    if other >= max(2 * own, 60 * s) and not any(self.overlap(box, b) for b in boxes):
                        seg.label, seg.label_xy = text, xy
                        boxes.append(box)
                        break
                if seg.label:
                    break
        self.circle = None
        for _ in range(200):
            c = (rng.uniform(0.2, 0.8) * self.w, rng.uniform(0.2, 0.8) * self.h)
            r = rng.uniform(150, 450) * s
            d = [math.dist(c, p) for p in self.pts]
            if all(abs(x - r) > 4 * self.dot for x in d) and sum(x < r for x in d) >= 1:
                self.circle = (c, r)
                break

    def dist_to_segment(self, p, i, j) -> float:
        (x0, y0), (x1, y1) = self.pts[i], self.pts[j]
        t = max(0.0, min(1.0, ((p[0] - x0) * (x1 - x0) + (p[1] - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2)))
        return math.dist(p, (x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    def point_label_xy(self, i: int) -> tuple[float, float]:
        return self.pts[i][0] + 3 * self.dot, self.pts[i][1] - 3 * self.dot

    def text_box(self, xy, text: str):
        return ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(xy, text, font=self.font, anchor="mm")

    def point_label_box(self, i: int):
        return self.text_box(self.point_label_xy(i), self.names[i])

    @staticmethod
    def overlap(a: Sequence[float], b: Sequence[float]) -> bool:
        return a[0] < b[2] + 6 and b[0] < a[2] + 6 and a[1] < b[3] + 6 and b[1] < a[3] + 6

    def connected(self, i: int, j: int) -> Segment | None:
        return next((g for g in self.segments if {g.a, g.b} == {i, j}), None)

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        if self.circle:
            (cx, cy), r = self.circle
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=COLORS["blue"], width=self.width)
        for g in self.segments:
            d.line([self.pts[g.a], self.pts[g.b]], fill=(60, 60, 60), width=self.width)
            if g.label:
                d.text(g.label_xy, g.label, fill=COLORS["red"], font=self.font, anchor="mm")
        for i, (x, y) in enumerate(self.pts):
            d.ellipse([x - self.dot, y - self.dot, x + self.dot, y + self.dot], fill=(0, 0, 0))
            d.text(self.point_label_xy(i), self.names[i], fill=(0, 0, 0), font=self.font, anchor="mm")
        return jpeg(img)

    def segment_mask(self, g: Segment) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).line([self.pts[g.a], self.pts[g.b]], fill=255, width=self.width)
        return rle.encode(m)

    def point_mask(self, i: int) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        x, y = self.pts[i]
        ImageDraw.Draw(m).ellipse([x - self.dot, y - self.dot, x + self.dot, y + self.dot], fill=255)
        return rle.encode(m)

    def seg_name(self, g: Segment) -> str:
        return self.names[g.a] + self.names[g.b]


@dataclass
class Curve:
    label: str  # legend text; differs from the drawn function in math_plot_mislabelled
    fn: Callable[[float], float]
    color: str


class PlotScene:
    """Two to four trigonometric curves on numbered axes, a legend to the right of the plot."""

    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(WIDE_SIZES)
        s = math.sqrt(self.w * self.h) / 1800
        self.x0, self.x1 = rng.choice([(-4, 4), (0, 7), (-2, 6)])
        self.y0, self.y1 = -2.5, 2.5
        self.left, self.right = round(0.08 * self.w), round(0.72 * self.w)
        self.top, self.bottom = round(0.07 * self.h), round(0.90 * self.h)
        self.width, self.font, self.letter_font = max(4, round(7 * s)), font(36 * s), font(48 * s)
        self.letter_h = 48 * s
        names = rng.sample(list(FUNCS), rng.randint(2, 4))
        self.curves = [Curve(n, FUNCS[n], c) for n, c in zip(names, rng.sample(CURVE_COLORS, len(names)), strict=True)]
        self.letters: list[tuple[str, tuple[float, float]]] = []  # region letters, set by gen_plot_region

    def px(self, x: float) -> float:
        return self.left + (x - self.x0) / (self.x1 - self.x0) * (self.right - self.left)

    def py(self, y: float) -> float:
        return self.bottom - (y - self.y0) / (self.y1 - self.y0) * (self.bottom - self.top)

    def xs(self, a: float, b: float) -> list[float]:
        n = max(2, round((self.px(b) - self.px(a)) / 2))
        return [a + (b - a) * i / n for i in range(n + 1)]

    def polyline(self, c: Curve, a: float | None = None, b: float | None = None) -> list[tuple[float, float]]:
        return [
            (self.px(x), self.py(c.fn(x))) for x in self.xs(self.x0 if a is None else a, self.x1 if b is None else b)
        ]

    def crossings(self, f: Curve, g: Curve) -> list[float] | None:
        """x of the transversal crossings of two curves; None if they also touch or nearly touch elsewhere."""
        xs = self.xs(self.x0, self.x1)
        d = [f.fn(x) - g.fn(x) for x in xs]
        cross = [
            xa + (xb - xa) * da / (da - db) for xa, xb, da, db in zip(xs, xs[1:], d, d[1:], strict=False) if da * db < 0
        ]
        if any(abs(v) < 0.12 and all(abs(x - c) > 0.3 for c in cross) for x, v in zip(xs, d, strict=True)):
            return None
        return cross

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        gap = 0.012 * self.h
        for gx in range(math.ceil(self.x0), math.floor(self.x1) + 1):
            d.line([(self.px(gx), self.top), (self.px(gx), self.bottom)], fill=(225, 225, 225), width=2)
            d.text((self.px(gx), self.bottom + gap), str(gx), fill=(0, 0, 0), font=self.font, anchor="ma")
        for i in range(-4, 5):
            gy = i / 2
            d.line([(self.left, self.py(gy)), (self.right, self.py(gy))], fill=(225, 225, 225), width=2)
            if i % 2 == 0:
                d.text((self.left - gap, self.py(gy)), str(i // 2), fill=(0, 0, 0), font=self.font, anchor="rm")
        d.line([(self.left, self.py(0)), (self.right, self.py(0))], fill=(120, 120, 120), width=3)
        if self.x0 < 0 < self.x1:
            d.line([(self.px(0), self.top), (self.px(0), self.bottom)], fill=(120, 120, 120), width=3)
        d.rectangle([self.left, self.top, self.right, self.bottom], outline=(0, 0, 0), width=3)
        for c in self.curves:
            d.line(self.polyline(c), fill=COLORS[c.color], width=self.width, joint="curve")
        for i, c in enumerate(self.curves):
            x, y = self.right + 0.025 * self.w, self.top + (1.5 + 2.2 * i) * self.letter_h
            d.line([(x, y), (x + 0.045 * self.w, y)], fill=COLORS[c.color], width=self.width)
            d.text((x + 0.055 * self.w, y), c.label, fill=(0, 0, 0), font=self.font, anchor="lm")
        for letter, xy in self.letters:
            d.text(xy, letter, fill=(0, 0, 0), font=self.letter_font, anchor="mm")
        return jpeg(img)

    def mask(self, c: Curve) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).line(self.polyline(c), fill=255, width=self.width, joint="curve")
        return rle.encode(m)

    def region_mask(self, f: Curve, g: Curve, a: float, b: float) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).polygon(self.polyline(f, a, b) + self.polyline(g, a, b)[::-1], fill=255)
        return rle.encode(m)

    def describe(self, c: Curve) -> str:
        return f"the {c.color} curve"


class PieScene:
    """A pie chart with a percentage in every section and category names around it, next to a table of values.

    Section angles follow the table; `shown` holds the printed percentages (changed by math_pie_inconsistent).
    """

    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(WIDE_SIZES)
        s = math.sqrt(self.w * self.h) / 1800
        self.title, pool = rng.choice(list(PIE_THEMES.items()))
        n = rng.randint(4, 6)
        self.names = rng.sample(pool, n)
        self.absent = [x for x in pool if x not in self.names]
        while True:
            cuts = sorted(rng.sample(range(1, 100), n - 1))
            self.pcts = [b - a for a, b in zip([0] + cuts, cuts + [100], strict=True)]
            if min(self.pcts) >= 8 and len(set(self.pcts)) == n:
                break
        k = rng.choice([1, 2, 5])
        self.values = [p * k for p in self.pcts]
        self.shown = list(self.pcts)
        self.colors = rng.sample(PIE_COLORS, n)
        self.start = rng.uniform(0, 360)
        self.rows = rng.sample(range(n), n)  # table order
        self.cx, self.cy, self.r = 0.37 * self.w, 0.55 * self.h, 0.22 * min(self.w, self.h)
        self.font, self.title_font, self.line = font(36 * s), font(48 * s), 60 * s

    def angles(self, i: int) -> tuple[float, float]:
        a = self.start + 3.6 * sum(self.pcts[:i])
        return a, a + 3.6 * self.pcts[i]

    def at(self, i: int, radius: float) -> tuple[float, float]:
        a0, a1 = self.angles(i)
        t = math.radians((a0 + a1) / 2)
        return self.cx + radius * math.cos(t), self.cy + radius * math.sin(t)

    def box(self) -> list[float]:
        return [self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r]

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        d.text((self.w / 2, 0.07 * self.h), self.title, fill=(0, 0, 0), font=self.title_font, anchor="mm")
        for i in range(len(self.names)):
            d.pieslice(self.box(), *self.angles(i), fill=COLORS[self.colors[i]], outline=(255, 255, 255), width=4)
        for i, name in enumerate(self.names):
            d.text(
                self.at(i, 0.62 * self.r),
                f"{self.shown[i]}%",
                fill=contrast(COLORS[self.colors[i]]),
                font=self.font,
                anchor="mm",
            )
            x, y = self.at(i, 1.08 * self.r)
            dx, dy = (x - self.cx) / self.r, (y - self.cy) / self.r
            anchor = ("l" if dx > 0.3 else "r" if dx < -0.3 else "m") + ("a" if dy > 0.3 else "d" if dy < -0.3 else "m")
            d.text((x, y), name, fill=(0, 0, 0), font=self.font, anchor=anchor)
        x0, x1, xm = 0.74 * self.w, 0.97 * self.w, 0.87 * self.w
        y0 = self.cy - (len(self.names) + 1) * self.line / 2
        for r, (left, right) in enumerate(
            [("Category", "Value")] + [(self.names[i], str(self.values[i])) for i in self.rows]
        ):
            y = y0 + r * self.line
            d.rectangle([x0, y, x1, y + self.line], outline=(0, 0, 0), width=2)
            d.line([(xm, y), (xm, y + self.line)], fill=(0, 0, 0), width=2)
            d.text(((x0 + xm) / 2, y + self.line / 2), left, fill=(0, 0, 0), font=self.font, anchor="mm")
            d.text(((xm + x1) / 2, y + self.line / 2), right, fill=(0, 0, 0), font=self.font, anchor="mm")
        return jpeg(img)

    def mask(self, i: int) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).pieslice(self.box(), *self.angles(i), fill=255)
        return rle.encode(m)


@dataclass
class Die:
    color: str
    value: int
    cx: float
    cy: float
    side: float
    angle: float

    def local(self, u: float, v: float) -> tuple[float, float]:
        """Die coordinates (half sides, origin at the centre) to image coordinates."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        return self.cx + 0.5 * self.side * (u * c - v * s), self.cy + 0.5 * self.side * (u * s + v * c)

    def corners(self) -> list[tuple[float, float]]:
        return [self.local(u, v) for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    def pips(self) -> list[list[float]]:
        r = 0.09 * self.side
        return [[x - r, y - r, x + r, y + r] for x, y in (self.local(0.55 * u, 0.55 * v) for u, v in PIPS[self.value])]


class DiceScene:
    """6-14 dice seen from above: tilted coloured squares with one to six pips."""

    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(SIZES)
        s = math.sqrt(self.w * self.h) / 1800
        self.stroke = max(3, round(4 * s))
        colors = rng.sample(DIE_COLORS, rng.randint(3, 5))
        n = rng.randint(6, 14)
        self.dice: list[Die] = []
        for _ in range(20000):
            if len(self.dice) == n:
                break
            side = rng.uniform(120, 200) * s
            cx, cy = rng.uniform(side, self.w - side), rng.uniform(side, self.h - side)
            if all(math.hypot(cx - d.cx, cy - d.cy) > 0.75 * (side + d.side) + 25 * s for d in self.dice):
                self.dice.append(Die(rng.choice(colors), rng.randint(1, 6), cx, cy, side, rng.uniform(-0.5, 0.5)))

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        for die in self.dice:
            d.polygon(die.corners(), fill=DIE_RGB[die.color], outline=(40, 40, 40), width=self.stroke)
            for pip in die.pips():
                d.ellipse(pip, fill=contrast(DIE_RGB[die.color]))
        return jpeg(img)

    def mask(self, die: Die) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).polygon(die.corners(), fill=255, outline=255, width=self.stroke)
        return rle.encode(m)

    def pip_masks(self, die: Die) -> list[dict]:
        out = []
        for pip in die.pips():
            m = Image.new("L", (self.w, self.h), 0)
            ImageDraw.Draw(m).ellipse(pip, fill=255)
            out.append(rle.encode(m))
        return out

    def of(self, color: str) -> list[Die]:
        return [d for d in self.dice if d.color == color]

    def unique(self) -> list[Die]:
        """Dice that are the only one of their colour."""
        return [d for d in self.dice if len(self.of(d.color)) == 1]


def distinct_colors(rng: random.Random, pool: list[str], n: int) -> list[str]:
    """n colours from pool, no two of which a reader may call by each other's name (NEAR_COLORS)."""
    while True:
        out: list[str] = []
        for c in rng.sample(pool, len(pool)):
            if not NEAR_COLORS[c] & set(out):
                out.append(c)
        if len(out) >= n:
            return out[:n]


def text_box(xy: tuple[float, float], text: str, fnt: ImageFont.FreeTypeFont) -> tuple[float, float, float, float]:
    return ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(xy, text, font=fnt, anchor="mm")


@dataclass
class Cell:
    row: int
    col: int
    color: str  # a COLORS name or "white"
    content: str  # a number, a letter or a SYMBOLS name


def symbol_points(name: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    if name == "triangle":
        return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    if name == "square":
        return [
            (cx - 0.75 * r, cy - 0.75 * r),
            (cx + 0.75 * r, cy - 0.75 * r),
            (cx + 0.75 * r, cy + 0.75 * r),
            (cx - 0.75 * r, cy + 0.75 * r),
        ]
    if name == "diamond":
        return [(cx, cy - r), (cx + 0.75 * r, cy), (cx, cy + r), (cx - 0.75 * r, cy)]
    if name == "star":
        return [
            (
                cx + (r if i % 2 == 0 else 0.45 * r) * math.cos(-math.pi / 2 + i * math.pi / 5),
                cy + (r if i % 2 == 0 else 0.45 * r) * math.sin(-math.pi / 2 + i * math.pi / 5),
            )
            for i in range(10)
        ]
    # heart curve: x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t, about 32 units wide
    k = r / 16
    return [
        (
            cx + k * 16 * math.sin(t) ** 3,
            cy - k * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t) + 2.5),
        )
        for t in (2 * math.pi * i / 64 for i in range(64))
    ]


class GridScene:
    """A table of 3-6 rows and 3-7 columns of white or coloured cells, each holding a number, a letter or a symbol.

    Cells tile the table without overlap; each draws its own half of the lines between cells.
    """

    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(SIZES)
        s = math.sqrt(self.w * self.h) / 1800
        self.rows, self.cols = rng.randint(3, 6), rng.randint(3, 7)
        self.kind = rng.choice(["number", "letter", "symbol"])
        size = min(0.84 * self.w / self.cols, 0.84 * self.h / self.rows, 300 * s)
        x0, y0 = (self.w - size * self.cols) / 2, (self.h - size * self.rows) / 2
        self.xs = [round(x0 + i * size) for i in range(self.cols + 1)]
        self.ys = [round(y0 + i * size) for i in range(self.rows + 1)]
        self.stroke, self.font, self.symbol_r = max(2, round(3 * s)), font(0.42 * size), 0.3 * size
        palette = distinct_colors(rng, GRID_COLORS, rng.randint(2, 4))
        pool = {"letter": LETTERS, "symbol": SYMBOLS}.get(self.kind)
        pool = rng.sample(pool, rng.randint(3, 6)) if pool else None
        self.cells = [
            Cell(
                r,
                c,
                "white" if rng.random() < 0.35 else rng.choice(palette),
                str(rng.randint(1, 99)) if pool is None else rng.choice(pool),
            )
            for r in range(self.rows)
            for c in range(self.cols)
        ]

    def rect(self, c: Cell) -> list[int]:
        return [self.xs[c.col], self.ys[c.row], self.xs[c.col + 1] - 1, self.ys[c.row + 1] - 1]

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        k = self.stroke  # outer border as thick as the lines between cells
        d.rectangle(
            [self.xs[0] - k, self.ys[0] - k, self.xs[-1] - 1 + k, self.ys[-1] - 1 + k], outline=(40, 40, 40), width=k
        )
        for c in self.cells:
            rgb = (255, 255, 255) if c.color == "white" else COLORS[c.color]
            x0, y0, x1, y1 = self.rect(c)
            d.rectangle([x0, y0, x1, y1], fill=rgb, outline=(40, 40, 40), width=self.stroke)
            if self.kind == "symbol":
                if c.content == "circle":
                    r = 0.8 * self.symbol_r
                    d.ellipse(
                        [(x0 + x1) / 2 - r, (y0 + y1) / 2 - r, (x0 + x1) / 2 + r, (y0 + y1) / 2 + r], fill=contrast(rgb)
                    )
                else:
                    d.polygon(symbol_points(c.content, (x0 + x1) / 2, (y0 + y1) / 2, self.symbol_r), fill=contrast(rgb))
            else:
                d.text(((x0 + x1) / 2, (y0 + y1) / 2), c.content, fill=contrast(rgb), font=self.font, anchor="mm")
        return jpeg(img)

    def mask(self, c: Cell) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).rectangle(self.rect(c), fill=255)
        return rle.encode(m)

    def of(self, color: str | None = None, content: str | None = None) -> list[Cell]:
        return [c for c in self.cells if color in (None, c.color) and content in (None, c.content)]

    def at(self, row: int, col: int) -> Cell | None:
        return self.cells[row * self.cols + col] if 0 <= row < self.rows and 0 <= col < self.cols else None

    def what(self, content: str) -> str:
        """The content as named in a question: the number 42, the letter K, a star."""
        return f"{article(content)} {content}" if self.kind == "symbol" else f"the {self.kind} {content}"

    def answer(self, content: str) -> str:
        return f"{article(content)} {content}" if self.kind == "symbol" else content

    def position(self, c: Cell) -> str:
        return f"the cell in the {ORDINALS[c.row]} row from the top and the {ORDINALS[c.col]} column from the left"

    def describe(self, c: Cell) -> str | None:
        """A phrase that picks out exactly this cell by its content or colour, or None."""
        if len(self.of(content=c.content)) == 1:
            return f"the cell containing {self.what(c.content)}"
        if c.color != "white" and len(self.of(color=c.color)) == 1:
            return f"the {c.color} cell"
        return None


@dataclass
class Angle:
    vertex: int
    arms: tuple[int, int]  # point indices
    measure: int  # degrees, equal to the drawn angle
    t1: float  # image directions of the arms, degrees clockwise from +x; the angle runs clockwise from t1 to t2
    t2: float
    color: str = ""
    label: str = ""  # written text: the measure, or "x" in math_angle_unknown
    label_xy: tuple[float, float] = (0, 0)


def angle_ok(a: int) -> bool:
    """Measures of marked angles: not tiny and not near a right angle, which would need a square mark."""
    return 25 <= a <= 150 and abs(a - 90) >= 10


class AngleScene:
    """Marked angles with written degree measures and labelled points.

    Layouts: a fan of rays from O with 2-4 adjacent marked angles, such a fan over a straight line (the angles sum to
    180), and a triangle with its three angles marked. A marked angle is a filled sector of radius `rm` with an arc;
    its measure is written on the bisector beyond the sector, clear of every line, sector and other text.
    Drawn angles equal the written measures.
    """

    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(SIZES)
        s = math.sqrt(self.w * self.h) / 1800
        self.lw, self.dot, self.pad = max(3, round(5 * s)), 9 * s, 10 * s
        self.font, self.name_font = font(38 * s), font(46 * s)
        while not self.build(rng):
            pass

    def build(self, rng: random.Random) -> bool:
        self.layout = rng.choice(["fan", "line", "triangle"])
        if self.layout == "triangle":
            measures = [rng.randint(30, 120), rng.randint(30, 120)]
            measures.append(180 - sum(measures))
            if min(measures) < 30 or not all(angle_ok(a) for a in measures):
                return False
            self.triangle(rng, measures)
        else:
            m = rng.randint(2, 4)
            measures = [rng.randint(25, 150) for _ in range(m - 1 if self.layout == "line" else m)]
            if self.layout == "line":
                measures.append(180 - sum(measures))
            if not all(angle_ok(a) for a in measures) or sum(measures) > 240:
                return False
            self.fan(rng, measures)
        for a, c in zip(self.angles, distinct_colors(rng, ANGLE_COLORS, len(self.angles)), strict=True):
            a.color, a.label = c, f"{a.measure}°"
        return self.place_labels()

    def direction(self, v: int, i: int) -> float:
        (x0, y0), (x1, y1) = self.pts[v], self.pts[i]
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    def fan(self, rng: random.Random, measures: list[int]) -> None:
        length = rng.uniform(0.36, 0.42) * min(self.w, self.h)
        dirs = [rng.uniform(0, 360)]
        for a in measures:
            dirs.append(dirs[-1] + a)
        ends = [(length * math.cos(math.radians(t)), length * math.sin(math.radians(t))) for t in dirs]
        xs, ys = [0.0] + [x for x, _ in ends], [0.0] + [y for _, y in ends]
        ox, oy = (self.w - min(xs) - max(xs)) / 2, (self.h - min(ys) - max(ys)) / 2
        self.pts = [(ox, oy)] + [(ox + x, oy + y) for x, y in ends]
        self.names = ["O"] + LETTERS[: len(ends)]
        self.lines = [(0, i) for i in range(1, len(self.pts))]
        self.angles = [Angle(0, (i + 1, i + 2), a, dirs[i], dirs[i + 1]) for i, a in enumerate(measures)]
        self.name_dirs = [dirs[-1] + (360 - sum(measures)) / 2] + dirs  # O's name on the unmarked side
        self.rm, self.reach = 0.27 * length, 0.85 * length

    def triangle(self, rng: random.Random, measures: list[int]) -> None:
        a, b, c = (math.radians(x) for x in measures)
        ac = math.sin(b) / math.sin(c)  # |AC| for |AB| = 1 (law of sines)
        pts = [(0.0, 0.0), (1.0, 0.0), (ac * math.cos(a), -ac * math.sin(a))]
        rot = rng.uniform(0, 2 * math.pi)
        pts = [(x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)) for x, y in pts]
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        k = min(0.75 * self.w / (max(xs) - min(xs)), 0.75 * self.h / (max(ys) - min(ys)))
        ox, oy = (self.w - k * (min(xs) + max(xs))) / 2, (self.h - k * (min(ys) + max(ys))) / 2
        self.pts = [(ox + k * x, oy + k * y) for x, y in pts]
        self.names = ["A", "B", "C"]
        self.lines = [(0, 1), (1, 2), (2, 0)]
        self.angles = []
        for v, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
            ti, tj = self.direction(v, i), self.direction(v, j)
            if (tj - ti) % 360 > 180:
                i, j, ti, tj = j, i, tj, ti
            self.angles.append(Angle(v, (i, j), measures[v], ti, ti + (tj - ti) % 360))
        cx, cy = sum(x for x, _ in self.pts) / 3, sum(y for _, y in self.pts) / 3
        self.name_dirs = [math.degrees(math.atan2(y - cy, x - cx)) for x, y in self.pts]
        shortest = min(math.dist(self.pts[i], self.pts[j]) for i, j in self.lines)
        self.rm, self.reach = 0.25 * shortest, shortest

    def inside(self, xy: tuple[float, float]) -> bool:
        """Measures of a triangle are written inside it."""
        if self.layout != "triangle":
            return True
        crosses = [
            (bx - ax) * (xy[1] - ay) - (by - ay) * (xy[0] - ax)
            for (ax, ay), (bx, by) in ((self.pts[i], self.pts[j]) for i, j in self.lines)
        ]
        return all(c > 0 for c in crosses) or all(c < 0 for c in crosses)

    def sector_box(self, a: Angle, extra: float = 0) -> list[float]:
        (x, y), r = self.pts[a.vertex], self.rm + extra
        return [x - r, y - r, x + r, y + r]

    def place_labels(self) -> bool:
        """Point names outside the figure and measures on the bisectors, each the nearest position clear of lines,
        sectors, dots and other text; False when some label does not fit."""
        blocked = Image.new("L", (self.w, self.h), 0)
        d = ImageDraw.Draw(blocked)
        pad = self.pad
        for i, j in self.lines:
            d.line([self.pts[i], self.pts[j]], fill=255, width=round(self.lw + 2 * pad))
        for a in self.angles:
            d.pieslice(self.sector_box(a, pad), a.t1, a.t2, fill=255)
        for x, y in self.pts:
            d.ellipse([x - self.dot - pad, y - self.dot - pad, x + self.dot + pad, y + self.dot + pad], fill=255)
        boxes: list[tuple[float, float, float, float]] = []

        def first_free(origin, t: float, text: str, fnt, start: float, stop: float, measure: bool):
            for dist in range(round(start), round(stop), max(1, round(pad / 4))):
                xy = (origin[0] + dist * math.cos(math.radians(t)), origin[1] + dist * math.sin(math.radians(t)))
                x0, y0, x1, y1 = box = text_box(xy, text, fnt)
                if (
                    pad <= x0
                    and pad <= y0
                    and x1 <= self.w - pad
                    and y1 <= self.h - pad
                    and (not measure or self.inside(xy))
                    and not blocked.crop((math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))).getbbox()
                    and not any(
                        x0 < b[2] + pad and b[0] < x1 + pad and y0 < b[3] + pad and b[1] < y1 + pad for b in boxes
                    )
                ):
                    boxes.append(box)
                    return xy
            return None

        self.name_xy = []
        for p, name, t in zip(self.pts, self.names, self.name_dirs, strict=True):
            xy = first_free(p, t, name, self.name_font, self.dot, self.dot + 6 * self.name_font.size, measure=False)
            if xy is None:
                return False
            self.name_xy.append(xy)
        for a in self.angles:
            xy = first_free(
                self.pts[a.vertex], (a.t1 + a.t2) / 2, a.label, self.font, self.rm, self.reach, measure=True
            )
            if xy is None:
                return False
            a.label_xy = xy
        return True

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        for a in self.angles:
            d.pieslice(self.sector_box(a), a.t1, a.t2, fill=COLORS[a.color])
            d.arc(self.sector_box(a), a.t1, a.t2, fill=(30, 30, 30), width=max(2, self.lw // 2))
        for i, j in self.lines:
            d.line([self.pts[i], self.pts[j]], fill=(30, 30, 30), width=self.lw)
        for (x, y), name, xy in zip(self.pts, self.names, self.name_xy, strict=True):
            d.ellipse([x - self.dot, y - self.dot, x + self.dot, y + self.dot], fill=(0, 0, 0))
            d.text(xy, name, fill=(0, 0, 0), font=self.name_font, anchor="mm")
        for a in self.angles:
            d.text(a.label_xy, a.label, fill=(0, 0, 0), font=self.font, anchor="mm")
        return jpeg(img)

    def mask(self, a: Angle) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).pieslice(self.sector_box(a), a.t1, a.t2, fill=255)
        return rle.encode(m)

    def name(self, a: Angle) -> str:
        x, y = sorted(self.names[i] for i in a.arms)
        return f"angle {x}{self.names[a.vertex]}{y}"


class BarScene:
    """A vertical bar chart: a title, 4-7 named categories, a value axis with grid lines, values written above the
    bars in 60% of the charts, and bars of one colour or of distinct colours."""

    def __init__(self, rng: random.Random):
        self.w, self.h = rng.choice(WIDE_SIZES)
        s = math.sqrt(self.w * self.h) / 1800
        self.title, pool = rng.choice(list(BAR_THEMES.items()))
        self.font, self.title_font = font(36 * s), font(48 * s)
        self.left, self.right = round(0.12 * self.w), round(0.96 * self.w)
        self.top, self.bottom = round(0.15 * self.h), round(0.84 * self.h)
        self.axis, self.gap = max(3, round(4 * s)), 0.012 * self.h
        while True:
            n = rng.randint(4, 7)
            self.names = rng.sample(pool, n)
            self.slot = (self.right - self.left) / n
            if all(
                text_box((0, 0), x, self.font)[2] - text_box((0, 0), x, self.font)[0] <= 0.9 * self.slot
                for x in self.names
            ):
                break
        self.absent = [x for x in pool if x not in self.names]
        self.step = rng.choice([5, 10, 20, 50])
        self.vmax = self.step * rng.randint(5, 8)
        self.values = rng.sample(range(round(0.1 * self.vmax), round(0.9 * self.vmax) + 1), n)
        self.show_values = rng.random() < 0.6
        self.colors = rng.sample(PIE_COLORS, n) if rng.random() < 0.5 else [rng.choice(PIE_COLORS)] * n
        self.clear = 0.06 * self.vmax  # a difference a reader sees without the printed values

    def py(self, v: float) -> float:
        return self.bottom - v / self.vmax * (self.bottom - self.top)

    def rect(self, i: int) -> list[int]:
        xc = self.left + (i + 0.5) * self.slot
        return [
            round(xc - 0.3 * self.slot),
            round(self.py(self.values[i])),
            round(xc + 0.3 * self.slot),
            self.bottom - 1,
        ]

    def render(self) -> bytes:
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        black = (0, 0, 0)
        d.text((self.w / 2, 0.07 * self.h), self.title, fill=black, font=self.title_font, anchor="mm")
        for t in range(0, self.vmax + 1, self.step):
            if t:
                d.line([(self.left, self.py(t)), (self.right, self.py(t))], fill=(220, 220, 220), width=2)
            d.text((self.left - 2 * self.gap, self.py(t)), str(t), fill=black, font=self.font, anchor="rm")
        for i, name in enumerate(self.names):
            x0, y0, x1, y1 = self.rect(i)
            d.rectangle([x0, y0, x1, y1], fill=COLORS[self.colors[i]], outline=(40, 40, 40), width=2)
            d.text(((x0 + x1) / 2, self.bottom + self.axis + self.gap), name, fill=black, font=self.font, anchor="ma")
            if self.show_values:
                d.text(((x0 + x1) / 2, y0 - self.gap), str(self.values[i]), fill=black, font=self.font, anchor="md")
        d.rectangle([self.left - self.axis, self.bottom, self.right, self.bottom + self.axis - 1], fill=black)
        d.rectangle([self.left - self.axis, self.top, self.left - 1, self.bottom], fill=black)
        return jpeg(img)

    def mask(self, i: int) -> dict:
        m = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(m).rectangle(self.rect(i), fill=255)
        return rle.encode(m)


# Each generator returns (task_type, question, answer, masks) or None when the scene does not support it.


def gen_count_shape(rng, sc: ShapeScene):
    kinds = [k for k in KINDS if 1 <= len(sc.of(k)) <= 15]
    if not kinds:
        return None
    k = rng.choice(kinds)
    return (
        "counting",
        f"How many {plural(k)} are there in the figure?",
        str(len(sc.of(k))),
        [sc.mask(s) for s in sc.of(k)],
    )


def gen_count_color(rng, sc: ShapeScene):
    opts = [(k, c) for k in KINDS for c in COLORS if sc.of(k, c) and len({s.color for s in sc.of(k)}) >= 2]
    if not opts:
        return None
    k, c = rng.choice(opts)
    shapes = sc.of(k, c)
    return "counting", f"How many {c} {plural(k)} are there?", str(len(shapes)), [sc.mask(s) for s in shapes]


def gen_color(rng, sc: ShapeScene):
    opts = [s for s in sc.shapes if s.label and len(sc.of(s.kind)) >= 2]
    if not opts:
        return None
    s = rng.choice(opts)
    return "attribute", f"What color is the {s.kind} labelled {s.label}?", s.color, [sc.mask(s)]


def gen_style(rng, sc: ShapeScene):
    opts = [(s, d) for s in sc.shapes if (d := sc.describe(s))]
    if not opts:
        return None
    s, d = rng.choice(opts)
    answer = "it is filled" if s.filled else "it is only outlined"
    return "attribute", f"Is {d} filled or only outlined?", answer, [sc.mask(s)]


def gen_sides(rng, sc: ShapeScene):
    opts = [s for s in sc.shapes if s.label and s.kind in SIDES]
    if not opts:
        return None
    s = rng.choice(opts)
    return "attribute", f"How many sides does the shape labelled {s.label} have?", str(SIDES[s.kind]), [sc.mask(s)]


def gen_identify_label(rng, sc: ShapeScene):
    opts = [s for s in sc.shapes if s.label]
    if not opts:
        return None
    s = rng.choice(opts)
    return "identification", f"What shape is labelled {s.label}?", f"{article(s.kind)} {s.kind}", [sc.mask(s)]


def gen_identify_largest(rng, sc: ShapeScene):
    by_area = sorted(sc.shapes, key=lambda s: -s.area())
    s = by_area[0]
    if s.area() < 1.5 * by_area[1].area() or len(sc.of(s.kind, s.color)) != 1:
        return None
    return "identification", "What is the largest shape in the figure?", f"the {s.color} {s.kind}", [sc.mask(s)]


def gen_spatial_relation(rng, sc: ShapeScene):
    gap = 0.02 * max(sc.w, sc.h)
    named = [(s, d) for s in sc.shapes if (d := sc.describe(s))]
    opts = []
    for a, da in named:
        for b, db in named:
            if a is b or a.kind == b.kind:
                continue
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a.box, b.box
            if ax1 + gap < bx0 or bx1 + gap < ax0:
                opts.append((da, db, "lr", "to the left" if ax1 < bx0 else "to the right", a))
            elif ay1 + gap < by0 or by1 + gap < ay0:
                opts.append((da, db, "ab", "above" if ay1 < by0 else "below", a))
    if not opts:
        return None
    da, db, axis, answer, a = rng.choice(opts)
    q = f"Is {da} to the left or to the right of {db}?" if axis == "lr" else f"Is {da} above or below {db}?"
    return "spatial", q, answer, [sc.mask(a)]


def gen_spatial_count(rng, sc: ShapeScene):
    opts = []
    for a in sc.shapes:
        if not (d := sc.describe(a)):
            continue
        others = [s for s in sc.shapes if s is not a]
        for axis in ("above", "below", "left", "right"):
            i0, i1 = (1, 3) if axis in ("above", "below") else (0, 2)
            before = [s for s in others if s.box[i1] < a.box[i0]]
            after = [s for s in others if s.box[i0] > a.box[i1]]
            if len(before) + len(after) != len(others):  # some shape straddles the line
                continue
            chosen = before if axis in ("above", "left") else after
            if 1 <= len(chosen) <= 12:
                opts.append((a, d, axis, chosen))
    if not opts:
        return None
    a, d, axis, chosen = rng.choice(opts)
    where = {"above": "above", "below": "below", "left": "to the left of", "right": "to the right of"}[axis]
    return "spatial", f"How many shapes are entirely {where} {d}?", str(len(chosen)), [sc.mask(s) for s in chosen]


def gen_nearest(rng, sc: ShapeScene):
    opts = []
    for a in sc.shapes:
        if not (da := sc.describe(a)):
            continue
        gaps = sorted(
            (math.hypot(a.cx - s.cx, a.cy - s.cy) - a.r - s.r, i) for i, s in enumerate(sc.shapes) if s is not a
        )
        near = sc.shapes[gaps[0][1]]
        if gaps[0][0] <= 0.6 * gaps[1][0] and (dn := sc.describe(near)):
            opts.append((da, dn, near))
    if not opts:
        return None
    da, dn, near = rng.choice(opts)
    return "spatial", f"Which shape is closest to {da}?", dn, [sc.mask(near)]


def gen_compare_size(rng, sc: ShapeScene):
    named = [(s, d) for s in sc.shapes if (d := sc.describe(s))]
    opts = [(a, da, b, db) for a, da in named for b, db in named if a is not b and a.area() >= 1.8 * b.area()]
    if not opts:
        return None
    a, da, _, db = rng.choice(opts)
    x, y = (da, db) if rng.random() < 0.5 else (db, da)
    return "comparison", f"Which is larger, {x} or {y}?", da, [sc.mask(a)]


def gen_compare_count(rng, sc: ShapeScene):
    opts = [(x, y) for x in KINDS for y in KINDS if x < y and sc.of(x) and sc.of(y) and len(sc.of(x)) != len(sc.of(y))]
    if not opts:
        return None
    x, y = rng.choice(opts)
    more = x if len(sc.of(x)) > len(sc.of(y)) else y
    return (
        "comparison",
        f"Are there more {plural(x)} or {plural(y)} in the figure?",
        plural(more),
        [sc.mask(s) for s in sc.of(more)],
    )


def gen_ocr_label(rng, sc: ShapeScene):
    opts = [(s, d) for s in sc.shapes if s.label and (d := sc.describe(s, use_label=False))]
    if not opts:
        return None
    s, d = rng.choice(opts)
    what = "letter" if s.label.isalpha() else "number"
    return "OCR", f"What {what} is written inside {d}?", s.label, [sc.mask(s)]


def gen_count_segments(rng, gs: GeometryScene):
    opts = [(i, segs) for i in range(len(gs.pts)) if len(segs := [g for g in gs.segments if i in (g.a, g.b)]) >= 1]
    if not opts:
        return None
    i, segs = rng.choice(opts)
    return (
        "counting",
        f"How many drawn segments have point {gs.names[i]} as an endpoint?",
        str(len(segs)),
        [gs.segment_mask(g) for g in segs],
    )


def gen_point_in_circle(rng, gs: GeometryScene):
    if not gs.circle:
        return None
    (c, r) = gs.circle
    inside = [i for i, p in enumerate(gs.pts) if math.dist(c, p) < r]
    if len(inside) != 1:
        return None
    return (
        "spatial",
        "Which labelled point lies inside the blue circle?",
        gs.names[inside[0]],
        [gs.point_mask(inside[0])],
    )


def gen_longer_segment(rng, gs: GeometryScene):
    length = {id(g): math.dist(gs.pts[g.a], gs.pts[g.b]) for g in gs.segments}
    opts = [
        (a, b)
        for a in gs.segments
        for b in gs.segments
        if a is not b and length[id(a)] >= 1.25 * length[id(b)] and not {a.a, a.b} & {b.a, b.b}
    ]
    if not opts:
        return None
    a, b = rng.choice(opts)
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    return (
        "comparison",
        f"Which segment is longer, {gs.seg_name(x)} or {gs.seg_name(y)}?",
        gs.seg_name(a),
        [gs.segment_mask(a)],
    )


def gen_side_length(rng, gs: GeometryScene):
    opts = [g for g in gs.segments if g.label]
    if not opts:
        return None
    g = rng.choice(opts)
    return "OCR", f"What length is written next to segment {gs.seg_name(g)}?", g.label, [gs.segment_mask(g)]


def listing(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def gen_plot_region(rng, ps: PlotScene):
    """Lettered regions between two curves; asked are the enclosed ones where the first curve is on top."""
    pairs = [
        (f, g, cross)
        for f in ps.curves
        for g in ps.curves
        if f is not g and (cross := ps.crossings(f, g)) and len(cross) >= 2
    ]
    if not pairs:
        return None
    f, g, cross = rng.choice(pairs)
    ps.curves = [f, g]
    unit = (ps.bottom - ps.top) / (ps.y1 - ps.y0)  # pixels per y unit
    room = 2.6 * ps.letter_h

    def widest(a: float, b: float, gap: Callable[[float], float]) -> float | None:
        """x on [a, b], away from its ends, where gap (in y units) is largest and leaves room for a letter."""
        xs = [x for x in ps.xs(a, b) if min(ps.px(x) - ps.px(a), ps.px(b) - ps.px(x)) >= 1.2 * ps.letter_h]
        x = max(xs, key=gap, default=None)
        return x if x is not None and gap(x) * unit >= room else None

    spots = []  # (letter position, enclosed interval or None, first curve on top)
    for a, b in zip([ps.x0] + cross, cross + [ps.x1], strict=True):
        x = widest(a, b, lambda x: abs(f.fn(x) - g.fn(x)))
        enclosed = a != ps.x0 and b != ps.x1
        if x is None:
            if enclosed:
                return None
            continue
        on_top = f.fn(x) > g.fn(x)
        if enclosed or not on_top:  # a region cut by the plot edge gets a letter only as a distractor
            spots.append(((ps.px(x), ps.py((f.fn(x) + g.fn(x)) / 2)), (a, b) if enclosed else None, on_top))
    for edge, outer in ((ps.y1, max), (ps.y0, min)):  # distractors above and below both curves
        # mypy cannot type min/max bound as a default argument
        x = widest(ps.x0, ps.x1, lambda x, edge=edge, outer=outer: abs(edge - outer(f.fn(x), g.fn(x))))  # type: ignore[misc]
        if x is not None:
            spots.append(((ps.px(x), ps.py((edge + outer(f.fn(x), g.fn(x))) / 2)), None, False))
    rng.shuffle(spots)
    hit = [(LETTERS[i], interval) for i, (_, interval, on_top) in enumerate(spots) if interval and on_top]
    if not hit or len(spots) < 3:
        return None
    ps.letters = [(LETTERS[i], xy) for i, (xy, _, _) in enumerate(spots)]
    fa, ga = f.label.removeprefix("y = "), g.label.removeprefix("y = ")
    names = [letter for letter, _ in hit]
    return (
        "spatial",
        f"Which labelled regions are enclosed between {f.label} and {g.label} where {ga} < {fa}?",
        ("region " if len(names) == 1 else "regions ") + listing(names),
        [ps.region_mask(f, g, *interval) for _, interval in hit],
    )


MISLABELLED = "Which curve in the figure is labelled incorrectly in the legend?"


def gen_plot_mislabelled(rng, ps: PlotScene):
    if len(ps.curves) < 3:
        return None
    c = rng.choice(ps.curves)
    c.label = rng.choice([n for n in FUNCS if n not in {x.label for x in ps.curves}])
    return "OCR", MISLABELLED, ps.describe(c), [ps.mask(c)]


def gen_no_mislabel(rng, ps: PlotScene, task: str):
    if len(ps.curves) < 3:
        return None
    return "OCR", MISLABELLED, "none of the curves is labelled incorrectly.", []


def gen_plot_compare(rng, ps: PlotScene):
    top = {id(c): max(c.fn(x) for x in ps.xs(ps.x0, ps.x1)) for c in ps.curves}
    opts = [(a, b) for a in ps.curves for b in ps.curves if top[id(a)] >= 1.5 * top[id(b)]]
    if not opts:
        return None
    a, b = rng.choice(opts)
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    return (
        "comparison",
        f"Which curve reaches a higher maximum value, {ps.describe(x)} or {ps.describe(y)}?",
        ps.describe(a),
        [ps.mask(a)],
    )


def gen_plot_color(rng, ps: PlotScene):
    c = rng.choice(ps.curves)
    return "attribute", f"What color is the curve labelled {c.label}?", c.color, [ps.mask(c)]


def gen_plot_legend(rng, ps: PlotScene):
    c = rng.choice(ps.curves)
    return "OCR", f"What is the legend label of {ps.describe(c)}?", c.label, [ps.mask(c)]


def gen_plot_identify(rng, ps: PlotScene):
    if not ps.x0 < 0 < ps.x1:
        return None
    at0 = [round(c.fn(0), 2) for c in ps.curves]
    opts = [c for c, v in zip(ps.curves, at0, strict=True) if v != 0 and at0.count(v) == 1]
    if not opts:
        return None
    c = rng.choice(opts)
    return "identification", f"Which plotted function has the value {c.fn(0):g} at x = 0?", c.label, [ps.mask(c)]


def gen_pie_inconsistent(rng, pie: PieScene):
    wrong = sorted(rng.sample(range(len(pie.names)), rng.randint(1, 2)))
    for i in wrong:
        delta = rng.randint(6, 15)
        pie.shown[i] = pie.pcts[i] + (delta if pie.pcts[i] - delta < 2 or rng.random() < 0.5 else -delta)
    return (
        "OCR",
        "Which sections of the pie chart show a percentage that is inconsistent with the values in the table?",
        listing([pie.names[i] for i in wrong]),
        [pie.mask(i) for i in wrong],
    )


def gen_pie_top(rng, pie: PieScene):
    first, second = sorted(range(len(pie.names)), key=lambda i: -pie.pcts[i])[:2]
    if pie.pcts[first] - pie.pcts[second] < 4:
        return None
    return (
        "identification",
        "Which category has the largest section of the pie chart?",
        pie.names[first],
        [pie.mask(first)],
    )


def gen_pie_compare(rng, pie: PieScene):
    n = len(pie.names)
    opts = [(i, j) for i in range(n) for j in range(n) if pie.pcts[i] - pie.pcts[j] >= 4]
    if not opts:
        return None
    i, j = rng.choice(opts)
    x, y = (i, j) if rng.random() < 0.5 else (j, i)
    return (
        "comparison",
        f"Which category has the larger share, {pie.names[x]} or {pie.names[y]}?",
        pie.names[i],
        [pie.mask(i)],
    )


def gen_pie_percent(rng, pie: PieScene):
    i = rng.randrange(len(pie.names))
    return (
        "OCR",
        f"What percentage is written in the {pie.names[i]} section of the pie chart?",
        f"{pie.pcts[i]}%",
        [pie.mask(i)],
    )


def gen_pie_color(rng, pie: PieScene):
    i = rng.randrange(len(pie.names))
    return "attribute", f"What color is the {pie.names[i]} section of the pie chart?", pie.colors[i], [pie.mask(i)]


def gen_pie_count(rng, pie: PieScene):
    opts = [t for t in (15, 20, 25, 30) if t not in pie.pcts and any(p > t for p in pie.pcts)]
    if not opts:
        return None
    t = rng.choice(opts)
    big = [i for i, p in enumerate(pie.pcts) if p > t]
    return (
        "counting",
        f"How many sections of the pie chart are larger than {t}%?",
        str(len(big)),
        [pie.mask(i) for i in big],
    )


def pip_phrase(v: int) -> str:
    return f"{NUMBER_WORDS[v]} {'pip' if v == 1 else 'pips'}"


def gen_dice_pips(rng, ds: DiceScene):
    colors = sorted({d.color for d in ds.dice})
    opts = [c for c in colors if len(ds.of(c)) <= 4 and sum(d.value for d in ds.of(c)) <= 20]
    if len(colors) < 2 or not opts:
        return None
    c = rng.choice(opts)
    dice = ds.of(c)
    q = f"How many pips are on the {c} die?" if len(dice) == 1 else f"What is the total number of pips on the {c} dice?"
    return "counting", q, str(sum(d.value for d in dice)), [m for d in dice for m in ds.pip_masks(d)]


def gen_dice_count_value(rng, ds: DiceScene):
    v = rng.choice(sorted({d.value for d in ds.dice}))
    dice = [d for d in ds.dice if d.value == v]
    return "counting", f"How many dice show exactly {pip_phrase(v)}?", str(len(dice)), [ds.mask(d) for d in dice]


def gen_dice_compare(rng, ds: DiceScene):
    uniq = ds.unique()
    opts = [(a, b) for a in uniq for b in uniq if a.value > b.value]
    if not opts:
        return None
    a, b = rng.choice(opts)
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    return (
        "comparison",
        f"Which die shows more pips, the {x.color} one or the {y.color} one?",
        f"the {a.color} die",
        [ds.mask(a)],
    )


def gen_dice_color(rng, ds: DiceScene):
    opts = [v for v in range(1, 7) if 1 <= sum(d.value == v for d in ds.dice) <= 3]
    if not opts:
        return None
    v = rng.choice(opts)
    dice = [d for d in ds.dice if d.value == v]
    q = (
        f"What color is the die showing {pip_phrase(v)}?"
        if len(dice) == 1
        else f"What colors are the dice showing {pip_phrase(v)}?"
    )
    return "attribute", q, listing(list(dict.fromkeys(d.color for d in dice))), [ds.mask(d) for d in dice]


def gen_dice_max(rng, ds: DiceScene):
    top = max(d.value for d in ds.dice)
    best = [d for d in ds.dice if d.value == top]
    if len(best) != 1 or len(ds.of(best[0].color)) != 1:
        return None
    return (
        "identification",
        "Which die shows the highest number of pips?",
        f"the {best[0].color} die",
        [ds.mask(best[0])],
    )


def absent(thing: str) -> str:
    return f"there is no {thing} in the figure."


def gen_absent_curve(rng, ps: PlotScene, task: str):
    label = rng.choice(ABSENT_FUNCS)
    color = rng.choice([c for c in CURVE_COLORS + ["pink", "brown"] if c not in {x.color for x in ps.curves}])
    other = ps.describe(rng.choice(ps.curves))
    by_label = {
        "attribute": f"What color is the curve labelled {label}?",
        "counting": f"How many times does the curve labelled {label} cross the x-axis in the plotted range?",
    }
    by_color = {
        "OCR": f"What is the legend label of the {color} curve?",
        "identification": f"Which function is drawn as the {color} curve?",
        "comparison": f"Which curve reaches a higher maximum value, the {color} curve or {other}?",
        "spatial": f"Is the {color} curve above or below {other} at x = 1?",
    }
    if task in by_label:
        return task, by_label[task], absent(f"curve labelled {label}"), []
    return task, by_color[task], absent(f"{color} curve"), []


def gen_absent_category(rng, pie: PieScene, task: str):
    name, other = rng.choice(pie.absent), rng.choice(pie.names)
    q = {
        "attribute": f"What color is the {name} section of the pie chart?",
        "OCR": f"What percentage is written in the {name} section of the pie chart?",
        "comparison": f"Which category has the larger share, {name} or {other}?",
        "counting": f"How many sections of the pie chart are larger than the {name} section?",
        "identification": f"Which section of the pie chart is next to the {name} section?",
        "spatial": f"Is the {name} section to the left or to the right of the {other} section?",
    }[task]
    return task, q, f"there is no {name} section in the pie chart.", []


def gen_absent_die(rng, ds: DiceScene, task: str):
    c = rng.choice([x for x in DIE_COLORS + ["pink"] if not ds.of(x)])
    if task == "attribute":
        missing = [v for v in range(1, 7) if all(d.value != v for d in ds.dice)]
        if not missing:
            return None
        v = rng.choice(missing)
        return task, f"What color is the die showing {pip_phrase(v)}?", absent(f"die showing {pip_phrase(v)}"), []
    if task in ("comparison", "spatial"):
        if not ds.unique():
            return None
        other = rng.choice(ds.unique()).color
        q = (
            f"Which die shows more pips, the {c} one or the {other} one?"
            if task == "comparison"
            else f"Is the {c} die above or below the {other} die?"
        )
        return task, q, absent(f"{c} die"), []
    if task == "identification":
        return task, f"Which die is closest to the {c} die?", absent(f"{c} die"), []
    return "counting", f"What is the total number of pips on the {c} dice?", f"there are no {c} dice in the figure.", []


def absent_colors(sc: ShapeScene) -> list[tuple[str, str]]:
    """(kind, colour) pairs where shapes of the kind exist, none in the colour or a colour near it."""
    return [
        (k, c)
        for k in KINDS
        if sc.of(k)
        for c in COLORS
        if not any(s.color == c or s.color in NEAR_COLORS[c] for s in sc.of(k))
    ]


def gen_absent_shape(rng, sc: ShapeScene, task: str):
    k = rng.choice(ABSENT_KINDS)
    named = [d for s in sc.shapes if (d := sc.describe(s))]
    if not named:
        return None
    other = rng.choice(named)
    q = {
        "counting": f"How many {plural(k)} are there in the figure?",
        "attribute": f"What color is the {k}?",
        "identification": f"What shape is directly to the left of the {k}?",
        "spatial": f"Is the {k} above or below {other}?",
        "comparison": f"Which is larger, the {k} or {other}?",
        "OCR": f"What is written inside the {k}?",
    }[task]
    return task, q, absent(k), []


def gen_absent_attribute(rng, sc: ShapeScene, task: str):
    opts = [(k, c) for k, c in absent_colors(sc)]
    named = [(s, d) for s in sc.shapes if (d := sc.describe(s))]
    if not opts or not named:
        return None
    k, c = rng.choice(opts)
    other = rng.choice([d for s, d in named if s.kind != k] or [d for _, d in named])
    q = {
        "counting": f"How many shapes are next to the {c} {k}?",
        "attribute": f"Is the {c} {k} filled or only outlined?",
        "identification": f"Which shape is closest to the {c} {k}?",
        "spatial": f"Is the {c} {k} to the left or to the right of {other}?",
        "comparison": f"Which is larger, the {c} {k} or {other}?",
        "OCR": f"What is written inside the {c} {k}?",
    }[task]
    return task, q, absent(f"{c} {k}"), []


def gen_absent_label(rng, sc: ShapeScene, task: str):
    used = {s.label for s in sc.shapes if s.label}
    label = rng.choice([c for c in LETTERS if c not in used])
    q = {
        "attribute": f"What color is the shape labelled {label}?",
        "identification": f"What shape is labelled {label}?",
        "counting": f"How many sides does the shape labelled {label} have?",
    }.get(task, f"What shape is labelled {label}?")
    task = task if task in ("attribute", "identification", "counting") else "identification"
    return task, q, absent(f"shape labelled {label}"), []


def gen_absent_segment(rng, gs: GeometryScene, task: str):
    opts = [(i, j) for i in range(len(gs.pts)) for j in range(i + 1, len(gs.pts)) if not gs.connected(i, j)]
    if not opts:
        return None
    i, j = rng.choice(opts)
    name = gs.names[i] + gs.names[j]
    q = {
        "OCR": f"What length is written next to segment {name}?",
        "comparison": f"Which segment is longer, {name} or the segment next to it?",
        "spatial": f"Does segment {name} cross the blue circle?",
    }.get(task, f"What length is written next to segment {name}?")
    task = task if task in ("OCR", "comparison", "spatial") else "OCR"
    return task, q, absent(f"segment {name}"), []


def gen_absent_reference(rng, sc: ShapeScene, task: str):
    """A present shape related to an absent one: the missing reference makes the whole question unanswerable."""
    named = [(s, d) for s in sc.shapes if (d := sc.describe(s))]
    recoloured = [f"{c} {k}" for k, c in absent_colors(sc)]
    if not named:
        return None
    ref = rng.choice(recoloured) if recoloured and rng.random() < 0.5 else rng.choice(ABSENT_KINDS)
    _, d = rng.choice(named)
    q = {
        "spatial": f"Is {d} to the left or to the right of the {ref}?",
        "counting": f"How many shapes are entirely above the {ref}?",
        "identification": f"Which shape is closest to the {ref}?",
        "comparison": f"Which is larger, {d} or the {ref}?",
    }
    task = task if task in q else rng.choice(sorted(q))
    return task, q[task], absent(ref), []


def gen_absent_shape_text(rng, sc: ShapeScene, task: str):
    """A present shape with nothing written inside: the place exists, the information does not."""
    # Stars never carry a label, so only shapes of kinds that can carry one are asked about.
    opts = [d for s in sc.shapes if not s.label and s.kind != "star" and (d := sc.describe(s, use_label=False))]
    if not opts:
        return None
    d = rng.choice(opts)
    return "OCR", f"What is written inside {d}?", f"nothing is written inside {d}.", []


def gen_absent_length(rng, gs: GeometryScene, task: str):
    """A drawn segment without a length label."""
    opts = [g for g in gs.segments if not g.label]
    if not opts:
        return None
    name = gs.seg_name(rng.choice(opts))
    return "OCR", f"What length is written next to segment {name}?", f"no length is written next to segment {name}.", []


def gen_grid_count(rng, g: GridScene):
    opts = [
        (f"How many {c} cells are there in the grid?", cells)
        for c in GRID_COLORS
        if 1 <= len(cells := g.of(color=c)) <= 12
    ]
    if g.kind != "number":
        opts += [
            (f"How many cells contain {g.what(x)}?", cells)
            for x in sorted({c.content for c in g.cells})
            if 1 <= len(cells := g.of(content=x)) <= 12
        ]
    if not opts:
        return None
    q, cells = rng.choice(opts)
    return "counting", q, str(len(cells)), [g.mask(c) for c in cells]


def gen_grid_color(rng, g: GridScene):
    opts = [(c, f"the cell containing {g.what(c.content)}") for c in g.cells if len(g.of(content=c.content)) == 1] or [
        (c, g.position(c)) for c in g.cells
    ]
    c, d = rng.choice(opts)
    return "attribute", f"What color is {d}?", c.color, [g.mask(c)]


def gen_grid_identify(rng, g: GridScene):
    if g.kind == "number":
        order = sorted(g.cells, key=lambda c: int(c.content))
        which, c = rng.choice([("largest", order[-1]), ("smallest", order[0])])
        if len(g.of(content=c.content)) != 1:
            return None
        return "identification", f"Which number in the grid is the {which}?", c.content, [g.mask(c)]
    contents = sorted({c.content for c in g.cells})
    once = [x for x in contents if len(g.of(content=x)) == 1]
    if len(once) != 1 or len(contents) < 3:
        return None
    c = g.of(content=once[0])[0]
    answer = f"the {c.content}" if g.kind == "symbol" else c.content
    return "identification", f"Which {g.kind} appears in exactly one cell?", answer, [g.mask(c)]


def gen_grid_neighbor(rng, g: GridScene):
    """The cell next to a uniquely described one; evidence is that cell, not the reference."""
    steps = {"above": (-1, 0), "below": (1, 0), "to the left of": (0, -1), "to the right of": (0, 1)}
    opts = [
        (d, rel, t)
        for c in g.cells
        if (d := g.describe(c))
        for rel, (dr, dc) in steps.items()
        if (t := g.at(c.row + dr, c.col + dc))
    ]
    if not opts:
        return None
    d, rel, t = rng.choice(opts)
    if rng.random() < 0.5:
        return "spatial", f"What color is the cell directly {rel} {d}?", t.color, [g.mask(t)]
    read = "Which symbol is in" if g.kind == "symbol" else "What is written in"
    return "spatial", f"{read} the cell directly {rel} {d}?", g.answer(t.content), [g.mask(t)]


def gen_grid_compare(rng, g: GridScene):
    present = [c for c in GRID_COLORS if g.of(color=c)]
    opts = [(a, b) for a in present for b in present if len(g.of(color=a)) >= len(g.of(color=b)) + 2]
    if not opts:
        return None
    a, b = rng.choice(opts)
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    return (
        "comparison",
        f"Are there more {x} cells or {y} cells in the grid?",
        f"{a} cells",
        [g.mask(c) for c in g.of(color=a)],
    )


def gen_grid_ocr(rng, g: GridScene):
    if g.kind == "symbol":
        return None
    single = [c for c in g.cells if c.color != "white" and len(g.of(color=c.color)) == 1]
    if single and rng.random() < 0.5:
        c = rng.choice(single)
        return "OCR", f"What {g.kind} is written in the {c.color} cell?", c.content, [g.mask(c)]
    c = rng.choice(g.cells)
    return "OCR", f"What is written in {g.position(c)}?", c.content, [g.mask(c)]


def gen_angle_measure(rng, sc: AngleScene):
    a = rng.choice(sc.angles)
    return "OCR", f"What degree measure is marked on {sc.name(a)}?", f"{a.measure}°", [sc.mask(a)]


def gen_angle_compare(rng, sc: AngleScene):
    opts = [(a, b) for a in sc.angles for b in sc.angles if a.measure >= b.measure + 15]
    if not opts:
        return None
    a, b = rng.choice(opts)
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    return "comparison", f"Which is larger, {sc.name(x)} or {sc.name(y)}?", sc.name(a), [sc.mask(a)]


def gen_angle_largest(rng, sc: AngleScene):
    if len(sc.angles) < 3:
        return None
    order = sorted(sc.angles, key=lambda a: a.measure)
    which, a, b = rng.choice([("largest", order[-1], order[-2]), ("smallest", order[0], order[1])])
    if abs(a.measure - b.measure) < 10:
        return None
    return "identification", f"Which marked angle is the {which}?", sc.name(a), [sc.mask(a)]


def gen_angle_count(rng, sc: AngleScene):
    opts = [
        (f"How many of the marked angles are {kind}?", angles)
        for kind, angles in (
            ("acute", [a for a in sc.angles if a.measure < 90]),
            ("obtuse", [a for a in sc.angles if a.measure > 90]),
        )
    ]
    opts = [(q, angles) for q, angles in opts if angles]
    opts += [
        (f"How many marked angles are larger than {t}°?", big)
        for t in range(30, 150, 10)
        if all(abs(a.measure - t) >= 5 for a in sc.angles)
        and 0 < len(big := [a for a in sc.angles if a.measure > t]) < len(sc.angles)
    ]
    q, angles = rng.choice(opts)
    return "counting", q, str(len(angles)), [sc.mask(a) for a in angles]


def gen_angle_type(rng, sc: AngleScene):
    a = rng.choice(sc.angles)
    if rng.random() < 0.5:
        answer = "it is acute" if a.measure < 90 else "it is obtuse"
        return "attribute", f"Is {sc.name(a)} acute or obtuse?", answer, [sc.mask(a)]
    return "attribute", f"What color is the marking of {sc.name(a)}?", a.color, [sc.mask(a)]


def gen_angle_unknown(rng, sc: AngleScene):
    """One angle of a triangle or of a straight line is marked x; its measure follows from the others."""
    if sc.layout == "fan":
        return None
    a = rng.choice(sc.angles)
    a.label = "x"
    if not sc.place_labels():
        return None
    return "attribute", "What is the measure of the angle marked x?", f"{a.measure}°", [sc.mask(a)]


def gen_angle_between(rng, sc: AngleScene):
    if sc.layout == "triangle":
        a = rng.choice(sc.angles)
        side = "".join(sorted(sc.names[i] for i in a.arms))
        return "spatial", f"Which angle of the triangle is opposite side {side}?", sc.name(a), [sc.mask(a)]
    if len(sc.angles) < 3:
        return None
    i = rng.randrange(1, len(sc.angles) - 1)
    before, a, after = sc.angles[i - 1 : i + 2]
    return (
        "spatial",
        f"Which marked angle lies between {sc.name(before)} and {sc.name(after)}?",
        sc.name(a),
        [sc.mask(a)],
    )


def gen_bar_value(rng, bs: BarScene):
    if not bs.show_values:
        return None
    i = rng.randrange(len(bs.names))
    return "OCR", f"What value is written above the {bs.names[i]} bar?", str(bs.values[i]), [bs.mask(i)]


def gen_bar_count(rng, bs: BarScene):
    """Thresholds are grid lines that no bar is close to, so the count is visible without the printed values."""
    opts = [
        t
        for t in range(bs.step, bs.vmax, bs.step)
        if all(abs(v - t) >= bs.clear for v in bs.values) and 0 < sum(v > t for v in bs.values) < len(bs.values)
    ]
    if not opts:
        return None
    t = rng.choice(opts)
    big = [i for i, v in enumerate(bs.values) if v > t]
    return "counting", f"How many bars have a value greater than {t}?", str(len(big)), [bs.mask(i) for i in big]


def gen_bar_color(rng, bs: BarScene):
    if len(set(bs.colors)) == 1:
        return None
    i = rng.randrange(len(bs.names))
    return "attribute", f"What color is the {bs.names[i]} bar?", bs.colors[i], [bs.mask(i)]


def gen_bar_identify(rng, bs: BarScene):
    order = sorted(range(len(bs.names)), key=lambda i: bs.values[i])
    which, i, j = rng.choice([("highest", order[-1], order[-2]), ("lowest", order[0], order[1])])
    if abs(bs.values[i] - bs.values[j]) < bs.clear:
        return None
    return "identification", f"Which category has the {which} value?", bs.names[i], [bs.mask(i)]


def gen_bar_spatial(rng, bs: BarScene):
    n = len(bs.names)
    side = rng.choice(["left", "right"])
    if rng.random() < 0.5:
        i = rng.randrange(n)
        idx = list(range(i)) if side == "left" else list(range(i + 1, n))
        if not idx:
            return None
        return (
            "spatial",
            f"How many bars are to the {side} of the {bs.names[i]} bar?",
            str(len(idx)),
            [bs.mask(j) for j in idx],
        )
    order = sorted(range(n), key=lambda i: bs.values[i])
    which, i, j = rng.choice([("tallest", order[-1], order[-2]), ("shortest", order[0], order[1])])
    k = i - 1 if side == "left" else i + 1
    if abs(bs.values[i] - bs.values[j]) < bs.clear or not 0 <= k < n:
        return None
    return "spatial", f"Which category's bar is directly to the {side} of the {which} bar?", bs.names[k], [bs.mask(k)]


def gen_bar_compare(rng, bs: BarScene):
    n = len(bs.names)
    opts = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if bs.values[i] >= 1.25 * bs.values[j] and bs.values[i] - bs.values[j] >= bs.clear
    ]
    if not opts:
        return None
    i, j = rng.choice(opts)
    which, answer = ("higher", i) if rng.random() < 0.5 else ("lower", j)
    x, y = (i, j) if rng.random() < 0.5 else (j, i)
    return (
        "comparison",
        f"Which category has the {which} value, {bs.names[x]} or {bs.names[y]}?",
        bs.names[answer],
        [bs.mask(answer)],
    )


def gen_absent_grid(rng, g: GridScene, task: str):
    """A cell content or a cell colour that is not in the grid."""
    contents, colors = {c.content for c in g.cells}, {c.color for c in g.cells}
    pool = [str(n) for n in range(1, 100)] if g.kind == "number" else LETTERS if g.kind == "letter" else SYMBOLS
    missing = [x for x in pool if x not in contents]
    gone = [c for c in GRID_COLORS + ["brown"] if c not in colors and not NEAR_COLORS[c] & colors]
    if not missing or not gone:
        return None
    x, color = rng.choice(missing), rng.choice(gone)
    thing = f"cell containing {g.what(x)}"
    read = "Which symbol is in" if g.kind == "symbol" else "What is written in"
    q = {
        "counting": (f"How many {color} cells are there in the grid?", f"there are no {color} cells in the figure."),
        "attribute": (f"What color is the {thing}?", absent(thing)),
        "identification": (f"{read} the cell directly to the right of the {thing}?", absent(thing)),
        "spatial": (f"What color is the cell directly below the {thing}?", absent(thing)),
        "OCR": (f"{read} the {color} cell?", absent(f"{color} cell")),
    }
    task = task if task in q else "spatial"
    return task, *q[task], []


def gen_absent_angle(rng, sc: AngleScene, task: str):
    """An angle at a point that is not in the figure, or a marking colour that is not used."""
    real = rng.choice(sc.angles)
    used = {a.color for a in sc.angles}
    gone = [c for c in ANGLE_COLORS if c not in used and not NEAR_COLORS[c] & used]
    if gone and rng.random() < 0.5:
        thing = f"angle marked in {rng.choice(gone)}"
        p = f"the {thing}"
    else:
        x, y = sorted([sc.names[rng.choice(real.arms)], rng.choice([c for c in LETTERS if c not in sc.names])])
        thing = p = f"angle {x}{sc.names[real.vertex]}{y}"
    q = {
        "counting": f"How many marked angles are larger than {p}?",
        "attribute": f"Is {p} acute or obtuse?",
        "identification": f"Which marked angle is next to {p}?",
        "spatial": f"Which marked angle lies between {p} and {sc.name(real)}?",
        "comparison": f"Which is larger, {p} or {sc.name(real)}?",
        "OCR": f"What degree measure is marked on {p}?",
    }[task]
    return task, q, absent(thing), []


def gen_absent_bar(rng, bs: BarScene, task: str):
    name, other = rng.choice(bs.absent), rng.choice(bs.names)
    q = {
        "counting": f"How many bars are taller than the {name} bar?",
        "attribute": f"What color is the {name} bar?",
        "identification": f"Which category's bar is directly to the right of the {name} bar?",
        "spatial": f"Is the {name} bar to the left or to the right of the {other} bar?",
        "comparison": f"Which category has the higher value, {name} or {other}?",
        "OCR": f"What value is written above the {name} bar?",
    }[task]
    return task, q, absent(f"{name} bar"), []


# name -> (scene class, generator); a generator returns (task, question, answer, masks) or None
Generators = dict[str, tuple[Callable[[random.Random], Any], Callable[..., Any]]]
GENERATORS: Generators = {
    "math_count_shape": (ShapeScene, gen_count_shape),
    "math_count_color": (ShapeScene, gen_count_color),
    "math_count_segments": (GeometryScene, gen_count_segments),
    "math_color": (ShapeScene, gen_color),
    "math_style": (ShapeScene, gen_style),
    "math_sides": (ShapeScene, gen_sides),
    "math_identify_label": (ShapeScene, gen_identify_label),
    "math_identify_largest": (ShapeScene, gen_identify_largest),
    "math_spatial_relation": (ShapeScene, gen_spatial_relation),
    "math_spatial_count": (ShapeScene, gen_spatial_count),
    "math_nearest": (ShapeScene, gen_nearest),
    "math_point_in_circle": (GeometryScene, gen_point_in_circle),
    "math_compare_size": (ShapeScene, gen_compare_size),
    "math_compare_count": (ShapeScene, gen_compare_count),
    "math_longer_segment": (GeometryScene, gen_longer_segment),
    "math_ocr_label": (ShapeScene, gen_ocr_label),
    "math_side_length": (GeometryScene, gen_side_length),
    "math_plot_region": (PlotScene, gen_plot_region),
    "math_plot_mislabelled": (PlotScene, gen_plot_mislabelled),
    "math_plot_compare": (PlotScene, gen_plot_compare),
    "math_plot_color": (PlotScene, gen_plot_color),
    "math_plot_legend": (PlotScene, gen_plot_legend),
    "math_plot_identify": (PlotScene, gen_plot_identify),
    "math_pie_inconsistent": (PieScene, gen_pie_inconsistent),
    "math_pie_top": (PieScene, gen_pie_top),
    "math_pie_compare": (PieScene, gen_pie_compare),
    "math_pie_percent": (PieScene, gen_pie_percent),
    "math_pie_color": (PieScene, gen_pie_color),
    "math_pie_count": (PieScene, gen_pie_count),
    "math_dice_pips": (DiceScene, gen_dice_pips),
    "math_dice_count_value": (DiceScene, gen_dice_count_value),
    "math_dice_compare": (DiceScene, gen_dice_compare),
    "math_dice_color": (DiceScene, gen_dice_color),
    "math_dice_max": (DiceScene, gen_dice_max),
    "math_grid_count": (GridScene, gen_grid_count),
    "math_grid_color": (GridScene, gen_grid_color),
    "math_grid_identify": (GridScene, gen_grid_identify),
    "math_grid_neighbor": (GridScene, gen_grid_neighbor),
    "math_grid_compare": (GridScene, gen_grid_compare),
    "math_grid_ocr": (GridScene, gen_grid_ocr),
    "math_angle_measure": (AngleScene, gen_angle_measure),
    "math_angle_compare": (AngleScene, gen_angle_compare),
    "math_angle_largest": (AngleScene, gen_angle_largest),
    "math_angle_count": (AngleScene, gen_angle_count),
    "math_angle_type": (AngleScene, gen_angle_type),
    "math_angle_unknown": (AngleScene, gen_angle_unknown),
    "math_angle_between": (AngleScene, gen_angle_between),
    "math_bar_value": (BarScene, gen_bar_value),
    "math_bar_count": (BarScene, gen_bar_count),
    "math_bar_color": (BarScene, gen_bar_color),
    "math_bar_identify": (BarScene, gen_bar_identify),
    "math_bar_spatial": (BarScene, gen_bar_spatial),
    "math_bar_compare": (BarScene, gen_bar_compare),
}
ABSENT: Generators = {
    "math_absent_shape": (ShapeScene, gen_absent_shape),
    "math_absent_attribute": (ShapeScene, gen_absent_attribute),
    "math_absent_label": (ShapeScene, gen_absent_label),
    "math_absent_segment": (GeometryScene, gen_absent_segment),
    "math_absent_reference": (ShapeScene, gen_absent_reference),
    "math_absent_shape_text": (ShapeScene, gen_absent_shape_text),
    "math_absent_length": (GeometryScene, gen_absent_length),
    "math_absent_curve": (PlotScene, gen_absent_curve),
    "math_no_mislabel": (PlotScene, gen_no_mislabel),
    "math_absent_category": (PieScene, gen_absent_category),
    "math_absent_die": (DiceScene, gen_absent_die),
    "math_absent_grid": (GridScene, gen_absent_grid),
    "math_absent_angle": (AngleScene, gen_absent_angle),
    "math_absent_bar": (BarScene, gen_absent_bar),
}
# Scenes added after the first build draw from their own random generator, so that adding them left the earlier
# samples and their dev/holdout split unchanged.
OWN_RNG_SCENES = (GridScene, AngleScene, BarScene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="writes <out>/dev and <out>/holdout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    own_rng = random.Random(f"grid-angle-bar-{args.seed}")  # OWN_RNG_SCENES
    images: dict[str, bytes] = {}
    by_gen: dict[str, list[Item]] = {}
    for name, n in TARGETS.items():
        scene_cls, gen = GENERATORS.get(name) or ABSENT[name]
        r = own_rng if scene_cls in OWN_RNG_SCENES else rng
        items: list[Item] = []
        while len(items) < n:
            scene = scene_cls(r)
            out = gen(r, scene, TASK_TYPES[len(items) % len(TASK_TYPES)]) if name in ABSENT else gen(r, scene)
            if out is None:
                continue
            task, question, answer, masks = out
            key = f"math/{len(images)}"
            images[key] = scene.render()
            items.append(
                Item(
                    name,
                    task,
                    question,
                    answer,
                    masks,
                    ("math", key),
                    key,
                    scene.w,
                    scene.h,
                    "math",
                    {"dataset": "synthetic math", "scene": scene_cls.__name__},
                )
            )
        by_gen[name] = items
    split_and_write(by_gen, TARGETS, rng, args.out, lambda it: images[it.image_src], args.seed)


if __name__ == "__main__":
    main()
