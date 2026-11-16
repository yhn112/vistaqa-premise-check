"""Find dev images that also appear in VistaQA, including resized, recompressed or cropped copies.

Reads only VistaQA image files (no questions, answers or masks).
SIFT keypoints of every VistaQA image go into one FLANN index; each keypoint of a dev image votes for the
VistaQA images of its nearest neighbours. The top candidates are matched pairwise (ratio test; valid even when
VistaQA repeats an image), fitted with a RANSAC similarity transform (scale, rotation, translation: covers
resizing and cropping), and confirmed by the pixel correlation of the warped dev image over the overlap.
Inlier counts alone are not enough: degenerate transforms collapse many matches onto one point.
Recall check and results: dev/README.md.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
LONG_SIDE = 640
NEIGHBOURS = 5
CANDIDATES = 5
RATIO = 0.75


def read_images(path: Path) -> dict[str, bytes]:
    """Image bytes from a directory (recursive) or a tar archive, keyed by name."""
    if path.is_dir():
        return {
            str(p.relative_to(path)): p.read_bytes()
            for p in sorted(path.rglob("*"))
            if p.suffix.lower() in IMAGE_SUFFIXES
        }
    with tarfile.open(path) as archive:
        return {
            m.name: archive.extractfile(m).read()  # type: ignore[union-attr]  # file members only (below)
            for m in archive.getmembers()
            if m.isfile() and Path(m.name).suffix.lower() in IMAGE_SUFFIXES
        }


Features = tuple[np.ndarray, np.ndarray, np.ndarray]  # grayscale image, keypoint coordinates, descriptors


def sift(data: bytes, detector: Any) -> Features:
    """Grayscale image resized to LONG_SIDE, its SIFT keypoint coordinates and descriptors."""
    with Image.open(io.BytesIO(data)) as img:
        gray = ImageOps.exif_transpose(img).convert("L")
    scale = LONG_SIDE / max(gray.size)
    g = np.asarray(gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.Resampling.LANCZOS))
    keypoints, descriptors = detector.detectAndCompute(g, None)
    if descriptors is None:
        return g, np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return g, np.array([k.pt for k in keypoints], np.float32), descriptors


def verify(a: Features, b: Features, matcher: Any) -> tuple[int, float, float]:
    """RANSAC similarity-transform inliers between images a and b, and the pixel correlation of a warped onto b
    over their overlap together with the overlap's share of the smaller image.

    Images sharing only a logo, text or a repeated pattern can give many inliers; a shared photograph also
    correlates over the whole overlap.
    """
    good = [
        m
        for pair in matcher.knnMatch(a[2], b[2], k=2)
        if len(pair) == 2
        for m in [pair[0]]
        if m.distance < RATIO * pair[1].distance
    ]
    if len(good) < 4:
        return 0, 0.0, 0.0
    src = a[1][[m.queryIdx for m in good]]
    dst = b[1][[m.trainIdx for m in good]]
    transform, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if transform is None:  # RANSAC found no transform; OpenCV's stubs do not allow None
        return 0, 0.0, 0.0  # type: ignore[unreachable]
    h, w = b[0].shape
    warped = cv2.warpAffine(a[0], transform, (w, h)).astype(np.float32)
    overlap = cv2.warpAffine(np.ones_like(a[0]), transform, (w, h), flags=cv2.INTER_NEAREST) > 0
    share = overlap.sum() / min(a[0].size, b[0].size)
    if overlap.sum() < 100:
        return int(mask.sum()), 0.0, float(share)
    x, y = warped[overlap], b[0].astype(np.float32)[overlap]
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
    return int(mask.sum()), corr, float(share)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vistaqa", type=Path, required=True, help="VistaQA directory (<id>.json + image)")
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="dev image directories or tar files")
    parser.add_argument("--min-inliers", type=int, default=12, help="report pairs with at least this many inliers")
    parser.add_argument("--min-corr", type=float, default=0.8, help="pixel correlation that confirms a shared image")
    args = parser.parse_args()
    detector = cv2.SIFT_create(nfeatures=1000)  # type: ignore[attr-defined]  # missing from OpenCV's stubs
    pair_matcher = cv2.BFMatcher(cv2.NORM_L2)

    vista = read_images(args.vistaqa)
    vista_names = sorted(vista)
    vista_feats = [sift(vista[n], detector) for n in vista_names]
    index = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
    index.add([f[2] for f in vista_feats])
    index.train()
    print(f"VistaQA: {len(vista_names)} images, {sum(len(f[2]) for f in vista_feats)} keypoints")

    for source in args.images:
        images = read_images(source)
        found = []
        for name in sorted(images):
            feats = sift(images[name], detector)
            if len(feats[2]) < NEIGHBOURS:
                continue
            votes = Counter(
                owner
                for neighbours in index.knnMatch(feats[2], k=NEIGHBOURS)
                for owner in {m.imgIdx for m in neighbours}
            )
            for j, _ in votes.most_common(CANDIDATES):
                n_inliers, corr, share = verify(feats, vista_feats[j], pair_matcher)
                if n_inliers >= args.min_inliers:
                    found.append((name, vista_names[j], n_inliers, corr, share))
        confirmed = [f for f in found if f[3] >= args.min_corr]
        print(f"\n{source}: {len(images)} images")
        print(
            f"candidate pairs with >= {args.min_inliers} inliers: {len(found)}; pixel correlation (bin start: count):",
            {
                round(float(b), 1): int(c)
                # one more bin edge than counts: each count goes with its bin's start
                for c, b in zip(*np.histogram([f[3] for f in found], bins=np.arange(-1, 1.01, 0.1)), strict=False)
                if c
            },
        )
        print(
            f"confirmed (correlation >= {args.min_corr}): {len(confirmed)} pairs, {len({f[0] for f in confirmed})} images"
        )
        for name, vista_name, n_inliers, corr, share in sorted(found, key=lambda f: -f[3])[: max(len(confirmed), 5)]:
            print(f"  {name}  VistaQA {vista_name}  inliers {n_inliers}  corr {corr:.3f}  overlap {share:.2f}")


if __name__ == "__main__":
    main()
