"""One frame from the middle of DROID episodes, from the first external camera (not part of the build).

Episodes are drawn with a seed from the public bucket gs://gresearch/robotics/droid_raw/1.0.1 (anonymous HTTP):
for every lab, successful episodes on distinct recording dates. The ext1 MP4 of an episode is downloaded,
its middle frame is saved as <out>/<uuid>.jpg (quality 95) and the MP4 is deleted.
MP4 decoding is not bit-reproducible across builds of OpenCV, so the frame list with sha256 is tracked in
dev/droid_frames.json and the frames themselves are the source of the robotics set (dev/README.md).
The run resumes: episodes already in the list are kept.

Usage: uv run python -m dev.fetch_droid_frames --out data/open_dev/src/droid_frames [--per-lab 20]
"""

import argparse
import hashlib
import json
import random
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

BUCKET = "gresearch"
ROOT = "robotics/droid_raw/1.0.1/"
LIST = Path(__file__).with_name("droid_frames.json")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        data: bytes = r.read()
    return data


def listing(prefix: str) -> tuple[list[str], list[str]]:
    """Sub-prefixes and object names directly under prefix."""
    prefixes, names, token = [], [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": "1000"} | ({"pageToken": token} if token else {})
        d = json.loads(get(f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urllib.parse.urlencode(q)}"))
        prefixes += d.get("prefixes", [])
        names += [o["name"] for o in d.get("items", [])]
        if not (token := d.get("nextPageToken")):
            return prefixes, names


def public(name: str) -> str:
    return f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}"


def middle_frame(mp4: bytes) -> tuple[bytes, int, int, int, int]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
    if not ok:
        raise ValueError("no frame decoded")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes(), n // 2, n, frame.shape[1], frame.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-lab", type=int, default=20)
    parser.add_argument("--labs", nargs="*", help="only these labs (default: all in the bucket)")
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/open_dev/src/droid/aggregated-annotations-030724.json")
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = json.loads(LIST.read_text()) if LIST.exists() else []
    done = {f["uuid"] for f in frames}
    instructions = json.loads(args.annotations.read_text())
    labs = args.labs or sorted(p.removeprefix(ROOT).strip("/") for p in listing(ROOT)[0])
    for lab in labs:
        rng = random.Random(f"{args.seed}-{lab}")
        dates = listing(f"{ROOT}{lab}/success/")[0]
        rng.shuffle(dates)
        have = sum(f["lab"] == lab for f in frames)
        for date in dates:
            if have >= args.per_lab:
                break
            episodes = sorted(listing(date)[0])
            if not episodes:
                continue
            episode = rng.choice(episodes)
            meta_names = [n for n in listing(episode)[1] if Path(n).name.startswith("metadata_")]
            if not meta_names:
                continue
            meta = json.loads(get(public(meta_names[0])))
            if meta["uuid"] in done:
                have += 1
                continue
            try:
                mp4 = f"{episode}recordings/MP4/{Path(meta['ext1_mp4_path']).name}"
                jpg, index, count, w, h = middle_frame(get(public(mp4)))
            except Exception as e:  # noqa: BLE001 - missing or broken recording: take another date
                print(f"skip {meta['uuid']}: {e}", flush=True)
                continue
            (args.out / f"{meta['uuid']}.jpg").write_bytes(jpg)
            frames.append(
                {
                    "uuid": meta["uuid"],
                    "lab": lab,
                    "episode": episode.removeprefix(ROOT),
                    "camera": meta["ext1_cam_serial"],
                    "frame": index,
                    "frames": count,
                    "width": w,
                    "height": h,
                    "sha256": hashlib.sha256(jpg).hexdigest(),
                    "task": meta.get("current_task"),
                    "instructions": sorted(
                        {v.strip() for v in instructions.get(meta["uuid"], {}).values() if v.strip()}
                    ),
                }
            )
            done.add(meta["uuid"])
            have += 1
            LIST.write_text("[\n" + ",\n".join(json.dumps(f) for f in frames) + "\n]\n")
            print(f"{lab}: {have}/{args.per_lab} {meta['uuid']} {w}x{h}", flush=True)
    print(f"{len(frames)} frames listed in {LIST}")


if __name__ == "__main__":
    main()
