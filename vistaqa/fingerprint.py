"""Fingerprints of a directory in VistaQA layout.

- inputs: image bytes, size and question of every sample, i.e. what the system sees.
  Equal inputs mean that predictions made on one directory are valid for the other.
- annotations: every sample JSON as stored (question, answer, masks and other fields).

Usage: uv run python -m vistaqa.fingerprint DIR [OTHER_DIR]
With two directories it also lists the image ids whose inputs or annotations differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vistaqa.data import Sample, load_samples


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_digest(sample: Sample) -> str:
    """What the system sees of one sample: image bytes, metadata size and question."""
    return _sha(json.dumps([sample.width, sample.height, sample.question]).encode() + sample.image_path.read_bytes())


def image_digest(sample: Sample) -> str:
    """The image of one sample as load_image produces it: image bytes and metadata size."""
    return _sha(json.dumps([sample.width, sample.height]).encode() + sample.image_path.read_bytes())


def digest(spec: dict) -> str:
    """Digest of a JSON-serializable specification, independent of key order."""
    return _sha(json.dumps(spec, sort_keys=True).encode())


def sample_digests(data_dir: Path) -> dict[str, dict[int, str]]:
    """Per-sample digests by image id: {"inputs": {...}, "annotations": {...}}."""
    inputs = {s.image_id: input_digest(s) for s in load_samples(data_dir)}
    annotations = {}
    for path in data_dir.glob("*.json"):
        data = path.read_bytes()
        annotations[int(json.loads(data)["image"]["image_id"])] = _sha(data)
    return {"inputs": inputs, "annotations": annotations}


def combine(digests: dict[int, str]) -> str:
    return _sha(json.dumps(sorted(digests.items())).encode())


def fingerprints(data_dir: Path) -> dict[str, str]:
    return {kind: combine(d) for kind, d in sample_digests(data_dir).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="+", help="one or two directories in VistaQA layout")
    args = parser.parse_args()
    if len(args.dirs) > 2:
        parser.error("give one or two directories")
    digests = [sample_digests(d) for d in args.dirs]
    for d, dg in zip(args.dirs, digests):
        print(f"{d}: {len(dg['inputs'])} samples, inputs {combine(dg['inputs'])}, "
              f"annotations {combine(dg['annotations'])}")
    if len(digests) == 2:
        for kind in ("inputs", "annotations"):
            a, b = digests[0][kind], digests[1][kind]
            differ = sorted(i for i in a.keys() | b.keys() if a.get(i) != b.get(i))
            print(f"{kind} differ for {len(differ)} ids: {differ}")


if __name__ == "__main__":
    main()
