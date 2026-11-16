"""Unpack the RoboRefIt testB parquet shards into image and mask files plus one JSON of rows (no depth).

Input: <src>/raw/data/testB-*.parquet from Hugging Face luyh20/RoboRefIt (dev/README.md, robotics dev set).
Output: <src>/testB/images/<image>.png (RGB, one per image), <src>/testB/masks/<sample_id>.png (target mask; any
nonzero channel is foreground, as in the released loader) and <src>/testB/rows.json.
pyarrow is not a project dependency:
  uv run --with pyarrow python -m dev.fetch_roborefit --src data/open_dev/src/roborefit
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

COLUMNS = ["sample_id", "image_id", "expression", "bbox", "scene", "category", "width", "height", "image", "mask"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="stop after this many rows (for a quick check)")
    args = parser.parse_args()
    out = args.src / "testB"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    rows = []
    for shard in sorted((args.src / "raw" / "data").glob("testB-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=COLUMNS):
            for r in batch.to_pylist():
                image = out / "images" / (r["image_id"].split("/")[-1] + ".png")
                if not image.exists():
                    image.write_bytes(r["image"]["bytes"])
                (out / "masks" / f"{r['sample_id']}.png").write_bytes(r["mask"]["bytes"])
                rows.append({k: r[k] for k in COLUMNS[:8]} | {"image": image.name})
                if args.limit and len(rows) == args.limit:
                    break
            if args.limit and len(rows) == args.limit:
                break
        if args.limit and len(rows) == args.limit:
            break
    (out / "rows.json").write_text(json.dumps(rows, indent=0))
    print(f"{len(rows)} rows, {len({r['image'] for r in rows})} images in {out}")


if __name__ == "__main__":
    main()
