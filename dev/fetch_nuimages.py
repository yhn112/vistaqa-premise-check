"""Keep the nuImages v1.0-val key frames from one streaming pass over the all-samples archive (15.3 GiB, not stored).

The archive is read over HTTP and only files listed as key frames in <src>/v1.0-val/sample_data.json are written,
under <src>/samples/... as in the archive; files already present are kept. Sources: dev/README.md, AV dev set.

Usage: uv run python -m dev.fetch_nuimages --src data/open_dev/src/nuimages
"""

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="directory with v1.0-val/")
    args = parser.parse_args()
    wanted = {
        s["filename"] for s in json.loads((args.src / "v1.0-val" / "sample_data.json").read_text()) if s["is_key_frame"]
    }
    todo = {f for f in wanted if not (args.src / f).exists()}
    print(f"{len(wanted)} key frames, {len(todo)} to fetch", flush=True)
    with urllib.request.urlopen(URL) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name in todo:
                dst = args.src / member.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(archive.extractfile(member).read())  # type: ignore[union-attr]  # a file member
                tmp.rename(dst)
                todo.discard(member.name)
                if len(todo) % 1000 == 0:
                    print(f"{len(todo)} left", flush=True)
                if not todo:
                    break
    if todo:
        raise SystemExit(f"{len(todo)} key frames not in the archive")
    print("done")


if __name__ == "__main__":
    main()
