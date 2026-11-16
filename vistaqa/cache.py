"""Per-sample results of the graph nodes, addressed by what produced them.

An entry is <dir>/<image_id>/<key>.<version>.json, where key is the digest of the node's specification:
every input and setting the result depends on (cache_spec in vistaqa/vlm.py and vistaqa/segment.py).
A changed image, question, prompt, model, effort, boxes or segmenter therefore misses the cache,
and results of different configurations sit side by side.
Each save creates an immutable version and atomically updates <key>.latest, which names that version.
Readers receive the exact version they loaded, so --redo leaves existing cache_index.json references valid.
Earlier unversioned <key>.json entries are still read and remain untouched when a new version is saved.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from vistaqa.fingerprint import digest
from vistaqa.runlog import REPO


class NodeCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _key_path(self, image_id: int, spec: dict) -> Path:
        return self.root / str(image_id) / f"{digest(spec)[:16]}.json"

    def path(self, image_id: int, spec: dict) -> Path:
        key_path = self._key_path(image_id, spec)
        latest = key_path.with_suffix(".latest")
        return key_path.parent / latest.read_text().strip() if latest.exists() else key_path

    def load(self, image_id: int, spec: dict) -> tuple[Path, dict] | None:
        """Pin the exact version being read, together with its result."""
        path = self.path(image_id, spec)
        if not path.exists():
            return None
        entry = json.loads(path.read_text())
        if entry["spec"] != spec:  # a 64-bit key prefix collided
            raise RuntimeError(f"{path}: stored spec differs from the requested one")
        return path, entry["result"]

    def save(self, image_id: int, spec: dict, result: dict) -> Path:
        key_path = self._key_path(image_id, spec)
        path = key_path.with_stem(f"{key_path.stem}.{uuid4().hex}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"spec": spec, "result": result}, ensure_ascii=False, indent=1))
        tmp.rename(path)
        latest_tmp = path.with_suffix(".latest.tmp")
        latest_tmp.write_text(path.name + "\n")
        latest_tmp.replace(key_path.with_suffix(".latest"))
        return path


def run_answers(run_dir: Path) -> dict[int, dict]:
    """The final answers/evidence behind predictions.jsonl, by image id.

    New indices pin a sample final; old indices pin the initial answer.
    Runs with keyed caches name their entries in cache_index.json (written with predictions.jsonl);
    earlier runs kept one answer per sample in <stage 1 dir>/<image_id>.json,
    the stage 1 dir being `vlm_dir` of the last invocation in run.jsonl, else <run_dir>/vlm.
    """
    index_path = run_dir / "cache_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        answers = {}
        for image_id, entries in index.items():
            if "final" in entries:
                answers[int(image_id)] = json.loads((REPO / entries["final"]).read_text())
            else:
                answers[int(image_id)] = json.loads((REPO / entries["answer"]).read_text())["result"]
        return answers
    stage1 = run_dir / "vlm"
    run_log = run_dir / "run.jsonl"
    if run_log.exists() and (vlm_dir := json.loads(run_log.read_text().splitlines()[-1])["args"].get("vlm_dir")):
        stage1 = Path(vlm_dir)
    return {int(p.stem): json.loads(p.read_text()) for p in stage1.glob("*.json")}
