"""The official VistaQA evaluation code, imported from its clone in third_party/VistaQA-Evaluation (eval/README.md)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "VistaQA-Evaluation"
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_and_merge_samples  # noqa: E402
from mask_evaluator import MaskEvaluator  # noqa: E402
from scorer import SAEScorer, _instance_group, compute_joint_score  # noqa: E402
from text_evaluator import TextEvaluator  # noqa: E402

__all__ = ["MaskEvaluator", "SAEScorer", "TextEvaluator", "_instance_group", "commit", "compute_joint_score",
           "load_and_merge_samples"]


def commit() -> dict:
    """Commit of the clone and whether its tracked files have local changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout

    modified = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    return {"commit": git("rev-parse", "HEAD").strip(), "modified": modified}
