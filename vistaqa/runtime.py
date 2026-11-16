"""Shared run dependencies; neither graph imports the other to access resources."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from vistaqa import reasoning, segment as sam, vlm
from vistaqa.cache import NodeCache
from vistaqa.data import Sample

NODES = ("answer", "segment", "finalize", "write", "evaluate")


@dataclass
class Run:
    samples: list[Sample]
    answers: NodeCache
    masks: NodeCache
    out_path: Path
    model: str
    effort: str
    timeout: int
    attempts: int
    slots: asyncio.Semaphore
    until: str = "write"
    data_dir: Path | None = None
    eval_name: str = "qwen72b"
    redo: frozenset[str] = frozenset()
    sam_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    segmenter: sam.BoxSegmenter | None = None
    variant: str = "a"
    max_corrections_per_target: int = 2
    max_agent_runs: int = 12
    initial_run: Path | None = None
    reuse_run: Path | None = None

    def answer_spec(self, sample: Sample) -> dict:
        return (vlm.cache_spec(sample, self.model, self.effort) if self.variant in ("a", "e") else
                reasoning.cache_spec(sample, "locate", self.model, self.effort, {}))

    def runs(self, node: str) -> bool:
        return NODES.index(node) <= NODES.index(self.until)

    @property
    def operations(self) -> NodeCache:
        return NodeCache(self.out_path.parent / "vlm")
