"""Map independent sample graphs, collect their immutable finals and export VistaQA JSONL.

Each sample graph owns reasoning, mask versions, verification and bounded repairs
(vistaqa/sample.py). The outer graph owns scheduling, errors, export and optional
evaluation. Until evaluate, no node reads dataset answers or masks.
--until answer/segment stop after the initial pass; finalize runs the sample workflow
without changing the submission. --redo answer/segment/solve creates a new attempt
and preserves all previously exported cache references.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import logfire
import pydantic_ai
from pydantic_graph import GraphBuilder, StepContext

from vistaqa import segment as sam, vlm
from vistaqa.cache import NodeCache
from vistaqa.data import Sample, load_samples
from vistaqa.fingerprint import fingerprints
from vistaqa.runlog import REPO, git_state, now, tee_output, trace_to
from vistaqa.runtime import NODES, Run
from vistaqa.sample import solve


@dataclass
class Progress:
    todo: int = 0
    completed: int = 0


@dataclass
class Done:
    image_id: int
    entries: dict[str, Path]


@dataclass
class Failed:
    image_id: int


@dataclass
class Outcome:
    done: dict[int, dict[str, Path]] = field(default_factory=dict)
    failed: list[int] = field(default_factory=list)


def _repo_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


g = GraphBuilder(name="predict", state_type=Progress, deps_type=Run, output_type=list[int])


@g.step
async def pending(ctx: StepContext[Progress, Run, None]) -> list[Sample]:
    if not ctx.deps.samples:
        raise ValueError("Cannot predict an empty sample selection")
    ctx.state.todo = len(ctx.deps.samples)
    logfire.notice("Solving {n} samples; variant {variant}, until {until}",
                   n=ctx.state.todo, variant=ctx.deps.variant, until=ctx.deps.until)
    return ctx.deps.samples


@g.step
async def solve_sample(ctx: StepContext[Progress, Run, Sample]) -> Done | Failed:
    sample = ctx.inputs
    try:
        state = await solve(sample, ctx.deps)
        entries = {"answer": Path(state.answer_entry)}
        if state.segment_entry:
            entries["segment"] = Path(state.segment_entry)
        if state.final is not None:
            entries["final"] = Path(state.attempt_dir) / "final.json"
        result = Done(sample.image_id, entries)
        logfire.notice("id={image_id} phase={phase} masks={masks} agent_runs={calls} answer={answer!r}",
                       image_id=sample.image_id, phase=state.phase, masks=len(state.selected),
                       calls=state.agent_runs, answer=state.final["answer"] if state.final else state.answer)
    except Exception:
        logfire.exception("Failed sample {image_id}", image_id=sample.image_id)
        result = Failed(sample.image_id)
    ctx.state.completed += 1
    return result


def _collect(outcome: Outcome, item: Done | Failed) -> Outcome:
    if isinstance(item, Done):
        outcome.done[item.image_id] = item.entries
    else:
        outcome.failed.append(item.image_id)
    return outcome


collect = g.join(_collect, initial_factory=Outcome, node_id="collect")


@g.step
async def write(ctx: StepContext[Progress, Run, Outcome]) -> list[int]:
    run, outcome = ctx.deps, ctx.inputs
    logfire.notice("{done} samples done, {failed} failed", done=len(outcome.done), failed=len(outcome.failed))
    if outcome.failed or not run.runs("write"):
        return outcome.failed
    records, index = [], {}
    for sample in run.samples:
        entries = outcome.done[sample.image_id]
        final = json.loads(entries["final"].read_text())
        records.append({"image": {"image_id": sample.image_id, "width": sample.width, "height": sample.height},
                        "prediction": {"text": final["answer"], "masks": final["masks"]}})
        index[sample.image_id] = {node: _repo_path(path) for node, path in entries.items()}
    # Build and validate the whole export before replacing either existing artifact.
    tmp = run.out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(record) + "\n" for record in records))
    index_path = run.out_path.parent / "cache_index.json"
    index_tmp = index_path.with_suffix(".json.tmp")
    index_tmp.write_text(json.dumps(index, indent=1))
    tmp.replace(run.out_path)
    index_tmp.replace(index_path)
    logfire.notice("Wrote {path}", path=str(run.out_path))
    return []


@g.step
async def evaluate(ctx: StepContext[Progress, Run, list[int]]) -> list[int]:
    if ctx.inputs or not ctx.deps.runs("evaluate"):
        return ctx.inputs
    from eval.evaluate import evaluate as score

    await asyncio.to_thread(score, ctx.deps.data_dir, ctx.deps.out_path.parent, ctx.deps.eval_name)
    return []


g.add(g.edge_from(g.start_node).to(pending),
      g.edge_from(pending).map(downstream_join_id=collect.id).to(solve_sample),
      g.edge_from(solve_sample).to(collect), g.edge_from(collect).to(write),
      g.edge_from(write).to(evaluate), g.edge_from(evaluate).to(g.end_node))
graph = g.build()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory with <id>.json and images")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6.1-sol", help="Codex model")
    parser.add_argument("--effort", default="xhigh", help="Reasoning effort")
    parser.add_argument("--workers", type=int, default=12, help="Parallel VLM calls")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Seconds per agent run; premise stages use at most their limit in vistaqa/reasoning.py")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--variant", choices=("a", "b", "c", "d", "e"), default="a")
    parser.add_argument("--max-corrections-per-target", type=int, default=2)
    parser.add_argument("--max-agent-runs", type=int, default=12)
    parser.add_argument("--vlm-dir", type=Path, help="Initial-answer cache to read and add to; later VLM operations stay in run-dir/vlm")
    parser.add_argument("--initial-run", type=Path,
                        help="Pin initial plans and masks from this completed run (B for C/D, A for E)")
    parser.add_argument("--reuse-run", type=Path, help="Read compatible model and SAM operation caches from an earlier run")
    parser.add_argument("--ids", help="Comma-separated subset of image ids")
    parser.add_argument("--until", choices=NODES, default="write")
    parser.add_argument("--redo", choices=("answer", "segment", "solve"), nargs="+", default=[])
    parser.add_argument("--eval-name", default="qwen72b")
    args = parser.parse_args()
    if args.vlm_dir and "answer" in args.redo:
        parser.error("--redo answer would add new answers to the --vlm-dir cache of another run")
    if args.initial_run and (args.vlm_dir or "answer" in args.redo or "segment" in args.redo):
        parser.error("--initial-run cannot be combined with --vlm-dir or --redo answer/segment")
    if min(args.workers, args.timeout, args.attempts) < 1 or args.max_corrections_per_target < 0 or args.max_agent_runs < 4:
        parser.error("workers, timeout and attempts must be positive; corrections >= 0; agent runs >= 4")
    try:
        samples = load_samples(args.data)
        if args.ids is not None:
            wanted = {int(x) for x in args.ids.split(",")}
            missing = wanted - {s.image_id for s in samples}
            if missing:
                raise ValueError(f"--ids contains unknown image ids: {sorted(missing)}")
            samples = [s for s in samples if s.image_id in wanted]
        if args.initial_run:
            initial_index = json.loads((args.initial_run / "cache_index.json").read_text())
            if any(str(s.image_id) not in initial_index for s in samples):
                raise ValueError("--initial-run does not contain every selected sample")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        asyncio.run(execute(args, samples))
    except RuntimeError as exc:
        sys.exit(str(exc))


async def execute(args: argparse.Namespace, samples: list[Sample], segmenter=None) -> Run:
    """Run one invocation; studies share an event loop, the Codex provider and SAM weights."""
    args.run_dir.mkdir(parents=True, exist_ok=True)
    tee_output(args.run_dir / "predict.log")
    trace_to(args.run_dir / "trace.jsonl")
    record = {"started": now(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "inputs_fingerprint": fingerprints(args.data)["inputs"], "n_samples": len(samples),
              "git": git_state(), "pydantic_ai": pydantic_ai.__version__, "logfire": logfire.__version__,
              "segmenter": {k: v for k, v in sam.cache_spec(samples[0], []).items() if k not in ("node", "image", "boxes")}}
    with (args.run_dir / "run.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    run = Run(samples=samples, answers=NodeCache(args.vlm_dir or args.run_dir / "vlm"),
              masks=NodeCache(args.run_dir / "masks"), out_path=args.run_dir / "predictions.jsonl",
              model=args.model, effort=args.effort, timeout=args.timeout, attempts=args.attempts,
              slots=asyncio.Semaphore(args.workers), until=args.until, data_dir=args.data,
              eval_name=args.eval_name, redo=frozenset(args.redo), variant=args.variant,
              max_corrections_per_target=args.max_corrections_per_target, max_agent_runs=args.max_agent_runs,
              initial_run=args.initial_run, reuse_run=args.reuse_run, segmenter=segmenter)
    if run.variant in ("b", "c", "d") and run.runs("finalize"):
        # Warm the small tokenizer off the event loop; output validation then stays local and fast.
        await asyncio.to_thread(sam.proposal_tokenizer)
    failures = await graph.run(state=Progress(), deps=run)
    if failures:
        raise RuntimeError(f"Failed {len(failures)} samples: {sorted(failures)}; rerun to retry")
    return run


if __name__ == "__main__":
    main()
