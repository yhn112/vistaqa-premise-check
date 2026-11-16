"""Score a run's predictions with the official VistaQA code, judging each sample once, for several epsilons.

Reads <run-dir>/predictions.jsonl and writes <run-dir>/eval_<name>/.
Same steps as third_party/VistaQA-Evaluation/src/main.py (official loader, TextEvaluator,
MaskEvaluator, joint score and report), plus a per-sample judge cache, per-sample outputs,
a breakdown by the dev sets' `generator` field and fingerprints of the GT directory (vistaqa/fingerprint.py).
If the run directory has run.jsonl (vistaqa.predict), its inputs fingerprint must equal the GT's.
The judge defaults to Qwen2.5-72B-Instruct through OpenRouter with the API key in the file `key`;
OPENAI_BASE_URL and OPENAI_API_KEY override the endpoint and the key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval import official
from eval.official import (MaskEvaluator, SAEScorer, TextEvaluator, _instance_group, compute_joint_score,
                           load_and_merge_samples)
from vistaqa.data import load_records
from vistaqa.fingerprint import digest, fingerprints
from vistaqa.runlog import REPO, git_state, tee_output

OPENROUTER = "https://openrouter.ai/api/v1"
JUDGE = "qwen/qwen-2.5-72b-instruct"


def check_run_inputs(run_dir: Path, gt: Path, gt_inputs: str) -> bool:
    """Exit if the run's recorded inputs differ from the GT's; False if the run recorded none."""
    run_log = run_dir / "run.jsonl"
    if not run_log.exists():
        print(f"no {run_log}: the inputs of the predictions are not checked", flush=True)
        return False
    recorded = {json.loads(line)["inputs_fingerprint"] for line in run_log.read_text().splitlines()}
    if recorded != {gt_inputs}:
        sys.exit(f"inputs of {gt} ({gt_inputs[:12]}) differ from those in {run_log} "
                 f"({', '.join(sorted(x[:12] for x in recorded))}): the predictions answer other questions or images")
    return True


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def by_generator(per_sample: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in per_sample:
        if "generator" in r:
            groups[r["generator"]].append(r)
    return {
        g: {
            "n": len(rs),
            "standard_average": mean([r["standard_average"] for r in rs]),
            "text_score": mean([r["text_score"] for r in rs]),
            "mask_score": mean([r["mask_score"] for r in rs]),
        }
        for g, rs in sorted(groups.items())
    }


def _judge_key(sample, evaluator: TextEvaluator, base_url: str) -> str:
    """Hash the actual judge prompt and request settings, independent of masks."""
    prompt = evaluator._build_prompt(
        question=sample.question, gt_answer=sample.gt_answer, pred_answer=sample.pred_text,
        task_type=sample.task_type, domain=sample.domain,
    )
    return digest({"prompt": prompt, "model": evaluator.model_name,
                   "base_url": base_url, "temperature": evaluator.temperature})


def evaluate(gt: Path, run_dir: Path, name: str, llm_model: str = JUDGE,
             epsilons: tuple[float, ...] = (0.1, 0.01), judge_workers: int = 16) -> None:
    """Score <run_dir>/predictions.jsonl against gt into <run_dir>/eval_<name>/."""
    args = argparse.Namespace(gt=gt, run_dir=run_dir, name=name, llm_model=llm_model,
                              epsilons=list(epsilons), judge_workers=judge_workers)
    _evaluate(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, required=True, help="GT directory in VistaQA layout")
    parser.add_argument("--run-dir", type=Path, required=True, help="run directory with predictions.jsonl")
    parser.add_argument("--name", required=True,
                        help="evaluation name, written to <run-dir>/eval_<name>: the judge (qwen72b), "
                             "plus _<suffix> when the GT changed")
    parser.add_argument("--llm-model", default=JUDGE)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.01])
    parser.add_argument("--judge-workers", type=int, default=16, help="Parallel judge requests")
    _evaluate(parser.parse_args())


def _evaluate(args: argparse.Namespace) -> None:
    output_dir = args.run_dir / f"eval_{args.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tee_output(output_dir / "eval.log")
    gt_fingerprints = fingerprints(args.gt)
    inputs_checked = check_run_inputs(args.run_dir, args.gt, gt_fingerprints["inputs"])
    samples = load_and_merge_samples(gt_path=str(args.gt), pred_path=str(args.run_dir / "predictions.jsonl"))
    generators = {i: r["generator"] for i, r in load_records(args.gt).items() if "generator" in r}

    judge_cache_path = output_dir / "judge.jsonl"
    judged = {}
    legacy = 0
    if judge_cache_path.exists():
        for line in judge_cache_path.read_text().splitlines():
            rec = json.loads(line)
            if "key" in rec:
                judged[(rec["image_id"], rec["key"])] = rec
            else:
                legacy += 1
    if legacy:
        print(f"judge cache: ignoring {legacy} legacy verdicts without an input/configuration key", flush=True)

    base_url = os.getenv("OPENAI_BASE_URL") or OPENROUTER
    text_evaluator = TextEvaluator(
        model_name=args.llm_model,
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY") or (REPO / "key").read_text().strip(),
        base_url=base_url,
    )
    keys = {s.image_id: _judge_key(s, text_evaluator, base_url) for s in samples}
    todo = {(s.image_id, keys[s.image_id]): s for s in samples if (s.image_id, keys[s.image_id]) not in judged}
    print(f"judge: {len(samples) - len(todo)} cached, {len(todo)} to query", flush=True)

    def judge(s):
        return text_evaluator.evaluate(s.image_id, s.question, s.gt_answer, s.pred_text, s.task_type, s.domain)

    with judge_cache_path.open("a") as cache, ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        futures = {pool.submit(judge, s): key for key, s in todo.items()}
        for done, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            judged[key] = {"image_id": key[0], "key": key[1], "pred_text": todo[key].pred_text, **future.result()}
            cache.write(json.dumps(judged[key]) + "\n")
            cache.flush()
            if done % 100 == 0:
                print(f"judged {done}/{len(todo)}", flush=True)

    mask_evaluator = MaskEvaluator()
    rows = []
    for s in samples:
        rows.append(
            {
                "image_id": s.image_id,
                "task_type": s.task_type,
                "domain": s.domain,
                "is_hallucination": bool(s.is_hallucination),
                "instance_group": _instance_group(len(s.gt_masks)),
                "text_score": int(judged[(s.image_id, keys[s.image_id])]["correct"]),
                "mask_score": mask_evaluator.score(gt_masks=s.gt_masks, pred_masks=s.pred_masks),
                "n_gt": len(s.gt_masks),
                "n_pred": len(s.pred_masks),
                **({"generator": generators[s.image_id]} if s.image_id in generators else {}),
            }
        )

    (output_dir / "per_sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    metadata = {
        "judge_model": text_evaluator.model_name,
        "judge_base_url": base_url,
        "judge_temperature": text_evaluator.temperature,
        "total_samples": len(samples),
        "gt": str(args.gt),
        "gt_fingerprints": gt_fingerprints,
        "run_inputs_checked": inputs_checked,
        "git": git_state(),
        "official_code": official.commit(),
    }
    summary = {}
    for eps in args.epsilons:
        scorer = SAEScorer(epsilon=eps)
        per_sample = [
            {**r, "standard_average": compute_joint_score(r["text_score"], r["mask_score"], eps)} for r in rows
        ]
        report = scorer.build_standard_average_report(per_sample)
        out = {
            "metadata": {"epsilon": eps, **metadata},
            "overall": report["overall"],
            "breakdown": report["breakdown"],
        }
        if generators:
            out["by_generator"] = by_generator(per_sample)
        (output_dir / f"evaluation_eps{eps}.json").write_text(json.dumps(out, indent=2))
        summary[eps] = out
    n = len(rows)
    print(f"GT {args.gt}: inputs {gt_fingerprints['inputs'][:12]}, annotations {gt_fingerprints['annotations'][:12]}")
    print(f"text accuracy {100 * sum(r['text_score'] for r in rows) / n:.2f}")
    print(f"mask score    {100 * sum(r['mask_score'] for r in rows) / n:.2f}")
    for eps, out in summary.items():
        print(f"GROVE eps={eps}: {100 * out['overall']['standard_average']:.2f}")
    first = summary[args.epsilons[0]]
    if "by_generator" in first:
        print(f"GROVE eps={args.epsilons[0]} by generator (n; Sa / Sm):")
        for g, v in first["by_generator"].items():
            print(f"  {g} {100 * v['standard_average']:.1f} ({v['n']}; "
                  f"{100 * v['text_score']:.1f} / {100 * v['mask_score']:.1f})")


if __name__ == "__main__":
    main()
