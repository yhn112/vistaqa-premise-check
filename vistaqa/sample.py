"""Per-image Pydantic Graph: locate, segment, check, verify, repair and finalize.

Variant E keeps A's answer and masks and adds one premise stage: a text-only call lists
what the question presupposes and a separate image call checks each claim.
Only a claim that clearly fails replaces A's result with an absence answer and no masks.
Checkpoints pin immutable node-cache entries. Each transition is saved before the
next external operation; an interrupted attempt resumes with its budgets intact.
Only the final artifact is consumed by the outer graph's writer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import logfire
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import AgentRunError
from pydantic_graph import GraphBuilder, StepContext

from vistaqa import evidence, reasoning, rle, segment as sam, vlm
from vistaqa.cache import NodeCache
from vistaqa.data import Sample, load_image
from vistaqa.evidence import (Accept, Action, EvidencePlan, FinalAnswer, MaskCandidate, MissingSearch,
                              ObjectRef, PremiseCheck, Presuppositions, Propose, Reconsider, Refine, Reject,
                              Relocate, SearchMissing, Selection, Verification, mask_checks, review_images,
                              valid_box)
from vistaqa.fingerprint import digest
from vistaqa.runtime import Run

VERSION = 3


class BudgetExhausted(RuntimeError):
    """No further perception call fits before the reserved finalization call."""


class SampleState(BaseModel):
    sample: Sample
    attempt_dir: str
    spec: dict
    phase: str = "locate"
    plan: EvidencePlan | None = None
    answer: str = ""
    answer_entry: str | None = None
    segment_entry: str | None = None
    candidates: list[MaskCandidate] = Field(default_factory=list)
    selected: list[str] = Field(default_factory=list)
    accepted: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    invalid: list[str] = Field(default_factory=list)
    comparison: list[str] = Field(default_factory=list)
    flags: list[dict] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    corrections: dict[str, int] = Field(default_factory=dict)
    proposal_keys: list[str] = Field(default_factory=list)
    proposal_rounds: int = 0
    search_status: str | None = None
    absence_answer: str | None = None
    complete: bool = False
    agent_runs: int = 0
    calls: list[dict] = Field(default_factory=list)
    pending_call: dict | None = None
    history: list[dict] = Field(default_factory=list)
    repair_prepared: bool = False
    repair_before: list[str] = Field(default_factory=list)
    repair_proposed: list[str] = Field(default_factory=list)
    repair_proposals: list[str] = Field(default_factory=list)
    iteration: int = 0
    final: dict | None = None

    def checkpoint(self):
        root = Path(self.attempt_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.iteration += 1
        content = self.model_dump_json(indent=1)
        (root / f"iteration_{self.iteration:04d}_{uuid4().hex[:8]}.json").write_text(content)
        tmp = root / "state.json.tmp"
        tmp.write_text(content)
        tmp.replace(root / "state.json")

    def by_id(self) -> dict[str, MaskCandidate]:
        return {c.id: c for c in self.candidates}


def workflow_spec(sample: Sample, run: Run) -> dict:
    initial = json.loads((run.initial_run / "cache_index.json").read_text())[str(sample.image_id)] if run.initial_run else None
    stages = ("presuppose", "premise_check") if run.variant == "e" else ("verify", "select", "search", "finalize")
    return {"version": VERSION, "checks_version": evidence.VERSION, "variant": run.variant, "answer": run.answer_spec(sample),
            "stages": {s: reasoning.cache_spec(sample, s, run.model, run.effort, {}) for s in stages},
            "segment": sam.operation_spec(sample, "workflow", {}),
            "max_corrections_per_target": run.max_corrections_per_target,
            "max_agent_runs": run.max_agent_runs, "attempts": run.attempts, "timeout": run.timeout,
            "answer_cache": str(run.answers.root.resolve()),
            "reuse_run": str(run.reuse_run.resolve()) if run.reuse_run else None,
            "initial_index": {k: initial[k] for k in ("answer", "segment")} if initial else None}


def prepare(sample: Sample, run: Run) -> SampleState:
    spec = workflow_spec(sample, run)
    root = run.out_path.parent / "samples" / str(sample.image_id) / digest(spec)[:16]
    latest = root / "latest"
    if latest.exists() and not run.redo:
        state = SampleState.model_validate_json((root / latest.read_text().strip() / "state.json").read_text())
        if state.spec != spec:
            raise RuntimeError("Workflow cache key collision")
        # A separately redone initial answer invalidates this attempt, even if the config is unchanged.
        current_answer = run.answers.path(sample.image_id, run.answer_spec(sample))
        if not current_answer.exists() and run.reuse_run:
            current_answer = NodeCache(run.reuse_run / "vlm").path(sample.image_id, run.answer_spec(sample))
        if run.initial_run or not state.answer_entry or Path(state.answer_entry) == current_answer:
            state.sample = sample
            final_path = Path(state.attempt_dir) / "final.json"
            if state.final is None and final_path.exists():
                final = json.loads(final_path.read_text())
                if final["workflow_spec"] != spec:
                    raise RuntimeError("Final artifact does not match checkpoint specification")
                state.final, state.phase = final, "finish"
            if state.phase == "finish" and state.final is None:
                if state.segment_entry is None and run.runs("segment"):
                    state.phase = "segment"
                elif state.segment_entry is not None and run.runs("finalize"):
                    state.phase = "check"
            if state.pending_call and len(state.pending_call["errors"]) >= run.attempts:
                state.pending_call.setdefault("previous_errors", []).extend(state.pending_call["errors"])
                state.pending_call["errors"] = []
            if (run.until == "answer" and state.answer_entry is not None or
                    run.until == "segment" and state.segment_entry is not None):
                # A temporary earlier stop must not replace the persisted later checkpoint.
                state.phase = "finish"
            return state
    attempt = uuid4().hex
    state = SampleState(sample=sample, spec=spec, attempt_dir=str(root / attempt))
    state.checkpoint()
    tmp = root / "latest.tmp"
    tmp.write_text(attempt + "\n")
    tmp.replace(latest)
    return state


def _source(run: Run, sample: Sample, node: str, spec: dict) -> tuple[Path, dict] | None:
    if not run.initial_run:
        return None
    from vistaqa.runlog import REPO

    index = json.loads((run.initial_run / "cache_index.json").read_text())
    path = REPO / index[str(sample.image_id)][node]
    entry = json.loads(path.read_text())
    if entry["spec"] != spec:
        raise ValueError(f"{path}: initial-run {node} spec differs from this run")
    return path, entry["result"]


def _retry_feedback(stage: str, payload: dict, error: str) -> str:
    if error and stage == "finalize":
        statuses = ["resolved", "unresolved"]
        if payload["independent_search"] in ("absent", "missing_information"):
            statuses.append("confirmed_absent")
        error += (f"\nFor this sample independent_search={payload['independent_search']!r}. "
                  f"Allowed status values are {json.dumps(statuses)}. "
                  "When absence is unconfirmed and uncertainty remains, use unresolved; "
                  "keep the best supported answer and available evidence.")
    return error


async def _ask(state: SampleState, run: Run, stage: str, payload: dict, candidates: list[MaskCandidate] = (),
               validator=None) -> dict:
    spec = reasoning.cache_spec(state.sample, stage, run.model, run.effort, payload)
    key = digest(spec)
    cache = run.answers if stage == "locate" else run.operations
    force = ("answer" in run.redo if stage == "locate" else bool(run.redo))
    pending = state.pending_call
    if pending is not None and pending["key"] != key:
        raise RuntimeError("Checkpoint has a different pending model operation")
    cached = None if force and pending is None else cache.load(state.sample.image_id, spec)
    if cached is None and run.reuse_run and not force:
        cached = NodeCache(run.reuse_run / "vlm").load(state.sample.image_id, spec)
    if cached and pending and str(cached[0]) == pending.get("previous_entry"):
        cached = None
    if cached is not None:
        path, result = cached
        if validator:
            validator(result["output"])
        charged = pending["attempts"] if pending else 0
        state.agent_runs += 0 if charged else 1
        state.calls.append({"stage": stage, "entry": str(path), "cache_hit": True,
                            "executed_attempts": charged, "executed_usage": result["usage"] if charged else {},
                            **result})
        state.pending_call = None
        return {**result, "entry": str(path)}
    image, images = None, []
    prior = cache.path(state.sample.image_id, spec)
    state.pending_call = pending or {"key": key, "stage": stage, "attempts": 0, "errors": [],
                                    "attempt_results": [], "previous_entry": str(prior) if prior.exists() else None}
    previous_errors = state.pending_call.get("previous_errors", []) + state.pending_call["errors"]
    feedback = _retry_feedback(stage, payload, previous_errors[-1] if previous_errors else "")
    reserve = 0 if stage == "finalize" else 1
    while len(state.pending_call["errors"]) < run.attempts and state.agent_runs < run.max_agent_runs - reserve:
        async with run.slots:
            if image is None:
                image = await asyncio.to_thread(load_image, state.sample)
                images = await asyncio.to_thread(review_images, image, list(candidates))
            state.pending_call["attempts"] += 1
            state.agent_runs += 1
            state.checkpoint()
            try:
                result = await reasoning.query(stage, state.sample, image, images, payload,
                                               run.model, run.effort, run.timeout, feedback)
                state.pending_call["attempt_results"].append({"usage": result["usage"], "elapsed_s": result["elapsed_s"]})
                if validator:
                    validator(result["output"])
            except (AgentRunError, ValueError) as exc:
                # A new attempt gets concrete schema/semantic feedback rather than the same failed request.
                error = f"\nPrevious attempt failed validation: {type(exc).__name__}: {str(exc)[:500]}. Correct it."
                state.pending_call["errors"].append(error)
                feedback = _retry_feedback(stage, payload, error)
                state.checkpoint()
                continue
        result["attempt"] = state.pending_call["attempts"]
        attempts = state.pending_call["attempt_results"]
        result["usage"] = {k: sum(a["usage"].get(k, 0) or 0 for a in attempts)
                           for k in {k for a in attempts for k in a["usage"]}}
        result["elapsed_s"] = sum(a["elapsed_s"] for a in attempts)
        result["unknown_usage_attempts"] = result["attempt"] - len(attempts)
        path = cache.save(state.sample.image_id, spec, result)
        state.calls.append({"stage": stage, "entry": str(path), "cache_hit": False,
                            "executed_attempts": result["attempt"], "executed_usage": result["usage"], **result})
        state.pending_call = None
        return {**result, "entry": str(path)}
    state.checkpoint()
    if stage not in ("locate", "finalize") and state.agent_runs >= run.max_agent_runs - reserve:
        raise BudgetExhausted(stage)
    raise RuntimeError(f"{stage}: no valid result within attempts/agent budget; {state.pending_call['errors']}")


async def _sam(state: SampleState, run: Run, operation: str, prompts: dict) -> tuple[Path, dict]:
    spec = sam.operation_spec(state.sample, operation, prompts)
    cached = run.masks.load(state.sample.image_id, spec)
    if cached is None and run.reuse_run:
        cached = NodeCache(run.reuse_run / "masks").load(state.sample.image_id, spec)
    if cached:
        return cached
    async with run.sam_lock:
        if run.segmenter is None:
            run.segmenter = await asyncio.to_thread(sam.BoxSegmenter)
        image = await asyncio.to_thread(load_image, state.sample)
        if operation == "proposal":
            results = await asyncio.to_thread(run.segmenter.proposals, image, **prompts)
            result = {"candidates": results}
        elif operation == "refine":
            mask = await asyncio.to_thread(run.segmenter.refine, image, **prompts)
            result = {"masks": [rle.encode(mask)]}
        else:
            masks = await asyncio.to_thread(run.segmenter.masks, image, prompts["boxes"])
            result = {"masks": [rle.encode(m) for m in masks]}
    return run.masks.save(state.sample.image_id, spec, result), result


def _candidate(state: SampleState, entry: Path, mask: dict, box: list[float], source: str,
               target: ObjectRef | None = None, parent: MaskCandidate | None = None,
               index: int = 0, score: float | None = None, **points) -> MaskCandidate:
    identity = {"entry": str(entry), "index": index, "target": target.id if target else None}
    candidate_id = "m_" + digest(identity)[:10]
    if candidate_id in state.by_id():
        return state.by_id()[candidate_id]
    c = MaskCandidate(id=candidate_id, number=len(state.candidates) + 1, target_id=target.id if target else None,
                      role="target" if target else "proposal", description=target.description if target else "candidate",
                      box=box, mask=mask, source=source, entry=str(entry), parent_id=parent.id if parent else None,
                      score=score, **points)
    if c.id not in state.by_id():
        state.candidates.append(c)
    if target is not None and source != "direct":
        state.plan.anchors = [o for o in state.plan.anchors if o.id != target.id]
        state.plan.targets = [o for o in state.plan.targets if o.id != target.id] + [target]
    return c


def _objects_valid(objects: list[ObjectRef], sample: Sample):
    for o in objects:
        if o.box is not None and not valid_box(o.box, sample.width, sample.height):
            raise ValueError(f"Invalid pixel box for {o.id}: {o.box}")


class Locate: pass
class Segment: pass
class Check: pass
class Verify: pass
class Repair: pass
class Search: pass
class Premise: pass
class Finalize: pass
class Finish: pass

g = GraphBuilder(name="sample", state_type=SampleState, deps_type=Run, output_type=SampleState)


@g.step
async def dispatch(ctx: StepContext[SampleState, Run, None]) -> Locate | Segment | Check | Verify | Repair | Search | Premise | Finalize | Finish:
    return {"locate": Locate, "segment": Segment, "check": Check, "verify": Verify, "repair": Repair,
            "search": Search, "premise": Premise, "finalize": Finalize, "finish": Finish}[ctx.state.phase]()


@g.step
async def locate(ctx: StepContext[SampleState, Run, Locate]) -> None:
    s, run = ctx.state, ctx.deps
    spec = run.answer_spec(s.sample)
    source = _source(run, s.sample, "answer", spec) if "answer" not in run.redo else None
    if run.variant in ("a", "e"):
        cached = source or (None if "answer" in run.redo else run.answers.load(s.sample.image_id, spec))
        if cached:
            entry, answer = cached
        else:
            async with run.slots:
                answer = await vlm.query_vlm(s.sample, run.model, run.effort, run.timeout, run.attempts)
            entry = run.answers.save(s.sample.image_id, spec, answer)
        s.calls.append({"stage": "locate", "entry": str(entry), "cache_hit": cached is not None,
                        "attempt": answer.get("attempt", 1), "usage": answer.get("usage", {}),
                        "elapsed_s": answer.get("elapsed_s", 0),
                        "executed_attempts": 0 if cached else answer.get("attempt", 1),
                        "executed_usage": {} if cached else answer.get("usage", {}),
                        "unknown_usage_attempts": 0 if cached else answer.get("attempt", 1) - 1})
        s.agent_runs = answer.get("attempt", 1)
        targets = [ObjectRef(id=f"target_{i + 1}", description=e["label"], box=e["box"], search_text=e["label"])
                   for i, e in enumerate(answer["evidence"])]
        s.plan = EvidencePlan(provisional_answer=answer["answer"], answerable=answer.get("target_present", bool(targets)),
                              task="other", targets=targets, anchors=[], expected_mask_count=len(targets), search_texts=[])
        s.answer_entry = str(entry)
    else:
        def validate(output):
            p = EvidencePlan.model_validate(output)
            _objects_valid(p.targets + p.anchors, s.sample)
        if source:
            entry, result = source
            validate(result["output"])
            s.agent_runs += 1
            s.calls.append({"stage": "locate", "entry": str(entry), "cache_hit": True,
                            "executed_attempts": 0, "executed_usage": {}, **result})
            result = {**result, "entry": str(entry)}
        else:
            result = await _ask(s, run, "locate", {}, validator=validate)
        s.plan = EvidencePlan.model_validate(result["output"])
        s.answer_entry = result["entry"]
    s.answer = s.plan.provisional_answer
    s.phase = "segment" if run.runs("segment") else "finish"
    s.checkpoint()


@g.step
async def segment(ctx: StepContext[SampleState, Run, Segment]) -> None:
    s, run = ctx.state, ctx.deps
    targets = [o for o in s.plan.targets if o.box is not None]
    boxes = [o.box for o in targets]
    spec = sam.cache_spec(s.sample, boxes)
    cached = _source(run, s.sample, "segment", spec) if "segment" not in run.redo else None
    cached = cached or (None if "segment" in run.redo else run.masks.load(s.sample.image_id, spec))
    if cached is None and run.reuse_run and "segment" not in run.redo:
        cached = NodeCache(run.reuse_run / "masks").load(s.sample.image_id, spec)
    if cached:
        entry, result = cached
    else:
        async with run.sam_lock:
            if boxes:
                if run.segmenter is None:
                    run.segmenter = await asyncio.to_thread(sam.BoxSegmenter)
                image = await asyncio.to_thread(load_image, s.sample)
                masks = await asyncio.to_thread(run.segmenter.masks, image, boxes)
            else:
                masks = []
        result = {"masks": [rle.encode(m) for m in masks]}
        entry = run.masks.save(s.sample.image_id, spec, result)
    if len(result["masks"]) != len(targets):
        raise ValueError("SAM3 did not return one mask per initial box")
    s.segment_entry = str(entry)
    s.selected = [_candidate(s, entry, mask, o.box, "direct", o, index=i).id
                  for i, (o, mask) in enumerate(zip(targets, result["masks"], strict=True))]
    s.phase = ("check" if run.runs("finalize") else "finish")
    s.checkpoint()


@g.step
async def check(ctx: StepContext[SampleState, Run, Check]) -> None:
    s = ctx.state
    if ctx.deps.variant in ("a", "e"):
        # E only checks answers that A gives as present; an absence answer has nothing to overturn.
        s.phase = "premise" if ctx.deps.variant == "e" and s.plan.answerable else "finalize"
        s.checkpoint()
        return
    s.flags, invalid = mask_checks(s.candidates, s.selected, s.plan.expected_mask_count,
                                  s.sample.width, s.sample.height)
    s.invalid = sorted(invalid)
    s.selected = [i for i in s.selected if i not in invalid]
    s.phase = "verify"
    if s.agent_runs >= ctx.deps.max_agent_runs - 1:
        s.complete = False
        s.phase = "finalize"
    s.checkpoint()


def _visible(s: SampleState) -> list[MaskCandidate]:
    if s.search_status in ("absent", "missing_information"):
        return []
    ids = set(s.selected + s.comparison)
    # A refinement request does not reject its parent. Keep promoted target versions
    # available for comparison and best-supported finalization at the repair limit.
    ids.update(c.id for c in s.candidates if c.role == "target")
    return [c for c in s.candidates if c.id in ids and c.id not in s.invalid and c.id not in s.rejected]


@g.step
async def verify(ctx: StepContext[SampleState, Run, Verify]) -> None:
    s, run = ctx.state, ctx.deps
    visible = _visible(s)
    payload = {"answer": s.answer, "masks": [{"mask_id": c.id, "number": c.number, "target_id": c.target_id} for c in visible],
               "checks": s.flags}
    def validate(output):
        v = Verification.model_validate(output)
        ids = {c.id for c in visible}
        targets = set()
        for a in v.actions:
            if isinstance(a, (Accept, Reject, Refine)) and a.mask_id not in ids:
                raise ValueError(f"Unknown mask ID: {a.mask_id}")
            if isinstance(a, Accept):
                target = s.by_id()[a.mask_id].target_id
                if target in targets:
                    raise ValueError("Accept only one version of each target")
                targets.add(target)
            if isinstance(a, Relocate) and not valid_box(a.new_box, s.sample.width, s.sample.height):
                raise ValueError("Relocate needs a valid original-image pixel box")
            if isinstance(a, (Relocate, SearchMissing)):
                sam.validate_search_text(a.search_text)
            if isinstance(a, Propose):
                sam.validate_search_text(a.search_text, allow_empty=bool(a.positive_boxes))
                for b in a.positive_boxes + a.negative_boxes:
                    if not valid_box(b, s.sample.width, s.sample.height):
                        raise ValueError("Proposal boxes must be valid original-image pixel boxes")
            if isinstance(a, Refine):
                if not a.positive_points and not a.negative_points:
                    raise ValueError("Refinement needs at least one point")
                for x, y in a.positive_points + a.negative_points:
                    if not (0 <= x < s.sample.width and 0 <= y < s.sample.height):
                        raise ValueError("Refinement points must be inside the original image")
    result = await _ask(s, run, "verify", payload, visible, validate)
    verdict = Verification.model_validate(result["output"])
    rejected = {a.mask_id for a in verdict.actions if isinstance(a, Reject)}
    s.rejected = sorted(set(s.rejected) | rejected)
    s.selected = [i for i in s.selected if i not in rejected]
    s.accepted = [i for i in s.accepted if i not in rejected]
    for a in verdict.actions:
        if isinstance(a, Accept):
            c = s.by_id()[a.mask_id]
            s.selected = [i for i in s.selected if s.by_id()[i].target_id != c.target_id] + [c.id]
            s.accepted = [i for i in s.accepted if s.by_id()[i].target_id != c.target_id] + [c.id]
        elif isinstance(a, Reconsider):
            s.answer = a.answer
    for event in s.history:
        if event.pop("awaiting_verdict", False):
            event["after"] = list(s.selected)
    s.comparison = []
    s.actions = [a for a in verdict.actions if isinstance(a, (Propose, Relocate, Refine, SearchMissing))]
    s.complete = verdict.complete and not s.actions and set(s.selected) <= set(s.accepted)
    need_search = not s.plan.answerable or not s.selected or any(isinstance(a, SearchMissing) for a in s.actions)
    if need_search and s.search_status is None and s.agent_runs < run.max_agent_runs - 1:
        s.phase = "search"
    elif s.actions and run.variant in ("c", "d") and s.agent_runs < run.max_agent_runs - 1:
        s.phase = "repair"
    else:
        s.phase = "finalize"
    s.checkpoint()


@g.step
async def search(ctx: StepContext[SampleState, Run, Search]) -> None:
    s, run = ctx.state, ctx.deps
    def validate(output):
        found = MissingSearch.model_validate(output)
        _objects_valid(found.targets, s.sample)
        for text in found.search_texts + [o.search_text for o in found.targets]:
            sam.validate_search_text(text)
    result = await _ask(s, run, "search", {}, validator=validate)
    found = MissingSearch.model_validate(result["output"])
    s.search_status = found.status
    if found.status in ("absent", "missing_information"):
        s.answer, s.selected, s.complete, s.actions = found.answer, [], True, []
        s.comparison = []
        s.phase = "finalize"
    else:
        s.actions = [a for a in s.actions if not isinstance(a, SearchMissing)]
        if run.variant == "d":
            s.actions.extend(Relocate(action="relocate", target_id=o.id, description=o.description,
                                      search_text=o.search_text, new_box=o.box)
                             for o in found.targets if o.box is not None)
        if run.variant in ("c", "d"):
            s.actions.extend(Propose(action="propose", search_text=t, positive_boxes=[], negative_boxes=[])
                             for t in found.search_texts)
        s.complete = False
        s.phase = "repair" if s.actions and run.variant in ("c", "d") else "finalize"
    s.checkpoint()


@g.step
async def premise(ctx: StepContext[SampleState, Run, Premise]) -> None:
    s, run = ctx.state, ctx.deps
    listed = await _ask(s, run, "presuppose", {}, validator=Presuppositions.model_validate)
    claims = Presuppositions.model_validate(listed["output"]).claims
    if claims:
        ids = sorted(c.id for c in claims)

        def validate(output):
            check = PremiseCheck.model_validate(output)
            if sorted(v.id for v in check.verdicts) != ids:
                raise ValueError(f"Return exactly one verdict for each claim ID: {ids}")
            if any(v.verdict == "fails" for v in check.verdicts) != bool(check.absence_answer.strip()):
                raise ValueError("absence_answer is required exactly when some claim fails")
        result = await _ask(s, run, "premise_check", {"claims": [c.model_dump() for c in claims]}, validator=validate)
        check = PremiseCheck.model_validate(result["output"])
        if any(v.verdict == "fails" for v in check.verdicts):
            s.absence_answer = check.absence_answer.strip()
    s.phase = "finalize"
    s.checkpoint()


def _can_correct(s: SampleState, run: Run, target_id: str) -> bool:
    return s.corrections.get(target_id, 0) < run.max_corrections_per_target


async def _prepare_repairs(s: SampleState, run: Run) -> tuple[list[str], list[str], list[str]]:
    before = list(s.selected)
    proposed = []
    actions = list(s.actions)
    if run.variant == "c":
        actions = [a if isinstance(a, Propose) else
                   Propose(action="propose", search_text=a.search_text, positive_boxes=[], negative_boxes=[])
                   for a in actions if isinstance(a, (Propose, Relocate))] if s.proposal_rounds == 0 else []
    proposal_actions = [a for a in actions if isinstance(a, Propose)]
    new_proposals = []
    for a in actions:
        if isinstance(a, Propose):
            prompts = {"text": a.search_text, "positive_boxes": a.positive_boxes, "negative_boxes": a.negative_boxes}
            key = digest(prompts)
            if key in s.proposal_keys or not (a.search_text or a.positive_boxes):
                continue
            entry, result = await _sam(s, run, "proposal", prompts)
            for i, item in enumerate(result["candidates"]):
                c = _candidate(s, entry, item["mask"], item["box"], "proposal", index=i, score=item["score"])
                new_proposals.append(c.id)
            s.proposal_keys.append(key)
        elif isinstance(a, Relocate) and run.variant == "d" and _can_correct(s, run, a.target_id):
            target = ObjectRef(id=a.target_id, description=a.description, box=a.new_box, search_text=a.search_text)
            parent = next((c for c in s.candidates if c.id in s.selected and c.target_id == a.target_id), None)
            if parent is not None and parent.box == a.new_box:
                continue
            entry, result = await _sam(s, run, "relocate", {"boxes": [a.new_box]})
            c = _candidate(s, entry, result["masks"][0], a.new_box, "relocate", target, parent)
            if c.id in s.comparison or c.id in s.rejected:
                continue
            s.comparison.append(c.id)
            proposed.append(c.id)
            s.corrections[a.target_id] = s.corrections.get(a.target_id, 0) + 1
        elif isinstance(a, Refine) and run.variant == "d":
            parent = s.by_id()[a.mask_id]
            if not _can_correct(s, run, parent.target_id):
                continue
            positive = parent.positive_points + [p for p in a.positive_points if p not in parent.positive_points]
            negative = parent.negative_points + [p for p in a.negative_points if p not in parent.negative_points]
            if positive == parent.positive_points and negative == parent.negative_points:
                continue
            prompts = {"box": parent.box, "positive_points": positive, "negative_points": negative}
            entry, result = await _sam(s, run, "refine", prompts)
            target = ObjectRef(id=parent.target_id, description=parent.description, box=parent.box,
                               search_text=next(o.search_text for o in s.plan.targets if o.id == parent.target_id))
            c = _candidate(s, entry, result["masks"][0], parent.box, "refine", target, parent,
                           positive_points=positive, negative_points=negative)
            if c.id in s.comparison or c.id in s.rejected:
                continue
            s.comparison.append(c.id)
            proposed.append(c.id)
            s.corrections[parent.target_id] = s.corrections.get(parent.target_id, 0) + 1
    if proposal_actions:
        s.proposal_rounds += 1
    return before, proposed, new_proposals


@g.step
async def repair(ctx: StepContext[SampleState, Run, Repair]) -> None:
    s, run = ctx.state, ctx.deps
    if not s.repair_prepared:
        s.repair_before, s.repair_proposed, s.repair_proposals = await _prepare_repairs(s, run)
        s.repair_prepared = True
        s.actions = []
        s.checkpoint()
    before, proposed, new_proposals = s.repair_before, s.repair_proposed, s.repair_proposals
    if new_proposals and s.agent_runs < run.max_agent_runs - 1:
        _, invalid = mask_checks(s.candidates, [], 0, s.sample.width, s.sample.height)
        available = [c for c in s.candidates if c.id in set(new_proposals + s.selected) and c.id not in invalid and c.id not in s.rejected]
        payload = {"answer": s.answer, "masks": [c.id for c in available],
                   "numbers": {str(c.number): c.id for c in available},
                   "known_targets": [{"id": o.id, "description": o.description} for o in s.plan.targets]}
        def validate(output):
            selection = Selection.model_validate(output)
            ids = [t.mask_id for t in selection.targets]
            if len(set(ids)) != len(ids) or not set(ids) <= {c.id for c in available}:
                raise ValueError("Select each available mask at most once")
            targets = [t.target_id for t in selection.targets]
            if len(set(targets)) != len(targets):
                raise ValueError("Select at most one mask version per target ID")
        result = await _ask(s, run, "select", payload, available, validate)
        selection = Selection.model_validate(result["output"])
        for t in selection.targets:
            original = s.by_id()[t.mask_id]
            if original.id in s.selected:
                continue
            if not _can_correct(s, run, t.target_id):
                continue
            search_text = json.loads(Path(original.entry).read_text())["spec"]["prompts"]["text"]
            target = ObjectRef(id=t.target_id, description=t.description, box=original.box, search_text=search_text)
            parent = next((c for c in s.candidates if c.id in s.selected and c.target_id == t.target_id), original)
            c = _candidate(s, Path(original.entry), original.mask, original.box, "proposal", target, parent,
                           index=int(digest({"id": original.id})[:8], 16), score=original.score)
            proposed.append(c.id)
            s.comparison.append(c.id)
            s.corrections[t.target_id] = s.corrections.get(t.target_id, 0) + 1
        s.answer = selection.answer
    s.history.append({"operation": "repair", "before": before, "proposed": proposed,
                      "proposal_candidates": new_proposals, "corrections": dict(s.corrections),
                      "awaiting_verdict": bool(proposed)})
    s.actions = []
    s.repair_prepared = False
    s.repair_before, s.repair_proposed, s.repair_proposals = [], [], []
    s.phase = "check" if proposed else "finalize"
    s.complete = False
    s.checkpoint()


@g.step
async def finalize(ctx: StepContext[SampleState, Run, Finalize]) -> None:
    s, run = ctx.state, ctx.deps
    if run.variant in ("a", "e") and s.absence_answer:
        selected = []
        final = FinalAnswer(answer=s.absence_answer, selected_mask_ids=[], status="confirmed_absent")
        target_present = False
    elif run.variant in ("a", "e"):
        answer = json.loads(Path(s.answer_entry).read_text())["result"]
        selected = s.selected
        final = FinalAnswer(answer=s.answer, selected_mask_ids=selected, status="resolved")
        target_present = answer.get("target_present", bool(selected))
    else:
        available = _visible(s)
        payload = {"answer": s.answer, "task": s.plan.task, "complete": s.complete,
                   "independent_search": s.search_status,
                   "masks": [{"mask_id": c.id, "number": c.number, "target_id": c.target_id,
                              "accepted": c.id in s.accepted} for c in available]}
        def validate(output):
            f = FinalAnswer.model_validate(output)
            ids = f.selected_mask_ids
            allowed = {c.id for c in available if c.role == "target"}
            if len(set(ids)) != len(ids) or not set(ids) <= allowed:
                raise ValueError("Final evidence must use unique, available target mask IDs")
            targets = [s.by_id()[i].target_id for i in ids]
            if len(set(targets)) != len(targets):
                raise ValueError("Select at most one version of an instance")
            if f.status == "confirmed_absent" and s.search_status not in ("absent", "missing_information"):
                raise ValueError("Absence must be confirmed by independent visual search")
            if f.status == "confirmed_absent" and ids:
                raise ValueError("Confirmed absence must have empty evidence")
        result = await _ask(s, run, "finalize", payload, available, validate)
        final = FinalAnswer.model_validate(result["output"])
        selected = final.selected_mask_ids
        if final.status == "resolved" and (not s.complete or not selected or not set(selected) <= set(s.accepted)):
            final.status = "unresolved"
        if s.plan.task == "instance_count" and (selected or final.status == "confirmed_absent"):
            final.answer = str(len(selected))
        target_present = final.status != "confirmed_absent"
    candidates = s.by_id()
    s.final = {**final.model_dump(), "image_id": s.sample.image_id,
               "target_present": target_present,
               "masks": [candidates[i].mask for i in selected],
               "evidence": [{"label": candidates[i].description, "box": candidates[i].box} for i in selected],
               "mask_entries": [candidates[i].entry for i in selected],
               "answer_entry": s.answer_entry, "segment_entry": s.segment_entry,
               "agent_runs": s.agent_runs, "calls": s.calls, "history": s.history,
               "candidates": [c.model_dump() for c in s.candidates], "workflow_spec": s.spec}
    s.selected, s.answer = selected, final.answer
    root = Path(s.attempt_dir)
    tmp = root / "final.json.tmp"
    tmp.write_text(json.dumps(s.final, ensure_ascii=False, indent=1))
    tmp.replace(root / "final.json")
    s.phase = "finish"
    s.checkpoint()


@g.step
async def finish(ctx: StepContext[SampleState, Run, Finish]) -> SampleState:
    return ctx.state


decision = g.decision()
for typ, step in ((Locate, locate), (Segment, segment), (Check, check), (Verify, verify), (Repair, repair),
                  (Search, search), (Premise, premise), (Finalize, finalize), (Finish, finish)):
    decision = decision.branch(g.match(typ).to(step))
g.add(g.edge_from(g.start_node).to(dispatch), g.edge_from(dispatch).to(decision),
      *(g.edge_from(step).to(dispatch) for step in (locate, segment, check, verify, repair, search, premise, finalize)),
      g.edge_from(finish).to(g.end_node))
sample_graph = g.build()


async def solve(sample: Sample, run: Run) -> SampleState:
    state = prepare(sample, run)
    with logfire.span("sample {image_id}", image_id=sample.image_id, variant=run.variant):
        try:
            return await sample_graph.run(state=state, deps=run)
        except BudgetExhausted:
            pending = state.pending_call
            attempts = pending["attempt_results"]
            usage = {k: sum(a["usage"].get(k, 0) or 0 for a in attempts)
                     for k in {k for a in attempts for k in a["usage"]}}
            state.calls.append({"stage": pending["stage"], "entry": None, "cache_hit": False,
                                "executed_attempts": pending["attempts"], "executed_usage": usage,
                                "usage": usage, "unknown_usage_attempts": pending["attempts"] - len(attempts),
                                "elapsed_s": sum(a["elapsed_s"] for a in attempts), "budget_exhausted": True})
            state.pending_call, state.complete, state.phase = None, False, "finalize"
            state.checkpoint()
            return await sample_graph.run(state=state, deps=run)
