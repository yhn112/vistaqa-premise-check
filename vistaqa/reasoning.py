"""Fresh-context structured agents for planning, verification, selection, finalization and premise checks.

Only explicit perception inputs cross stage boundaries. Verifiers never receive
the locator's object descriptions, rationale or earlier verdict explanations.
The premise lister sees only the question; the premise checker sees the question,
the listed claims and the image, never the answer it may overturn.
"""

from __future__ import annotations

import json
import time
from functools import cache

from PIL import Image
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

from vistaqa import evidence, vlm
from vistaqa.evidence import (EvidencePlan, FinalAnswer, MissingSearch, PremiseCheck, Presuppositions, Selection,
                              Verification)
from vistaqa.fingerprint import digest, input_digest
from vistaqa.data import Sample

VERSION = 1
COMMON = """You solve an image-grounded visual question. Return structured JSON in English.
Do not browse. Coordinates are pixels of the original image, origin top-left.
Only the objects asked about are exportable evidence. Spatial references are anchors.
For comparison, only the selected object(s) are evidence. For reading a label that
names an object, evidence is that object; for reading scene text, evidence is the text.
Check false premises, missing reference objects, missing information and zero counts.
Do not infer absence from segmentation failure. Use crop for a closer look.
"""
SAM_SEARCH = """
Every search_text is sent directly to SAM3's small text encoder. Write only a short
visual object noun phrase, usually 1-6 words, such as "small coloured flag" or
"pink and white motor scooter". The limit is 32 tokens including special tokens.
Do not send sentences, commands, coordinates, spatial relations, ordinals, exclusions
or the full question to SAM3. Keep that reasoning in descriptions, box prompts and
selection among numbered candidates. For several categories, return separate short queries.
For relocation, keep the detailed instance description separate from search_text.
"""
PROMPTS = {
    "locate": COMMON + vlm.PROMPT_TEMPLATE.replace("`target_present`", "`answerable`").replace(
        "`evidence` list", "`targets` list") + """
Return an EvidencePlan instead of the old answer/evidence schema. The answer is provisional.
Put reference and losing comparison objects in anchors, never in targets.
Give one target per evidence instance, including every counted instance.
Give stable unique IDs, descriptions, tight boxes, and short SAM search noun phrases.
Boxes may be null only when localization genuinely needs proposals.
Use task=instance_count only when the answer counts separate evidence instances,
not for a number read from text or a numeric property such as the sides of one shape.
expected_mask_count is an initial expectation, not a constraint to force an answer.
""",
    "verify": COMMON + SAM_SEARCH + """Independently inspect the question, proposed answer and numbered masks.
You receive the original image, mask panels and sometimes enlarged context crops.
Judge both object identity and visible boundaries, and search for omitted instances.
Mask IDs label candidate versions, not distinct objects. Prefer at most one version per object.
Accept correct evidence explicitly; reject irrelevant or incorrect versions explicitly.
For wrong or missing instances request proposals or relocate with a tight new box.
For a correct instance with bad edges request refine_mask with positive/negative points.
For an uncertain premise request search_missing. Reconsider an answer if evidence contradicts it.
Do not repeat an earlier operation without a different box, points or search hypothesis.
complete means every required instance and premise was checked, not merely all supplied masks.
""",
    "select": COMMON + """Select the numbered candidate masks that answer the question.
Compare the original image and all panels; do not select reference or losing comparison objects.
Return an object ID and description for every selected mask, keeping known target IDs
when they refer to the same instance. New instances need new IDs.
Do not force the provisional count. Return a revised concise answer and completeness.
""",
    "search": COMMON + SAM_SEARCH + """Independently search for the objects and conditions required by the question.
The previous segmentation may have failed. Inspect the original image and use crops.
Report present, absent, missing_information or uncertain; absence requires visual evidence,
not an empty list of detector results. When found, return tight target boxes and search phrases.
""",
    "finalize": COMMON + """Finalize the answer using the chosen evidence and original image.
Answer the image question in a short phrase or sentence. Keep pipeline commentary
(mask IDs, rejected candidates, verification and correction limits) out of the answer.
Re-evaluate counting, spatial relations, comparisons and the premise of the question.
Select only available target mask IDs; never select explicitly rejected or invalid candidates.
Keep answer and masks consistent. Do not turn an empty segmentation into a claim of absence.
confirmed_absent is allowed only after the independent search confirmed absence/missing information.
If uncertainty remains, return the best supported answer and masks with status=unresolved.
Do not replace uncertainty with a mandatory abstention. No further corrections are possible here.
""",
    "presuppose": """List what a visual question takes for granted about its image. You do not see the image.
Return structured JSON in English. A presupposition must hold in the image for the question to have
a non-empty answer. Write each as a short declarative claim:
- object: every object, person or region the question refers to exists, including reference objects
  ("the cup to the left of the knife": "There is a cup.", "There is a knife.");
- attribute: every property the question ascribes to them, such as colour, material, size, state,
  action or clothing ("the pink bucket": "The bucket is pink.");
- relation: every spatial or other relation the question asserts
  ("the cup to the left of the knife": "A cup is to the left of a knife.");
- text_or_information: text, a label or information the question assumes is visible
  ("the departure time for Toronto on the board": "The board lists Toronto.").
For a counting question, the counted things with all their stated properties exist at least once.
List only what the question states or directly implies, never the unknown it asks for.
Claims are visual facts. Read loose or idiomatic wording by its visual meaning
("the number he likes to wear" presupposes only that he wears a number);
never list preferences, intentions or feelings, and never a claim that the image shows the answer.
One claim per object, attribute and relation. Return an empty list if nothing is presupposed.
""",
    "premise_check": """Check presuppositions of a visual question against the image. Return structured JSON in English.
Do not browse. Coordinates are pixels of the original image, origin top-left.
Use crop to inspect small, distant or ambiguous regions and search the whole image before deciding.
For every listed claim return one verdict with the visual observation behind it:
- holds: the image shows that the claim is true; a synonym or a close variant of the named thing counts;
- fails: the image clearly shows that the claim is false: the thing is not there anywhere,
  it visibly lacks the attribute (a white bucket where the claim says pink), the relation is another one,
  or the text or information is not where the claim places it;
- unclear: occlusion, blur, resolution or ambiguous wording prevent a confident decision.
A similar object with a different attribute does not make an attribute claim hold.
Judge the visual content of each claim; loose wording, or a preference or intention that cannot be seen,
is unclear at most and never fails on that ground alone.
If any claim fails, absence_answer is one sentence that states only what is absent, in the question's words,
for example "There is no pink bucket in the image.", "There is no information about Toronto on the board."
or "There are no performers wearing yellow vests."; it says nothing about what is there instead.
Otherwise absence_answer is empty.
""",
}
SCHEMAS = {"locate": EvidencePlan, "verify": Verification, "select": Selection,
           "search": MissingSearch, "finalize": FinalAnswer,
           "presuppose": Presuppositions, "premise_check": PremiseCheck}
TEXT_ONLY = {"presuppose"}
# On open dev (2026-11-13) premise calls took at most 36 s (lister) and 84 s (checker);
# a longer wait is a stalled connection, so retry sooner than the run's general timeout.
STAGE_TIMEOUTS = {"presuppose": 180, "premise_check": 180}


def stage_timeout(stage: str, timeout: int) -> int:
    """The run's --timeout, capped by the stage's own limit."""
    return min(timeout, STAGE_TIMEOUTS.get(stage, timeout))


@cache
def agent(stage: str) -> Agent:
    tools = [] if stage in TEXT_ONLY else [vlm.CROP]
    return Agent(output_type=NativeOutput(SCHEMAS[stage]), deps_type=Image.Image, tools=tools)


def cache_spec(sample: Sample, stage: str, model: str, effort: str, payload: dict) -> dict:
    spec = vlm.cache_spec(sample, model, effort)
    return {**spec, "node": stage, "version": VERSION, "inputs": input_digest(sample),
            "prompt": digest({"text": PROMPTS[stage]}), "schema": digest(SCHEMAS[stage].model_json_schema()),
            "payload": payload, "render_version": evidence.VERSION}


async def query(stage: str, sample: Sample, image: Image.Image, images: list[Image.Image],
                payload: dict, model: str, effort: str, timeout: int, feedback: str = "") -> dict:
    """One agent.run; orchestration owns retries, budgets and cache provenance."""
    prompt = PROMPTS[stage]
    if stage == "locate":
        prompt = prompt.format(width=sample.width, height=sample.height, question=sample.question)
    if stage not in TEXT_ONLY:
        prompt += f"\nOriginal image: {sample.width}x{sample.height} px."
    prompt += f"\nQuestion: {sample.question}\n"
    prompt += json.dumps(payload, ensure_ascii=False) + feedback
    content = [prompt] if stage in TEXT_ONLY else [prompt, vlm._png(image), *[vlm._png(i) for i in images]]
    started = time.monotonic()
    result = await agent(stage).run(content, model=vlm._model(model), deps=image, timeout=stage_timeout(stage, timeout),
                                    model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort=effort))
    usage = result.usage
    return {"output": result.output.model_dump(), "elapsed_s": round(time.monotonic() - started, 3),
            "usage": {k: getattr(usage, k) for k in ("requests", "input_tokens", "cache_read_tokens",
                                                     "output_tokens", "output_reasoning_tokens", "tool_calls")}}
