"""Stage 1: ask the VLM (a PydanticAI agent on the Codex backend) for an answer and per-instance evidence boxes."""

from __future__ import annotations

import asyncio
import io
import time
from functools import cache
from pathlib import Path

import pydantic_ai
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelRetry, NativeOutput, RunContext, Tool
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.models.openai_codex import OpenAICodexModel

from vistaqa.data import Sample, load_image
from vistaqa.fingerprint import digest, input_digest

# Bump when the answer node changes in a way cache_spec does not see (code in this module).
VERSION = 1
PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompt.md").read_text()
# Images go to the model unscaled; `original` is the only detail level without a pixel budget.
IMAGE_DETAIL = "original"
CROP_MIN_SIDE = 1024  # smaller crops are enlarged to this long side
pydantic_ai.BANNER_ENABLED = False  # keep the first-run banner out of predict.log


class Evidence(BaseModel):
    label: str
    box: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[x_min, y_min, x_max, y_max] in pixels of the original image, origin top-left.",
    )


class Answer(BaseModel):
    answer: str = Field(description="Concise English answer to the question.")
    target_present: bool = Field(
        description="False when what the question refers to is not in the image, the requested information "
        "is missing or cannot be determined, or the count asked for is zero."
    )
    evidence: list[Evidence] = Field(
        description="One entry per evidence instance; empty when target_present is false."
    )


def _png(image: Image.Image) -> BinaryContent:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return BinaryContent(data=buffer.getvalue(), media_type="image/png", vendor_metadata={"detail": IMAGE_DETAIL})


def crop(ctx: RunContext[Image.Image], x_min: float, y_min: float, x_max: float, y_max: float) -> BinaryContent:
    """Return the region [x_min, y_min, x_max, y_max] of the image for a closer look.

    Coordinates are pixels of the image, origin at the top-left corner. Small regions are returned enlarged.
    """
    image = ctx.deps
    left, right = sorted((min(max(x_min, 0.0), image.width), min(max(x_max, 0.0), image.width)))
    top, bottom = sorted((min(max(y_min, 0.0), image.height), min(max(y_max, 0.0), image.height)))
    if right - left < 1 or bottom - top < 1:
        raise ModelRetry(f"The region is empty; the image is {image.width}x{image.height} px.")
    region = image.crop((int(left), int(top), round(right), round(bottom)))
    if max(region.size) < CROP_MIN_SIDE:
        zoom = CROP_MIN_SIDE / max(region.size)
        region = region.resize((round(region.width * zoom), round(region.height * zoom)), Image.Resampling.LANCZOS)
    return _png(region)


CROP = Tool(crop, takes_ctx=True)
agent = Agent(output_type=NativeOutput(Answer), deps_type=Image.Image, tools=[CROP])


@cache
def _model(name: str) -> OpenAICodexModel:
    """One model per name: its provider refreshes the Codex token in memory, and refresh tokens are single-use."""
    return OpenAICodexModel(name)


def cache_spec(sample: Sample, model: str, effort: str) -> dict:
    """Everything an answer depends on; its digest is the key of the answer cache (vistaqa/cache.py)."""
    tool = CROP.tool_def  # what the model is told about the tool
    return {
        "node": "answer",
        "version": VERSION,
        "inputs": input_digest(sample),
        "prompt": digest({"text": PROMPT_TEMPLATE}),
        "schema": digest(Answer.model_json_schema()),
        "tools": digest({"name": tool.name, "description": tool.description,
                         "parameters": tool.parameters_json_schema}),
        "image_detail": IMAGE_DETAIL,
        "crop_min_side": CROP_MIN_SIDE,
        "model": model,
        "effort": effort,
        "pydantic_ai": pydantic_ai.__version__,
    }


def _clean_boxes(evidence: list[dict], width: int, height: int) -> list[dict]:
    """Clip boxes to the image and drop degenerate ones."""
    cleaned = []
    for item in evidence:
        x1, y1, x2, y2 = (float(v) for v in item["box"])
        x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
        y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            cleaned.append({"label": item["label"], "box": [x1, y1, x2, y2]})
    return cleaned


async def query_vlm(
    sample: Sample,
    model: str,
    effort: str,
    timeout_s: int,
    attempts: int,
) -> dict:
    image = await asyncio.to_thread(load_image, sample)
    prompt = PROMPT_TEMPLATE.format(width=image.width, height=image.height, question=sample.question)
    content = [prompt, await asyncio.to_thread(_png, image)]
    last_error = ""
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            result = await agent.run(
                content,
                model=_model(model),
                model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort=effort),
                deps=image,
                timeout=timeout_s,
            )
        except AgentRunError as exc:  # API errors, invalid output, timeout
            last_error = f"attempt {attempt}: {type(exc).__name__}: {str(exc)[:500]}"
            continue
        raw = result.output.model_dump()
        usage = result.usage
        return {
            "image_id": sample.image_id,
            "answer": raw["answer"],
            "target_present": raw["target_present"],
            "evidence": _clean_boxes(raw["evidence"], sample.width, sample.height)
            if raw["target_present"]
            else [],
            "raw": raw,
            "model": model,
            "effort": effort,
            "sent_size": list(image.size),
            "elapsed_s": round(time.monotonic() - started, 1),
            "attempt": attempt,
            "tool_calls": usage.tool_calls,
            "usage": {
                "requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "output_tokens": usage.output_tokens,
                "output_reasoning_tokens": usage.output_reasoning_tokens,
            },
        }
    raise RuntimeError(f"image_id {sample.image_id}: VLM failed: {last_error}")
