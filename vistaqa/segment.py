"""Stage 2: turn evidence boxes into instance masks with SAM 3."""

from __future__ import annotations

from functools import cache

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

from vistaqa.data import Sample
from vistaqa.fingerprint import image_digest

# SAM 3 box prompts go through its tracker (promptable visual segmentation) head.
MODEL_ID = "facebook/sam3"
MULTIMASK_OUTPUT = False
# Bump when segmentation changes in a way cache_spec does not see (code in this module).
VERSION = 1
OPERATION_VERSION = 1
PROPOSAL_THRESHOLD = 0.3
MASK_THRESHOLD = 0.5
PROPOSAL_TEXT_MAX_LENGTH = 32


@cache
def proposal_tokenizer():
    """Load only SAM3's text tokenizer, without model weights or image inference."""
    return Sam3Processor.from_pretrained(MODEL_ID).tokenizer


def _check_text_length(tokens: int):
    if tokens > PROPOSAL_TEXT_MAX_LENGTH:
        raise ValueError(
            f"SAM3 search_text has {tokens} tokens; the limit is {PROPOSAL_TEXT_MAX_LENGTH} including special tokens. "
            "Rewrite it as a short object noun phrase (usually 1-6 words). "
            "Put location and exclusion constraints in boxes or candidate selection, not in search_text."
        )


def validate_search_text(text: str, *, allow_empty: bool = False):
    if not text.strip():
        if allow_empty:
            return
        raise ValueError("SAM3 search_text must name an object; use a short visual noun phrase.")
    _check_text_length(len(proposal_tokenizer()(text, add_special_tokens=True)["input_ids"]))


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cache_spec(sample: Sample, boxes: list[list[float]]) -> dict:
    """Everything the masks of these boxes depend on; its digest is the key of the mask cache (vistaqa/cache.py)."""
    return {
        "node": "segment",
        "version": VERSION,
        "image": image_digest(sample),
        "boxes": boxes,
        "model": MODEL_ID,
        "multimask_output": MULTIMASK_OUTPUT,
        "device": default_device(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }


def operation_spec(sample: Sample, operation: str, prompts: dict) -> dict:
    spec = {**cache_spec(sample, []), "node": operation, "version": OPERATION_VERSION, "box_segmenter_version": VERSION,
            "prompts": prompts, "proposal_threshold": PROPOSAL_THRESHOLD, "mask_threshold": MASK_THRESHOLD}
    if operation == "proposal":
        spec["text_tokenization"] = {"max_length": PROPOSAL_TEXT_MAX_LENGTH, "truncation": False}
    return spec


class BoxSegmenter:
    def __init__(self) -> None:
        self.device = default_device()
        self.processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
        self.model = Sam3TrackerModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self.proposal_model = None
        self.proposal_processor = None

    @torch.inference_mode()
    def masks(self, image: Image.Image, boxes: list[list[float]]) -> list[np.ndarray]:
        """One boolean HxW mask per [x1, y1, x2, y2] box, in original image resolution."""
        if not boxes:
            return []
        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, multimask_output=MULTIMASK_OUTPUT)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        return [m[0].numpy().astype(bool) for m in masks]

    @torch.inference_mode()
    def proposals(self, image: Image.Image, text: str, positive_boxes: list[list[float]],
                  negative_boxes: list[list[float]]) -> list[dict]:
        """Concept segmentation is a separate SAM3 head; no proposals means no detection, not absence."""
        if self.proposal_model is None:
            self.proposal_processor = Sam3Processor.from_pretrained(MODEL_ID)
            self.proposal_model = Sam3Model.from_pretrained(MODEL_ID).to(self.device).eval()
        boxes = positive_boxes + negative_boxes
        prompts = {"input_boxes": [boxes], "input_boxes_labels": [[1] * len(positive_boxes) +
                                                                  [0] * len(negative_boxes)]} if boxes else {}
        inputs = self.proposal_processor(images=image, text=text or None, **prompts, return_tensors="pt")
        # Model outputs are validated upstream and corrected there, never truncated.
        _check_text_length(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.device)
        outputs = self.proposal_model(**inputs)
        result = self.proposal_processor.post_process_instance_segmentation(
            outputs, threshold=PROPOSAL_THRESHOLD, mask_threshold=MASK_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]
        from vistaqa import rle

        return [{"mask": rle.encode(m.cpu().numpy()), "box": b.cpu().tolist(), "score": float(s.cpu())}
                for m, b, s in zip(result["masks"], result["boxes"], result["scores"], strict=True)]

    @torch.inference_mode()
    def refine(self, image: Image.Image, box: list[float], positive_points: list[list[float]],
               negative_points: list[list[float]]) -> np.ndarray:
        """Re-run the tracker head with the instance box and accumulated point constraints.

        This also works for PCS proposals without pretending their logits are tracker logits.
        The parent mask remains immutable and is compared with this new candidate by Verify.
        """
        points = positive_points + negative_points
        inputs = self.processor(images=image, input_boxes=[[box]], input_points=[[points]],
                                input_labels=[[[1] * len(positive_points) + [0] * len(negative_points)]],
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, multimask_output=MULTIMASK_OUTPUT)
        mask = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        return mask[0, 0].numpy().astype(bool)
