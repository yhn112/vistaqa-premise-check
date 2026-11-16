# Answer, Segment, Then Check the Premise

System of team YHN112 for the [VistaQA Zero-Shot Challenge](https://vlm4rwd.github.io/challenges.html#vistaqa)
(VLM4RWD workshop, NeurIPS 2026).
Author: Michael Diskin, HSE University.

The system is training-free:

1. A reasoning VLM (gpt-6.1-sol at reasoning effort xhigh, through the `openai-codex:` provider of PydanticAI)
   returns an answer, a presence flag and one box per evidence instance (`vistaqa/vlm.py`, prompt in `vistaqa/prompt.md`).
2. The tracker head of SAM 3 (`facebook/sam3`) turns each box into a mask (`vistaqa/segment.py`).
3. A premise check in fresh contexts lists what the question takes for granted (text only)
   and checks each claim in the image without seeing the answer.
   Only a clearly failed claim replaces the answer with an absence answer and no masks (`vistaqa/reasoning.py`, `vistaqa/sample.py`).

Variant `a` is steps 1–2; variant `e`, the submitted system, adds step 3.

## Submitted predictions and evaluation

| run | GROVE, ε = 0.01 | GROVE, ε = 0.1 | Sa | Sm |
|---|---:|---:|---:|---:|
| `results/vistaqa_premise-a-xhigh` (A) | 73.68 | 76.83 | 80.81 | 75.60 |
| `results/vistaqa_premise-e-xhigh` (E, submitted) | **75.31** | 78.29 | 82.28 | 77.08 |

All 1157 public VistaQA samples; judge Qwen2.5-72B-Instruct through OpenRouter at temperature 0,
with the official [VistaQA-Evaluation](https://github.com/VistaQA/VistaQA-Evaluation) code at commit `6ad724f`, unmodified.
E reuses A's answers and masks and changes only the answers whose premise check fails.

Each run directory contains:

- `predictions.jsonl`: the predictions (E's file is the submission);
- `run.jsonl`: invocation conditions (code commit, inputs fingerprint, library and SAM 3 versions);
- `eval_qwen72b/`: `evaluation_eps0.1.json` and `evaluation_eps0.01.json` (overall scores and the official breakdowns),
  `per_sample.jsonl` (per-sample answer and mask scores), `judge.jsonl` (cached judge verdicts) and `eval.log`.

### Verification with the official evaluator

`predictions.jsonl` is in the format of the official evaluator and can be scored by it unchanged
(GT: the [VistaQA dataset](https://huggingface.co/datasets/mozhina/VistaQA) in `data/vistaqa/VistaQA`):

```sh
git clone https://github.com/VistaQA/VistaQA-Evaluation third_party/VistaQA-Evaluation
git -C third_party/VistaQA-Evaluation checkout 6ad724f
export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=<OpenRouter key>
uv run python third_party/VistaQA-Evaluation/src/main.py --gt data/vistaqa/VistaQA \
  --pred results/vistaqa_premise-e-xhigh/predictions.jsonl \
  --llm_model qwen/qwen-2.5-72b-instruct --epsilon 0.01 --output_dir outputs/official_eps0.01
```

The official script judges every answer anew.
OpenRouter does not fix the provider that serves the judge, so answer scores can differ slightly from our saved verdicts;
mask scores do not depend on the judge.
Our evaluator (`eval/evaluate.py`) calls the official loader, judge, mask metric and report functions and caches verdicts;
in an earlier check on a full VistaQA prediction file with one shared judge,
the unmodified script gave the same overall GROVE and breakdowns.

## Development data

[`dev/`](dev/README.md) holds the builders, our annotations and the manifest of the final development mix
(923 questions in six domains) used to select the system; source images are downloaded separately.

## External resources

- **Training or fine-tuning:** none.
- **VLM:** gpt-6.1-sol at reasoning effort xhigh, through Codex
  (the `openai-codex:` provider of PydanticAI 2.67.0, which needs a Codex login).
- **Segmenter:** SAM 3 (`facebook/sam3`) through transformers 5.22.2 and torch 2.15.0, float32;
  the weights are gated on Hugging Face (access is granted on request, under their own license).
- **Orchestration:** pydantic-graph 2.67.0; local tracing with logfire 5.6.0.
- **Judge:** Qwen2.5-72B-Instruct through OpenRouter at temperature 0,
  with the official VistaQA-Evaluation code at commit `6ad724f`.
- **Development data:** SA-Co/Gold, SA-1B, SA-Text, AI2D, nuImages, DROID and RoboRefIt for the final mix;
  an early open-domain set also used COCO, COCO-ReM, LVIS, PACO, TextOCR, TextVQA and VQA-AnswerTherapy.
  Sources and licenses: [`dev/README.md`](dev/README.md).
- **Annotation agents:** Claude Opus 5.5 agents wrote and blindly checked the reviewed dev questions
  and selected and described the DROID frames.

## Code

The code is a snapshot of `vistaqa/` and `eval/` at commit `d10bd97f` of our development repository;
the development history is not published.
E and the completing invocation of A recorded this commit in `run.jsonl`;
A's first invocation recorded `d533d81d`, whose `vistaqa/` and `eval/` are identical,
and no invocation had uncommitted changes in these directories.
At this commit `--variant` defaults to `a`, and variants `b`–`d` belong to an abandoned study.

Setup (Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)):

```sh
uv sync
git clone https://github.com/VistaQA/VistaQA-Evaluation third_party/VistaQA-Evaluation
```

- Data: download the [VistaQA dataset](https://huggingface.co/datasets/mozhina/VistaQA)
  so that `data/vistaqa/VistaQA` holds the `<id>.json` files and images.
- VLM: the `openai-codex:` provider reads the Codex CLI login in `~/.codex/auth.json` (`codex login`).
- SAM 3: the `facebook/sam3` weights are loaded from Hugging Face; they run on Apple MPS when available, otherwise on CPU.
- Judge: set `OPENAI_API_KEY` to an OpenRouter key; `OPENAI_BASE_URL` and `--llm-model` override the endpoint and model.

Prediction and evaluation of both runs:

```sh
uv run python -m vistaqa.predict --data data/vistaqa/VistaQA --run-dir results/vistaqa_premise-a-xhigh --variant a --until evaluate
uv run python -m vistaqa.predict --data data/vistaqa/VistaQA --run-dir results/vistaqa_premise-e-xhigh --variant e --initial-run results/vistaqa_premise-a-xhigh --until evaluate
```

The model is a closed, sampled API, so a new run produces different answers; run new predictions into new run directories.
To rescore the saved predictions instead, run the evaluator on the included run directory;
it reuses the cached verdicts in `judge.jsonl` and queries the judge only for missing ones:

```sh
uv run python -m eval.evaluate --gt data/vistaqa/VistaQA --run-dir results/vistaqa_premise-e-xhigh --name qwen72b
```

## License

Our code and annotations are released under the [MIT License](LICENSE).
The source datasets, models and evaluation code keep their own licenses.
