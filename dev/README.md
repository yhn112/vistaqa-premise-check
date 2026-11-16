# Development data

Our development sets are in VistaQA layout (`<id>.json` plus image) and are built from public sources and synthetic figures.
No VistaQA sample, label or mask is part of them.
This directory holds the builders and our own annotations for the final sets;
source images and source annotations are not redistributed and must be downloaded under their licenses.

The final development mix is the `dev` split of five sets, 923 questions in six task domains
(`final_mix.json` lists every image ID and both fingerprints).
The builders also write `holdout` splits, which the report does not use.

| set | domains | images | questions and masks | dev |
|---|---|---|---|---:|
| `saco_dev` | indoor 62, outdoor 196 | SA-1B via SA-Co/Gold | templates on SA-Co/Gold annotations; reviewed questions; OCR from SA-Text words | 258 |
| `math_dev` | math | synthetic (`build_math_dev.py`) | programmatic answers and masks | 233 |
| `science_dev` | science | AI2D diagrams | templates on AI2D annotations; reviewed part questions on SAM 3 masks | 181 |
| `av_dev` | AV | nuImages key frames | templates on nuImages masks; reviewed questions | 115 |
| `robotics_dev` | robotics | DROID frames, RoboRefIt testB | reviewed questions; SAM 3 masks (DROID), human masks (RoboRefIt) | 136 |

About a quarter of the questions in every set have a false premise:
the answer states what is absent ("there is no ... in the image") and there is no mask.

## Annotations

- Evidence is what the question asks about: for comparisons only the selected object,
  for counting one mask per counted instance, for a label that names an object the object rather than the label
  (the text region only when the question asks what a text says), no reference or context objects.
- Reviewed questions (`*_reviewed.json`, `saco_identification.json`, `satext_ocr.json`, `saco_absent_info.json`,
  `science_parts.json`) were written by Claude Opus 5.5 agents from sheets of candidate masks under explicit criteria
  and checked blindly by another agent (all false-premise questions, all counts of four or more and a quarter of the rest).
  The tools that rendered candidate sheets and merged the reviews are not included; the builders read only their results.
- DROID and AI2D part masks are SAM 3 outputs chosen by the agents, not refined by hand.
- `saco_scene_labels.json` assigns SA-Co/Gold images to the indoor and outdoor domains.
- `*_in_vistaqa.json` and the `vistaqa_corr` field of `droid_frames.json` list source images that match VistaQA images;
  the builders exclude them.
  `check_vistaqa_overlap.py` found them by SIFT matching, a RANSAC similarity fit and correlation after alignment,
  reading VistaQA image pixels only.

## Sources

Place each source under `data/open_dev/src/`:

| source | location under `data/open_dev/src/` | license (as stated by the source) |
|---|---|---|
| [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold) annotations (gated) | `saco_gold/` | SAM License |
| SA-Co/Gold images `sa_co_gold.tar` and SA-1B shard `sa_000000.tar` from the [Segment Anything downloads](https://ai.meta.com/datasets/segment-anything-downloads/) | `saco_gold_images/sa_co_gold.tar`, `sa1b/sa_000000.tar.gz` | see the downloads page |
| [AI2D](https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip), unzipped | `ai2d/` | non-commercial research use, no redistribution |
| nuImages v1.0: `nuimages-v1.0-mini.tgz` (into `mini/`), `nuimages-v1.0-all-metadata.tgz` (`v1.0-val/`) from the [public bucket](https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/nuimages-v1.0/); key frames via `fetch_nuimages.py` | `nuimages/` | CC BY-NC-SA 4.0 with the nuScenes terms of use |
| DROID raw 1.0.1 (`gs://gresearch/robotics/droid_raw/1.0.1`), one frame per listed episode | `droid_frames/<uuid>.jpg` | see the DROID release |
| [RoboRefIt](https://huggingface.co/datasets/luyh20/RoboRefIt) testB parquet shards in `roborefit/raw/data/`, unpacked by `fetch_roborefit.py` | `roborefit/testB/` | CC-BY 4.0 (dataset card) |

```sh
uv run python -m dev.fetch_nuimages --src data/open_dev/src/nuimages
uv run --with pyarrow python -m dev.fetch_roborefit --src data/open_dev/src/roborefit
```

`droid_frames.json` lists each DROID frame with its episode, frame index and the sha256 of our JPEG.
`fetch_droid_frames.py` drew and decoded these frames;
it skips episodes already in the list, so recreating them means extracting the listed frame of each episode's
first external-camera MP4 into `droid_frames/<uuid>.jpg` (JPEG quality 95).
MP4 decoding is not bit-reproducible across OpenCV builds,
so recreated frames can differ from ours and change the robotics inputs fingerprint.

## Build and verify

```sh
uv run python -m dev.build_saco_dev --src data/open_dev/src --out data/saco_dev
uv run python -m dev.build_math_dev --out data/math_dev
uv run python -m dev.build_science_dev --src data/open_dev/src --out data/science_dev
uv run python -m dev.build_av_dev --src data/open_dev/src/nuimages --out data/av_dev
uv run python -m dev.build_robotics_dev --src data/open_dev/src --out data/robotics_dev
uv run python -m vistaqa.fingerprint data/<set>/dev   # compare with final_mix.json
```

With our sources, these commands reproduce all five `dev` splits of `final_mix.json` exactly (both fingerprints).
Predict and evaluate on a split as on VistaQA, with `--data data/<set>/dev` and `--gt data/<set>/dev`.
