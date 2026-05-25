# Copilot Instructions

## Project focus

This repository is currently a **local face-grouping POC**, not a full media app yet.

Current goal:

- ingest a local directory of images
- detect multiple faces per image
- generate local embeddings
- cluster likely same-person faces
- emit HTML and JSON artifacts for review

Out of scope for now:

- EXIF work
- video parsing
- desktop packaging
- broader product architecture

## Environment and tooling

- **Conda only. Do not create virtualenvs.**
- Python dependencies use **`pyproject.toml` as the source of truth**.
- `requirements.txt` is a mirror for convenience.
- Tests run with `pytest` and `pytest.ini` sets `pythonpath = src`.

## Entry points

- Prefer the explicit script entrypoint: `python run_face_poc.py ...`
- Main user-facing commands:
  - `python run_face_poc.py run ...`
  - `python run_face_poc.py recluster ...`

Do not hide the effective start point behind packaging-only conventions when a plain script will do.

## Current implementation notes

- Root script: `run_face_poc.py`
- Package code: `src/face_poc/`
- Main runtime stack: **InsightFace + ONNX Runtime**
- Correction layer: **SQLite** with persistent must-link/cannot-link constraints and optional labels
- Reports/artifacts go under `reports/`
- Default corrections DB path: `data/corrections.sqlite3`
- Current detector threshold default is **0.6**

## Known-good workflow

Setup:

```bash
conda create -n face-poc python=3.13 -y
conda activate face-poc
pip install -r requirements.txt
pip install -e . --no-deps
```

Initial run:

```bash
python run_face_poc.py run \
  --input-dir /path/to/images \
  --output-dir reports/run-001
```

Recluster an existing run:

```bash
python run_face_poc.py recluster \
  --input-run-dir reports/run-002 \
  --output-dir reports/recluster-060 \
  --agglomerative-distance-threshold 0.6
```

## Current baseline

- `reports/run-002/` is the current working baseline run on `sample_photos/`
- `reports/recluster-060/` is the current best local clustering result so far
- That result produced **27 images**, **96 faces**, **41 clusters**, **0 problem images**

## Preferences to preserve

- Keep local AI workloads local; no external AI APIs.
- Prefer explicit, obvious entrypoints over implicit CLI/package wiring.
- Do not assume API/UI/product structure work is the priority until the face-grouping pipeline is proven.
