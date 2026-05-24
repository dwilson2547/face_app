# face_app

Local face-grouping POC for clustering the same person across a folder of images.

## Scope

This repository currently targets a narrow first milestone:

- read a local directory of images
- detect one or more faces per image
- generate face embeddings locally
- cluster likely same-person faces across the dataset
- emit JSON artifacts and an HTML report for review

EXIF, video processing, desktop packaging, and broader app architecture are intentionally out of scope for this first POC.

## Setup

```bash
conda create -n face-poc python=3.13 -y
conda activate face-poc
pip install -r requirements.txt
pip install -e . --no-deps
```

## Run

```bash
python run_face_poc.py run --input-dir /path/to/images --output-dir reports/run-001
```

Useful tuning flags:

```bash
python run_face_poc.py run \
  --input-dir /path/to/images \
  --output-dir reports/run-002 \
  --clusterer agglomerative \
  --agglomerative-distance-threshold 0.4 \
  --detector-threshold 0.6
```

To rerun clustering only from an existing run:

```bash
python run_face_poc.py recluster \
  --input-run-dir reports/run-002 \
  --output-dir reports/recluster-050 \
  --agglomerative-distance-threshold 0.5
```

## Outputs

Each run writes:

- `run.json` - config, model info, and summary counts
- `images.json` - per-image ingest and detection status
- `faces.json` - per-face detections and cluster assignments
- `clusters.json` - grouped face summaries
- `embeddings.npy` - normalized face embeddings
- `report.html` - local review report
- `thumbnails/` - face crops used in the report

## Notes

- If CUDA is available through ONNX Runtime, the script uses it automatically unless you override `--device`.
- Multiple faces per photo are supported.
- Bad files or undetected images are recorded in the output instead of stopping the run.
- The first run will download the local InsightFace model pack and cache it locally.