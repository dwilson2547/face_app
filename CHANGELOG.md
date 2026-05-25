# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.0] - 2026-05-24

### Added

- SQLite-backed correction layer for must-link/cannot-link constraints, cluster merge/split actions, and optional person labels.
- `corrections` CLI commands for summary, linking, splitting, merging, and labeling.
- Automatic application of saved corrections on future `run` and `recluster` commands.

### Changed

- Reclustering now supports a persistent truth layer on top of raw embeddings instead of being a one-off operation.

## [0.1.0] - 2026-05-24

### Added

- Initial local face-grouping POC for clustering the same person across image sets.
- Explicit root entry script at `run_face_poc.py`.
- Detection, embedding, clustering, JSON artifact generation, and HTML report generation under `src/face_poc/`.
- `recluster` workflow for reusing existing embeddings without rerunning face detection.
- `requirements.txt` and `pytest.ini`.

### Changed

- Standardized setup around **conda** instead of virtualenv.
- Lowered the default detector threshold from `0.95` to `0.6` after the sample set showed the original value was too strict.

### Current baseline

- Sample set: `sample_photos/`
- Baseline run: `reports/run-002/`
- Current best recluster sweep: `reports/recluster-060/`
- Sample result at current best sweep: **27 images**, **96 faces**, **41 clusters**, **0 problem images**
