# Clustering Tuning Reference

## Sample Set Baseline

- Input: `sample_photos/` (27 images, 96 faces, 0 problem images)
- Baseline run: `reports/run-002/`
- Best known recluster: `reports/recluster-060/`

## Agglomerative Clustering Results (threshold sweep)

| threshold | clusters | noise | notes |
|---|---|---|---|
| 0.40 | ~20 | low | over-merges — different people grouped together |
| 0.45 | ~27 | low | still too aggressive |
| 0.50 | ~34 | 0 | reasonable starting point |
| 0.60 | **41** | **0** | **current best — 41 clusters from 96 faces** |

Threshold 0.6 with average linkage and cosine metric appears to be the sweet spot for the sample set.

## DBSCAN Notes

- DBSCAN with `eps=0.35, min_samples=1` works for small-to-medium sets but tends to fragment.
- Agglomerative generally outperforms DBSCAN for this use case.
- Start with Agglomerative at threshold=0.6 for new datasets.

## Detector Threshold Notes

- Default: 0.6 (lowered from 0.95)
- 0.5-0.65 is good for casual photos (outdoor, angled, small faces)
- Raise toward 0.8 only if getting too many false positives on clear portrait-style photos

## Face Key Stability

- face_key is derived from `image_path + bounding_box` (normalized)
- Stays stable across recluster operations on the same source run
- Changes if images are moved or bounding boxes shift (i.e. different detection run)
