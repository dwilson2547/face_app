---
title: Face app detector threshold
date: 2026-05-31
tags: insightface, thresholds
---

The initial InsightFace detector threshold of 0.95 was too strict for the sample_photos set and produced zero faces. Lowering the default threshold to 0.6 yielded detections on all 27 images in the sample set (96 faces total, 0 problem images), so 0.6 is the current default baseline.
