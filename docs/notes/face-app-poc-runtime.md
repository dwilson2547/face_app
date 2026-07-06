---
title: Face app POC runtime
date: 2026-05-31
tags: insightface, python, environment
---

The face-grouping POC uses InsightFace locally instead of facenet-pytorch because facenet-pytorch hit a Python 3.13 packaging issue through Pillow during install. InsightFace + onnxruntime-gpu installed cleanly, supports local model caching, and the scaffold smoke run succeeded.
