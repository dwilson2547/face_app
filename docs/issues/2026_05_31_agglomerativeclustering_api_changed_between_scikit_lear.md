# AgglomerativeClustering API changed between scikit-learn versions (affinity vs metric parameter)

**Severity:** sev2  
**Date:** 2026-05-31

## Root cause

scikit-learn renamed the 'affinity' parameter to 'metric' in a later release. Code using 'affinity' raises TypeError on newer versions.

## Resolution

Added a try/except in cluster_embeddings() that falls back from 'metric=' to 'affinity=' if TypeError is raised. Supports both old and new scikit-learn API.
