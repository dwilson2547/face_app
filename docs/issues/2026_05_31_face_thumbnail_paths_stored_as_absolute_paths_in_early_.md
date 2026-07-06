# Face thumbnail paths stored as absolute paths in early runs break portability

**Severity:** sev3  
**Date:** 2026-05-31

## Root cause

Early versions stored thumbnail_path as an absolute filesystem path in faces.json. This breaks when the reports/ directory is moved or the machine changes.

## Resolution

Changed to store thumbnail_path as relative path from the run output directory (e.g. 'thumbnails/img-0000-face-00.jpg'). thumbnail_relpath is also computed relative to the consuming run dir for recluster.
