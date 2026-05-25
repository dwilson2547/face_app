from __future__ import annotations

import hashlib


def normalize_box(box: list[float] | tuple[float, ...]) -> list[float]:
    return [round(float(value), 2) for value in box]


def make_face_key(image_path: str, box: list[float] | tuple[float, ...]) -> str:
    normalized = ",".join(f"{value:.2f}" for value in normalize_box(box))
    digest = hashlib.sha1(f"{image_path}|{normalized}".encode("utf-8")).hexdigest()[:16]
    return f"face-{digest}"
