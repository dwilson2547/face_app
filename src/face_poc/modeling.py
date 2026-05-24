from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import insightface
import numpy as np
from PIL import Image


@dataclass(slots=True)
class DetectedFace:
    box: list[float]
    confidence: float
    embedding: np.ndarray
    thumbnail_image: Image.Image


class FacePipeline:
    def __init__(self, device: str, detector_threshold: float, model_name: str, det_size: int) -> None:
        self.device = device
        self.detector_threshold = detector_threshold
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        self.app = insightface.app.FaceAnalysis(
            name=model_name,
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect_and_embed(self, image: Image.Image) -> list[DetectedFace]:
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)

        results: list[DetectedFace] = []
        for face in faces:
            probability = float(face.det_score)
            if probability < self.detector_threshold:
                continue
            box = np.asarray(face.bbox, dtype=np.float32)
            if hasattr(face, "normed_embedding"):
                embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            else:
                raw = np.asarray(face.embedding, dtype=np.float32)
                norm = np.linalg.norm(raw)
                embedding = raw if norm == 0 else raw / norm
            thumbnail = crop_face_thumbnail(image, box)
            results.append(
                DetectedFace(
                    box=[float(value) for value in box.tolist()],
                    confidence=probability,
                    embedding=embedding.astype(np.float32),
                    thumbnail_image=thumbnail,
                )
            )
        return results


def crop_face_thumbnail(image: Image.Image, box: np.ndarray, margin_ratio: float = 0.2) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = [float(value) for value in box.tolist()]
    face_width = max(right - left, 1.0)
    face_height = max(bottom - top, 1.0)
    margin_x = face_width * margin_ratio
    margin_y = face_height * margin_ratio

    crop_box = (
        max(0, int(round(left - margin_x))),
        max(0, int(round(top - margin_y))),
        min(width, int(round(right + margin_x))),
        min(height, int(round(bottom + margin_y))),
    )
    return image.crop(crop_box).convert("RGB").resize((160, 160))


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
