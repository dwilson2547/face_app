from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from face_poc.clustering import cluster_embeddings, summarize_clusters
from face_poc.config import RunConfig
from face_poc.corrections import apply_saved_corrections, ensure_face_keys, load_run_artifacts
from face_poc.discovery import discover_images
from face_poc.identity import make_face_key, normalize_box
from face_poc.modeling import FacePipeline, load_image
from face_poc.reporting import write_html_report, write_json


def run_pipeline(config: RunConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir = config.output_dir / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(config.input_dir, recursive=config.recursive)
    model = None
    if image_paths:
        model = FacePipeline(
            device=config.device,
            detector_threshold=config.detector_threshold,
            model_name=config.model_name,
            det_size=config.det_size,
        )

    image_records: list[dict[str, object]] = []
    face_records: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []

    for image_index, image_path in enumerate(image_paths):
        image_id = f"img-{image_index:04d}"
        rel_path = image_path.relative_to(config.input_dir)
        try:
            image = load_image(image_path)
        except Exception as exc:
            image_records.append(
                {
                    "image_id": image_id,
                    "path": str(rel_path),
                    "status": "load_error",
                    "error": str(exc),
                    "face_count": 0,
                }
            )
            continue

        try:
            assert model is not None
            detected_faces = model.detect_and_embed(image)
        except Exception as exc:
            image_records.append(
                {
                    "image_id": image_id,
                    "path": str(rel_path),
                    "status": "processing_error",
                    "error": str(exc),
                    "face_count": 0,
                }
            )
            continue

        if not detected_faces:
            image_records.append(
                {
                    "image_id": image_id,
                    "path": str(rel_path),
                    "status": "no_faces",
                    "error": None,
                    "face_count": 0,
                }
            )
            continue

        image_records.append(
            {
                "image_id": image_id,
                "path": str(rel_path),
                "status": "ok",
                "error": None,
                "face_count": len(detected_faces),
            }
        )

        for face_index, detected_face in enumerate(detected_faces):
            face_id = f"{image_id}-face-{face_index:02d}"
            thumbnail_name = f"{face_id}.jpg"
            thumbnail_path = thumbnail_dir / thumbnail_name
            detected_face.thumbnail_image.save(thumbnail_path, format="JPEG", quality=90)

            embeddings.append(detected_face.embedding)
            face_records.append(
                {
                    "face_id": face_id,
                    "face_key": make_face_key(str(rel_path), normalize_box(detected_face.box)),
                    "image_id": image_id,
                    "image_path": str(rel_path),
                    "box": normalize_box(detected_face.box),
                    "confidence": round(detected_face.confidence, 6),
                    "thumbnail_path": str(Path("thumbnails") / thumbnail_name),
                    "thumbnail_relpath": str(Path("thumbnails") / thumbnail_name),
                }
            )

    embedding_array = np.stack(embeddings).astype(np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)
    labels = cluster_embeddings(
        embeddings=embedding_array,
        method=config.clusterer,
        dbscan_eps=config.dbscan_eps,
        dbscan_min_samples=config.dbscan_min_samples,
        agglomerative_distance_threshold=config.agglomerative_distance_threshold,
    )
    labels, label_map, correction_warnings = apply_saved_corrections(
        face_records=face_records,
        labels=labels,
        db_path=config.corrections_db,
    )

    for record, label in zip(face_records, labels):
        record["cluster_id"] = int(label)
        record["thumbnail_relpath"] = str(record["thumbnail_path"])
        record["person_label"] = label_map.get(str(record["face_key"]))

    clusters = summarize_clusters(face_records)
    cluster_face_index = Counter(int(record["cluster_id"]) for record in face_records)
    run_payload = {
        "config": config.to_json(),
        "model": {
            "detector": f"InsightFace({config.model_name}) detector",
            "embedder": f"InsightFace({config.model_name}) recognizer",
        },
        "summary": {
            "total_images": len(image_paths),
            "total_faces": len(face_records),
            "total_clusters": len([cluster for cluster in clusters if cluster["cluster_id"] != -1]),
            "noise_faces": cluster_face_index.get(-1, 0),
            "problem_images": len([record for record in image_records if record["status"] != "ok"]),
        },
        "corrections": {
            "db_path": str(config.corrections_db),
            "applied": bool(label_map or correction_warnings),
            "warnings": correction_warnings,
        },
        "input_dir": str(config.input_dir),
    }

    np.save(config.output_dir / "embeddings.npy", embedding_array)
    write_json(config.output_dir / "run.json", run_payload)
    write_json(config.output_dir / "images.json", image_records)
    write_json(config.output_dir / "faces.json", face_records)
    write_json(config.output_dir / "clusters.json", clusters)
    write_html_report(
        path=config.output_dir / "report.html",
        run_payload=run_payload,
        image_records=image_records,
        cluster_payloads=clusters,
        face_records=face_records,
    )
    return config.output_dir / "report.html"


def recluster_run(
    input_run_dir: Path,
    output_dir: Path,
    corrections_db: Path,
    clusterer: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    agglomerative_distance_threshold: float,
) -> Path:
    input_run_dir = input_run_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_payload, image_records, face_records, embeddings = load_run_artifacts(input_run_dir)
    ensure_face_keys(face_records)

    labels = cluster_embeddings(
        embeddings=embeddings,
        method=clusterer,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        agglomerative_distance_threshold=agglomerative_distance_threshold,
    )
    labels, label_map, correction_warnings = apply_saved_corrections(
        face_records=face_records,
        labels=labels,
        db_path=corrections_db,
    )

    for record, label in zip(face_records, labels):
        record["cluster_id"] = int(label)
        original_thumbnail = input_run_dir / str(record["thumbnail_path"])
        record["thumbnail_relpath"] = os.path.relpath(original_thumbnail, output_dir)
        record["person_label"] = label_map.get(str(record["face_key"]))

    clusters = summarize_clusters(face_records)
    cluster_face_index = Counter(int(record["cluster_id"]) for record in face_records)
    updated_config = dict(run_payload["config"])
    updated_config["clusterer"] = clusterer
    updated_config["dbscan_eps"] = dbscan_eps
    updated_config["dbscan_min_samples"] = dbscan_min_samples
    updated_config["agglomerative_distance_threshold"] = agglomerative_distance_threshold
    updated_config["output_dir"] = str(output_dir)
    updated_config["corrections_db"] = str(corrections_db)
    run_payload["config"] = updated_config
    run_payload["summary"] = {
        "total_images": len(image_records),
        "total_faces": len(face_records),
        "total_clusters": len([cluster for cluster in clusters if cluster["cluster_id"] != -1]),
        "noise_faces": cluster_face_index.get(-1, 0),
        "problem_images": len([record for record in image_records if record["status"] != "ok"]),
    }
    run_payload["corrections"] = {
        "db_path": str(corrections_db),
        "applied": bool(label_map or correction_warnings),
        "warnings": correction_warnings,
    }
    run_payload["source_run_dir"] = str(input_run_dir)

    np.save(output_dir / "embeddings.npy", embeddings)
    write_json(output_dir / "run.json", run_payload)
    write_json(output_dir / "images.json", image_records)
    write_json(output_dir / "faces.json", face_records)
    write_json(output_dir / "clusters.json", clusters)
    write_html_report(
        path=output_dir / "report.html",
        run_payload=run_payload,
        image_records=image_records,
        cluster_payloads=clusters,
        face_records=face_records,
    )
    return output_dir / "report.html"
