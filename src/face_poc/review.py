from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from face_poc.corrections import load_run_json_artifacts
from face_poc.corrections_db import default_db_path, summary as corrections_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = REPO_ROOT / "reports"


def resolve_within(base_dir: Path, candidate: Path) -> Path:
    base = base_dir.resolve()
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"Resolved path escapes base directory: {candidate}")
    return resolved


def reports_root() -> Path:
    return REPORTS_ROOT


def resolve_run_dir(run_name: str) -> Path:
    run_dir = resolve_within(reports_root(), reports_root() / run_name)
    if not (run_dir / "run.json").exists():
        raise FileNotFoundError(f"Unknown run: {run_name}")
    return run_dir


def resolve_corrections_db(run_payload: dict[str, object]) -> Path:
    config = dict(run_payload.get("config", {}))
    configured = config.get("corrections_db")
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else (REPO_ROOT / path)
    return REPO_ROOT / default_db_path()


def discover_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for run_json in reports_root().glob("*/run.json"):
        run_dir = run_json.parent
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        runs.append(
            {
                "name": run_dir.name,
                "path": run_dir,
                "payload": payload,
                "summary": dict(payload.get("summary", {})),
                "modified_at": datetime.fromtimestamp(run_json.stat().st_mtime),
                "has_report": (run_dir / "report.html").exists(),
            }
        )
    runs.sort(key=lambda item: item["modified_at"], reverse=True)
    return runs


def load_run_review(run_name: str) -> dict[str, object]:
    run_dir = resolve_run_dir(run_name)
    run_payload, image_records, face_records = load_run_json_artifacts(run_dir)
    clusters = json.loads((run_dir / "clusters.json").read_text(encoding="utf-8"))

    face_by_id = {str(face["face_id"]): dict(face) for face in face_records}
    for face in face_by_id.values():
        face["thumbnail_artifact_path"] = str(face["thumbnail_path"])

    cluster_views: list[dict[str, object]] = []
    for cluster in clusters:
        faces = [face_by_id[str(face_id)] for face_id in cluster["face_ids"] if str(face_id) in face_by_id]
        cluster_views.append(
            {
                **cluster,
                "faces": faces,
                "preview_faces": faces[:8],
            }
        )
    cluster_views.sort(key=lambda item: (item["cluster_id"] == -1, -int(item["size"]), int(item["cluster_id"])))

    image_faces: dict[str, list[dict[str, object]]] = {}
    for face in face_by_id.values():
        image_faces.setdefault(str(face["image_id"]), []).append(face)

    image_views: list[dict[str, object]] = []
    for image in image_records:
        faces = sorted(
            image_faces.get(str(image["image_id"]), []),
            key=lambda face: int(face["cluster_id"]),
        )
        image_views.append({**image, "faces": faces})

    image_views.sort(key=lambda item: str(item["path"]))
    cluster_by_id = {int(cluster["cluster_id"]): cluster for cluster in cluster_views}
    image_by_id = {str(image["image_id"]): image for image in image_views}

    corrections_db = resolve_corrections_db(run_payload)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "run_payload": run_payload,
        "clusters": cluster_views,
        "cluster_by_id": cluster_by_id,
        "images": image_views,
        "image_by_id": image_by_id,
        "face_by_id": face_by_id,
        "problem_images": [image for image in image_views if image["status"] != "ok"],
        "corrections_db": corrections_db,
        "corrections_summary": corrections_summary(corrections_db),
    }


def resolve_run_artifact(run_name: str, artifact_path: str) -> Path:
    run_dir = resolve_run_dir(run_name)
    return resolve_within(run_dir, run_dir / artifact_path)


def resolve_input_image(run_name: str, image_path: str) -> Path:
    run_dir = resolve_run_dir(run_name)
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    input_dir = Path(str(run_payload.get("input_dir", "")))
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    return resolve_within(input_dir, input_dir / image_path)


def build_recluster_output_dir(run_name: str, threshold: float) -> Path:
    slug_threshold = str(threshold).replace(".", "")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return reports_root() / f"{run_name}-recluster-{slug_threshold}-{stamp}"
