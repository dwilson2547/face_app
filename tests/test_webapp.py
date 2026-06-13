from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from face_poc.corrections_db import summary
from face_poc.webapp import create_app


def create_sample_run(tmp_path: Path) -> tuple[Path, Path]:
    reports_dir = tmp_path / "reports"
    run_dir = reports_dir / "sample-run"
    thumbs_dir = run_dir / "thumbnails"
    input_dir = tmp_path / "sample_photos"
    thumbs_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    (input_dir / "one.jpg").write_bytes(b"fake-image")
    (input_dir / "two.jpg").write_bytes(b"fake-image")
    (thumbs_dir / "img-0000-face-00.jpg").write_bytes(b"thumb-a")
    (thumbs_dir / "img-0001-face-00.jpg").write_bytes(b"thumb-b")
    (run_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")

    run_payload = {
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(run_dir),
            "corrections_db": str(tmp_path / "data" / "corrections.sqlite3"),
            "clusterer": "agglomerative",
            "dbscan_eps": 0.35,
            "dbscan_min_samples": 1,
            "agglomerative_distance_threshold": 0.6,
        },
        "summary": {
            "total_images": 2,
            "total_faces": 2,
            "total_clusters": 1,
            "problem_images": 0,
        },
        "input_dir": str(input_dir),
    }
    images = [
        {"image_id": "img-0000", "path": "one.jpg", "status": "ok", "error": None, "face_count": 1},
        {"image_id": "img-0001", "path": "two.jpg", "status": "ok", "error": None, "face_count": 1},
    ]
    faces = [
        {
            "face_id": "img-0000-face-00",
            "face_key": "one.jpg::0,0,1,1",
            "image_id": "img-0000",
            "image_path": "one.jpg",
            "box": [0.0, 0.0, 1.0, 1.0],
            "confidence": 0.99,
            "thumbnail_path": "thumbnails/img-0000-face-00.jpg",
            "cluster_id": 0,
        },
        {
            "face_id": "img-0001-face-00",
            "face_key": "two.jpg::0,0,1,1",
            "image_id": "img-0001",
            "image_path": "two.jpg",
            "box": [0.0, 0.0, 1.0, 1.0],
            "confidence": 0.98,
            "thumbnail_path": "thumbnails/img-0001-face-00.jpg",
            "cluster_id": 0,
        },
    ]
    clusters = [
        {
            "cluster_id": 0,
            "size": 2,
            "image_count": 2,
            "image_paths": ["one.jpg", "two.jpg"],
            "face_ids": ["img-0000-face-00", "img-0001-face-00"],
            "face_keys": ["one.jpg::0,0,1,1", "two.jpg::0,0,1,1"],
            "thumbnail_paths": ["thumbnails/img-0000-face-00.jpg", "thumbnails/img-0001-face-00.jpg"],
            "person_labels": [],
        }
    ]

    (run_dir / "run.json").write_text(json.dumps(run_payload), encoding="utf-8")
    (run_dir / "images.json").write_text(json.dumps(images), encoding="utf-8")
    (run_dir / "faces.json").write_text(json.dumps(faces), encoding="utf-8")
    (run_dir / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    return reports_dir, tmp_path


def test_runs_index_lists_available_runs(tmp_path: Path, monkeypatch) -> None:
    reports_dir, repo_root = create_sample_run(tmp_path)
    import face_poc.review as review
    import face_poc.webapp as webapp

    monkeypatch.setattr(review, "REPORTS_ROOT", reports_dir)
    monkeypatch.setattr(review, "REPO_ROOT", repo_root)
    monkeypatch.setattr(webapp, "REPO_ROOT", repo_root)

    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "sample-run" in response.text
    assert "2" in response.text


def test_label_cluster_persists_labels(tmp_path: Path, monkeypatch) -> None:
    reports_dir, repo_root = create_sample_run(tmp_path)
    import face_poc.review as review
    import face_poc.webapp as webapp

    monkeypatch.setattr(review, "REPORTS_ROOT", reports_dir)
    monkeypatch.setattr(review, "REPO_ROOT", repo_root)
    monkeypatch.setattr(webapp, "REPO_ROOT", repo_root)

    client = TestClient(create_app())
    response = client.post(
        "/runs/sample-run/clusters/0/label",
        data={"label": "Alice", "note": "verified by review"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].endswith("/runs/sample-run/clusters/0")
    assert summary(repo_root / "data" / "corrections.sqlite3")["labels"] == 2


def test_recluster_endpoint_redirects_to_named_output(tmp_path: Path, monkeypatch) -> None:
    reports_dir, repo_root = create_sample_run(tmp_path)
    import face_poc.review as review
    import face_poc.webapp as webapp

    monkeypatch.setattr(review, "REPORTS_ROOT", reports_dir)
    monkeypatch.setattr(review, "REPO_ROOT", repo_root)
    monkeypatch.setattr(webapp, "REPO_ROOT", repo_root)

    captured: dict[str, Path] = {}

    def fake_recluster_run(**kwargs):
        captured["output_dir"] = kwargs["output_dir"]
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"] / "report.html"

    monkeypatch.setattr(webapp, "recluster_run", fake_recluster_run)

    client = TestClient(create_app())
    response = client.post(
        "/runs/sample-run/recluster",
        data={
            "clusterer": "agglomerative",
            "agglomerative_distance_threshold": "0.55",
            "dbscan_eps": "0.35",
            "dbscan_min_samples": "1",
            "output_name": "manual-recluster",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].endswith("/runs/manual-recluster")
    assert captured["output_dir"] == repo_root / "reports" / "manual-recluster"
