from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from face_poc.corrections import resolve_cluster_faces, resolve_face_key
from face_poc.corrections_db import add_pair_constraint, record_action, set_face_label
from face_poc.pipeline import recluster_run
from face_poc.review import (
    REPO_ROOT,
    build_recluster_output_dir,
    discover_runs,
    load_run_review,
    reports_root,
    resolve_input_image,
    resolve_run_artifact,
    resolve_run_dir,
    resolve_within,
)


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_app() -> FastAPI:
    app = FastAPI(title="face_app review workstation")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def redirect_to(request: Request, route_name: str, **path_params: object) -> RedirectResponse:
        url = request.url_for(route_name, **path_params)
        return RedirectResponse(url=url, status_code=303)

    @app.get("/", response_class=HTMLResponse, name="runs_index")
    def runs_index(request: Request) -> HTMLResponse:
        runs = discover_runs()
        return templates.TemplateResponse(
            request,
            "runs_index.html",
            {
                "runs": runs,
                "active_nav": "runs",
            },
        )

    @app.get("/runs/{run_name}", response_class=HTMLResponse, name="run_detail")
    def run_detail(request: Request, run_name: str) -> HTMLResponse:
        try:
            review = load_run_review(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        top_clusters = review["clusters"][:12]
        recent_images = review["images"][:18]
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                **review,
                "top_clusters": top_clusters,
                "recent_images": recent_images,
                "active_nav": "runs",
            },
        )

    @app.get("/runs/{run_name}/clusters", response_class=HTMLResponse, name="cluster_index")
    def cluster_index(request: Request, run_name: str) -> HTMLResponse:
        review = load_run_review(run_name)
        return templates.TemplateResponse(
            request,
            "cluster_index.html",
            {
                **review,
                "active_nav": "clusters",
            },
        )

    @app.get("/runs/{run_name}/clusters/{cluster_id}", response_class=HTMLResponse, name="cluster_detail")
    def cluster_detail(request: Request, run_name: str, cluster_id: int) -> HTMLResponse:
        review = load_run_review(run_name)
        cluster = review["cluster_by_id"].get(cluster_id)
        if cluster is None:
            raise HTTPException(status_code=404, detail=f"Unknown cluster: {cluster_id}")
        merge_targets = [item for item in review["clusters"] if int(item["cluster_id"]) not in {-1, cluster_id}]
        return templates.TemplateResponse(
            request,
            "cluster_detail.html",
            {
                **review,
                "cluster": cluster,
                "merge_targets": merge_targets,
                "active_nav": "clusters",
            },
        )

    @app.get("/runs/{run_name}/images/{image_id}", response_class=HTMLResponse, name="image_detail")
    def image_detail(request: Request, run_name: str, image_id: str) -> HTMLResponse:
        review = load_run_review(run_name)
        image = review["image_by_id"].get(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail=f"Unknown image: {image_id}")
        return templates.TemplateResponse(
            request,
            "image_detail.html",
            {
                **review,
                "image": image,
                "active_nav": "images",
            },
        )

    @app.get("/runs/{run_name}/artifacts/{artifact_path:path}", name="run_artifact")
    def run_artifact(run_name: str, artifact_path: str) -> FileResponse:
        try:
            path = resolve_run_artifact(run_name, artifact_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")
        return FileResponse(path)

    @app.get("/runs/{run_name}/inputs/{image_path:path}", name="run_input_image")
    def run_input_image(run_name: str, image_path: str) -> FileResponse:
        try:
            path = resolve_input_image(run_name, image_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Input image not found: {image_path}")
        return FileResponse(path)

    @app.post("/runs/{run_name}/clusters/{cluster_id}/label", name="label_cluster")
    def label_cluster(
        request: Request,
        run_name: str,
        cluster_id: int,
        label: str = Form(...),
        note: str | None = Form(default=None),
    ) -> RedirectResponse:
        run_dir = resolve_run_dir(run_name)
        review = load_run_review(run_name)
        faces = resolve_cluster_faces(run_dir, cluster_id)
        for face in faces:
            set_face_label(review["corrections_db"], str(face["face_key"]), label, "ui_label_cluster", note)
        record_action(
            review["corrections_db"],
            "ui_label_cluster",
            {
                "run_name": run_name,
                "cluster_id": cluster_id,
                "label": label,
                "faces": [str(face["face_key"]) for face in faces],
                "note": note,
            },
        )
        return redirect_to(request, "cluster_detail", run_name=run_name, cluster_id=cluster_id)

    @app.post("/runs/{run_name}/clusters/{cluster_id}/merge", name="merge_cluster")
    def merge_cluster(
        request: Request,
        run_name: str,
        cluster_id: int,
        target_cluster_id: int = Form(...),
        note: str | None = Form(default=None),
    ) -> RedirectResponse:
        if target_cluster_id == cluster_id:
            raise HTTPException(status_code=400, detail="Target cluster must be different from the source cluster.")
        run_dir = resolve_run_dir(run_name)
        review = load_run_review(run_name)
        faces_a = resolve_cluster_faces(run_dir, cluster_id)
        faces_b = resolve_cluster_faces(run_dir, target_cluster_id)
        for face_a in faces_a:
            for face_b in faces_b:
                add_pair_constraint(
                    review["corrections_db"],
                    str(face_a["face_key"]),
                    str(face_b["face_key"]),
                    "must_link",
                    "ui_merge_clusters",
                    note,
                )
        record_action(
            review["corrections_db"],
            "ui_merge_clusters",
            {
                "run_name": run_name,
                "cluster_a": cluster_id,
                "cluster_b": target_cluster_id,
                "faces_a": [str(face["face_key"]) for face in faces_a],
                "faces_b": [str(face["face_key"]) for face in faces_b],
                "note": note,
            },
        )
        return redirect_to(request, "cluster_detail", run_name=run_name, cluster_id=cluster_id)

    @app.post("/runs/{run_name}/clusters/{cluster_id}/split", name="split_cluster")
    def split_cluster(
        request: Request,
        run_name: str,
        cluster_id: int,
        face_a: str = Form(...),
        face_b: str = Form(...),
        note: str | None = Form(default=None),
    ) -> RedirectResponse:
        if face_a == face_b:
            raise HTTPException(status_code=400, detail="Choose two different faces to split.")
        run_dir = resolve_run_dir(run_name)
        review = load_run_review(run_name)
        left = resolve_face_key(run_dir, face_a)
        right = resolve_face_key(run_dir, face_b)
        add_pair_constraint(
            review["corrections_db"],
            left,
            right,
            "cannot_link",
            "ui_split_cluster",
            note,
        )
        record_action(
            review["corrections_db"],
            "ui_split_cluster",
            {
                "run_name": run_name,
                "cluster_id": cluster_id,
                "face_a": left,
                "face_b": right,
                "note": note,
            },
        )
        return redirect_to(request, "cluster_detail", run_name=run_name, cluster_id=cluster_id)

    @app.post("/runs/{run_name}/faces/{face_id}/label", name="label_face")
    def label_face(
        request: Request,
        run_name: str,
        face_id: str,
        label: str = Form(...),
        note: str | None = Form(default=None),
        next_url: str | None = Form(default=None),
    ) -> RedirectResponse:
        run_dir = resolve_run_dir(run_name)
        review = load_run_review(run_name)
        face_key = resolve_face_key(run_dir, face_id)
        set_face_label(review["corrections_db"], face_key, label, "ui_label_face", note)
        record_action(
            review["corrections_db"],
            "ui_label_face",
            {
                "run_name": run_name,
                "face_id": face_id,
                "face_key": face_key,
                "label": label,
                "note": note,
            },
        )
        if next_url:
            return RedirectResponse(url=next_url, status_code=303)
        return redirect_to(request, "run_detail", run_name=run_name)

    @app.post("/runs/{run_name}/recluster", name="recluster_from_ui")
    def recluster_from_ui(
        request: Request,
        run_name: str,
        clusterer: str = Form(...),
        agglomerative_distance_threshold: float = Form(0.6),
        dbscan_eps: float = Form(0.35),
        dbscan_min_samples: int = Form(1),
        output_name: str | None = Form(default=None),
    ) -> RedirectResponse:
        run_dir = resolve_run_dir(run_name)
        output_dir = (
            resolve_within(reports_root(), reports_root() / output_name.strip())
            if output_name and output_name.strip()
            else build_recluster_output_dir(run_name, agglomerative_distance_threshold)
        )
        if output_dir.exists():
            raise HTTPException(status_code=400, detail=f"Output directory already exists: {output_dir.name}")
        recluster_run(
            input_run_dir=run_dir,
            output_dir=output_dir,
            corrections_db=load_run_review(run_name)["corrections_db"],
            clusterer=clusterer,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            agglomerative_distance_threshold=agglomerative_distance_threshold,
        )
        return redirect_to(request, "run_detail", run_name=output_dir.name)

    return app
