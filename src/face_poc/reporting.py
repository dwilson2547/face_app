from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template


REPORT_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face POC Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #111827; color: #f3f4f6; }
    h1, h2, h3 { margin-bottom: 0.4rem; }
    a { color: #93c5fd; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .card { background: #1f2937; border-radius: 10px; padding: 12px 16px; }
    .cluster { margin: 24px 0; padding: 16px; background: #1f2937; border-radius: 10px; }
    .faces { display: flex; flex-wrap: wrap; gap: 12px; }
    .face { width: 180px; background: #111827; border: 1px solid #374151; border-radius: 8px; overflow: hidden; }
    .face img { width: 100%; display: block; background: #000; }
    .face .meta { padding: 8px; font-size: 0.85rem; }
    .problem-list li { margin-bottom: 6px; }
    .image-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .image-card { background: #1f2937; border-radius: 10px; padding: 12px 16px; }
    pre { white-space: pre-wrap; background: #0f172a; padding: 12px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>Face-grouping POC report</h1>
  <p>Generated from <strong>{{ run.input_dir }}</strong> using <strong>{{ run.model.detector }}</strong> and <strong>{{ run.model.embedder }}</strong>.</p>

  <div class="summary">
    <div class="card"><h3>Images</h3><div>{{ run.summary.total_images }}</div></div>
    <div class="card"><h3>Faces</h3><div>{{ run.summary.total_faces }}</div></div>
    <div class="card"><h3>Clusters</h3><div>{{ run.summary.total_clusters }}</div></div>
    <div class="card"><h3>Problem files</h3><div>{{ run.summary.problem_images }}</div></div>
  </div>

  <h2>Run configuration</h2>
  <pre>{{ config_json }}</pre>

  <h2>Problem files</h2>
  {% if problem_images %}
  <ul class="problem-list">
    {% for image in problem_images %}
    <li><strong>{{ image.path }}</strong> — {{ image.status }}{% if image.error %}: {{ image.error }}{% endif %}</li>
    {% endfor %}
  </ul>
  {% else %}
  <p>No problem files recorded.</p>
  {% endif %}

  <h2>Images</h2>
  <div class="image-list">
    {% for image in images %}
    <div class="image-card">
      <h3>{{ image.path }}</h3>
      <p>status={{ image.status }} | faces={{ image.face_count }}</p>
      {% if image.error %}
      <p>{{ image.error }}</p>
      {% endif %}
      {% if image.faces %}
      <div class="faces">
        {% for face in image.faces %}
        <div class="face">
          <img src="{{ face.thumbnail_relpath }}" alt="{{ face.face_id }}">
          <div class="meta">
            <div><strong>{{ face.face_id }}</strong></div>
            <div>cluster={{ face.cluster_id }}</div>
            <div>confidence={{ "%.3f"|format(face.confidence) }}</div>
          </div>
        </div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
    {% endfor %}
  </div>

  <h2>Clusters</h2>
  {% for cluster in clusters %}
  <div class="cluster">
    <h3>Cluster {{ cluster.cluster_id }} {% if cluster.cluster_id == -1 %}(noise/unassigned){% endif %}</h3>
    <p>{{ cluster.size }} faces across {{ cluster.image_count }} images</p>
    <div class="faces">
      {% for face in cluster.faces %}
      <div class="face">
        <img src="{{ face.thumbnail_relpath }}" alt="{{ face.face_id }}">
        <div class="meta">
          <div><strong>{{ face.face_id }}</strong></div>
          <div>{{ face.image_path }}</div>
          <div>confidence={{ "%.3f"|format(face.confidence) }}</div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endfor %}
</body>
</html>
"""
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_html_report(
    path: Path,
    run_payload: dict[str, object],
    image_records: list[dict[str, object]],
    cluster_payloads: list[dict[str, object]],
    face_records: list[dict[str, object]],
) -> None:
    face_index = {str(face["face_id"]): face for face in face_records}
    clusters_for_render = []
    for cluster in cluster_payloads:
        faces = []
        for face_id in cluster["face_ids"]:
            face = dict(face_index[str(face_id)])
            faces.append(face)
        clusters_for_render.append({**cluster, "faces": faces})

    image_faces: dict[str, list[dict[str, object]]] = {}
    for face in face_records:
        image_faces.setdefault(str(face["image_id"]), []).append(dict(face))
    images_for_render = []
    for image in image_records:
        images_for_render.append({**image, "faces": image_faces.get(str(image["image_id"]), [])})

    problem_images = [image for image in image_records if image["status"] != "ok"]
    html = REPORT_TEMPLATE.render(
        run=run_payload,
        config_json=json.dumps(run_payload["config"], indent=2),
        problem_images=problem_images,
        images=images_for_render,
        clusters=clusters_for_render,
    )
    path.write_text(html, encoding="utf-8")
