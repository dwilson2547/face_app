from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
    agglomerative_distance_threshold: float,
) -> np.ndarray:
    if embeddings.size == 0:
        return np.empty((0,), dtype=int)
    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    if method == "dbscan":
        model = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            metric="cosine",
        )
        return model.fit_predict(embeddings)

    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=agglomerative_distance_threshold,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=agglomerative_distance_threshold,
            affinity="cosine",
            linkage="average",
        )
    return model.fit_predict(embeddings)


def summarize_clusters(face_records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in face_records:
        cluster_id = int(record["cluster_id"])
        grouped[cluster_id].append(record)

    summaries: list[dict[str, object]] = []
    for cluster_id, members in sorted(grouped.items(), key=lambda item: (item[0] == -1, item[0])):
        image_paths = sorted({str(member["image_path"]) for member in members})
        labels = sorted({str(member["person_label"]) for member in members if member.get("person_label")})
        summaries.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "image_count": len(image_paths),
                "image_paths": image_paths,
                "face_ids": [str(member["face_id"]) for member in members],
                "face_keys": [str(member.get("face_key", member["face_id"])) for member in members],
                "thumbnail_paths": [str(member["thumbnail_path"]) for member in members],
                "person_labels": labels,
            }
        )
    return summaries
