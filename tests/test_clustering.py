import numpy as np

from face_poc.clustering import cluster_embeddings, summarize_clusters


def test_dbscan_clusters_close_points() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = cluster_embeddings(
        embeddings=embeddings,
        method="dbscan",
        dbscan_eps=0.05,
        dbscan_min_samples=1,
        agglomerative_distance_threshold=0.4,
    )
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_summarize_clusters_groups_members() -> None:
    face_records = [
        {
            "face_id": "a",
            "cluster_id": 0,
            "image_path": "one.jpg",
            "thumbnail_path": "thumbs/a.jpg",
        },
        {
            "face_id": "b",
            "cluster_id": 0,
            "image_path": "two.jpg",
            "thumbnail_path": "thumbs/b.jpg",
        },
        {
            "face_id": "c",
            "cluster_id": -1,
            "image_path": "three.jpg",
            "thumbnail_path": "thumbs/c.jpg",
        },
    ]
    summary = summarize_clusters(face_records)
    assert summary[0]["cluster_id"] == 0
    assert summary[0]["size"] == 2
    assert summary[1]["cluster_id"] == -1
