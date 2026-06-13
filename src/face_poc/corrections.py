from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from face_poc.corrections_db import load_constraints_for_faces, load_labels_for_faces
from face_poc.identity import make_face_key


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def ensure_face_keys(face_records: list[dict[str, object]]) -> None:
    for record in face_records:
        if "face_key" not in record:
            record["face_key"] = make_face_key(str(record["image_path"]), list(record["box"]))


def load_run_json_artifacts(run_dir: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    run_payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    image_records = json.loads((run_dir / "images.json").read_text(encoding="utf-8"))
    face_records = json.loads((run_dir / "faces.json").read_text(encoding="utf-8"))
    ensure_face_keys(face_records)
    return run_payload, image_records, face_records


def load_run_artifacts(run_dir: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], np.ndarray]:
    run_payload, image_records, face_records = load_run_json_artifacts(run_dir)
    embeddings = np.load(run_dir / "embeddings.npy")
    return run_payload, image_records, face_records, embeddings


def resolve_face_key(run_dir: Path, face_identifier: str) -> str:
    _, _, face_records = load_run_json_artifacts(run_dir)
    for record in face_records:
        if record["face_id"] == face_identifier or record["face_key"] == face_identifier:
            return str(record["face_key"])
    raise ValueError(f"Could not resolve face identifier: {face_identifier}")


def resolve_cluster_faces(run_dir: Path, cluster_id: int) -> list[dict[str, object]]:
    _, _, face_records = load_run_json_artifacts(run_dir)
    return [record for record in face_records if int(record["cluster_id"]) == cluster_id]


def apply_saved_corrections(
    face_records: list[dict[str, object]],
    labels: np.ndarray,
    db_path: Path,
) -> tuple[np.ndarray, dict[str, str], list[str]]:
    ensure_face_keys(face_records)
    face_keys = {str(record["face_key"]) for record in face_records}
    must_links, cannot_links = load_constraints_for_faces(db_path, face_keys)
    label_map = load_labels_for_faces(db_path, face_keys)
    if not must_links and not cannot_links and not label_map:
        return labels, label_map, []

    index_by_key = {str(record["face_key"]): index for index, record in enumerate(face_records)}
    dsu = DisjointSet(len(face_records))
    for left, right in must_links:
        if left in index_by_key and right in index_by_key:
            dsu.union(index_by_key[left], index_by_key[right])

    corrected = labels.copy()
    root_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(face_records)):
        root_members[dsu.find(index)].append(index)

    for members in root_members.values():
        member_labels = [int(corrected[index]) for index in members if int(corrected[index]) != -1]
        target_label = min(member_labels) if member_labels else min(int(corrected[index]) for index in members)
        for index in members:
            corrected[index] = target_label

    next_label = int(corrected.max()) + 1 if len(corrected) else 0
    warnings: list[str] = []
    assigned_component_labels: dict[int, int] = {}

    for left, right in cannot_links:
        if left not in index_by_key or right not in index_by_key:
            continue
        left_index = index_by_key[left]
        right_index = index_by_key[right]
        left_root = dsu.find(left_index)
        right_root = dsu.find(right_index)
        if left_root == right_root:
            warnings.append(f"Conflicting correction: {left} and {right} are both must-link and cannot-link")
            continue
        if int(corrected[left_index]) != int(corrected[right_index]):
            continue
        if right_root not in assigned_component_labels:
            assigned_component_labels[right_root] = next_label
            next_label += 1
        for member_index in root_members[right_root]:
            corrected[member_index] = assigned_component_labels[right_root]

    return corrected, label_map, warnings
