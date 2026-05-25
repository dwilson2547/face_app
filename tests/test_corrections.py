from pathlib import Path

import numpy as np

from face_poc.corrections import apply_saved_corrections
from face_poc.corrections_db import add_pair_constraint, set_face_label


def _face(face_key: str, face_id: str) -> dict[str, object]:
    return {
        "face_key": face_key,
        "face_id": face_id,
        "image_id": face_id.split("-face-")[0],
        "image_path": f"{face_id}.jpg",
        "box": [0.0, 0.0, 1.0, 1.0],
    }


def test_must_link_merges_cluster_labels(tmp_path: Path) -> None:
    db_path = tmp_path / "corrections.sqlite3"
    add_pair_constraint(db_path, "face-a", "face-b", "must_link", "test")
    labels = np.array([0, 1], dtype=int)
    corrected, label_map, warnings = apply_saved_corrections(
        face_records=[_face("face-a", "img-0000-face-00"), _face("face-b", "img-0001-face-00")],
        labels=labels,
        db_path=db_path,
    )
    assert list(corrected) == [0, 0]
    assert label_map == {}
    assert warnings == []


def test_cannot_link_splits_labels_and_applies_person_label(tmp_path: Path) -> None:
    db_path = tmp_path / "corrections.sqlite3"
    add_pair_constraint(db_path, "face-a", "face-b", "cannot_link", "test")
    set_face_label(db_path, "face-a", "Alice", "test")
    labels = np.array([3, 3], dtype=int)
    corrected, label_map, warnings = apply_saved_corrections(
        face_records=[_face("face-a", "img-0000-face-00"), _face("face-b", "img-0001-face-00")],
        labels=labels,
        db_path=db_path,
    )
    assert corrected[0] != corrected[1]
    assert label_map["face-a"] == "Alice"
    assert warnings == []
