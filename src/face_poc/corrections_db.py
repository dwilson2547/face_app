from __future__ import annotations

import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_key_a TEXT NOT NULL,
    face_key_b TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('must_link', 'cannot_link')),
    source TEXT,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(face_key_a, face_key_b, relation)
);

CREATE TABLE IF NOT EXISTS face_labels (
    face_key TEXT PRIMARY KEY,
    person_label TEXT NOT NULL,
    source TEXT,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def default_db_path() -> Path:
    return Path("data/corrections.sqlite3")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def normalize_pair(face_key_a: str, face_key_b: str) -> tuple[str, str]:
    return tuple(sorted((face_key_a, face_key_b)))


def add_pair_constraint(
    db_path: Path,
    face_key_a: str,
    face_key_b: str,
    relation: str,
    source: str,
    note: str | None = None,
) -> None:
    left, right = normalize_pair(face_key_a, face_key_b)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO pair_constraints (face_key_a, face_key_b, relation, source, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (left, right, relation, source, note),
        )


def set_face_label(
    db_path: Path,
    face_key: str,
    person_label: str,
    source: str,
    note: str | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO face_labels (face_key, person_label, source, note)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(face_key) DO UPDATE SET
              person_label = excluded.person_label,
              source = excluded.source,
              note = excluded.note,
              updated_at = CURRENT_TIMESTAMP
            """,
            (face_key, person_label, source, note),
        )


def record_action(db_path: Path, action_type: str, payload: dict[str, object]) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO action_log (action_type, payload_json) VALUES (?, ?)",
            (action_type, json.dumps(payload, sort_keys=True)),
        )


def load_constraints_for_faces(
    db_path: Path,
    face_keys: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not db_path.exists():
        return [], []
    placeholders = ",".join("?" for _ in face_keys)
    query = f"""
        SELECT face_key_a, face_key_b, relation
        FROM pair_constraints
        WHERE face_key_a IN ({placeholders}) AND face_key_b IN ({placeholders})
    """
    params = list(face_keys) + list(face_keys)
    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    must_links: list[tuple[str, str]] = []
    cannot_links: list[tuple[str, str]] = []
    for left, right, relation in rows:
        if relation == "must_link":
            must_links.append((left, right))
        else:
            cannot_links.append((left, right))
    return must_links, cannot_links


def load_labels_for_faces(db_path: Path, face_keys: set[str]) -> dict[str, str]:
    if not db_path.exists() or not face_keys:
        return {}
    placeholders = ",".join("?" for _ in face_keys)
    query = f"SELECT face_key, person_label FROM face_labels WHERE face_key IN ({placeholders})"
    with connect(db_path) as conn:
        rows = conn.execute(query, list(face_keys)).fetchall()
    return {face_key: label for face_key, label in rows}


def summary(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {"must_link": 0, "cannot_link": 0, "labels": 0, "actions": 0}
    with connect(db_path) as conn:
        must_link = conn.execute(
            "SELECT COUNT(*) FROM pair_constraints WHERE relation = 'must_link'"
        ).fetchone()[0]
        cannot_link = conn.execute(
            "SELECT COUNT(*) FROM pair_constraints WHERE relation = 'cannot_link'"
        ).fetchone()[0]
        labels = conn.execute("SELECT COUNT(*) FROM face_labels").fetchone()[0]
        actions = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]
    return {
        "must_link": int(must_link),
        "cannot_link": int(cannot_link),
        "labels": int(labels),
        "actions": int(actions),
    }
