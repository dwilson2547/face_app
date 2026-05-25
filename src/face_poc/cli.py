from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort

from face_poc import __version__
from face_poc.config import RunConfig
from face_poc.corrections import resolve_cluster_faces, resolve_face_key
from face_poc.corrections_db import (
    add_pair_constraint,
    default_db_path,
    record_action,
    set_face_label,
    summary as corrections_summary,
)
from face_poc.pipeline import recluster_run, run_pipeline


def default_device() -> str:
    providers = set(ort.get_available_providers())
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="face-poc", description="Local face-grouping POC")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the face-grouping pipeline")
    run_parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing images")
    run_parser.add_argument("--output-dir", required=True, type=Path, help="Directory for report artifacts")
    run_parser.add_argument(
        "--corrections-db",
        type=Path,
        default=default_db_path(),
        help="SQLite database for saved corrections",
    )
    run_parser.add_argument(
        "--device",
        default=default_device(),
        choices=("cuda", "cpu"),
        help="Execution target for InsightFace",
    )
    run_parser.add_argument(
        "--model-name",
        default="buffalo_l",
        help="InsightFace model pack name",
    )
    run_parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Square detector input size for InsightFace",
    )
    run_parser.add_argument(
        "--clusterer",
        choices=("agglomerative", "dbscan"),
        default="agglomerative",
        help="Clustering algorithm for face embeddings",
    )
    run_parser.add_argument(
        "--detector-threshold",
        type=float,
        default=0.6,
        help="Minimum confidence required to keep a detected face",
    )
    run_parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.35,
        help="DBSCAN eps when --clusterer=dbscan",
    )
    run_parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=1,
        help="DBSCAN min_samples when --clusterer=dbscan",
    )
    run_parser.add_argument(
        "--agglomerative-distance-threshold",
        type=float,
        default=0.4,
        help="Cosine distance threshold when --clusterer=agglomerative",
    )
    run_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subdirectories for images",
    )

    recluster_parser = subparsers.add_parser("recluster", help="Recluster an existing run without rerunning detection")
    recluster_parser.add_argument("--input-run-dir", required=True, type=Path, help="Existing run directory")
    recluster_parser.add_argument("--output-dir", required=True, type=Path, help="Directory for reclustered artifacts")
    recluster_parser.add_argument(
        "--corrections-db",
        type=Path,
        default=default_db_path(),
        help="SQLite database for saved corrections",
    )
    recluster_parser.add_argument(
        "--clusterer",
        choices=("agglomerative", "dbscan"),
        default="agglomerative",
        help="Clustering algorithm for face embeddings",
    )
    recluster_parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.35,
        help="DBSCAN eps when --clusterer=dbscan",
    )
    recluster_parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=1,
        help="DBSCAN min_samples when --clusterer=dbscan",
    )
    recluster_parser.add_argument(
        "--agglomerative-distance-threshold",
        type=float,
        default=0.4,
        help="Cosine distance threshold when --clusterer=agglomerative",
    )

    corrections_parser = subparsers.add_parser("corrections", help="Manage saved clustering corrections")
    corrections_sub = corrections_parser.add_subparsers(dest="corrections_command", required=True)

    summary_parser = corrections_sub.add_parser("summary", help="Show correction database counts")
    summary_parser.add_argument("--db-path", type=Path, default=default_db_path())

    must_link_parser = corrections_sub.add_parser("must-link", help="Add a must-link between two faces")
    must_link_parser.add_argument("--run-dir", required=True, type=Path)
    must_link_parser.add_argument("--face-a", required=True)
    must_link_parser.add_argument("--face-b", required=True)
    must_link_parser.add_argument("--db-path", type=Path, default=default_db_path())
    must_link_parser.add_argument("--note")

    cannot_link_parser = corrections_sub.add_parser("cannot-link", help="Add a cannot-link between two faces")
    cannot_link_parser.add_argument("--run-dir", required=True, type=Path)
    cannot_link_parser.add_argument("--face-a", required=True)
    cannot_link_parser.add_argument("--face-b", required=True)
    cannot_link_parser.add_argument("--db-path", type=Path, default=default_db_path())
    cannot_link_parser.add_argument("--note")

    merge_parser = corrections_sub.add_parser("merge-clusters", help="Merge two clusters by storing must-links")
    merge_parser.add_argument("--run-dir", required=True, type=Path)
    merge_parser.add_argument("--cluster-a", required=True, type=int)
    merge_parser.add_argument("--cluster-b", required=True, type=int)
    merge_parser.add_argument("--db-path", type=Path, default=default_db_path())
    merge_parser.add_argument("--note")

    split_parser = corrections_sub.add_parser("split-cluster", help="Split a cluster by storing a cannot-link")
    split_parser.add_argument("--run-dir", required=True, type=Path)
    split_parser.add_argument("--face-a", required=True)
    split_parser.add_argument("--face-b", required=True)
    split_parser.add_argument("--db-path", type=Path, default=default_db_path())
    split_parser.add_argument("--note")

    label_face_parser = corrections_sub.add_parser("label-face", help="Assign a person label to one face")
    label_face_parser.add_argument("--run-dir", required=True, type=Path)
    label_face_parser.add_argument("--face", required=True)
    label_face_parser.add_argument("--label", required=True)
    label_face_parser.add_argument("--db-path", type=Path, default=default_db_path())
    label_face_parser.add_argument("--note")

    label_cluster_parser = corrections_sub.add_parser("label-cluster", help="Assign a person label to all faces in a cluster")
    label_cluster_parser.add_argument("--run-dir", required=True, type=Path)
    label_cluster_parser.add_argument("--cluster-id", required=True, type=int)
    label_cluster_parser.add_argument("--label", required=True)
    label_cluster_parser.add_argument("--db-path", type=Path, default=default_db_path())
    label_cluster_parser.add_argument("--note")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = RunConfig(
            input_dir=args.input_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            corrections_db=args.corrections_db.resolve(),
            device=args.device,
            recursive=not args.no_recursive,
            model_name=args.model_name,
            det_size=args.det_size,
            detector_threshold=args.detector_threshold,
            clusterer=args.clusterer,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            agglomerative_distance_threshold=args.agglomerative_distance_threshold,
        )
        report_path = run_pipeline(config)
        print(f"Report written to {report_path}")
    elif args.command == "recluster":
        report_path = recluster_run(
            input_run_dir=args.input_run_dir,
            output_dir=args.output_dir,
            corrections_db=args.corrections_db.resolve(),
            clusterer=args.clusterer,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            agglomerative_distance_threshold=args.agglomerative_distance_threshold,
        )
        print(f"Report written to {report_path}")
    elif args.command == "corrections":
        if args.corrections_command == "summary":
            print(corrections_summary(args.db_path.resolve()))
        elif args.corrections_command in {"must-link", "cannot-link"}:
            left = resolve_face_key(args.run_dir.resolve(), args.face_a)
            right = resolve_face_key(args.run_dir.resolve(), args.face_b)
            relation = "must_link" if args.corrections_command == "must-link" else "cannot_link"
            add_pair_constraint(
                db_path=args.db_path.resolve(),
                face_key_a=left,
                face_key_b=right,
                relation=relation,
                source=args.corrections_command,
                note=args.note,
            )
            record_action(
                args.db_path.resolve(),
                args.corrections_command,
                {"run_dir": str(args.run_dir.resolve()), "face_a": left, "face_b": right, "note": args.note},
            )
            print(f"Saved {relation}: {left} <-> {right}")
        elif args.corrections_command == "merge-clusters":
            faces_a = resolve_cluster_faces(args.run_dir.resolve(), args.cluster_a)
            faces_b = resolve_cluster_faces(args.run_dir.resolve(), args.cluster_b)
            for face_a in faces_a:
                for face_b in faces_b:
                    add_pair_constraint(
                        db_path=args.db_path.resolve(),
                        face_key_a=str(face_a["face_key"]),
                        face_key_b=str(face_b["face_key"]),
                        relation="must_link",
                        source="merge_clusters",
                        note=args.note,
                    )
            record_action(
                args.db_path.resolve(),
                "merge_clusters",
                {
                    "run_dir": str(args.run_dir.resolve()),
                    "cluster_a": args.cluster_a,
                    "cluster_b": args.cluster_b,
                    "faces_a": [str(face["face_key"]) for face in faces_a],
                    "faces_b": [str(face["face_key"]) for face in faces_b],
                    "note": args.note,
                },
            )
            print(f"Saved merge between clusters {args.cluster_a} and {args.cluster_b}")
        elif args.corrections_command == "split-cluster":
            left = resolve_face_key(args.run_dir.resolve(), args.face_a)
            right = resolve_face_key(args.run_dir.resolve(), args.face_b)
            add_pair_constraint(
                db_path=args.db_path.resolve(),
                face_key_a=left,
                face_key_b=right,
                relation="cannot_link",
                source="split_cluster",
                note=args.note,
            )
            record_action(
                args.db_path.resolve(),
                "split_cluster",
                {"run_dir": str(args.run_dir.resolve()), "face_a": left, "face_b": right, "note": args.note},
            )
            print(f"Saved split constraint: {left} x {right}")
        elif args.corrections_command == "label-face":
            face_key = resolve_face_key(args.run_dir.resolve(), args.face)
            set_face_label(args.db_path.resolve(), face_key, args.label, "label_face", args.note)
            record_action(
                args.db_path.resolve(),
                "label_face",
                {"run_dir": str(args.run_dir.resolve()), "face": face_key, "label": args.label, "note": args.note},
            )
            print(f"Labeled {face_key} as {args.label}")
        elif args.corrections_command == "label-cluster":
            faces = resolve_cluster_faces(args.run_dir.resolve(), args.cluster_id)
            for face in faces:
                set_face_label(args.db_path.resolve(), str(face["face_key"]), args.label, "label_cluster", args.note)
            record_action(
                args.db_path.resolve(),
                "label_cluster",
                {
                    "run_dir": str(args.run_dir.resolve()),
                    "cluster_id": args.cluster_id,
                    "label": args.label,
                    "faces": [str(face["face_key"]) for face in faces],
                    "note": args.note,
                },
            )
            print(f"Labeled cluster {args.cluster_id} as {args.label}")
