from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort

from face_poc.config import RunConfig
from face_poc.pipeline import recluster_run, run_pipeline


def default_device() -> str:
    providers = set(ort.get_available_providers())
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="face-poc", description="Local face-grouping POC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the face-grouping pipeline")
    run_parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing images")
    run_parser.add_argument("--output-dir", required=True, type=Path, help="Directory for report artifacts")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = RunConfig(
            input_dir=args.input_dir.resolve(),
            output_dir=args.output_dir.resolve(),
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
            clusterer=args.clusterer,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            agglomerative_distance_threshold=args.agglomerative_distance_threshold,
        )
        print(f"Report written to {report_path}")
