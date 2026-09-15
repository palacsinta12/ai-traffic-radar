"""Coordinate the end-to-end radar BEV training workflow."""

import argparse
import subprocess
import sys
from pathlib import Path

from config import PROJECT_DIR, SCRIPTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Run the configured radar BEV pipeline stages.")
    parser.add_argument("--skip-annotation", action="store_true", help="Skip the annotation and homography stage.")
    parser.add_argument("--skip-grid", action="store_true", help="Skip the radar grid mapping stage.")
    parser.add_argument("--skip-prep", action="store_true", help="Skip the dataset preparation stage.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the YOLO training stage.")
    return parser.parse_args()


def run_cmd(description, cmd_list):
    """Run a subprocess command and stop the workflow if it fails."""
    print(f"\n{'=' * 80}")
    print(f"STARTING: {description}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd_list, cwd=str(PROJECT_DIR), check=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed. Stopping the pipeline before continuing.")
        sys.exit(result.returncode)

    print(f"\nFINISHED: {description}\n")


if __name__ == "__main__":
    args = parse_args()
    print("PIPELINE INITIATED. Starting...\n")

    if not args.skip_annotation:
        annotation_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "annotation_pipeline.py"),
            "--run_batch",
            "--run_inference",
            "--run_homography",
            "--run_visualize",
            "--model",
            "yolo11m.pt",
            "--conf",
            "0.15",
            "--iou",
            "0.45",
            "--tracker_type",
            "botsort",
            "--track_buffer",
            "100",
            "--no-use_ransac",
            "--augment",
            "--allow_skip",
        ]
        run_cmd("Annotation and homography", annotation_cmd)

    if not args.skip_grid:
        grid_cmd = [sys.executable, str(SCRIPTS_DIR / "grid_mapping_pipeline.py")]
        run_cmd("Radar grid mapping", grid_cmd)

    if not args.skip_prep:
        prep_cmd = [sys.executable, str(SCRIPTS_DIR / "dataset_preparation.py")]
        run_cmd("Dataset preparation", prep_cmd)

    if not args.skip_train:
        train_cmd = [sys.executable, str(SCRIPTS_DIR / "training.py")]
        run_cmd("YOLO model training", train_cmd)

    print(f"\n{'*' * 80}")
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY.")
    print(f"{'*' * 80}\n")
   