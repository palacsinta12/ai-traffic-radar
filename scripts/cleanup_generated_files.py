"""Cleanup utility for generated measurement artifacts."""

import argparse
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_DIR = Path(__file__).resolve().parent.parent
MEASUREMENTS_DIR = PROJECT_DIR / "measurements"

GENERATED_ITEMS = [
    "detections.csv",
    "video1_annot.mp4",
    "bev_images",
    "time_sync.csv",
    "detections_synced.csv",
    "radar_synced.csv",
    "fusion_bev.mp4",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Remove generated outputs from measurement folders.")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Report target artifacts without deleting them.")
    return parser.parse_args()


def purge_generated_data(force: bool = False, dry_run: bool = False):
    """Delete generated labels, media, and BEV artifacts while preserving raw inputs."""
    if not MEASUREMENTS_DIR.exists():
        logging.error("Measurements directory not found: %s", MEASUREMENTS_DIR)
        return

    if not force:
        confirm = input("This deletes generated artifacts from all measurement folders. Continue? (y/n): ")
        if confirm.lower() != "y":
            logging.info("Cleanup cancelled.")
            return

    logging.info("Starting cleanup in: %s", MEASUREMENTS_DIR)
    folders = [f for f in MEASUREMENTS_DIR.iterdir() if f.is_dir() and f.name not in {"reference_points", "ref_pts"}]

    total_deleted = 0
    for folder in folders:
        logging.info("Checking folder: %s", folder.name)

        for item_name in GENERATED_ITEMS:
            item_path = folder / item_name
            if not item_path.exists():
                continue

            if dry_run:
                logging.info("Would remove %s", item_path)
                continue

            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                    logging.info("Deleted directory: %s", item_name)
                else:
                    os.remove(item_path)
                    logging.info("Deleted file: %s", item_name)
                total_deleted += 1
            except Exception as exc:
                logging.error("Could not delete %s: %s", item_name, exc)

    dataset_dir = PROJECT_DIR / "dataset"
    if dataset_dir.exists() and not dry_run:
        logging.info("Removing consolidated dataset directory.")
        shutil.rmtree(dataset_dir)

    logging.info("Cleanup complete. Total items removed: %s", total_deleted)


if __name__ == "__main__":
    args = parse_args()
    purge_generated_data(force=args.force, dry_run=args.dry_run)
