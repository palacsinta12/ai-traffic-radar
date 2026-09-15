import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

from config import PROJECT_DIR, RUNS_DIR, EXPORT_IMGSZ

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Export the latest YOLO training weights to ONNX.")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="Directory containing YOLO training runs.")
    parser.add_argument("--imgsz", nargs=2, type=int, default=EXPORT_IMGSZ, help="Model image size [width height].")
    parser.add_argument("--half", action="store_true", help="Use FP16 export when supported.")
    return parser.parse_args()


def get_latest_run_weights(runs_dir: Path) -> Path:
    """Return the newest YOLO weights file found under the training run tree."""
    all_runs = list(runs_dir.rglob("best.pt"))
    if not all_runs:
        raise FileNotFoundError(f"No best.pt found in {runs_dir}")

    all_runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return all_runs[0]


def export_to_onnx(args=None):
    if args is None:
        args = parse_args()

    runs_dir = Path(args.runs_dir)
    best_weights_path = get_latest_run_weights(runs_dir)

    logging.info("Loading weights from latest run: %s", best_weights_path.parent.parent.name)
    model = YOLO(str(best_weights_path))

    logging.info("Exporting to ONNX format...")
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        simplify=True,
        opset=12,
    )

    logging.info("Export successful. ONNX model saved to: %s", onnx_path)


if __name__ == "__main__":
    export_to_onnx(parse_args())