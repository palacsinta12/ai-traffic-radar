import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from config import ASSETS_DIR, PROJECT_DIR, RUNS_DIR, SYNTHETIC_GIF_OUTPUT
from grid_mapping_pipeline import point_cloud_to_bev

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_synthetic_radar_frame(frame_idx: int) -> pd.DataFrame:
    """Generate one synthetic 40 ms radar sweep as a point-cloud dataframe."""
    points = []

    num_clutter = 50
    for _ in range(num_clutter):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(0, 70)
        amp = np.random.uniform(0.01, 0.05)
        freq = 32 + np.random.normal(0, 0.5)
        points.append([x, y, amp, freq])

    car_y = 60.0 - (frame_idx * 0.48)
    car_x = 2.5
    if 0 < car_y < 70:
        for _ in range(8):
            x_jitter = car_x + np.random.normal(0, 0.5)
            y_jitter = car_y + np.random.normal(0, 1.0)
            amp = np.random.uniform(1.5, 4.0)
            freq = np.random.normal(40.3, 0.5)
            points.append([x_jitter, y_jitter, amp, freq])

    ped_x = -5.0 + (frame_idx * 0.06)
    ped_y = 25.0
    if -10 < ped_x < 10:
        for _ in range(3):
            x_jitter = ped_x + np.random.normal(0, 0.2)
            y_jitter = ped_y + np.random.normal(0, 0.2)
            amp = np.random.uniform(0.1, 0.5)
            freq = 32 + np.sin(frame_idx * 0.5) * 2.0
            points.append([x_jitter, y_jitter, amp, freq])

    return pd.DataFrame(points, columns=["RealXData", "RealYData", "AmpData", "Frequency"])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic BEV radar demo GIF from a trained YOLO model.")
    parser.add_argument("--num-frames", type=int, default=80, help="Number of synthetic radar frames.")
    parser.add_argument("--model", default=None, help="Optional path to a best.pt file. Defaults to the latest run.")
    parser.add_argument("--output", default=str(SYNTHETIC_GIF_OUTPUT), help="Destination GIF path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO prediction confidence threshold.")
    return parser.parse_args()


def get_latest_model(runs_dir: Path) -> Path:
    model_paths = list(runs_dir.rglob("best.pt"))
    if not model_paths:
        raise FileNotFoundError("No best.pt found under the training run directory.")

    model_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return model_paths[0]


def run_synthetic_inference(args=None):
    if args is None:
        args = parse_args()

    runs_dir = RUNS_DIR / "radar"
    model_path = Path(args.model) if args.model else get_latest_model(runs_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logging.info("Loading YOLO model from: %s", model_path)
    model = YOLO(str(model_path))

    frames = []
    logging.info("Generating synthetic radar point clouds, mapping them to BEV, and running inference...")

    for i in range(args.num_frames):
        df = generate_synthetic_radar_frame(i)
        bev_img = point_cloud_to_bev(df)
        results = model.predict(bev_img, conf=args.conf, verbose=False)

        annotated_img = bev_img.copy()
        colors = {0: (0, 200, 0), 1: (0, 165, 255), 2: (0, 220, 100)}
        names = {0: "Ped", 1: "Car", 2: "Cyc"}

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = colors.get(cls_id, (255, 255, 255))

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label = f"{names.get(cls_id, 'Obj')} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated_img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        frames.append(Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=40,
        loop=0,
    )
    logging.info("Synthetic inference GIF saved to: %s", output_path)


if __name__ == "__main__":
    run_synthetic_inference(parse_args())