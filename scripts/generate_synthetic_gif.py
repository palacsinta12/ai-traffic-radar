import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import ASSETS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a synthetic radar BEV gif demonstration.")
    parser.add_argument("--output", default=str(ASSETS_DIR / "inference_demo.gif"), help="Destination gif path.")
    parser.add_argument("--frames", type=int, default=60, help="Number of synthetic frames to render.")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate used to compute frame duration.")
    return parser.parse_args()


def create_synthetic_radar_gif(args=None):
    """Create a synthetic radar BEV gif for documentation or demo purposes."""
    if args is None:
        args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 224, 704
    duration_ms = int(1000 / args.fps)
    frames = []

    for i in range(args.frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = 127

        noise_x = np.random.randint(0, width, 400)
        noise_y = np.random.randint(0, height, 400)
        img[noise_y, noise_x, 2] = np.random.randint(10, 40, 400)
        img[noise_y, noise_x, 1] = np.random.randint(10, 40, 400)

        car_y = int(550 - (i * 5))
        car_x = 112 + int(np.sin(i / 4.0) * 3)
        car_bgr = (60, 220, 180)

        cv2.ellipse(img, (car_x, car_y), (8, 18), 0, 0, 360, car_bgr, -1)
        cv2.circle(img, (car_x - 6, car_y + 10), 4, car_bgr, -1)
        cv2.circle(img, (car_x + 6, car_y + 10), 4, car_bgr, -1)

        ped_y = 350
        ped_x = int(40 + (i * 2.5))
        ped_bgr = (170, 90, 60)
        pulse = int(np.sin(i * 1.5) * 20)
        ped_bgr_pulsed = (min(255, 170 + pulse), 90, 60)
        cv2.circle(img, (ped_x, ped_y), 4, ped_bgr_pulsed, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ch0, ch1, ch2 = cv2.split(img)
        ch2_dilated = cv2.dilate(ch2, kernel)
        ch1_dilated = cv2.dilate(ch1, kernel)
        ch0_dilated = cv2.dilate(ch0, kernel)

        ch0_dilated = np.where(ch2_dilated > 0, ch0_dilated, 127)
        img = cv2.merge([ch0_dilated, ch1_dilated, ch2_dilated])

        c_w, c_h = 24, 48
        cx1, cy1 = car_x - c_w // 2, car_y - c_h // 2 - 5
        cx2, cy2 = car_x + c_w // 2, car_y + c_h // 2 + 5
        car_conf = 0.88 + np.random.rand() * 0.05
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)

        car_label = f"Car {car_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(car_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (cx1, cy1 - th - 4), (cx1 + tw, cy1), (0, 165, 255), -1)
        cv2.putText(img, car_label, (cx1, cy1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        p_w, p_h = 8, 8
        px1, py1 = ped_x - p_w // 2 - 2, ped_y - p_h // 2 - 2
        px2, py2 = ped_x + p_w // 2 + 2, ped_y + p_h // 2 + 2
        ped_conf = 0.72 + np.random.rand() * 0.08
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 200, 0), 1)

        ped_label = f"Ped {ped_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(ped_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (px1, py1 - th - 4), (px1 + tw, py1), (0, 200, 0), -1)
        cv2.putText(img, ped_label, (px1, py1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(img_rgb))

    logging.info(f"Saving animated GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )
    logging.info("Synthetic gif generation complete.")


if __name__ == "__main__":
    create_synthetic_radar_gif(parse_args())
