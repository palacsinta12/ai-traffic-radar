import gc
import os
import torch
from ultralytics import YOLO

from config import DEFAULT_DATA_YAML, RUNS_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def train_radar_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🚀 Starting Fresh YOLO11s Training for Radar Object Detection")

    # 1. Start fresh from official YOLO11s optical pretrained weights
    model = YOLO("yolo11s.pt")

    model.train(
        data=DEFAULT_DATA_YAML,
        epochs=150,
        patience=35,
        batch=32,
        
        imgsz=[640, 192],
        rect=True,
        
        device="0" if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR / "radar"),
        name="yolo11s_radar",
        
        # Optimizer and Schedule
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        
        # High cls loss to heavily penalize missing/confusing sparse VRUs (peds/cyclists)
        cls=1.6,
        box=7.0,
        dfl=0.6,
        
        copy_paste=0.4,
        
        mosaic=0.0,
        mixup=0.0,
        
        # BEV Spatial Symmetries:
        # Left-Right inversion is physically valid across the sensor bore-sight
        fliplr=0.5,
        flipud=0.0,
        
        # Geometry shifts: slight translation & minor scale to avoid memorizing grid bins
        translate=0.08,
        scale=0.05,
        degrees=0.0,
        
        cache=False,
        workers=4,
        resume=False,
    )


if __name__ == "__main__":
    train_radar_model()