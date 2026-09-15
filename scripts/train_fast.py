import os
import gc
import torch
from ultralytics import YOLO

from config import DEFAULT_DATA_YAML, RUNS_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Path to the best.pt from the run you just finished
PREV_BEST_WEIGHTS = RUNS_DIR / "radar" / "yolo11n_unified_vru-2" / "weights" / "best.pt"

def train_fast():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Starting FAST Fine-Tuning Run from: {PREV_BEST_WEIGHTS.name}")

    # Load the already-trained radar weights (NOT optical yolo11n.pt)
    model = YOLO(str(PREV_BEST_WEIGHTS))
    
    model.train(
        data=DEFAULT_DATA_YAML,
        epochs=40,              # Short, focused runway
        patience=15,            # Stop if it overfits after peak
        batch=32,
        imgsz=640,
        
        device="0" if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR / "radar"),
        name="yolo11n_fast_finetune",
        
        # Fine-tuning optimizer settings
        optimizer="AdamW",
        lr0=0.0005,             # 4x lower LR to avoid destroying pre-learned features
        lrf=0.05,               # Final LR = 0.05 * lr0
        cos_lr=True,
        weight_decay=0.02,
        
        # Loss balance: increase box and dfl priority so small targets bind tightly
        cls=1.0,                
        box=7.5,                
        dfl=2.0,                # Sharpen edge boundaries for tiny 8x8 pedestrian boxes
        
        # Augmentations
        mosaic=0.0,
        scale=0.0,
        translate=0.0,
        flipud=0.0,
        fliplr=0.5,
        mixup=0.1,              # Light blend to prevent memorization
        close_mosaic=10,        # Turn off mixup for the last 10 epochs
        
        cache=False,
        workers=2,
    )

if __name__ == "__main__":
    train_fast()