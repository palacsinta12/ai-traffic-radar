import os
import gc
import torch
from ultralytics import YOLO

from config import DEFAULT_DATA_YAML, RUNS_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def train_fast_radar():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🚀 Starting FAST YOLO BEV Radar Training (Targeting < 2 Hours)")

    # 1. UPGRADE TO 'SMALL' MODEL: Much better at sparse radar features than 'Nano'
    model = YOLO("yolo11s.pt") 
    
    model.train(
        data=DEFAULT_DATA_YAML,
        epochs=35,               # 35 epochs is plenty for fine-tuning to reach 0.40+ mAP
        patience=12,             # Stop early if it plateaus
        batch=32,                # Keep high to maximize GPU
        
        imgsz=[640, 192],  
        
        device="0" if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR / "radar"),
        name="yolo11s_fast_radar",
        
        # SPEED BOOSTERS
        cache=True,              # Loads dataset into RAM -> MASSIVE speedup per epoch!
        workers=4,               # Speeds up dataloading
        
        # Optimizer and Schedule
        optimizer="AdamW",
        lr0=0.001,               # Lowered from 0.002: more stable, precise learning
        cos_lr=True,
        weight_decay=0.0005,     # Standard, healthy weight decay
        
        # Adjusted Loss Weights for Precision
        cls=1.0,                 # Raised to 1.0 to stop false positive "Pedestrian" guesses
        box=8.5,                 # Raised to force tighter bounding boxes
        dfl=0.5,                 # Kept low so it doesn't punish fuzzy radar edges
        
        # Augmentations
        mosaic=0.0,
        scale=0.0,
        translate=0.1,  
        flipud=0.0,
        fliplr=0.5,
        copy_paste=0.3,          # Kept per your constraints
        mixup=0.0,      
        close_mosaic=5,          # Disables heavy augments in the final 5 epochs to settle the weights
    )

if __name__ == "__main__":
    train_fast_radar()