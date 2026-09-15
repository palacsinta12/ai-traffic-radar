"""import argparse
import os
import gc
from pathlib import Path
import torch
from ultralytics import YOLO

from config import DATASET_DIR, DEFAULT_DATA_YAML, DEFAULT_MODEL, RUNS_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def train_radar_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🚀 Starting YOLO BEV Radar Training (Unified Stage)")
    print("Applying Mixup & Copy-Paste to prevent background memorization.")

    model = YOLO(DEFAULT_MODEL) # Loads yolo11n.pt
    
    model.train(
        data=DEFAULT_DATA_YAML,
        epochs=120,
        patience=30,
        batch=32,
        
        # USE A SINGLE INT: YOLO will map longest side to 640 and keep 192 width automatically
        imgsz=640, 
        
        device="0" if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR / "radar"),
        name="yolo11n_unified_vru",
        
        # Optimizer and Schedule
        optimizer="AdamW",
        lr0=0.002,                
        cos_lr=True,
        # Doubled weight decay to force the network to forget the background
        weight_decay=0.03, 
        
        # Loss weights (punish classification errors heavily to drop val/cls_loss)
        cls=2.0,                  
        box=7.0,                  
        dfl=1.5,                  
        
        # Physics-Preserving Spatial Augmentations (Keep these disabled!)
        mosaic=0.0,
        scale=0.0,
        translate=0.0,
        flipud=0.0,
        fliplr=0.5,
        
        # THE CURE FOR MEMORIZATION: Synthetic Augmentations
        copy_paste=0.3, # Randomly pastes pedestrians into the road
        mixup=0.15,     # Blends two frames together to smooth decision boundaries
        
        cache=False,
        workers=2,
    )

if __name__ == "__main__":
    train_radar_model()"""

import argparse
import os
import gc
from pathlib import Path
import torch
from ultralytics import YOLO

from config import DATASET_DIR, DEFAULT_DATA_YAML, DEFAULT_MODEL, RUNS_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def resume_radar_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Define the path to the interrupted run's last checkpoint
    last_checkpoint = RUNS_DIR / "radar" / "yolo11n_unified_vru-2" / "weights" / "last.pt"

    if not last_checkpoint.exists():
        print(f"❌ Error: Could not find checkpoint at {last_checkpoint}")
        print("Make sure the path is correct!")
        return

    print(f"🚀 RESUMING YOLO BEV Radar Training from {last_checkpoint}")

    # 1. Load the interrupted model's checkpoint
    model = YOLO(str(last_checkpoint)) 
    
    # 2. Call train with resume=True
    # Note: YOLO automatically reads your saved args.yaml, so you don't need 
    # to pass epochs, imgsz, mixup, or learning rate here! It remembers everything.
    model.train(resume=True)

if __name__ == "__main__":
    resume_radar_model()