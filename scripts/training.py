import os
import gc
import torch
from ultralytics import YOLO

from config import DEFAULT_DATA_YAML, DEFAULT_MODEL, RUNS_DIR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def train_radar_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🚀 Starting YOLO BEV Radar Training on CORRECTED Labels")
    last_checkpoint = RUNS_DIR / "radar" / "yolo11n_corrected_vru_fast-3" / "weights" / "last.pt"

    # START FRESH: Load standard optical weights (e.g., yolo11n.pt)
    # Do NOT load the previous best.pt, as it learned the delayed ghost labels.
    model = YOLO(DEFAULT_MODEL) 
    model=YOLO(str(last_checkpoint))
    model.train(
        data=DEFAULT_DATA_YAML,
        epochs=120,
        patience=30,
        batch=32,
        
        # Constraint honored: Keeping imgsz at 640
        imgsz=640, 
        
        device="0" if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR / "radar"),
        name="yolo11n_corrected_vru",
        
        # Optimizer and Schedule
        optimizer="AdamW",
        lr0=0.002,                
        cos_lr=True,
        
        # CRITICAL FIX 1: Restored standard weight decay (was 0.03)
        # 0.03 was choking the network's capacity to differentiate classes.
        weight_decay=0.0005, 
        
        # CRITICAL FIX 2: Rebalanced loss weights
        # Lowered 'cls' to stop it from defaulting to "Pedestrian" for everything.
        # Lowered 'dfl' because radar point clouds have inherently fuzzy edges.
        cls=0.5,                  
        box=7.5,                  
        dfl=0.5,                  
        
        # Augmentations
        mosaic=0.0,
        scale=0.0,
        translate=0.1,  # Added slight shift so it doesn't memorize absolute grid coordinates
        flipud=0.0,
        fliplr=0.5,
        
        # Constraint honored: Kept copy_paste
        copy_paste=0.3, 
        mixup=0.0,      # Kept off (prevents impossible radar color/velocity bleeding)
        
        cache=False,
        workers=2,
        resume=True,
    )

if __name__ == "__main__":
    train_radar_model()