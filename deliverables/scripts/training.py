import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from ultralytics import YOLO
from pathlib import Path

torch.cuda.set_per_process_memory_fraction(0.8, device=0)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_DIR / "dataset" / "data.yaml"

def train_base_model():
    # Start fresh from a nano model to avoid overfitting on the sparse radar data
    model = YOLO('yolo11n.pt') 
    
    model.train(
        data=str(DATA_YAML),
        epochs=60,               
        patience=15,             
        batch=32,               
        imgsz=[704, 224],        # CRITICAL FIX: Must be multiples of 32!
        device=0,
        project='runs/radar',
        name='yolo11n_radar_final_pls_work',
        
        optimizer='AdamW',      
        lr0=0.002,               
        cos_lr=True,             # TWEAK: Helps AdamW settle into the minimum loss
        weight_decay=0.01,      
        warmup_epochs=3,         
        
        # --- RADAR PHYSICS AUGMENTATION RESTRAINTS ---
        mosaic=0.0,              
        scale=0.0,               
        translate=0.0,           
        flipud=0.0,              # NEVER flip up/down (breaks approaching/receding doppler)
        fliplr=0.5,              # Left/Right is perfectly symmetrical
        
        # --- LOSS WEIGHTS ---
        cls=0.5,                 
        box=7.5,                 # TWEAK: Standard YOLO default for tighter bounding boxes
        dfl=1.5,                 # TWEAK: Standard YOLO default for Distribution Focal Loss
        cache=True,              # Cache images for faster training (critical for large datasets)
        half=True,
        workers=4,
    )

if __name__ == "__main__":
    train_base_model()