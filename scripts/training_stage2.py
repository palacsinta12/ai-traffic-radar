import argparse
import os
import gc
import torch
from pathlib import Path
from ultralytics import YOLO

# Adjust this path to point to the exact best.pt from your Stage 1 run
STAGE1_BEST_WEIGHTS = "runs/radar/yolo11n_stage1_baseline/weights/best.pt"

def train_stage2():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Starting STAGE 2: VRU Sensitivity and Synthetic Augmentation")
    
    # 1. Initialize from Stage 1 weights
    model = YOLO(STAGE1_BEST_WEIGHTS)
    
    # 2. Execute fine-tuning regime
    model.train(
        data="dataset/data.yaml",
        epochs=60,
        patience=20,
        batch=32,
        imgsz=[640, 192],
        device="0" if torch.cuda.is_available() else "cpu",
        project="runs/radar",
        name="yolo11n_stage2_vru_optimized",
        
        # Lower learning rate for fine-tuning
        optimizer="AdamW",
        lr0=0.0005,                
        cos_lr=True,
        weight_decay=0.015,
        
        # Increased classification penalty to suppress background false negatives
        cls=2.0,                  
        box=7.0,                  
        dfl=1.5,                  
        
        # Physics-Preserving base augmentations
        mosaic=0.0,
        scale=0.0,
        translate=0.0,
        flipud=0.0,
        fliplr=0.5,
        
        # AGGRESSIVE SYNTHETIC AUGMENTATIONS RE-ENABLED
        copy_paste=0.4, # Crucial for bicycle/pedestrian mAP gain  
        mixup=0.2,      # Simulates dense multi-target traffic          
        
        cache=False,
        workers=2,
    )

if __name__ == "__main__":
    train_stage2()