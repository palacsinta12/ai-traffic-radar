"""
Dataset Preparation & Synchronization Pipeline
==============================================
Synchronizes optical bounding box tracks with physical radar point-cloud clusters.
Projects camera annotations into the Radar BEV grid and filters out ghost/occluded
objects by requiring minimum radar signature support.
"""

import os
import shutil
import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    DATASET_DIR,
    MEASUREMENTS_DIR,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    IMG_WIDTH_M,
    IMG_HEIGHT_M,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Bounding Box Physical Extent Priors (Width, Length) in meters
CLASS_PRIORS = {
    0: (1.2, 1.2),  # Pedestrian: expanded from 0.8 to absorb radar limb smearing
    1: (1.0, 2.2),  # Bicycle
    2: (2.2, 4.8),  # Car
    3: (1.2, 2.2)   # Cyclist: expanded width
}

def setup_directories():
    """Initializes clean output directories for YOLO format datasets."""
    if DATASET_DIR.exists(): 
        shutil.rmtree(DATASET_DIR)
    
    for cache_file in MEASUREMENTS_DIR.glob("**/*.cache"):
        try: os.remove(cache_file)
        except OSError: pass
            
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def sync_and_extract_labels(folder: Path):
    """
    Pairs camera annotations with radar points. 
    Filters labels lacking radar support and calculates normalized bounding box geometries.
    """
    bev_dir = folder / "bev_images"
    csv_path = folder / "detections.csv"
    radar_path = folder / "export" / "radar1_resp.csv"
    ts_path = folder / "export" / "video1.timestamps"

    missing_files = [p.name for p in [bev_dir, csv_path, radar_path, ts_path] if not p.exists()]
    if missing_files:
        logging.warning(f"Skipping {folder.name}: Missing {missing_files}")
        return []

    dets = pd.read_csv(csv_path)
    if dets.empty or 'cam_X' not in dets.columns:
        logging.warning(f"Skipping {folder.name}: detections.csv is empty or missing 'cam_X' (Homography not run).")
        return []
    
    radar_df = pd.read_csv(radar_path)
    with open(ts_path, 'r') as f:
        cam_ts = [float(line.strip()) for line in f.readlines()]

    unique_clocks = sorted(radar_df['Window Clock'].unique())
    
    # Reconstruct the sliding accumulation window (10 frames / 400ms)
    clock_to_pts = {}
    accumulation_buffer = []
    for c in unique_clocks:
        current_points = radar_df[radar_df['Window Clock'] == c]
        accumulation_buffer.append(current_points)
        if len(accumulation_buffer) > 10:
            accumulation_buffer.pop(0)
        clock_to_pts[c] = pd.concat(accumulation_buffer, ignore_index=True)

    labels_data = []
    
    # Subsample based on domain
    for clock in unique_clocks[::5]:
        img_file = bev_dir / f"{int(clock)}.png"
        if not img_file.exists(): continue
    
        closest_frame = np.argmin(np.abs(np.array(cam_ts) - clock))
        if abs(cam_ts[closest_frame] - clock) > 100: 
            continue

        frame_dets = dets[dets['frame'] == closest_frame]
        r_pts = clock_to_pts[clock] 
        
        label_lines = []
        for _, row in frame_dets.iterrows():
            c_x, c_y = row['cam_X'], row['cam_Y']
            cls_id = int(row['cls_id'])
            
            # Radar Support Validation & Position Snapping
            has_radar_support = False
            if not r_pts.empty:
                dist = np.sqrt((r_pts['RealXData'] - c_x)**2 + (r_pts['RealYData'] - c_y)**2)
                close_pts = r_pts[dist < 2.0]
                if not close_pts.empty:
                    c_x = close_pts['RealXData'].median()
                    
                    # Only snap Y to the most recent radar returns in the cluster
                    current_pts = close_pts[close_pts['Window Clock'] == clock]
                    if not current_pts.empty:
                        c_y = current_pts['RealYData'].min() # Front edge of the CURRENT position
                    else:
                        c_y = close_pts['RealYData'].median() # Fallback
                    
                    has_radar_support = True 
            
            if not has_radar_support:
                continue

            if not (X_MIN <= c_x <= X_MAX) or not (Y_MIN <= c_y <= Y_MAX):
                continue
                
            rad_W, rad_L = CLASS_PRIORS.get(cls_id, (1.0, 1.0))
            c_y_center = c_y + (rad_L / 2.0)
            
            # Normalize to padded dimensions
            x_norm = max(0.001, min(0.999, (c_x - X_MIN) / IMG_WIDTH_M))
            y_norm = max(0.001, min(0.999, (Y_MAX - c_y_center) / IMG_HEIGHT_M))
            w_norm = rad_W / IMG_WIDTH_M
            h_norm = rad_L / IMG_HEIGHT_M
            
            # Map to target classes: 0=Pedestrian, 1=Car, 2=Cyclist
            if cls_id == 0: yolo_cls = 0      
            elif cls_id == 2: yolo_cls = 1    
            elif cls_id in [1, 3]: yolo_cls = 2 
            else: continue
            
            label_lines.append(f"{yolo_cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        labels_data.append((img_file, "\n".join(label_lines)))
    
    return labels_data

def build_dataset():
    """Compiles the final YOLO dataset with train/val splits."""
    setup_directories()
    folders = [f for f in MEASUREMENTS_DIR.iterdir() if f.is_dir() and f.name not in ["reference_points", "ref_pts"]]
    
    folder_cache = {}
    for f in tqdm(folders, desc="Extracting Target Labels"):
        labels = sync_and_extract_labels(f)
        if len(labels) > 0:
            folder_cache[f] = labels

    valid_folders = list(folder_cache.keys())
    if not valid_folders:
        raise RuntimeError("No valid, annotated measurement folders found.")

    # --- RANDOM TRAIN/VAL FOLDER SPLIT ---
    # Sort first to ensure deterministic behavior across systems, then shuffle
    # 1. Separate clean target environment (202605) from older legacy data
    target_folders = [f for f in valid_folders if "202605" in f.name]
    legacy_folders = [f for f in valid_folders if "202605" not in f.name]
    
    # 2. Randomly split the clean 202605 folders (80% train / 20% val)
    target_folders.sort()
    random.seed(42)  # Deterministic seed
    random.shuffle(target_folders)
    
    split_idx = max(1, int(len(target_folders) * 0.8))
    target_train = target_folders[:split_idx]
    val_folders  = target_folders[split_idx:]  # Clean, uncorrupted validation set
    
    # 3. Training set gets legacy data + target training data
    train_folders = legacy_folders + target_train
    
    logging.info(f"--- DATASET COMPILATION ---")
    logging.info(f"Training on {len(train_folders)} folders (Legacy + 80% of 202605).")
    logging.info(f"Randomly validating on {len(val_folders)} clean 202605 folders: {[f.name for f in val_folders]}")
    
    def process_split(split_folders, split_name):
        total = 0
        for i, folder in enumerate(tqdm(split_folders, desc=f"Building {split_name}")):
            clean_folder_name = folder.name.replace("#", "_")
            for img_path, label_content in folder_cache[folder]:
                new_img_name = f"{i}_{clean_folder_name}_{img_path.name}"
                shutil.copy(img_path, DATASET_DIR / 'images' / split_name / new_img_name)
                
                label_path = DATASET_DIR / 'labels' / split_name / new_img_name.replace('.png', '.txt')
                with open(label_path, 'w') as f:
                    f.write(label_content)
                total += 1
        return total

    train_count = process_split(train_folders, 'train')
    val_count = process_split(val_folders, 'val')
    
    yaml_content = f"path: {DATASET_DIR.absolute()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: pedestrian\n  1: car\n  2: cyclist\n"
    with open(DATASET_DIR / 'data.yaml', 'w') as f:
        f.write(yaml_content)
        
    logging.info(f"Dataset compiled! Train: {train_count} images, Val: {val_count} images.")
    
if __name__ == "__main__":
    build_dataset()