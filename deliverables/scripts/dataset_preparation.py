import os
import shutil
import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PROJECT_DIR = Path(__file__).resolve().parent.parent
MEASUREMENTS_DIR = PROJECT_DIR / "measurements"
DATASET_DIR = PROJECT_DIR / "dataset"

# --- EXACT GRID ALIGNMENT (0.1m/px -> 224x704 pixels) ---
X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = 0.0, 70.0
WIDTH_M = X_MAX - X_MIN    
HEIGHT_M = Y_MAX - Y_MIN  

# --- EXPANDED BOX PRIORS (To catch scattered radar points) ---
CLASS_PRIORS = {
    0: (0.8, 0.8),  # Pedestrian (Tighter)
    1: (0.8, 2.0),  # Bicycle
    2: (2.0, 4.8),  # Car
    3: (1.0, 2.0)   # Cyclist
}

def setup_directories():
    if DATASET_DIR.exists(): shutil.rmtree(DATASET_DIR)
    
    # CRITICAL: Purge any old YOLO caches that corrupt class IDs
    for cache_file in MEASUREMENTS_DIR.glob("**/*.cache"):
        try: os.remove(cache_file)
        except OSError: pass
            
    for split in ['train', 'val']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def sync_and_extract_labels(folder: Path):
    bev_dir = folder / "bev_images"
    csv_path = folder / "detections.csv"
    radar_path = folder / "export" / "radar1_resp.csv"
    ts_path = folder / "export" / "video1.timestamps"

    if not all(p.exists() for p in [bev_dir, csv_path, radar_path, ts_path]): return []

    dets = pd.read_csv(csv_path)
    if dets.empty or 'cam_X' not in dets.columns: return []
    
    radar_df = pd.read_csv(radar_path)
    with open(ts_path, 'r') as f:
        cam_ts = [float(line.strip()) for line in f.readlines()]

    unique_clocks = sorted(radar_df['Window Clock'].unique())[::5]
    labels_data = []

    for clock in unique_clocks:
        img_file = bev_dir / f"{int(clock)}.png"
        if not img_file.exists(): continue

        closest_frame = np.argmin(np.abs(np.array(cam_ts) - clock))
        if abs(cam_ts[closest_frame] - clock) > 100: continue

        frame_dets = dets[dets['frame'] == closest_frame]
        r_pts = radar_df[radar_df['Window Clock'] == clock].copy()
        
        label_lines = []
        for _, row in frame_dets.iterrows():
            c_x, c_y = row['cam_X'], row['cam_Y']
            cls_id = int(row['cls_id'])
            
            has_radar_support = False
            if not r_pts.empty:
                dist = np.sqrt((r_pts['RealXData'] - c_x)**2 + (r_pts['RealYData'] - c_y)**2)
                close_pts = r_pts[dist < 1.5] # Tighter radius (1.5m instead of 2.0m)
                if not close_pts.empty:
                    c_x = close_pts['RealXData'].median()
                    # Snapping to min() is brilliant for cars (front bumper reflection)
                    c_y = close_pts['RealYData'].min() 
                    has_radar_support = True
            
            # FIX 2: If the object is invisible to radar, DO NOT train on it!
            if not has_radar_support:
                continue
                
            rad_W, rad_L = CLASS_PRIORS.get(cls_id, (1.0, 1.0))
            c_y_center = c_y + (rad_L / 2.0)
            
            x_norm = max(0.001, min(0.999, (c_x - X_MIN) / WIDTH_M))
            y_norm = max(0.001, min(0.999, (Y_MAX - c_y_center) / HEIGHT_M))
            w_norm = rad_W / WIDTH_M
            h_norm = rad_L / HEIGHT_M
            
            # --- EXACT 3-CLASS MAPPING ---
            if cls_id == 0: yolo_cls = 0      # Pedestrian
            elif cls_id == 2: yolo_cls = 1    # Car
            elif cls_id in [1, 3]: yolo_cls = 2 # Cyclist (Bikes mapped to cyclist)
            else: continue
            
            label_lines.append(f"{yolo_cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        labels_data.append((img_file, "\n".join(label_lines)))
    return labels_data

def build_dataset():
    setup_directories()
    folders = [f for f in MEASUREMENTS_DIR.iterdir() if f.is_dir() and f.name not in ["reference_points", "ref_pts"]]
    
    # --- FIX: CACHE THE EXTRACTED LABELS ---
    # sync_and_extract_labels is computationally heavy. We run it exactly ONCE per folder 
    # and store the result in memory to speed up validation checks and oversampling.
    folder_cache = {}
    for f in tqdm(folders, desc="Extracting Labels from CSVs"):
        labels = sync_and_extract_labels(f)
        if len(labels) > 0:
            folder_cache[f] = labels

    valid_folders = list(folder_cache.keys())
    if not valid_folders:
        raise RuntimeError("No valid, annotated measurement folders found.")

    # Separate older campaigns from the target May 6th benchmark campaign
    target_folders = [f for f in valid_folders if "202605" in f.name]
    base_folders = [f for f in valid_folders if "202605" not in f.name]
    
    # --- FIXED DIVERSE TARGET SPLIT FOR VALIDATION ---
    guaranteed_val_names = [
        "20260506-151215_bike+ped",
        "20260506-153613_car+bikes",
        "20260506-161743_mix",
        "20260506-151900-ped+bike-mix",
        "20260506-150006_car",
        "20260506-151731-bike-change+bike-pusher"
    ]
    
    val_folders = [f for f in target_folders if f.name in guaranteed_val_names]
    target_train = [f for f in target_folders if f.name not in guaranteed_val_names]

    # --- ONE-STEP TARGET BIASING ---
    # Training set = All old data + (May 6th training data * 3 copies)
    train_folders = base_folders + (target_train * 3)
    
    logging.info(f"--- 1-STEP BENCHMARK DATASET COMPILATION ---")
    logging.info(f"Validation strictly on target environment: {[f.name for f in val_folders]}")
    logging.info(f"Training uses {len(base_folders)} base folders and heavily oversamples {len(target_train)} target folders.")

    def process_split(split_folders, split_name):
        total = 0
        for i, folder in enumerate(tqdm(split_folders, desc=f"Building {split_name}")):
            
            # FIX: Clean the folder name of special characters forbidden by Kaggle
            clean_folder_name = folder.name.replace("#", "_")
            
            for img_path, label_content in folder_cache[folder]:
                # Using the sanitized name
                new_img_name = f"{i}_{clean_folder_name}_{img_path.name}"
                
                shutil.copy(img_path, DATASET_DIR / 'images' / split_name / new_img_name)
                with open(DATASET_DIR / 'labels' / split_name / new_img_name.replace('.png', '.txt'), 'w') as f:
                    f.write(label_content)
                total += 1
        return total

    train_count = process_split(train_folders, 'train')
    val_count = process_split(val_folders, 'val')
    
    yaml_content = f"path: {DATASET_DIR.absolute()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: pedestrian\n  1: car\n  2: cyclist\n"
    with open(DATASET_DIR / 'data.yaml', 'w') as f:
        f.write(yaml_content)
        
    logging.info(f"Dataset generated! Train={train_count} images, Val={val_count} images.")
    
if __name__ == "__main__":
    build_dataset()