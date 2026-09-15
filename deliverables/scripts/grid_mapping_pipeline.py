"""
Radar BEV Grid Mapping Pipeline
===============================
Transforms sparse radar point clouds into dense, 3-channel pseudo-images.
Implements Temporal Accumulation, Amplitude Normalization, Doppler Skewing,
and Cell Spreading.
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = 0.0, 70.0
RESOLUTION = 0.1

IMG_W = int((X_MAX - X_MIN) / RESOLUTION)  # 200 pixels
IMG_H = int((Y_MAX - Y_MIN) / RESOLUTION)  # 700 pixels

def point_cloud_to_bev(accumulated_points_df: pd.DataFrame) -> np.ndarray:
    """
    Accepts a pandas DataFrame containing accumulated radar point clouds 
    and returns a normalized, padded 3-channel BEV pseudo-image (704x224x3)
    ready for YOLO inference.
    """
    # Filter strictly to our 70m x 20m ROI
    roi_pts = accumulated_points_df[
        (accumulated_points_df['RealXData'] >= X_MIN) & (accumulated_points_df['RealXData'] <= X_MAX) &
        (accumulated_points_df['RealYData'] >= Y_MIN) & (accumulated_points_df['RealYData'] <= Y_MAX)
    ].copy()

    # Initialize the 3-channel image (H, W, 3)
    grid = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)

    if roi_pts.empty:
        # Return a blank padded image if no points exist
        return np.zeros((704, 224, 3), dtype=np.uint8)

    # Map metric coordinates to pixel indices
    u_idx = np.floor((roi_pts['RealXData'] - X_MIN) / RESOLUTION).astype(int)
    v_idx = np.floor((Y_MAX - roi_pts['RealYData']) / RESOLUTION).astype(int)
    
    u_idx = np.clip(u_idx, 0, IMG_W - 1)
    v_idx = np.clip(v_idx, 0, IMG_H - 1)

    np.add.at(grid[:, :, 0], (v_idx, u_idx), 1)

    R_sq = roi_pts['RealXData']**2 + roi_pts['RealYData']**2
    norm_amp = roi_pts['AmpData'] * R_sq
    log_amp = np.log1p(norm_amp) 
    
    raw_vel = (roi_pts['Frequency'] - 32) * 1.435986
    skewed_vel = np.sign(raw_vel) * np.sqrt(np.abs(raw_vel))

    for i in range(len(u_idx)):
        u, v = u_idx.iloc[i], v_idx.iloc[i]
        grid[v, u, 1] = max(grid[v, u, 1], log_amp.iloc[i])
        grid[v, u, 2] = skewed_vel.iloc[i] if abs(skewed_vel.iloc[i]) > abs(grid[v, u, 2]) else grid[v, u, 2]

    ch0 = np.clip(grid[:, :, 0] / 5.0, 0, 1) * 255
    
    max_amp = np.max(grid[:, :, 1]) if np.max(grid[:, :, 1]) > 0 else 1
    ch1 = (grid[:, :, 1] / max_amp) * 255
    
    ch2 = np.clip((grid[:, :, 2] / 5.5) * 127 + 127, 0, 255)
    ch2[grid[:, :, 0] == 0] = 0  # Background has 0 velocity

    # Stack raw channels
    img = np.stack([ch0, ch1, ch2], axis=-1).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ch0_dilated = cv2.dilate(img[:, :, 0], kernel, iterations=1)
    ch1_dilated = cv2.dilate(img[:, :, 1], kernel, iterations=1)
    ch2_dilated = cv2.dilate(img[:, :, 2], kernel, iterations=1)
    
    ch1_masked = np.where(ch0_dilated > 0, ch1_dilated, 0)
    ch2_masked = np.where(ch0_dilated > 0, ch2_dilated, 0)
    img_spread = np.stack([ch0_dilated, ch1_masked, ch2_masked], axis=-1).astype(np.uint8)

    # Padding to Stride-32 compliant size (704x224x3) for YOLO
    padded_img = cv2.copyMakeBorder(
        img_spread, 
        top=0, bottom=4, 
        left=0, right=24, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )

    return padded_img

def process_radar_to_grid(radar_df: pd.DataFrame, out_dir: Path, frames_to_accumulate: int = 10):
    """
    Converts a dataframe of raw radar points into a series of BEV PNG images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_clocks = sorted(radar_df['Window Clock'].unique())
    accumulation_buffer = []

    for idx, clock in enumerate(tqdm(unique_clocks, desc="Generating BEV Grids")):
        current_points = radar_df[radar_df['Window Clock'] == clock].copy()
        
        accumulation_buffer.append(current_points)
        if len(accumulation_buffer) > frames_to_accumulate:
            accumulation_buffer.pop(0)
        
        # Step every 5 frames to match training dataset decimation
        if idx % 5 != 0:
            continue
            
        dense_pc = pd.concat(accumulation_buffer, ignore_index=True)
        
        # Process the point cloud into our standard padded BEV grid
        padded_bev = point_cloud_to_bev(dense_pc)

        # Slice the padding off when saving to disk to preserve the 700x200 raw output shape
        # (YOLO's dataloader handles padding dynamically during training)
        img_to_save = padded_bev[0:700, 0:200]

        filename = out_dir / f"{int(clock)}.png"
        cv2.imwrite(str(filename), img_to_save)

def run_batch_grid_mapping(root_dir: str):
    root = Path(root_dir)
    measurement_folders = sorted(f for f in root.iterdir() if f.is_dir() and f.name not in ["reference_points", "ref_pts"])
    
    for folder in measurement_folders:
        radar_csv = folder / "export" / "radar1_resp.csv"
        if not radar_csv.exists():
            continue
            
        logging.info(f"Mapping: {folder.name}")
        out_dir = folder / "bev_images"
        
        df = pd.read_csv(radar_csv)
        df.columns = df.columns.str.strip()
        
        process_radar_to_grid(df, out_dir, frames_to_accumulate=10) #10 frames * 40ms = 400ms temporal window

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "measurements"))
    args = parser.parse_args()
    
    run_batch_grid_mapping(args.root_dir)