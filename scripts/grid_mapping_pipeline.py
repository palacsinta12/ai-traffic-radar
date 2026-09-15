import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

from config import (
    IMG_H,
    IMG_W,
    RESOLUTION,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    BEV_IMAGE_HEIGHT,
    BEV_IMAGE_WIDTH,
    BEV_PAD_BOTTOM,
    BEV_PAD_RIGHT,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def point_cloud_to_bev(accumulated_points_df: pd.DataFrame) -> np.ndarray:
    roi_pts = accumulated_points_df[
        (accumulated_points_df['RealXData'] >= X_MIN) & (accumulated_points_df['RealXData'] <= X_MAX) &
        (accumulated_points_df['RealYData'] >= Y_MIN) & (accumulated_points_df['RealYData'] <= Y_MAX)
    ].copy()

    grid = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)

    if roi_pts.empty:
        # Base background state: neutral velocity (127) for zero-Doppler areas
        blank = np.zeros((BEV_IMAGE_HEIGHT, BEV_IMAGE_WIDTH, 3), dtype=np.uint8)
        blank[:, :, 2] = 127
        return blank

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

    # Feature Normalization
    ch0 = np.clip(grid[:, :, 0] / 5.0, 0, 1) * 255
    ch1 = np.clip(grid[:, :, 1] / 15.0, 0, 1) * 255
    ch2 = np.clip((grid[:, :, 2] / 5.5) * 127 + 127, 0, 255)
    
    # Mask background during dilation to preserve negative (receding) velocities
    ch2[grid[:, :, 0] == 0] = 0  
    img = np.stack([ch0, ch1, ch2], axis=-1).astype(np.uint8)

    # Spatial Cell Spreading (Morphological Dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ch0_dilated = cv2.dilate(img[:, :, 0], kernel, iterations=1)
    ch1_dilated = cv2.dilate(img[:, :, 1], kernel, iterations=1)
    ch2_dilated = cv2.dilate(img[:, :, 2], kernel, iterations=1)
    
    ch1_masked = np.where(ch0_dilated > 0, ch1_dilated, 0)
    ch2_masked = np.where(ch0_dilated > 0, ch2_dilated, 127) 
    
    img_spread = np.stack([ch0_dilated, ch1_masked, ch2_masked], axis=-1).astype(np.uint8)

    # Apply asymmetric spatial padding to maintain standard tensor dimensions.
    # Background padded with neutral velocity (127) to prevent edge artifacts.
    padded_img = cv2.copyMakeBorder(
        img_spread,
        top=0,
        bottom=BEV_PAD_BOTTOM,
        left=0,
        right=BEV_PAD_RIGHT,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 127],
    )

    return padded_img

def process_radar_to_grid(radar_df: pd.DataFrame, out_dir: Path, frames_to_accumulate: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_clocks = sorted(radar_df['Window Clock'].unique())
    accumulation_buffer = []

    for idx, clock in enumerate(tqdm(unique_clocks, desc="Generating BEV Grids")):
        current_points = radar_df[radar_df['Window Clock'] == clock].copy()
        
        accumulation_buffer.append(current_points)
        if len(accumulation_buffer) > frames_to_accumulate:
            accumulation_buffer.pop(0)
        
        if idx % 5 != 0:
            continue
            
        dense_pc = pd.concat(accumulation_buffer, ignore_index=True)
        padded_bev = point_cloud_to_bev(dense_pc)

        filename = out_dir / f"{int(clock)}.png"
        cv2.imwrite(str(filename), padded_bev)

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
        
        process_radar_to_grid(df, out_dir, frames_to_accumulate=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "measurements"))
    args = parser.parse_args()
    
    run_batch_grid_mapping(args.root_dir)