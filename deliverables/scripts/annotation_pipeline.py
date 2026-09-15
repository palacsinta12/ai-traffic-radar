"""
Road User Detection Pipeline — Traffic-Monitoring Camera
=========================================================

Detects and localises pedestrians, cyclists, and cars in video from a fixed
traffic camera. Optionally maps detections to Bird's-Eye View (BEV) coordinates
via homography.

Pipeline steps
--------------
1. INFERENCE    — Run YOLO ± tracker on video1.avi → detections.csv
2. HOMOGRAPHY   — Map pixel detections to BEV coordinates (cam_X, cam_Y)
                  → columns appended to detections.csv
3. VISUALIZATION — Render detections.csv onto video1.avi → annotated video

Folder structure
----------------
radar_AI_teamName/
├── scripts/
├── models/
├── dataset/  (images/, labels/, data.yaml)
├── docs/
└── measurements/
    ├── reference_points/
    │   ├── REF_PTS_2025.csv        semicolon-delimited, comma decimal (European Excel)
    │   ├── REF_PTS_202602.csv      columns: y;x;u;v  →  (y,x)=BEV [m], (u,v)=pixel
    │   ├── REF_PTS_202603.csv
    │   └── REF_PTS_202605.csv
    └── <meas_name>/                e.g. 20260218-134121_mix
        ├── export/
        │   ├── radar1_resp.csv
        │   └── video1.timestamps
        ├── video1.avi
        ├── detections.csv          (pipeline output)
        └── video1_annot.mp4        (pipeline output)

Typical usage
-------------
# Full run — single measurement:
python scripts/annotation_pipeline.py --run_inference --run_homography --run_visualize --meas_name 20260218-134121_mix
python scripts/annotation_pipeline.py  --run_visualize 

# With tracker:
python scripts/annotation_pipeline.py --run_inference --run_homography --run_visualize --tracker_type botsort --meas_name 20260218-134121_mix

# Batch (all subfolders):
python scripts/annotation_pipeline.py --run_batch --run_inference --run_homography --run_visualize

# Custom model:
python scripts/annotation_pipeline.py --run_inference --run_visualize --meas_name 20260218-134121_mix --model yolo11l.pt --conf 0.20 --imgsz 1280 960
python scripts/annotation_pipeline.py --run_inference --run_homography --run_visualize --meas_name 20260218-134121_mix --model yolo11n.pt

Run with --help for a full list of arguments.
"""

import time
import os
import cv2
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# CONSTANTS
# ==============================================================================
LOG_WIDTH   = 60
PROJECT_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ==============================================================================
# TRACKER CONFIGURATION
# ==============================================================================
def create_custom_tracker(output_path: str, args: argparse.Namespace) -> str:
    """
    Writes a tracker YAML config file for the selected tracker type.

    botsort   : Full config including ReID, GMC, and appearance thresholds.
    bytetrack : Lightweight config using only confidence/buffer/match thresholds.
    """
    config = {
        "tracker_type":      args.tracker_type,
        "track_high_thresh": args.track_high_thresh,
        "track_low_thresh":  args.track_low_thresh,
        "new_track_thresh":  args.new_track_thresh,
        "track_buffer":      args.track_buffer,
        "match_thresh":      args.match_thresh,
    }

    if args.tracker_type == "botsort":
        config.update({
            "fuse_score":        args.fuse_score,
            "gmc_method":        args.gmc_method,
            "with_reid":         args.with_reid,
            "proximity_thresh":  args.proximity_thresh,
            "appearance_thresh": args.appearance_thresh,
            "model":             args.reid_model,   # ReID model weights (not the YOLO model)
        })
    elif args.tracker_type == "bytetrack":
        pass  # no additional fields needed
    else:
        logging.warning(
            f"[TRACKER] Unknown tracker_type '{args.tracker_type}' — "
            "writing shared fields only. Check YOLO docs for required fields."
        )

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.info(f"[TRACKER] '{args.tracker_type}' config written → '{output_path}'")
    return output_path


# ==============================================================================
# [STUDENT TASK 1]  MODEL & INFERENCE PARAMETER SELECTION
# ------------------------------------------------------------------------------
# Compare at least three candidate COCO-pretrained YOLO architectures
# (e.g., YOLOv8, YOLOv11, YOLOv12) and record their parameter counts and
# expected trade-offs in the evaluation table before committing to one.
# Apply the selected model to the camera recordings to detect road users,
# tuning inference parameters such as confidence threshold, test-time
# augmentation, and multi-object tracker settings. Justify your final model
# and parameter selection, and document detection performance.
#
# Steps:
#   1. Test at least three COCO-pretrained YOLO variants (different versions
#      and/or sizes). For each, record parameter count, inference speed, and
#      qualitative detection quality on the measurement videos.
#   2. Run the pipeline in both predict mode and track mode (botsort /
#      bytetrack). Compare ID consistency across frames.
#   3. Systematically vary key inference and tracker settings (conf, imgsz,
#      iou, track_high_thresh, etc.) and observe the effect on detections.
#
# Report:
#   - Chosen model and justification (accuracy vs. speed trade-off)
#   - Predict vs. track mode decision and rationale
#   - Which parameters had the largest effect and in which direction
#     (more/fewer detections, more/fewer false positives)
#   - Final chosen parameter values and the reason for each
# ==============================================================================
# STEP 1 — INFERENCE
# ==============================================================================
def run_inference(measurement_folder, model: YOLO, tracker_yaml: str | None, args: argparse.Namespace):
    """
    Runs YOLO on the source video and saves detections to detections.csv.
    Handles device allocation and prints model architecture statistics for Task 1.
    """
    # Setup paths and timing for the current measurement folder
    start_time = time.time()
    folder_path = Path(measurement_folder)
    measurement_name = folder_path.name
    video_path = folder_path / args.source
    csv_path = folder_path / args.out_csv
    
    # Determine if we are tracking or just predicting based on args
    use_tracker = args.tracker_type is not None
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    batch = args.batch if args.batch is not None else 1
    
    logging.info("-" * LOG_WIDTH)
    logging.info(f"[INFERENCE] Processing: {measurement_name}")
    logging.info(f"[INFERENCE] Mode: {'track (' + args.tracker_type + ')' if use_tracker else 'predict (no tracker)'}")
    
    # Check CUDA availability and print GPU VRAM to ensure we don't OOM
    _cuda = torch.cuda.is_available()
    if _cuda:
        _dev_idx = int(device) if str(device).isdigit() else 0
        _prop = torch.cuda.get_device_properties(_dev_idx)
        logging.info(f"[DEVICE] GPU: {_prop.name} | VRAM: {_prop.total_memory / 1e9:.1f} GB | batch={batch}")

    if not video_path.exists():
        logging.warning(f"[INFERENCE] Skipping: {args.source} not found.")
        return

    # --- TASK 1 AUTOMATION: Force YOLO to print Params & FLOPs ---
    logging.info(f"\n[TASK 1 INFO] Architecture details for {args.model}:")
    model.info(verbose=True)  # Prints layers, parameters, and GFLOPs directly to console
    logging.info("\n")

    # Quickly read the video metadata to setup the progress bar
    cap_temp = cv2.VideoCapture(str(video_path))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    # Configure YOLO parameters, prioritizing recall (lower conf/iou) for radar fusion
    infer_kwargs = dict(
        source=str(video_path), stream=True, verbose=False,
        classes=args.classes, conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, rect=args.rect, augment=args.augment,
        device=device, half=args.half, batch=batch,
    )

    # Initialize the generator (Track mode assigns persistent IDs across frames)
    if use_tracker:
        results_gen = model.track(tracker=tracker_yaml, **infer_kwargs)
    else:
        results_gen = model.predict(**infer_kwargs)

    detections = []
    # Process the video frame-by-frame as the YOLO generator yields results
    for frame_idx, result in tqdm(enumerate(results_gen), total=total_frames, desc="Inference"):
        if result.boxes is not None:
            # Extract bounding boxes, confidence, and class IDs
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                
                # Assign track_id if available, otherwise mark as untracked (-2)
                track_id = int(box.id[0]) if (use_tracker and box.id is not None) else -2
                x_c, y_c, w, h = box.xywh[0].tolist()

                # Append to our local list for CSV export
                detections.append({
                    "frame": frame_idx, "track_id": track_id,
                    "class": model.names[cls_id], "cls_id": cls_id,
                    "conf": conf_val, "x_c": x_c, "y_c": y_c, "w": w, "h": h
                })
        # Release tensors to prevent VRAM memory leaks during long videos
        result.cpu() 

    # Calculate actual processing FPS (important for the <40ms inference constraint)
    duration = time.time() - start_time
    fps = total_frames / duration if duration > 0 else 0
    
    logging.info(f"[INFERENCE] Done in {duration:.2f}s ({fps:.1f} FPS) — {len(detections)} detections")
    if detections:
        pd.DataFrame(detections).to_csv(csv_path, index=False)
# ==============================================================================
# HOMOGRAPHY HELPERS
# ==============================================================================
def load_reference_points(file_path: str) -> pd.DataFrame:
    """
    Loads reference point correspondences from a CSV file.
    Dynamically scans the first data row to auto-detect both the column 
    delimiter and the decimal separator to prevent parsing/coercion errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reference points file not found: {file_path}")

    file_name = Path(file_path).name
    logging.info("-" * LOG_WIDTH)
    logging.info(f"[HOMOGRAPHY] Loading reference points from {file_name}")

    # --- ADVANCED AUTO-DETECTION ---
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(3)]
    
    # 1. Detect column separator
    sep = ";" if ";" in lines[0] else ","
    
    # 2. Detect decimal separator by scanning the first data row
    decimal = "."
    if len(lines) > 1:
        data_row = lines[1]
        # If there is a comma in the data row and we are using semicolon delimiter,
        # it is highly likely a European comma-decimal
        if "," in data_row and sep == ";":
            decimal = ","

    logging.info(f"[FORMAT] {file_name} -> Delimiter: '{sep}' | Decimal: '{decimal}'")

    df = pd.read_csv(file_path, sep=sep, decimal=decimal)
    df.columns = df.columns.str.strip()

    req_cols = ["y", "x", "u", "v"]
    missing  = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_name}: {missing}. "
            f"Found: {list(df.columns)}"
        )

    for c in req_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If this drops to 0, our parser failed
    df = df.dropna(subset=req_cols).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"Failed to parse any valid reference points in {file_name}. Check separators.")

    display_cols = (["id"] + req_cols) if "id" in df.columns else req_cols
    logging.info(f"Loaded {len(df)} reference points:\n{df[display_cols]}")
    logging.info("-" * LOG_WIDTH)

    return df.rename(columns={"y": "y_m", "x": "x_m"})[["x_m", "y_m", "u", "v"]]

def compute_homography(ref: pd.DataFrame, args: argparse.Namespace):
    """
    Computes homography mapping pixel (u, v) → BEV world (x_m, y_m).
    Evaluates both DLT and RANSAC errors to populate the student report.
    """
    # Convert pandas columns to numpy arrays for OpenCV processing
    src = ref[["u", "v"]].to_numpy(dtype=np.float64)
    dst = ref[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    
    # Pad source points with ones to convert them to homogeneous coordinates
    src_hom = np.hstack([src, np.ones((src.shape[0], 1))])

    # 1. Calculate standard DLT (method=0) and its reprojection error
    H_dlt, _ = cv2.findHomography(src, dst, method=0)
    err_dlt = float('inf')
    if H_dlt is not None:
        # Project source pixels through the DLT matrix and calculate Euclidean distance
        pred_dlt_hom = (H_dlt @ src_hom.T).T
        pred_dlt = pred_dlt_hom[:, :2] / pred_dlt_hom[:, 2:]
        err_dlt = np.mean(np.linalg.norm(dst - pred_dlt, axis=1))

    # 2. Calculate RANSAC and its reprojection error
    H_ran, inliers_ran = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=args.ransac_thresh)
    err_ran = float('inf')
    inliers_count = 0
    if H_ran is not None:
        # Project source pixels through the RANSAC matrix and calculate Euclidean distance
        pred_ran_hom = (H_ran @ src_hom.T).T
        pred_ran = pred_ran_hom[:, :2] / pred_ran_hom[:, 2:]
        err_ran = np.mean(np.linalg.norm(dst - pred_ran, axis=1))
        inliers_count = int(inliers_ran.sum()) if inliers_ran is not None else 0

    # Log the mathematical comparison for the lab report
    logging.info(f"\n--- [TASK 2] HOMOGRAPHY STATS ---")
    logging.info(f"Total Reference Points: {len(src)}")
    logging.info(f"DLT Method    -> Mean Reproj Error: {err_dlt:.4f} meters")
    logging.info(f"RANSAC Method -> Mean Reproj Error: {err_ran:.4f} meters | Inliers: {inliers_count}/{len(src)}")
    
    # Select the final matrix based on CLI arguments
    H = H_ran if args.use_ransac else H_dlt
    if H is None:
        raise RuntimeError("findHomography failed — check point correspondences.")
    return H, (inliers_ran if args.use_ransac else None)

def apply_homography_to_csv(csv_path: Path, H: np.ndarray) -> None:
    """
    Enforces the 70m (Length) x 20m (Width) roadway corridor.
    """
    df = pd.read_csv(csv_path)
    if df.empty: return

    # Project the bottom-center of the bbox (contact point with road)
    u  = df["x_c"].to_numpy(dtype=np.float32)
    v  = (df["y_c"] + df["h"].astype(float) / 2.0).to_numpy(dtype=np.float32)
    uv = np.column_stack((u, v)).reshape(-1, 1, 2)
    
    # Apply transformation
    xy_bev = cv2.perspectiveTransform(uv, H).reshape(-1, 2)

    # Based on REF_PTS: cam_X is lateral (width), cam_Y is longitudinal (distance)
    df["cam_X"] = xy_bev[:, 0]
    df["cam_Y"] = xy_bev[:, 1]

    # --- RIGOROUS SPATIAL MASK ---
    # Width (cam_X): -10m to +10m (Total 20m)
    # Distance (cam_Y): 0m to 70m
    initial_count = len(df)
    
    # We use a 2-meter buffer (safety margin) for the mask
    df = df[
        (df["cam_X"] >= -12) & (df["cam_X"] <= 12) & 
        (df["cam_Y"] >= -2)  & (df["cam_Y"] <= 72)
    ]
    
    removed = initial_count - len(df)
    if removed > 0:
        logging.info(f"[SPATIAL MASK] Deleted {removed} off-road detections. Remaining: {len(df)}")
        
    df.to_csv(csv_path, index=False)


def _select_ref_file(meas_name: str, ref_points_dir: str) -> str | None:
    """Returns the correct reference CSV path based on measurement name prefix."""
    prefix_map = {
        "202603": "REF_PTS_202603.csv",
        "202602": "REF_PTS_202602.csv",
        "202605": "REF_PTS_202605.csv",
        "2025":   "REF_PTS_2025.csv",
    }
    for prefix, filename in prefix_map.items():
        if meas_name.startswith(prefix):
            return os.path.join(ref_points_dir, filename)
    return None


# ==============================================================================
# PARKED CAR FILTER
# ==============================================================================
# Lab-specific calibration data: known parked-car pixel centres for each
# measurement period at the FETI traffic camera location.
# Detections within PIXEL_TOL pixels of these positions are removed before
# visualisation to suppress false positives from stationary vehicles.
#
# NOTE: These coordinates are specific to this camera installation.
#       If you move the camera or use a different measurement site,
#       update or clear this dictionary.
_PARKED_CARS = {
    "2025":   [(538, 345), (588, 419), (373, 188)],
    "202602": [(585, 404), (496, 295), (380, 198)],
    "202603": [(592, 427), (378, 202)],
} 
_PIXEL_TOL = 10


def filter_parked_cars(df: pd.DataFrame, meas_name: str) -> pd.DataFrame:
    """Removes detections whose bbox centre falls within ±PIXEL_TOL px of any known parked-car position."""
    parked_centers = next(
        (v for k, v in _PARKED_CARS.items() if meas_name.startswith(k)),
        []
    )
    if not parked_centers:
        return df

    mask_remove = pd.Series(False, index=df.index)
    for (pu, pv) in parked_centers:
        mask_remove |= (
            (df["x_c"].between(pu - _PIXEL_TOL, pu + _PIXEL_TOL)) &
            (df["y_c"].between(pv - _PIXEL_TOL, pv + _PIXEL_TOL))
        )

    n = mask_remove.sum()
    logging.info(f"[PARKED CAR FILTER] Removed {n} detections near parked car positions.")
    return df[~mask_remove].reset_index(drop=True)


# ==============================================================================
# [STUDENT TASK 3]  CYCLIST DETECTION MERGE
# ------------------------------------------------------------------------------
# YOLO detects the bicycle frame (cls_id=1, 'bic') and the rider (cls_id=0,
# 'ped') as two separate bounding boxes on the same frame. For radar fusion
# and dataset labelling, a single 'cyclist' detection enclosing both is needed.
# ==============================================================================
def merge_cyclist_detections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Pedestrians (0) and Bicycles (1) into 'Cyclist' (3) using 
    generous 2D camera-space overlap to bypass BEV perspective distortion.
    """
    if df.empty: return df
    
    frames = df['frame'].unique()
    new_rows, drop_indices = [], []

    for f in frames:
        f_df = df[df['frame'] == f]
        peds = f_df[f_df['cls_id'] == 0]
        bics = f_df[f_df['cls_id'] == 1]
        
        if peds.empty or bics.empty: continue
            
        matched_peds = set()

        for b_idx, bic in bics.iterrows():
            # Expand the bicycle camera box by 50% to catch nearby pedestrians
            pad_x = bic['w'] * 0.5
            pad_y = bic['h'] * 0.5
            bx_min = bic['x_c'] - (bic['w']/2) - pad_x
            bx_max = bic['x_c'] + (bic['w']/2) + pad_x
            by_min = bic['y_c'] - (bic['h']/2) - pad_y
            by_max = bic['y_c'] + (bic['h']/2) + pad_y
            
            for p_idx, ped in peds.iterrows():
                if p_idx in matched_peds: continue
                
                # If the pedestrian's center falls inside this expanded bike zone
                if (bx_min <= ped['x_c'] <= bx_max) and (by_min <= ped['y_c'] <= by_max):
                    
                    u_x1 = min(bic['x_c'] - bic['w']/2, ped['x_c'] - ped['w']/2)
                    u_y1 = min(bic['y_c'] - bic['h']/2, ped['y_c'] - ped['h']/2)
                    u_x2 = max(bic['x_c'] + bic['w']/2, ped['x_c'] + ped['w']/2)
                    u_y2 = max(bic['y_c'] + bic['h']/2, ped['y_c'] + ped['h']/2)
                    
                    new_row = bic.copy()
                    new_row['cls_id'] = 3
                    new_row['class']  = 'cyclist'
                    new_row['x_c'] = (u_x1 + u_x2) / 2
                    new_row['y_c'] = (u_y1 + u_y2) / 2
                    new_row['w']   = u_x2 - u_x1
                    new_row['h']   = u_y2 - u_y1
                    new_row['conf'] = max(bic['conf'], ped['conf'])
                    new_row['track_id'] = ped['track_id']
                    
                    if 'cam_X' in df.columns:
                        new_row['cam_X'] = (bic['cam_X'] + ped['cam_X']) / 2
                        new_row['cam_Y'] = (bic['cam_Y'] + ped['cam_Y']) / 2
                    
                    new_rows.append(new_row)
                    drop_indices.extend([b_idx, p_idx])
                    matched_peds.add(p_idx)
                    break 

    df = df.drop(index=list(set(drop_indices)))
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    return df.sort_values(by=['frame']).reset_index(drop=True)
# ==============================================================================
# STEP 2 — VISUALIZATION
# ==============================================================================

def _overlaps(rl, rt, rr, rb, zones):
    """Return True if rectangle (rl,rt,rr,rb) overlaps any rectangle in zones."""
    return any(
        min(rr, z[2]) - max(rl, z[0]) > 0 and
        min(rb, z[3]) - max(rt, z[1]) > 0
        for z in zones
    )


def _place(tx, rb, block_w, block_h, pad):
    """Given left anchor tx and desired rect_bottom rb, return full rect."""
    return (tx - pad, rb - block_h, tx - pad + block_w, rb)


_CLS_COLOR = {0: (0, 255, 0), 1: (255, 0, 255), 2: (0, 128, 255), 3: (0, 200, 255)}
# 0=ped: green, 1=bic: magenta, 2=car: orange, 3=cyclist: yellow-green
_CLS_NAME  = {0: "ped",       1: "bic",          2: "car",          3: "cyclist"}


def run_visualization(measurement_folder, args: argparse.Namespace):
    """
    Renders bounding boxes and labels from detections.csv onto the source video.
    Applies parked-car filter. Shows cam_X/cam_Y BEV labels if present in CSV (--run_homography).
    """
    start_time       = time.time()
    folder_path      = Path(measurement_folder)
    measurement_name = folder_path.name
    video_path       = folder_path / args.source
    csv_path         = folder_path / args.out_csv
    out_video_path   = folder_path / args.out_video

    logging.info("-" * LOG_WIDTH)
    logging.info(f"[VISUALIZATION] Processing: {measurement_name}")

    if not video_path.exists():
        logging.warning(f"[VISUALIZATION] Skipping: {args.source} not found in {measurement_name}")
        return
    if not csv_path.exists():
        logging.warning(f"[VISUALIZATION] Skipping: {args.out_csv} not found — run inference first.")
        return

    # --- Load & filter detections ---
    df = pd.read_csv(csv_path)
    df = filter_parked_cars(df, folder_path.name)
    df = merge_cyclist_detections(df)

    parked_centers = next((v for k, v in _PARKED_CARS.items() if folder_path.name.startswith(k)), [])
    df["is_parked"] = False
    for (pu, pv) in parked_centers:
        df.loc[(df["x_c"].between(pu - _PIXEL_TOL, pu + _PIXEL_TOL)) & 
               (df["y_c"].between(pv - _PIXEL_TOL, pv + _PIXEL_TOL)), "is_parked"] = True
        

    detections_by_frame = {idx: grp for idx, grp in df.groupby("frame")}

    has_bev = ("cam_X" in df.columns) and ("cam_Y" in df.columns)
    if not has_bev:
        logging.info("[VISUALIZATION] cam_X/cam_Y columns not found — BEV labels will not be shown. Run with --run_homography first.")
    
    # --- Video I/O ---
    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    logging.info(f"[VISUALIZATION] {width}x{height} @ {fps:.2f} FPS")

    font           = cv2.FONT_HERSHEY_SIMPLEX
    font_scale     = 0.30
    font_thickness = 1
    pad            = 2


    for frame_idx in tqdm(range(total_frames), desc=f"Visualize: {measurement_name[:15]}"):
        ret, frame = cap.read()
        if not ret:
            break

        # ── Pass 1: compute label layout for all detections ──────────────────
        frame_dets = detections_by_frame.get(frame_idx, pd.DataFrame())
        label_data = []   # (x1,y1,x2,y2, color, line1, line2, rect_*, text_x, text_y1, th2)

        # Pre-populate occupied zones with all bbox rectangles (to avoid label–bbox overlap)
        # Each entry is (left, top, right, bottom)
        bbox_zones = [
            (int(r["x_c"] - r["w"] / 2), int(r["y_c"] - r["h"] / 2),
             int(r["x_c"] + r["w"] / 2), int(r["y_c"] + r["h"] / 2))
            for _, r in frame_dets.iterrows()
        ]
        drawn_labels = list(bbox_zones)  # start with all bboxes as occupied

        for det_idx, (_, row) in enumerate(frame_dets.iterrows()):
            cls_id   = int(row["cls_id"])
            conf     = float(row["conf"])
            track_id = int(row["track_id"])
            x_c, y_c, w, h = float(row["x_c"]), float(row["y_c"]), float(row["w"]), float(row["h"])

            x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
            x2, y2 = int(x_c + w / 2), int(y_c + h / 2)

            color    = _CLS_COLOR.get(cls_id, (0, 128, 255))
            cls_name = _CLS_NAME.get(cls_id, "car")

            id_part = "" if track_id == -2 else f"ID:{track_id} "
            line1 = f"{id_part}{cls_name} {conf:.2f}"
            line2 = f"X:{float(row['cam_X']):.1f} Y:{float(row['cam_Y']):.1f}" if has_bev else None

            (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, font_thickness)
            (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, font_thickness) if line2 else ((0, 0), None)

            line_gap = 4
            block_w  = max(tw1, tw2) + 2 * pad
            block_h  = th1 + (line_gap + th2 if line2 else 0) + 2 * pad

            # All occupied zones (including own bbox — label must not overlap ANY bbox)
            other_zones = drawn_labels

            # Default: label bottom flush with bbox top (rect_bottom = y1 - 1)
            default_tx = max(pad, min(x1, width - block_w - pad))
            default_rb = y1 - 1   # label sits just above bbox, no overlap

            rl, rt, rr, rb = _place(default_tx, default_rb, block_w, block_h, pad)
            was_pushed = False

            if rt >= 0 and not _overlaps(rl, rt, rr, rb, other_zones):
                # No collision — place directly above bbox, no leader line
                text_x  = default_tx
                rect_left, rect_top, rect_right, rect_bottom = rl, rt, rr, rb
                text_y1 = rect_bottom - (line_gap + th2 + pad if line2 else pad)

            else:
                # Try right side: anchor to x2
                right_tx = max(pad, min(x2, width - block_w - pad))
                rl2, rt2, rr2, rb2 = _place(right_tx, default_rb, block_w, block_h, pad)
                if rt2 >= 0 and not _overlaps(rl2, rt2, rr2, rb2, other_zones):
                    text_x  = right_tx
                    rect_left, rect_top, rect_right, rect_bottom = rl2, rt2, rr2, rb2
                    text_y1 = rect_bottom - (line_gap + th2 + pad if line2 else pad)
                    was_pushed = True
                else:
                    # Fall back: push upward from default position
                    text_x  = default_tx
                    cur_rb  = default_rb
                    while True:
                        rl, rt, rr, rb = _place(text_x, cur_rb, block_w, block_h, pad)
                        if rt < 0:
                            cur_rb = block_h
                            rl, rt, rr, rb = _place(text_x, cur_rb, block_w, block_h, pad)
                            drawn_labels.append((rl, rt, rr, rb))
                            was_pushed = True
                            break
                        if not _overlaps(rl, rt, rr, rb, other_zones):
                            drawn_labels.append((rl, rt, rr, rb))
                            was_pushed = True
                            break
                        cur_rb -= (block_h + 1)
                    rect_left, rect_top, rect_right, rect_bottom = rl, rt, rr, rb
                    text_y1 = rect_bottom - (line_gap + th2 + pad if line2 else pad)

            drawn_labels.append((rect_left, rect_top, rect_right, rect_bottom))
            label_data.append((x1, y1, x2, y2, color, line1, line2,
                                rect_left, rect_top, rect_right, rect_bottom,
                                text_x, text_y1, th2, line_gap, was_pushed))

        # ── Pass 2: draw all bounding boxes ──────────────────────────────────
        for (x1, y1, x2, y2, color, *_) in label_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # ── Pass 3: draw leader lines (behind labels) ─────────────────────────
        for (x1, y1, x2, y2, color, line1, line2,
             rect_left, rect_top, rect_right, rect_bottom,
             text_x, text_y1, th2, line_gap, was_pushed) in label_data:
            if was_pushed:
                label_cx = (rect_left + rect_right) // 2
                anchor_x = x1 if abs(x1 - label_cx) <= abs(x2 - label_cx) else x2
                cv2.line(frame, (anchor_x, y1), (label_cx, rect_bottom), color, 1, cv2.LINE_AA)

        # ── Pass 4: draw all label backgrounds + text on top ─────────────────
        for (x1, y1, x2, y2, color, line1, line2,
             rect_left, rect_top, rect_right, rect_bottom,
             text_x, text_y1, th2, line_gap, was_pushed) in label_data:

            # Semi-transparent dark background
            overlay = frame.copy()
            cv2.rectangle(overlay, (rect_left, rect_top), (rect_right, rect_bottom), (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            # Text with outline
            for text, ty in [(line1, text_y1)] + ([(line2, text_y1 + line_gap + th2)] if line2 else []):
                cv2.putText(frame, text, (text_x + pad, ty),
                            font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, text, (text_x + pad, ty),
                            font, font_scale, color, font_thickness, cv2.LINE_AA)

        # Measurement name — top-left, small dark badge
        (_mw, _mh), _ml = cv2.getTextSize(measurement_name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (6, 8), (6 + _mw + 8, 8 + _mh + _ml + 2), (20, 20, 20), -1)
        cv2.putText(frame, measurement_name, (10, 8 + _mh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    duration = time.time() - start_time
    logging.info(f"[VISUALIZATION] Done in {duration:.2f}s → {out_video_path}")
    logging.info("-" * LOG_WIDTH)


# ==============================================================================
# BATCH WRAPPER
# ==============================================================================
def run_batch(root_data_dir: str, model: YOLO | None, tracker_yaml: str | None,
              ref_points_dir: str, args: argparse.Namespace):
    """
    Runs configured pipeline steps on every measurement subfolder.
    Checks args.allow_skip to determine whether to overwrite existing data.
    """
    root = Path(root_data_dir)
    measurement_folders = sorted(
        f for f in root.iterdir()
        if f.is_dir() and f.name not in ["reference_points", "ref_pts"]
    )
    
    if not measurement_folders:
        logging.warning(f"[BATCH] No subfolders found in {root_data_dir}")
        return

    for folder in tqdm(measurement_folders, desc="Batch Progress", position=0):
        # 1. INFERENCE
        if args.run_inference:
            csv_path = folder / args.out_csv
            if args.allow_skip and csv_path.exists():
                logging.info(f"[BATCH] Skipping Inference for {folder.name} (Found {args.out_csv})")
            else:
                run_inference(folder, model, tracker_yaml, args)

        # 2. HOMOGRAPHY
        if args.run_homography:
            # We check for the output column 'cam_X' in the CSV to see if homography was already done
            skip_homography = False
            if args.allow_skip and (folder / args.out_csv).exists():
                temp_df = pd.read_csv(folder / args.out_csv, nrows=1)
                if "cam_X" in temp_df.columns:
                    skip_homography = True

            if skip_homography:
                logging.info(f"[BATCH] Skipping Homography for {folder.name} (Already applied)")
            else:
                ref_file = _select_ref_file(folder.name, ref_points_dir)
                if ref_file:
                    try:
                        ref_df = load_reference_points(ref_file)
                        H, _   = compute_homography(ref_df, args)
                        apply_homography_to_csv(folder / args.out_csv, H)
                    except Exception as e:
                        logging.error(f"[HOMOGRAPHY] Failed for {folder.name}: {e}")

        # 3. VISUALIZATION
        if args.run_visualize:
            video_out = folder / args.out_video
            if args.allow_skip and video_out.exists():
                logging.info(f"[BATCH] Skipping Visualization for {folder.name} (Found {args.out_video})")
            else:
                run_visualization(folder, args)


# ==============================================================================
# CLI
# ==============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Road User Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Pipeline switches ---
    sw = p.add_argument_group("Pipeline Switches")
    sw.add_argument("--run_inference",  action="store_true", help="Run YOLO inference (skipped if CSV exists)")
    sw.add_argument("--run_homography", action="store_true", help="Compute homography and map detections to BEV")
    sw.add_argument("--run_visualize",  action="store_true", help="Render annotated output video")
    sw.add_argument("--run_batch",      action="store_true", help="Process all subfolders in root_dir")
    sw.add_argument("--logging",        type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    sw.add_argument("--allow_skip", action="store_true", 
                    help="If set, skips tasks where the output file already exists. "
                         "Default is False (overwrite everything for consistency).")
    # --- Paths ---
    pa = p.add_argument_group("Paths & Filenames")
    pa.add_argument("--root_dir",       type=str, default=str(PROJECT_DIR / "measurements"),
                    help="Root directory containing measurement subfolders")
    pa.add_argument("--meas_name",      type=str, default="20260218-134121_mix",
                    help="Single measurement folder name inside root_dir")
    pa.add_argument("--source",         type=str, default="video1.avi",    help="Input video filename")
    pa.add_argument("--out_video",      type=str, default="video1_annot.mp4", help="Output annotated video filename")
    pa.add_argument("--out_csv",        type=str, default="detections.csv",   help="Output detections CSV filename")
    pa.add_argument("--tracker_yaml",   type=str, default=None,
                    help="Tracker YAML path (default: <script_dir>/custom_tracker.yaml)")
    pa.add_argument("--ref_points_dir", type=str, default=None,
                    help="Reference points folder (default: <root_dir>/reference_points)")

    # --- YOLO / Inference ---
    yi = p.add_argument_group("YOLO / Inference Settings")
    yi.add_argument("--model",   type=str,   default="yolo11m.pt",  help="YOLO model path or name")
    yi.add_argument("--conf",    type=float, default=0.25,          help="Confidence threshold")
    yi.add_argument("--iou",     type=float, default=0.7,           help="NMS IoU threshold")
    yi.add_argument("--imgsz",   type=int,   nargs="+", default=[640], metavar="N",
                    help="Inference image size: single int or H W pair")
    yi.add_argument("--device",  type=str,   default=None,          help="Device: cuda / cpu / index (default: 0 if CUDA available, else cpu)")
    yi.add_argument("--augment", action="store_true",               help="Test-Time Augmentation (TTA)")
    yi.add_argument("--rect",    action="store_true",               help="Rectangular inference (preserves aspect ratio)")
    yi.add_argument("--half",    action=argparse.BooleanOptionalAction,
                    default=False,
                    help="FP16 half-precision inference (GPU only). Default: False. Enable with --half.")
    yi.add_argument("--batch",   type=int,   default=None,
                    help="Inference batch size. Default: 1. Override with --batch N.")
    yi.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2],
                    help="COCO class IDs to detect (0=person 1=bicycle 2=car)")

    # --- Tracker ---
    tr = p.add_argument_group("Tracker (omit --tracker_type to use predict mode without tracking)")
    tr.add_argument("--tracker_type",      type=str,   default=None,
                    help="Tracker: 'botsort' or 'bytetrack'. If not set, runs in predict mode (track_id = -2).")
    tr.add_argument("--track_high_thresh", type=float, default=0.5,   help="High-confidence detection threshold")
    tr.add_argument("--track_low_thresh",  type=float, default=0.1,   help="Low-confidence detection threshold")
    tr.add_argument("--new_track_thresh",  type=float, default=0.6,   help="Threshold to initialise a new track")
    tr.add_argument("--track_buffer",      type=int,   default=30,    help="Frames to keep a lost track alive")
    tr.add_argument("--match_thresh",      type=float, default=0.8,   help="IoU threshold for track-detection matching")
    # BoT-SORT only
    tr.add_argument("--fuse_score",        action="store_true", default=False, help="[botsort] Fuse detection score into matching")
    tr.add_argument("--gmc_method",        type=str,   default="sparseOptFlow", help="[botsort] Global motion compensation method")
    tr.add_argument("--with_reid",         action="store_true", default=False,  help="[botsort] Enable ReID appearance features")
    tr.add_argument("--proximity_thresh",  type=float, default=0.5,   help="[botsort] Proximity threshold for ReID")
    tr.add_argument("--appearance_thresh", type=float, default=0.25,  help="[botsort] Appearance similarity threshold")
    tr.add_argument("--reid_model",        type=str,   default="auto", help="[botsort] ReID model filename")

    # --- Homography ---
    ho = p.add_argument_group("Homography Settings")
    ho.add_argument("--use_ransac",    action=argparse.BooleanOptionalAction, default=True, help="Use RANSAC for robust homography estimation (disable with --no-use-ransac)")
    ho.add_argument("--ransac_thresh", type=float, default=3.0,           help="RANSAC reprojection threshold in pixels")

    return p.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()

    logging.getLogger().setLevel(getattr(logging, args.logging))

    # --- Resolve late-bound path defaults ---
    TRACKER_YAML_PATH = args.tracker_yaml   or str(PROJECT_DIR / "custom_tracker.yaml")
    REF_POINTS_DIR    = args.ref_points_dir or str(Path(args.root_dir) / "reference_points")

    # --- Build tracker YAML (only when a tracker is selected) ---
    if args.tracker_type is not None:
        create_custom_tracker(output_path=TRACKER_YAML_PATH, args=args)
        tracker_yaml = TRACKER_YAML_PATH
    else:
        tracker_yaml = None

    # --- Load model if inference is needed ---
    yolo_model = None
    if args.run_inference:
        logging.info(f"[INIT] Loading YOLO model: {args.model} ...")
        yolo_model = YOLO(args.model)

    logging.info("=" * LOG_WIDTH)

    if args.run_batch:
        logging.info("Starting Batch Processing ...")
        run_batch(
            root_data_dir  = args.root_dir,
            model          = yolo_model,
            tracker_yaml   = tracker_yaml,
            ref_points_dir = REF_POINTS_DIR,
            args           = args,
        )
        logging.info("Batch Processing Complete.")

    else:
        logging.info(f"Starting Single Measurement Run: {args.meas_name}")
        test_folder = Path(args.root_dir) / args.meas_name

        if args.run_inference:
            run_inference(test_folder, yolo_model, tracker_yaml, args)

        if args.run_homography:
            ref_file = _select_ref_file(args.meas_name, REF_POINTS_DIR)
            if ref_file is None:
                raise ValueError(
                    f"Cannot determine reference year from meas_name='{args.meas_name}'. "
                    "Name must start with '2025', '202602', or '202603'."
                )
            ref_df = load_reference_points(ref_file)
            H, _   = compute_homography(ref_df, args)
            apply_homography_to_csv(test_folder / args.out_csv, H)

        if args.run_visualize:
            run_visualization(test_folder, args)

        logging.info("Single Run Complete.")

    logging.info("=" * LOG_WIDTH)