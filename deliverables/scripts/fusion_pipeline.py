"""
Radar–Camera Fusion Pipeline
============================

Synchronises radar scans and camera frames by timestamp, then generates a
side-by-side visualisation: camera panel | radar BEV tile | range scale.

detections.csv (from annotation_pipeline.py) is optional — when absent the
visualiser still renders every synchronised scan (camera + radar only).

See annotation_pipeline.py for the expected folder structure.

Typical usage
-------------
# Single measurement, annotated video panel:
python scripts/fusion_pipeline.py
    --meas_name 20260218-134121_mix --source_video video1_annot.mp4

# Without detections (raw video):
python scripts/fusion_pipeline.py
    --meas_name 20260218-134121_mix --source_video video1.avi

# Batch run:
python scripts/fusion_pipeline.py --run_batch --source_video video1_annot.mp4

# With debug CSVs:
python scripts/fusion_pipeline.py --meas_name 20260218-134121_mix --debug

Run with --help for a full list of arguments.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# LOGGING & PROJECT DIR
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOG_WIDTH   = 60
PROJECT_DIR = Path(__file__).resolve().parent.parent


# ==============================================================================
# 1.  DATA LOADING
# ==============================================================================

def load_radar_data(file_path: str) -> pd.DataFrame:
    """
    Load raw radar data from radar1_resp.csv.
    Converts Doppler frequency to radial velocity [m/s]: vel = (freq - 32) × 1.436
    Returns DataFrame['scan_idx', 'radar_ts', 'amp', 'amp_db', 'vel', 'X', 'Y']

    Amplitude note
    --------------
    'amp' is the raw |Ehh| electric field amplitude — a radar analogue of
    image brightness. It is range-dependent (|Ehh|² ∝ 1/R⁴) and affected by
    AGC, so two points with equal amp are not necessarily physically similar.
    'amp_db' (20·log10(amp)) is added for human-readable inspection only.
    Neither column is normalised here — range compensation and per-scan
    normalisation must be applied in the grid-mapping step before use as
    a machine learning feature.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Radar file not found: {file_path}")

    logging.info("-" * LOG_WIDTH)
    logging.info("LOADING: radar data")

    df = pd.read_csv(
        file_path, sep=',',
        usecols=['Window Clock', 'AmpData', 'Frequency', 'RealXData', 'RealYData']
    )
    df.rename(columns={
        'Window Clock': 'radar_ts',
        'AmpData':      'amp',
        'Frequency':    'vel',
        'RealXData':    'X',
        'RealYData':    'Y',
    }, inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df['vel']    = (df['vel'] - 32) * 1.435986161        # Hz -> m/s (positive = approaching)
    df['amp_db'] = 20 * np.log10(df['amp'].clip(lower=1e-9))  # dB, informational only
    df.insert(0, 'scan_idx', df['radar_ts'].factorize(sort=False)[0])

    logging.info(f"Loaded Radar Data: {len(df)} rows, {df['scan_idx'].nunique()} unique scans.")
    logging.info(f"First 5 rows:\n{df.head().to_string()}")
    logging.info("-" * LOG_WIDTH)
    return df


def load_cam_ts(file_path: str, video_path: str) -> pd.DataFrame:
    """
    Load camera timestamps from video1.timestamps and clip to the number of
    actually decodable frames in video1.avi.
    Returns DataFrame['frame', 'cam_ts']
    """
    for p in [file_path, video_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    logging.info("-" * LOG_WIDTH)
    logging.info("LOADING: camera timestamps")

    df = pd.read_csv(file_path, header=None, names=["cam_ts"])
    df["cam_ts"] = pd.to_numeric(df["cam_ts"], errors="coerce")
    df.insert(0, 'frame', range(len(df)))

    # Count decodable frames — use CAP_PROP_FRAME_COUNT for speed (O(1)).
    # If the container reports 0 or a negative value (corrupted header), fall
    # back to decoding every frame (slow but reliable).
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        logging.warning("[CAM TS] CAP_PROP_FRAME_COUNT unreliable — counting frames by decoding (slow).")
        n_frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n_frames += 1
    cap.release()

    df = df.iloc[:n_frames].copy()
    logging.info(f"Loaded {len(df)} camera timestamps ({n_frames} frames in video).")
    logging.info(f"First 5 rows:\n{df.head().to_string()}")
    logging.info("-" * LOG_WIDTH)
    return df


def load_detections(file_path: str) -> pd.DataFrame | None:
    """
    Load YOLO detections from detections.csv (produced by annotation_pipeline.py).
    Returns None if file is absent — pipeline continues without detections.

    Column names on load: x_c→x_center, y_c→y_center, w→width, h→height
    Logs whether cam_X/cam_Y BEV columns are present.
    """
    if not os.path.exists(file_path):
        logging.warning(
            f"Detections file not found: {Path(file_path).name} — "
            "running without detections. Visualiser will show radar + camera only."
        )
        return None

    logging.info("-" * LOG_WIDTH)
    logging.info(f"LOADING: detections from {Path(file_path).name}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required = ['frame', 'track_id', 'class', 'cls_id', 'conf', 'x_c', 'y_c', 'w', 'h']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing} in {Path(file_path).name}.\n"
            f"Columns found: {list(df.columns)}"
        )

    df = df.rename(columns={
        'x_c': 'x_center',
        'y_c': 'y_center',
        'w':   'width',
        'h':   'height',
    })

    has_bev = ('cam_X' in df.columns) and ('cam_Y' in df.columns)
    if has_bev:
        logging.info(
            "BEV columns (cam_X, cam_Y) found in detections.csv — "
            "computed by annotation_pipeline.py --run_homography."
        )
    else:
        logging.info(
            "BEV columns (cam_X, cam_Y) not found in detections.csv — "
            "annotation_pipeline.py was run without --run_homography."
        )

    logging.info(f"Loaded {len(df)} detections across {df['frame'].nunique()} frames.")
    logging.info(f"First 5 rows:\n{df.head().to_string()}")
    logging.info("-" * LOG_WIDTH)
    return df


# ==============================================================================
# 2.  TIME SYNCHRONISATION
# ==============================================================================

def synchronize_time(
    radar_df:      pd.DataFrame,
    detections_df: pd.DataFrame | None,
    cam_ts_df:     pd.DataFrame,
    tolerance_ms:  int = 35,
):
    """
    Match radar scans to camera frames by nearest timestamp within tolerance_ms.

    Returns (sync_map, detections_synched, radar_synched):
      sync_map           — DataFrame['scan_idx', 'frame', 'radar_ts', 'cam_ts', 'dt_ms']
      detections_synched — detections joined to sync_map on frame (empty DF if no detections)
      radar_synched      — radar points whose scan_idx appears in sync_map
    """
    logging.info("-" * LOG_WIDTH)
    logging.info("SYNCHRONIZING: radar & camera timestamps")

    radar_s = (
        radar_df[['scan_idx', 'radar_ts']]
        .drop_duplicates(subset=['scan_idx'])
        .sort_values('radar_ts')
    )
    cam_s = cam_ts_df.sort_values('cam_ts').reset_index(drop=True)

    sync = pd.merge_asof(
        left      = radar_s[['scan_idx', 'radar_ts']],
        right     = cam_s[['frame', 'cam_ts']],
        left_on   = 'radar_ts',
        right_on  = 'cam_ts',
        direction = 'nearest',
        tolerance = tolerance_ms,
    )

    sync['frame']    = sync['frame'].round().astype('Int64')
    sync['cam_ts']   = sync['cam_ts'].round().astype('Int64')
    sync['radar_ts'] = sync['radar_ts'].round().astype('Int64')
    sync['dt_ms']     = sync['cam_ts'] - sync['radar_ts']

    sync_map   = sync.dropna(subset=['frame'])
    match_rate = len(sync_map) / len(radar_s) * 100
    logging.info(
        f"Time Sync: matched {len(sync_map)}/{len(radar_s)} scans "
        f"({match_rate:.1f}%) within {tolerance_ms} ms."
    )

    # Join detections onto sync_map when available
    if detections_df is not None:
        detections_synched = pd.merge(
            detections_df, sync_map[['scan_idx', 'frame']], on='frame', how='inner'
        )
        col = detections_synched.pop('scan_idx')
        detections_synched.insert(0, 'scan_idx', col)
        logging.info(f"Detections after sync: {len(detections_synched)} rows.")
    else:
        # Empty DataFrame so downstream code can safely call .empty without branching
        detections_synched = pd.DataFrame(columns=['scan_idx', 'frame'])
        logging.info("No detections loaded — detections_synched is empty.")

    radar_synched = pd.merge(radar_df, sync_map[['scan_idx']], on='scan_idx', how='inner')

    logging.info(
        f"After sync — radar samples: {len(radar_synched)}, "
        f"detections: {len(detections_synched)}"
    )
    logging.info(f"Sync map (first 5 rows):\n{sync_map.head().to_string()}")
    logging.info("-" * LOG_WIDTH)
    return sync_map, detections_synched, radar_synched


# ==============================================================================
# 3.  VISUALISATION
# ==============================================================================

def render_fusion_video(
    sync_map:          pd.DataFrame,
    radar_synched:     pd.DataFrame,
    video_path:        str,
    output_path:       str,
    fps:               float = 30.0,
    meas_name:         str   = "",
    radar_point_mode:  str   = 'doppler_amp',
    bev_detections:    "pd.DataFrame | None" = None,   # Task 4: BEV box overlay
):
    """
    Render a side-by-side video: camera panel | radar BEV tile | range scale.

    Frame iteration is driven by sync_map so the video is always produced even
    when detections_df_s is empty (no detections.csv was loaded).

    Camera panel
    ------------
    The matched camera frame for each radar scan is shown resized to 640x480.
    No additional labels or bounding boxes are drawn here — when the annotated
    video is used as source those annotations are already baked into the frames.

    Radar BEV tile
    --------------
    Radar points for the current scan are plotted as small blue dots on a white
    120x480 px tile. Road-edge guidelines drawn at x = −3 m and x = +2 m (y = 20–80 m).

    If no camera frame is available for a scan, the previous frame is shown (freeze).
    """
    logging.info("-" * LOG_WIDTH)
    logging.info("GENERATING: visualisation helper video")

    # --- Layout constants ---
    VID_W, VID_H   = 640, 480
    TILE_W, TILE_H = 120, 480
    SCALE_W        = 52
    BORDER         = 3

    # --- Colour palette (BGR) ---
    C_BG        = (18,  18,  18)   # near-black canvas
    C_TILE_BG   = (30,  30,  30)   # dark radar tile background
    C_GRID      = (55,  55,  55)   # subtle grid lines
    C_ROAD_EDGE = (60, 160,  60)   # muted green lane guides
    C_POINT     = (0,  210, 255)   # cyan radar echo
    C_TICK      = (130,130, 130)   # scale tick marks
    C_LABEL     = (200,200, 200)   # scale text
    C_TITLE     = (180,180, 180)   # "Radar BEV" header text


    FINAL_W = BORDER + VID_W + BORDER + TILE_W + BORDER + SCALE_W + BORDER
    FINAL_H = BORDER + VID_H + BORDER

    # --- Radar coordinate bounds ---
    RADAR_Y_MAX = 80.0
    RADAR_X_MIN = -10.0
    RADAR_X_MAX =  10.0
    x_range     = RADAR_X_MAX - RADAR_X_MIN

    cap    = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (FINAL_W, FINAL_H))

    # --- Pre-normalise amp/vel across the whole measurement (stable colour scale) ---
    VEL_MAX = 10.0   # fixed Doppler scale [m/s] — covers slow traffic scenarios
    amp_min = amp_max = 0.0
    if radar_point_mode in ('amp', 'doppler_amp'):
        amp_vals = radar_synched['amp_db'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(amp_vals):
            amp_min = float(amp_vals.quantile(0.05))
            amp_max = float(amp_vals.quantile(0.95))

    def _pt_style(amp_db: float, vel: float):
        """Return (bgr_color, radius) for one radar echo."""
        # Amplitude → radius 1–5 px  +  colour (amp mode only)
        t_amp = 0.0
        if radar_point_mode in ('amp', 'doppler_amp'):
            t_amp = float(np.clip((amp_db - amp_min) / max(amp_max - amp_min, 1e-6), 0.0, 1.0))
            radius = max(1, int(1 + t_amp * 4))
        else:
            radius = 2
        # Colour
        if radar_point_mode == 'amp':
            # weak → blue, strong → red
            b = int(220 * (1 - t_amp) + 60 * t_amp)
            r = int(60  * (1 - t_amp) + 220 * t_amp)
            color = (b, 60, r)           # BGR
        elif radar_point_mode in ('vel', 'doppler_amp'):
            # Velocity → colour  (red = approaching, blue = receding)
            if vel > 0:    # approaching → red
                t = float(np.clip(vel / VEL_MAX, 0.0, 1.0))
                r = 220; g = b = int(60 * (1 - t))
            elif vel < 0:  # receding → dark blue
                t = float(np.clip(abs(vel) / VEL_MAX, 0.0, 1.0))
                b = 220; g = r = int(60 * (1 - t))
            else:
                r = g = b = 120
            color = (b, g, r)            # BGR
        else:
            color = C_POINT
        return color, radius

    # Drive iteration from sync_map so every matched radar scan gets a frame,
    # independent of whether detections were loaded.
    unique_scans  = sorted(sync_map['scan_idx'].unique())
    last_vid_panel = np.zeros((VID_H, VID_W, 3), dtype=np.uint8)

    # Pre-build frame lookup from sync_map for O(1) access inside the loop
    scan_to_frame = sync_map.set_index('scan_idx')['frame'].to_dict()

    for scan_idx in tqdm(unique_scans, desc=f"Visualize: {meas_name[:20]}"):
        scan_radar = radar_synched[radar_synched['scan_idx'] == scan_idx]

        # ── 1. Camera panel ──────────────────────────────────────────────────
        vid_panel = last_vid_panel.copy()
        frame_idx = scan_to_frame.get(scan_idx)

        if frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret:
                vid_panel = cv2.resize(frame, (VID_W, VID_H))
                last_vid_panel = vid_panel.copy()

        # Measurement name — top-left, small dark badge
        if meas_name:
            (_mw, _mh), _ml = cv2.getTextSize(meas_name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(vid_panel, (6, 8), (6 + _mw + 8, 8 + _mh + _ml + 2), (20, 20, 20), -1)
            cv2.putText(vid_panel, meas_name, (10, 8 + _mh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        # Scan badge — top-right, small, dark pill background
        _badge_txt = f"#{scan_idx}"
        (_bw, _bh), _bl = cv2.getTextSize(_badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        _bx = VID_W - _bw - 10;  _by = 10
        cv2.rectangle(vid_panel,
                      (_bx - 4, _by), (_bx + _bw + 4, _by + _bh + _bl + 2),
                      (20, 20, 20), -1)
        cv2.putText(vid_panel, _badge_txt,
                    (_bx, _by + _bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 2. Radar BEV tile ─────────────────────────────────────────────────
        radar_tile = np.full((TILE_H, TILE_W, 3), C_TILE_BG, dtype=np.uint8)

        # Subtle horizontal grid lines every 20 m
        for gm in range(0, int(RADAR_Y_MAX) + 1, 20):
            gy = TILE_H - int((gm / RADAR_Y_MAX) * TILE_H)
            cv2.line(radar_tile, (0, gy), (TILE_W, gy), C_GRID, 1)

        # Ego vehicle axis (x = 0 m)
        ego_px = int(((0.0 - RADAR_X_MIN) / x_range) * TILE_W)
        cv2.line(radar_tile, (ego_px, 0), (ego_px, TILE_H), C_GRID, 1)

        # Road-edge guidelines at x = -3 m and x = +2 m (between y = 10 m and 80 m)
        for road_x in [-3.0, 2.0]:
            px       = int(((road_x - RADAR_X_MIN) / x_range) * TILE_W)
            py_start = TILE_H - int((80.0 / RADAR_Y_MAX) * TILE_H)
            py_end   = TILE_H - int((10.0 / RADAR_Y_MAX) * TILE_H)
            if 0 <= px < TILE_W:
                cv2.line(radar_tile, (px, py_start), (px, py_end), C_ROAD_EDGE, 1)

        for _, pt in scan_radar.iterrows():
            px = int(((pt.get('X', 0.0) - RADAR_X_MIN) / x_range) * TILE_W)
            py = TILE_H - int((pt.get('Y', 0.0) / RADAR_Y_MAX) * TILE_H)
            if 0 <= px < TILE_W and 0 <= py < TILE_H:
                color, radius = _pt_style(pt.get('amp_db', 0.0), pt.get('vel', 0.0))
                cv2.circle(radar_tile, (px, py), radius, color, -1)

        # Colour legend
        if radar_point_mode == 'amp':
            bar_x0, bar_x1 = 4, TILE_W - 4
            bar_y0, bar_y1 = TILE_H - 26, TILE_H - 18
            bar_w = bar_x1 - bar_x0
            for i in range(bar_w):
                frac = i / max(bar_w - 1, 1)
                b_c = int(220 * (1 - frac) + 60 * frac)
                r_c = int(60  * (1 - frac) + 220 * frac)
                cv2.line(radar_tile,
                         (bar_x0 + i, bar_y0), (bar_x0 + i, bar_y1),
                         (b_c, 60, r_c), 1)
            cv2.rectangle(radar_tile, (bar_x0, bar_y0), (bar_x1, bar_y1), C_GRID, 1)
            cv2.putText(radar_tile, "weak",   (bar_x0, bar_y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 100, 100), 1)
            (tw, _), _ = cv2.getTextSize("strong", cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(radar_tile, "strong", (bar_x1 - tw, bar_y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 200), 1)

        elif radar_point_mode in ('vel', 'doppler_amp'):
            # Gradient bar: blue (receding, left) → gray (0) → red (approaching, right)
            bar_x0, bar_x1 = 4, TILE_W - 4
            bar_y0, bar_y1 = TILE_H - 26, TILE_H - 18
            bar_w = bar_x1 - bar_x0
            for i in range(bar_w):
                frac = i / max(bar_w - 1, 1)   # 0 = left/receding, 1 = right/approaching
                if frac < 0.5:                  # receding side → blue
                    t_b = 1.0 - frac * 2        # 1 at far left, 0 at center
                    b_c = 220; g_c = r_c = int(60 * (1 - t_b))
                else:                           # approaching side → red
                    t_r = (frac - 0.5) * 2      # 0 at center, 1 at far right
                    r_c = 220; g_c = b_c = int(60 * (1 - t_r))
                cv2.line(radar_tile,
                         (bar_x0 + i, bar_y0), (bar_x0 + i, bar_y1),
                         (b_c, g_c, r_c), 1)
            # Thin border around bar
            cv2.rectangle(radar_tile, (bar_x0, bar_y0), (bar_x1, bar_y1), C_GRID, 1)
            # Direction labels above bar: receding (left/blue) — approaching (right/red)
            cv2.putText(radar_tile, "receding", (bar_x0, bar_y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 100, 100), 1)
            (tw, _), _ = cv2.getTextSize("approaching", cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(radar_tile, "approaching", (bar_x1 - tw, bar_y0 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 200), 1)

        _font, _scale, _thick = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        (_tw, _), _ = cv2.getTextSize("Radar BEV", _font, _scale, _thick)
        cv2.putText(radar_tile, "Radar BEV", ((TILE_W - _tw) // 2, 13), _font, _scale, C_TITLE, _thick)

        # ── 3. [STUDENT TASK 4]  BEV bounding box overlay ───────────────────
        # Once map_detections_to_radar_bev() is implemented, bev_detections will
        # contain the columns rad_X, rad_Y, rad_W, rad_L (all in metres, same
        # coordinate frame as the radar points above).
        #
        # Steps:
        #   a) Filter bev_detections to rows where scan_idx == scan_idx.
        #   b) For each row, convert (rad_X, rad_Y, rad_W, rad_L) to pixel
        #      coordinates using the same world→pixel mapping used for radar
        #      points (see px / py computation above).
        #   c) Draw a cv2.rectangle on radar_tile.
        #      Colour scheme (BGR): ped=(0,200,0), bic=(255,0,255),
        #                           car=(0,165,255), cyclist=(0,220,100)
        #   d) Optionally add a small class label above the rectangle.
        #
        # TODO (Task 4): implement this block.
        # ── 3. [STUDENT TASK 4]  BEV bounding box overlay ───────────────────
        if bev_detections is not None and not bev_detections.empty:
            # Filter detections to the current scan
            frame_dets = bev_detections[bev_detections['scan_idx'] == scan_idx]
            
            for _, det in frame_dets.iterrows():
                if pd.isna(det.get('rad_X')): continue
                
                cls_id = int(det['cls_id'])
                # Standard BGR color mapping
                color_map = {
                    0: (0, 200, 0),    # ped (Green)
                    1: (255, 0, 255),  # bic (Magenta)
                    2: (0, 165, 255),  # car (Orange)
                    3: (0, 220, 100)   # cyclist (Yellow-Green)
                }
                color = color_map.get(cls_id, (255, 255, 255))
                
                # Calculate metric boundaries
                # Note: rad_Y represents the *front* of the object because we projected 
                # the bottom-center of the camera box. So the box extends "up" (rad_L) from rad_Y.
                left_m   = det['rad_X'] - det['rad_W'] / 2
                right_m  = det['rad_X'] + det['rad_W'] / 2
                front_m  = det['rad_Y']
                back_m   = det['rad_Y'] + det['rad_L']
                
                # Convert Metric coordinates to Pixel coordinates on the Radar Tile
                px_left  = int(((left_m - RADAR_X_MIN) / x_range) * TILE_W)
                px_right = int(((right_m - RADAR_X_MIN) / x_range) * TILE_W)
                py_front = TILE_H - int((front_m / RADAR_Y_MAX) * TILE_H)
                py_back  = TILE_H - int((back_m / RADAR_Y_MAX) * TILE_H)
                
                # Draw the rectangle
                cv2.rectangle(radar_tile, (px_left, py_front), (px_right, py_back), color, 1)
                
                # Draw a small dot at the geometric center
                py_center = (py_front + py_back) // 2
                px_center = (px_left + px_right) // 2
                cv2.circle(radar_tile, (px_center, py_center), 1, color, -1)

        # ── 4. Range scale tile ───────────────────────────────────────────────
        scale_tile = np.full((TILE_H, SCALE_W, 3), C_TILE_BG, dtype=np.uint8)
        cv2.line(scale_tile, (0, 0), (0, TILE_H - 1), C_TICK, 1)
        for m in range(0, int(RADAR_Y_MAX) + 1, 10):
            ty = TILE_H - int((m / RADAR_Y_MAX) * (TILE_H - 1))
            ty = max(1, min(ty, TILE_H - 1))
            tick_len = 7 if m % 20 == 0 else 4
            cv2.line(scale_tile, (0, ty), (tick_len, ty), C_TICK, 1)
            if m % 20 == 0:
                label_y = max(10, min(ty + 4, TILE_H - 4))
                cv2.putText(
                    scale_tile, f"{m}m", (tick_len + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_LABEL, 1,
                )

        # ── 4. Assemble final frame ───────────────────────────────────────────
        canvas = np.full((FINAL_H, FINAL_W, 3), C_BG, dtype=np.uint8)

        vid_xs = BORDER
        til_xs = vid_xs + VID_W + BORDER
        sca_xs = til_xs + TILE_W + BORDER

        canvas[BORDER:BORDER + VID_H,  vid_xs:vid_xs + VID_W]   = vid_panel
        canvas[BORDER:BORDER + TILE_H, til_xs:til_xs + TILE_W]  = radar_tile
        canvas[BORDER:BORDER + TILE_H, sca_xs:sca_xs + SCALE_W] = scale_tile

        writer.write(canvas)

    cap.release()
    writer.release()
    logging.info(f"Visualisation saved -> {output_path}")
    logging.info("-" * LOG_WIDTH)


# ==============================================================================
# 4.  SINGLE MEASUREMENT PROCESSING
# ==============================================================================

def process_single_measurement(args, measurement_folder: Path):
    """
    Full pipeline for one measurement folder:
      1. Locate required files (radar CSV, camera timestamps, video)
      2. Optionally load detections CSV (skipped gracefully if absent)
      3. Load radar data and camera timestamps
      4. Time-synchronise radar <-> camera (detections joined onto sync if available)
      5. Camera-to-Radar BEV mapping via map_detections_to_radar_bev() (Task 4)
      6. Optionally save intermediate CSVs (--debug)
      7. Generate the side-by-side visualisation video
    """
    measurement_name = measurement_folder.name
    export_path      = measurement_folder / "export"

    logging.info("=" * LOG_WIDTH)
    logging.info(f"Processing: {measurement_name}")
    logging.info("=" * LOG_WIDTH)

    # ── Locate required files in export/ ────────────────────────────────────
    try:
        radar_file = next(export_path.glob('radar1_resp*.csv'))
        ts_file    = next(export_path.glob('video1.timestamps'))
    except StopIteration as e:
        logging.warning(f"Skipping {measurement_name}: missing file in export/. {e}")
        return

    # video1.avi is always needed for the camera-timestamp decoding check
    raw_video = measurement_folder / "video1.avi"
    if not raw_video.exists():
        logging.warning(f"Skipping {measurement_name}: video1.avi not found.")
        return

    # ── Select video for the camera panel ───────────────────────────────────
    # The user supplies the exact filename; any video present in the folder works.
    viz_video = measurement_folder / args.source_video
    if not viz_video.exists():
        logging.warning(
            f"Source video '{args.source_video}' not found in {measurement_name}. "
            "Falling back to video1.avi."
        )
        viz_video = raw_video

    logging.info(f"Camera panel source: {viz_video.name}")

    # ── Load detections (optional) ───────────────────────────────────────────
    det_file      = measurement_folder / args.in_csv
    detections_df = load_detections(str(det_file))  # returns None if file absent

    # ── Load radar data and camera timestamps ────────────────────────────────
    radar_df = load_radar_data(str(radar_file))
    cam_df   = load_cam_ts(str(ts_file), str(raw_video))  # decoding check always on video1.avi

    # ── Time synchronisation ─────────────────────────────────────────────────
    sync_map, detections_synched, radar_synched = synchronize_time(
        radar_df, detections_df, cam_df, tolerance_ms=args.sync_tol
    )

    # ── Camera-to-Radar BEV mapping (Task 4) ─────────────────────────────────
    # map_detections_to_radar_bev() is a student task — it returns None until
    # implemented. The pipeline runs normally either way.
    bev_detections = None
    if not detections_synched.empty:
        bev_detections = map_detections_to_radar_bev(detections_synched, radar_synched)
        if bev_detections is not None:
            logging.info(f"BEV mapping complete: {len(bev_detections)} rows with rad_X/Y/W/L.")
        else:
            logging.info("BEV mapping skipped (map_detections_to_radar_bev not yet implemented).")

    # ── Optional debug CSVs ──────────────────────────────────────────────────
    if args.debug:
        p_sync = measurement_folder / "time_sync.csv"
        sync_map[['scan_idx', 'frame', 'radar_ts', 'cam_ts', 'dt_ms']].to_csv(p_sync, index=False)
        logging.info(f"Saved time sync -> {p_sync}")

        if not detections_synched.empty:
            p_det = measurement_folder / "detections_synced.csv"
            detections_synched.to_csv(p_det, sep=';', index=False)
            logging.info(f"Saved synced detections -> {p_det}")

        p_rad = measurement_folder / "radar_synced.csv"
        radar_synched.to_csv(p_rad, sep=';', index=False)
        logging.info(f"Saved synced radar -> {p_rad}")

    # ── Visualisation ────────────────────────────────────────────────────────
    out_video_path = measurement_folder / args.out_video
    render_fusion_video(
        sync_map          = sync_map,
        radar_synched     = radar_synched,
        video_path        = str(viz_video),
        output_path       = str(out_video_path),
        fps               = args.fps,
        meas_name         = measurement_name,
        radar_point_mode  = args.radar_point,
        bev_detections    = bev_detections,   # None until Task 4 is implemented
    )

    logging.info("=" * LOG_WIDTH)
    logging.info("Processing complete!")
    logging.info("=" * LOG_WIDTH)


# ==============================================================================
# 5.  BATCH PROCESSING
# ==============================================================================

def run_batch(args):
    """
    Iterate over all measurement subfolders in root_dir and run the full pipeline
    on each one.  The 'reference_points' subfolder is always skipped.
    Folders where the output video already exists are skipped.
    """
    root = Path(args.root_dir)
    measurement_folders = sorted([
        f for f in root.iterdir()
        if f.is_dir() and f.name != "reference_points"
    ])

    if not measurement_folders:
        logging.warning(f"No measurement subfolders found in {root}")
        return

    logging.info("=" * LOG_WIDTH)
    logging.info(f"Batch: {len(measurement_folders)} folder(s) in {root}")
    logging.info("=" * LOG_WIDTH)

    for folder in tqdm(measurement_folders, desc="Batch Progress", position=0):
        out_video_path = folder / args.out_video
        if out_video_path.exists():
            logging.info(f"Skipping {folder.name}: {args.out_video} already exists.")
            continue
        process_single_measurement(args, folder)


# ==============================================================================
# [STUDENT TASK 4]  CAMERA-TO-RADAR BEV MAPPING
# ------------------------------------------------------------------------------
# For every camera detection, estimate its position and physical size in the
# radar Bird's-Eye View (BEV) coordinate frame. The resulting labels will be
# used to generate training data for a radar-based object detector.
# ==============================================================================

'''def map_detections_to_radar_bev(detections_df, radar_synched=None):
    """
    [STUDENT TASK 4]  CAMERA-TO-RADAR BEV MAPPING
    ------------------------------------------------------------------
    For every detection in detections_df, compute four new columns:
        rad_X  — object centre X in radar BEV frame [m]
        rad_Y  — object centre Y in radar BEV frame [m]
        rad_W  — estimated object width  (cross-range extent) [m]
        rad_L  — estimated object length (down-range extent)  [m]

    Rows where a reliable estimate cannot be made should have NaN.

    Inputs
    ------
    detections_df : pd.DataFrame
        Columns: scan_idx, class, cls_id, x_center, y_center, width, height,
                 cam_X, cam_Y  (BEV position from homography, may be absent)

    radar_synched : pd.DataFrame, optional
        Synced radar points: scan_idx, X, Y, amp, vel

    Steps:
      1. Use cam_X, cam_Y (from homography) as the initial BEV position.
      2. Estimate rad_W (i.e. form the bbox width projected through the homography).
      3. Estimate rad_L (i.e. using a class-dependent aspect ratio, or by fitting
         a radar cluster associated with the detection).
      4. Visualise the results: draw each BEV bounding box (rad_X/Y/W/L) as a
         coloured rectangle on the radar BEV tile in render_fusion_video(). Use
         the same class-colour scheme as annotation_pipeline.py
         (ped=green, bic=magenta, car=orange, cyclist=yellow-green).

    Report:
      - Chosen approach and justification
      - How rad_W and rad_L are estimated per class (ped / cyclist / car)
      - Quantitative or qualitative evaluation of the BEV position accuracy
    """
    pass'''


def map_detections_to_radar_bev(detections_df, radar_synched=None):
    """
    [STUDENT TASK 4] CAMERA-TO-RADAR BEV MAPPING (PERFECTIONIST VERSION)
    Uses class-based priors for physical extent and employs 'Radar Snapping'
    to correct homography drift by fitting the box to the nearest radar cluster.
    """
    df = detections_df.copy()
    if 'cam_X' not in df.columns or 'cam_Y' not in df.columns:
        logging.warning("[TASK 4] cam_X/cam_Y not found. Homography was skipped.")
        return df

    # 1. Set base coordinates from camera homography
    df['rad_X'] = df['cam_X']
    df['rad_Y'] = df['cam_Y']
    
    # 2. Assign Physical Extents (rad_W, rad_L) based on class priors
    # Ped=0, Bic=1, Car=2, Cyclist=3
    # Note: Radar only sees the contour facing the sensor. 
    priors = {
        0: (0.6, 0.6),  # Pedestrian (0.6m x 0.6m)
        1: (0.6, 1.8),  # Bicycle (0.6m x 1.8m)
        2: (1.8, 4.5),  # Car (1.8m x 4.5m)
        3: (0.8, 1.8),  # Cyclist (0.8m x 1.8m)
    }
    df['rad_W'] = df['cls_id'].map(lambda x: priors.get(x, (1.0, 1.0))[0])
    df['rad_L'] = df['cls_id'].map(lambda x: priors.get(x, (1.0, 1.0))[1])

    # 3. SENSOR FUSION: "Radar Snapping"
    if radar_synched is not None and not radar_synched.empty:
        snapped_count = 0
        for idx, row in df.iterrows():
            # Get radar points from the exact same synchronized scan
            r_pts = radar_synched[radar_synched['scan_idx'] == row['scan_idx']]
            if r_pts.empty: continue
            
            # Calculate distance from camera-estimated position to all radar points
            dist = np.sqrt((r_pts['X'] - row['cam_X'])**2 + (r_pts['Y'] - row['cam_Y'])**2)
            
            # If there are radar points within a 2.5m radius, snap to them!
            close_pts = r_pts[dist < 2.5]
            if not close_pts.empty:
                # Snap X (Width) to the median of the radar cluster
                df.at[idx, 'rad_X'] = close_pts['X'].median()
                # Snap Y (Range) to the closest radar point (front bumper of the car)
                df.at[idx, 'rad_Y'] = close_pts['Y'].min()
                snapped_count += 1
                
        logging.info(f"[TASK 4] Sensor Fusion: Snapped {snapped_count}/{len(df)} boxes to radar clusters.")

    return df

# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Radar-Camera Fusion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Pipeline switches ---
    p.add_argument(
        "--run_batch", action="store_true",
        help="Process all measurement subfolders in root_dir instead of a single folder.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Save intermediate CSVs: time_sync.csv, detections_synced.csv, radar_synced.csv.",
    )

    # --- Paths ---
    p.add_argument(
        "--root_dir", type=str,
        default=str(PROJECT_DIR / "measurements"),
        help="Root directory containing measurement subfolders.",
    )
    p.add_argument(
        "--meas_name", type=str,
        default="20260218-134121_mix",
        help="Single measurement folder name inside root_dir (ignored when --run_batch is set).",
    )
    p.add_argument(
        "--in_csv", type=str,
        default="detections.csv",
        help=(
            "Input detections CSV filename inside the measurement folder. "
            "If the file does not exist the pipeline runs without detections — "
            "the visualiser still renders radar + camera for every synced scan."
        ),
    )
    p.add_argument(
        "--out_video", type=str,
        default="fusion_bev.mp4",
        help="Output visualisation video filename.",
    )

    # --- Visualisation ---
    p.add_argument(
        "--source_video", type=str,
        default="video1.avi",
        help=(
            "Filename of the video inside the measurement folder to use for the camera panel. "
            "Can be any video present in the folder, e.g. 'video1.avi', 'video1_annot.mp4', "
            "or any custom filename. Falls back to video1.avi if the specified file is missing."
        ),
    )
    p.add_argument(
        "--fps", type=float,
        default=30.0,
        help="Output video frame rate.",
    )

    # --- Radar BEV point style ---
    p.add_argument(
        "--radar_point",
        choices=['amp', 'vel', 'doppler_amp'],
        default='doppler_amp',
        help=(
            "Radar point colouring/sizing in BEV tile. "
            "'amp': colour+size~amplitude; "
            "'vel': colour~Doppler; 'doppler_amp': colour~Doppler, size~amplitude."
        ),
    )

    # --- Synchronisation ---
    p.add_argument(
        "--sync_tol", type=int,
        default=35,
        help="Max radar-camera timestamp difference in ms for a sync match to be accepted.",
    )

    return p.parse_args()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.run_batch:
        run_batch(args)
    else:
        measurement_folder = Path(args.root_dir) / args.meas_name
        process_single_measurement(args, measurement_folder)
