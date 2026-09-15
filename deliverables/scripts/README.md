# AI-Based Road User Detection Using Traffic-Monitoring Radar

This repository contains the end-to-end machine learning and sensor fusion pipeline developed for the **Data Science Aided Measurement (BMETEFTBsPATTM-00)** course at the **Budapest University of Technology and Economics (BME)**, in cooperation with **FETI (Furukawa Electric Technológiai Intézet Kft.)**.

The project implements a deep learning pipeline to detect and classify moving road users (pedestrians, cars, and cyclists) using a stationary, overhead 24 GHz pulse-Doppler radar.

---

## 1. Directory Structure

Ensure your workspace is structured as follows for the scripts to locate the data paths automatically:

```text
radar_AI_project/
├── scripts/
│   ├── annotation_pipeline.py   # Step 1: Camera YOLO, tracking, & homography
│   ├── grid_mapping_pipeline.py # Step 2: Dense Radar BEV pseudo-image generation
│   ├── dataset_preparation.py   # Step 3: Radar snapping, labeling, & oversampling
│   ├── training.py              # Step 4: YOLO training on BEV images
│   ├── export_model.py          # Step 5: Exporting weights to ONNX format
│   └── inference_rehearsal.py   # Step 6: Simulation & inference video generation
├── dataset/                     # Generated automatically by Step 3
│   ├── data.yaml                # YOLO configuration metadata
│   ├── images/                  # Split train/val directories
│   └── labels/                  # YOLO txt annotations
├── measurements/                # Raw input data directory
│   ├── reference_points/        # Homography calibration files
│   │   ├── REF_PTS_2025.csv
│   │   ├── REF_PTS_202602.csv
│   │   ├── REF_PTS_202603.csv
│   │   └── REF_PTS_202605.csv
│   └── <measurement_folders>/   # e.g., 20260506-161743_mix
│       ├── video1.avi
│       └── export/
│           ├── radar1_resp.csv
│           └── video1.timestamps
└── README.md
```

---

## 2. Installation & Setup

This pipeline is optimized for Python 3.10+ and CUDA-enabled execution.

### Option A: Standard Virtual Environment (`venv` and `pip`)
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install --upgrade pip
pip install numpy pandas opencv-python tqdm ultralytics torch torchvision torchaudio onnx onnxruntime
```

### Option B: Fast Setup with `uv` (Recommended)
If you have the `uv` package manager installed, you can execute the scripts directly without manual environment management:
```bash
# Example run utilizing uv
uv run python scripts/grid_mapping_pipeline.py
```

---

## 3. Step-by-Step Pipeline Execution

Execute the scripts chronologically to build your dataset, train the model, and generate deliverables.

### Step 1: Camera Annotation, Tracking & Homography
Runs object detection (YOLOv11m) on the overhead camera footage, applies ByteTrack multi-object tracking, merges cyclists, filters out static parked cars, and projects the bottom-center of bounding boxes to the ground plane via camera-radar homography.

```bash
python scripts/annotation_pipeline.py --run_batch --run_inference --run_homography --conf 0.10 --tracker_type bytetrack
```
*   **Input**: `video1.avi`, `video1.timestamps`, and corresponding `REF_PTS` CSV files.
*   **Output**: `detections.csv` inside each measurement subfolder.

### Step 2: Radar BEV Grid Mapping
Accumulates sparse radar point-clouds over a rolling 480 ms window (12 frames) to generate dense BEV pseudo-images. The script maps metric coordinates to a $22.4\,\text{m} \times 70.4\,\text{m}$ grid, computes Doppler velocity skewing, normalizes received electric field amplitudes, and applies spatial cell spreading.

```bash
python scripts/grid_mapping_pipeline.py
```
*   **Input**: `radar1_resp.csv`
*   **Output**: PNG BEV images saved in a newly created `<folder>/bev_images/` directory.

### Step 3: Dataset Preparation
Pairs synchronized camera annotations with physical radar point returns. If an object contains no corresponding radar reflections within a $1.5\,\text{m}$ radius, it is ignored (eliminating empty training frames). Valid objects are snapped to the median of local radar clusters. 

To offset severe class imbalance, the training directory oversamples the target campaign folders (May 6th) three times ($3\times$), while keeping a distinct, non-overlapping validation set reserved exclusively for May 6th scenarios.

```bash
python scripts/dataset_preparation.py
```
*   **Input**: Synchronized `detections.csv` and `bev_images/`.
*   **Output**: Generates `/dataset/images/`, `/dataset/labels/` (train/val splits), and `/dataset/data.yaml` in YOLO format.

### Step 4: Model Training
Fine-tunes a `yolo11n` object detector on the custom multi-channel radar BEV dataset. 

Physical spatial relations are preserved by disabling geometric augmentations (`mosaic=0.0`, `scale=0.0`, `translate=0.0`) and preventing vertical flips (`flipud=0.0`), as approaching and receding targets have asymmetric Doppler signs. Loss weights are carefully scaled (`box=7.5`, `cls=0.5`, `dfl=1.5`) to prevent classification gradients from overriding bounding box regression.

```bash
python scripts/training.py
```
*   **Input**: `/dataset/data.yaml`
*   **Output**: Training metrics and weights exported to `runs/radar/yolo11n_radar_stage1/`.

### Step 5: ONNX Model Export
Exports the trained PyTorch model (`best.pt`) to an optimized ONNX graph for final implementation.

```bash
python scripts/export_model.py
```
*   **Input**: `runs/radar/yolo11n_radar_stage1/weights/best.pt`
*   **Output**: `runs/radar/yolo11n_radar_stage1/weights/best.onnx`

### Step 6: Inference Rehearsal & Video Output
Replays a held-out test dataset in 40 ms cycles (matching real-time sensor output). The script runs inference on each BEV frame using the exported ONNX model, draws bounding boxes, and records the output.

```bash
python scripts/inference_rehearsal.py
```
*   **Input**: Test folder (e.g., `20260506-161743_mix`) and `best.onnx`.
*   **Output**: `inference_rehearsal_20260506-161743_mix.mp4`.

---

## 4. Key Design Decisions & Parameter Reference

*   **Spatial Extent**: $22.4\,\text{m} \times 70.4\,\text{m}$ mapped onto a $224 \times 704$ pixel grid. Resolution is strictly configured at $0.1\,\text{m/px}$ across all scripts.
*   **Feature Channels**:
    *   **Channel 0 (Red)**: Point occupancy density (capped at 5 points).
    *   **Channel 1 (Green)**: Normalized log-scale received electric field amplitude ($E_{hh}$ compensated by $R^2$).
    *   **Channel 2 (Blue)**: Sign-preserved square-root skewed Doppler radial velocity.
*   **Tighter Bounding Box Priors**: Built around actual physical dimensions to minimize empty background coverage:
    *   **Pedestrian**: $0.8\,\text{m} \times 0.8\,\text{m}$
    *   **Bicycle / Cyclist**: $1.0\,\text{m} \times 2.0\,\text{m}$
    *   **Car**: $2.0\,\text{m} \times 4.5\,\text{m}$