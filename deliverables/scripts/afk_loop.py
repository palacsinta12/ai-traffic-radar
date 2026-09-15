"""
The Overnight Master Script
Executes Phase 1 (Annotation) -> Phase 2 (Grid Mapping) -> Phase 3 (Dataset Prep)
-> Phase 4 (YOLO Training) -> System Shutdown
"""
import os
import subprocess
import sys
from pathlib import Path

def run_cmd(description, cmd_list):
    print(f"\n{'='*80}")
    print(f"STARTING: {description}")
    print(f"{'='*80}\n")
    
    # Run the command and stream output to terminal
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed! Stopping pipeline to prevent unresolved issues.")
        sys.exit(1)
    
    print(f"\nFINISHED: {description}\n")

if __name__ == "__main__":
    print("PIPELINE INITIATED. Starting...\n")

    # 1. RUN THE ANNOTATION BATCH (Tasks 1-3)
    # Using DLT (no-use_ransac), botsort tracking, and recall-biased confidence [5].
    # Added '--allow_skip' to prevent re-annotating already-processed video folders [5].
    annotation_cmd = [
        "uv", "run", "python", "scripts/annotation_pipeline.py", 
        "--run_batch", "--run_inference", "--run_homography", "--run_visualize", 
        "--model", "yolo11m.pt", 
        "--conf", "0.15", 
        "--iou", "0.45", 
        "--tracker_type", "botsort", 
        "--track_buffer", "100",
        "--no-use_ransac",  # DLT performs better for this camera mounting angle [5].
        "--augment",
        "--allow_skip"      # Skip folders where annotations/videos already exist [5].
    ]
    run_cmd("Task 1-3: YOLO Annotation & Homography", annotation_cmd)

    # 2. RUN THE GRID MAPPING (Task 5)
    # Generates the 3-channel Radar images (Accumulation + Spreading)
    grid_cmd = ["uv", "run", "python", "scripts/grid_mapping_pipeline.py"]
    run_cmd("Task 5: Radar Grid Mapping", grid_cmd)

    # 3. RUN THE DATASET PREP (Tasks 4 & 6)
    # Snaps boxes to radar, splits train/val, applies target domain oversampling
    prep_cmd = ["uv", "run", "python", "scripts/dataset_preparation.py"]
    run_cmd("Task 6: Dataset Preparation & Oversampling", prep_cmd)

    # 4. RUN THE MODEL TRAINING (Task 7)
    # Fine-tunes YOLO11n on the prepared and aligned radar grids [8].
    train_cmd = ["uv", "run", "python", "scripts/training.py"]
    run_cmd("Task 7: YOLO Model Training on Radar BEV Grids", train_cmd)

    print(f"\n{'*'*80}")
    print("ALL PIPELINE STAGES & TRAINING COMPLETE!")
    print("Initiating system shutdown...")
    print(f"{'*'*80}\n")

    # 5. CROSS-PLATFORM SYSTEM SHUTDOWN
    # Shuts down the machine to preserve power/hardware once overnight work finishes.
    os.system("shutdown /s /t 10")  # Shutdown with 10-second warning
   