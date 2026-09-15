import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PROJECT_DIR = Path(__file__).resolve().parent.parent

CLASS_COLORS = {
    0: (0, 200, 0),    # Pedestrian: Green
    1: (0, 165, 255),  # Car: Orange
    2: (0, 220, 100)   # Cyclist: Yellow-Green
}
CLASS_NAMES = {0: "Ped", 1: "Car", 2: "Cyc"}

def record_inference_video(meas_folder_name: str, conf_threshold: float = 0.3):
    test_folder = PROJECT_DIR / "measurements" / meas_folder_name
    bev_dir = test_folder / "bev_images"
    
    # Path to the exported ONNX model
    onnx_model_path =  PROJECT_DIR / "runs" /"detect"/"runs"/ "radar" / "yolo11n_instructor_killer-9" / "weights" / "best.onnx"
    out_video_path = PROJECT_DIR / f"inference_rehearsal_{meas_folder_name}.mp4"

    if not bev_dir.exists():
        raise FileNotFoundError(f"BEV directory not found. Did you run grid mapping on {meas_folder_name}?")
    if not onnx_model_path.exists():
        raise FileNotFoundError("ONNX model not found. Run export_model.py first!")

    logging.info(f"Loading ONNX model: {onnx_model_path.name}")
    model = YOLO(str(onnx_model_path), task='detect')

    image_paths = sorted(bev_dir.glob("*.png"), key=lambda x: int(x.stem))
    if not image_paths:
        raise ValueError("No BEV images found in the target folder.")

    # 40ms cycle -> 25 Frames Per Second
    fps = 25.0  
    
    first_frame = cv2.imread(str(image_paths[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    logging.info(f"Generating inference video at 25 FPS: {out_video_path.name}")
    
    for img_path in tqdm(image_paths, desc="Running Inference"):
        frame = cv2.imread(str(img_path))
        
        # Run YOLO ONNX model
        results = model.predict(source=frame, conf=conf_threshold, verbose=False, imgsz=[704, 224])
        
        # Draw bounding boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {conf:.2f}"
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background and text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    logging.info(f"Successfully recorded inference video: {out_video_path.absolute()}")

if __name__ == "__main__":
    HELD_OUT_FOLDER = "20260506-161743_mix" 
    
    record_inference_video(HELD_OUT_FOLDER, conf_threshold=0.3)