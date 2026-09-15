import os
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PROJECT_DIR = Path(__file__).resolve().parent.parent

def export_to_onnx():
    best_weights_path = PROJECT_DIR / "runs" /"detect"/"runs"/ "radar" / "yolo11n_instructor_killer-9" / "weights" / "best.pt"

    logging.info(f"Loading best weights from: {best_weights_path}")
    model = YOLO(str(best_weights_path))

    logging.info("Exporting to ONNX format...")
    onnx_path = model.export(
        format='onnx', 
        imgsz=[704, 224], 
        half=True, 
        simplify=True,
        opset=12 
    )
    
    logging.info(f"Export successful! ONNX model saved to: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()