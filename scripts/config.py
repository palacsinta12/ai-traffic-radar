from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
MEASUREMENTS_DIR = PROJECT_DIR / "measurements"
DATASET_DIR = PROJECT_DIR / "dataset"
RUNS_DIR = PROJECT_DIR / "runs"
ASSETS_DIR = PROJECT_DIR / "assets"

DEFAULT_DATA_YAML = DATASET_DIR / "data.yaml"
DEFAULT_MODEL = PROJECT_DIR / "yolo11n.pt"
DEFAULT_TRAIN_RUN_OUTPUT = RUNS_DIR / "radar"
DEFAULT_TRAIN_WEIGHTS = RUNS_DIR / "radar" / "yolo11n_radar_vru_optimized-2" / "weights" / "best.pt"
DEFAULT_ONNX_MODEL = (
    RUNS_DIR
    / "detect"
    / "runs"
    / "radar"
    / "yolo11n_instructor_killer-9"
    / "weights"
    / "best.onnx"
)

SYNTHETIC_GIF_OUTPUT = ASSETS_DIR / "actual_inference_demo.gif"
SYNTHETIC_DEMO_OUTPUT = ASSETS_DIR / "inference_demo.gif"

# Shared grid geometry and model output defaults.
# Match benchmark physical limits
X_MIN, X_MAX = -9.6, 9.6      # Total width = 19.2m (192 pixels)
Y_MIN, Y_MAX = 0.0, 64.0      # Total length = 64.0m (640 pixels)
RESOLUTION = 0.1
IMG_W = int(round((X_MAX - X_MIN) / RESOLUTION))
IMG_H = int(round((Y_MAX - Y_MIN) / RESOLUTION))
IMG_WIDTH_M = float(X_MAX - X_MIN)
IMG_HEIGHT_M = float(Y_MAX - Y_MIN)

# Padded BEV export / inference dimensions shared by the mapping and rehearsal scripts.
BEV_PAD_RIGHT = 0
BEV_PAD_BOTTOM = 0
BEV_IMAGE_WIDTH = IMG_W + BEV_PAD_RIGHT
BEV_IMAGE_HEIGHT = IMG_H + BEV_PAD_BOTTOM
# Change this:
DEFAULT_IMGSZ = [ BEV_IMAGE_HEIGHT,BEV_IMAGE_WIDTH]
EXPORT_IMGSZ = [ BEV_IMAGE_HEIGHT,BEV_IMAGE_WIDTH]


VERIFICATION_PATTERNS = []
