
"""
SilentVoice - Computer Vision Configuration

Central configuration file for the Computer Vision system.

Stores:
    1. Project paths.
    2. Camera settings.
    3. MediaPipe settings.
    4. Preprocessing settings.
    5. Data collection settings.
"""

from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

# Main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project directories
VISION_DIR = PROJECT_ROOT / "vision"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# MediaPipe model
HAND_LANDMARKER_MODEL = (
    MODELS_DIR / "hand_landmarker.task"
)

# Dataset file
DATASET_FILE = DATA_DIR / "sign_samples.jsonl"


# ==========================================
# 2. CAMERA SETTINGS
# ==========================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_WINDOW_NAME = "SilentVoice - Camera"


# ==========================================
# 3. MEDIAPIPE HAND TRACKING SETTINGS
# ==========================================

MAX_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

HAND_LANDMARK_COUNT = 21


# ==========================================
# 4. PREPROCESSING SETTINGS
# ==========================================

# Supported values: "world" or "image"
COORDINATE_TYPE = "world"

# Number of coordinates per landmark
COORDINATES_PER_LANDMARK = 3

# Total features per hand: 21 * 3
FEATURE_VECTOR_SIZE = (
    HAND_LANDMARK_COUNT * COORDINATES_PER_LANDMARK
)


# ==========================================
# 5. DATA COLLECTION SETTINGS
# ==========================================

DATASET_FORMAT = "jsonl"

SAVE_KEY = ord(" ")
EXIT_KEY = ord("q")

COLLECTION_WINDOW_NAME = (
    "SilentVoice - Data Collection"
)


# ==========================================
# 6. DIRECTORY MANAGEMENT
# ==========================================

def ensure_directories():
    """
    Create required project directories if they do not exist.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def validate_model():
    """
    Verify that the MediaPipe model exists.

    Returns:
        bool: True if the model exists.
    """

    return HAND_LANDMARKER_MODEL.is_file()


# ==========================================
# 7. CONFIGURATION TEST
# ==========================================

if __name__ == "__main__":

    ensure_directories()

    print("=" * 45)
    print("       SilentVoice - Configuration")
    print("=" * 45)

    print(f"\nProject root: {PROJECT_ROOT}")

    print(f"\nCamera:")
    print(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Camera index: {CAMERA_INDEX}")

    print(f"\nHand tracking:")
    print(f"Maximum hands: {MAX_HANDS}")
    print(
        f"Detection confidence: "
        f"{MIN_HAND_DETECTION_CONFIDENCE}"
    )
    print(f"Tracking confidence: {MIN_TRACKING_CONFIDENCE}")

    print(f"\nPreprocessing:")
    print(f"Coordinate type: {COORDINATE_TYPE}")
    print(f"Features per hand: {FEATURE_VECTOR_SIZE}")

    print(f"\nModel:")
    print(f"Model path: {HAND_LANDMARKER_MODEL}")
    print(f"Model exists: {validate_model()}")

    print(f"\nDataset:")
    print(f"Dataset path: {DATASET_FILE}")

    print("\nConfiguration loaded successfully!")
