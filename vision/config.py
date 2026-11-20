"""
SilentVoice - Computer Vision Configuration

Central configuration file for the Computer Vision system.
All other vision modules read their default values from here,
so a setting only has to be changed in one place.

Stores:
    1. Project paths.
    2. Camera settings.
    3. MediaPipe settings.
    4. Preprocessing settings.
    5. Data collection settings.

IMPORTANT: CAMERA_MIRROR, COORDINATE_TYPE and MIRROR_LEFT_HAND
must be identical when collecting data, when training the model
and in the app. Changing them after collecting data means the
old samples were made with different settings.
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
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

# Dataset file
DATASET_FILE = DATA_DIR / "sign_samples.jsonl"


# ==========================================
# 2. CAMERA SETTINGS
# ==========================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Flip frames horizontally (selfie / mirror view).
CAMERA_MIRROR = True

# Tested on our setup: with a mirrored image, MediaPipe reports the
# hands the wrong way around (right hand = "Left"). Swapping the labels
# whenever the image is mirrored makes "Right" mean the signer's real
# right hand. If you change CAMERA_MIRROR, re-check the labels with
# "python -m vision.hand_tracking".
SWAP_HANDEDNESS = CAMERA_MIRROR

# Optional OpenCV backend. None = OpenCV default.
# On Windows, cv2.CAP_DSHOW (= 700) may open the camera faster.
CAMERA_BACKEND = None

CAMERA_WINDOW_NAME = "SilentVoice - Camera"


# ==========================================
# DISPLAY WINDOW SETTINGS
# ==========================================

# Starting window size = camera resolution * WINDOW_SCALE
WINDOW_SCALE = 1.5

# Open windows in full screen
START_FULLSCREEN = False

# Toggle full screen
FULLSCREEN_KEYS = frozenset({ord("f"), ord("F")})


# ==========================================
# 3. MEDIAPIPE HAND TRACKING SETTINGS
# ==========================================

MAX_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

HAND_LANDMARK_COUNT = 21

TRACKING_WINDOW_NAME = "SilentVoice - Hand Tracking"


# ==========================================
# 4. PREPROCESSING SETTINGS
# ==========================================

# Supported values: "world" (recommended) or "image"
COORDINATE_TYPE = "world"

# Flip left hands so they look like right hands.
# Lets one model work for right- and left-handed signers.
MIRROR_LEFT_HAND = False

# Number of coordinates per landmark
COORDINATES_PER_LANDMARK = 3

# Total features per hand: 21 * 3 = 63
FEATURE_VECTOR_SIZE = HAND_LANDMARK_COUNT * COORDINATES_PER_LANDMARK

# Two-hand feature vector: left hand + right hand = 126
TWO_HAND_FEATURE_VECTOR_SIZE = FEATURE_VECTOR_SIZE * 2


# ==========================================
# 5. DATA COLLECTION SETTINGS
# ==========================================

DATASET_SCHEMA_VERSION = 2

# Continuous recording: minimum time between two saved samples
CAPTURE_INTERVAL_MS = 150

SAVE_KEY = ord(" ")

# Q, q, ESC, and "/" (the Q key on a Hebrew keyboard layout)
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27, ord("/")})

RECORD_KEYS = frozenset({ord("r"), ord("R")})

COLLECTION_WINDOW_NAME = "SilentVoice - Data Collection"


# ==========================================
# 6. DIRECTORY MANAGEMENT AND VALIDATION
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


def validate_config():
    """
    Check that the configuration values make sense.

    Returns:
        list: Error messages. Empty if everything is valid.
    """

    errors = []

    if WINDOW_SCALE <= 0:
        errors.append("WINDOW_SCALE must be positive.")

    if COORDINATE_TYPE not in ("world", "image"):
        errors.append("COORDINATE_TYPE must be 'world' or 'image'.")

    if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
        errors.append("Camera resolution must be positive.")

    if MAX_HANDS < 1:
        errors.append("MAX_HANDS must be at least 1.")

    confidences = (
        MIN_HAND_DETECTION_CONFIDENCE,
        MIN_HAND_PRESENCE_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE
    )

    if any(not 0.0 <= value <= 1.0 for value in confidences):
        errors.append("Confidence values must be between 0 and 1.")

    if CAPTURE_INTERVAL_MS <= 0:
        errors.append("CAPTURE_INTERVAL_MS must be positive.")

    return errors


# ==========================================
# 7. CONFIGURATION TEST
# ==========================================

if __name__ == "__main__":

    ensure_directories()

    print("=" * 45)
    print("       SilentVoice - Configuration")
    print("=" * 45)

    print(f"\nProject root: {PROJECT_ROOT}")

    print("\nCamera:")
    print(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Camera index: {CAMERA_INDEX}")
    print(f"Mirror: {CAMERA_MIRROR}")
    print(f"Swap handedness labels: {SWAP_HANDEDNESS}")

    print("\nHand tracking:")
    print(f"Maximum hands: {MAX_HANDS}")
    print(f"Detection confidence: {MIN_HAND_DETECTION_CONFIDENCE}")
    print(f"Presence confidence: {MIN_HAND_PRESENCE_CONFIDENCE}")
    print(f"Tracking confidence: {MIN_TRACKING_CONFIDENCE}")

    print("\nPreprocessing:")
    print(f"Coordinate type: {COORDINATE_TYPE}")
    print(f"Mirror left hand: {MIRROR_LEFT_HAND}")
    print(f"Features per hand: {FEATURE_VECTOR_SIZE}")
    print(f"Features for two hands: {TWO_HAND_FEATURE_VECTOR_SIZE}")

    print("\nModel:")
    print(f"Model path: {HAND_LANDMARKER_MODEL}")
    print(f"Model exists: {validate_model()}")

    if not validate_model():
        print(f"Download it from:\n{HAND_LANDMARKER_URL}")

    print("\nDataset:")
    print(f"Dataset path: {DATASET_FILE}")

    errors = validate_config()

    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration loaded successfully!")