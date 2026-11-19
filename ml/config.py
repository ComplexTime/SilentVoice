
"""Central configuration for the SilentVoice ML component (ASL MVP)."""
from pathlib import Path

# ---------- Paths ----------
ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "training.npz"
MODELS_DIR = ML_DIR / "models"

# ---------- Data contract (agreed with the team - do not change) ----------
NUM_FRAMES = 15
NUM_FEATURES = 128
FLAT_FEATURES = NUM_FRAMES * NUM_FEATURES  # 1920

LEFT_HAND_SLICE = slice(0, 63)
RIGHT_HAND_SLICE = slice(63, 126)
LEFT_DETECTED_IDX = 126
RIGHT_DETECTED_IDX = 127

# ---------- Prediction ----------
UNKNOWN_LABEL = "UNKNOWN"
CONFIDENCE_THRESHOLD = 0.80  # starting value, to be tuned on real test data

# ---------- Model (starting parameters, not guaranteed optimal) ----------
N_ESTIMATORS = 100
RANDOM_STATE = 42
N_JOBS = -1
TEST_SIZE = 0.25  # fraction of groups held out for testing

# ---------- Sign language ----------
DEFAULT_LANGUAGE = "ASL"
SUPPORTED_LANGUAGES = ("ASL",)  # ISL will be added later, after ASL is complete

# One separately trained model per language.
MODEL_FILENAMES = {
    "ASL": "asl_model.joblib",
    # "ISL": "isl_model.joblib",  # later
}

# PROPOSED vocabulary - must be confirmed by the team before data collection.
# The UNKNOWN class is handled separately via UNKNOWN_LABEL.
VOCABULARY = {
    "ASL": ["I_LOVE_YOU", "YES", "NO", "MORE", "STOP", "FINISH"],
}


def get_model_path(language: str = DEFAULT_LANGUAGE) -> Path:
    language = language.upper()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported sign language {language!r}; supported: {SUPPORTED_LANGUAGES}"
        )
    return MODELS_DIR / MODEL_FILENAMES[language]


MODEL_PATH = get_model_path()
