from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "alphabet"
DATA_PATH = PROJECT_ROOT / "data" / "alphabet_training.npz"

MODEL_PATH = ML_DIR / "models" / "alphabet_model.joblib"

LETTERS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")

NUM_FEATURES = 63
MIN_VALID_FRAMES = 5

UNKNOWN_LABEL = "UNKNOWN"
CONFIDENCE_THRESHOLD = 0.35

N_ESTIMATORS = 100
RANDOM_STATE = 42
N_JOBS = -1
TEST_SIZE = 0.20