from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

PROJECT_NAME = "SilentVoice"
VERSION = "1.0.0"

HOST = "127.0.0.1"
PORT = 8000

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

SIGN_LANGUAGE = "ASL"

SEQUENCE_LENGTH = 15
FEATURE_VECTOR_SIZE = 128
MIN_CONFIDENCE = 0.80

MOCK_MODE = True

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")