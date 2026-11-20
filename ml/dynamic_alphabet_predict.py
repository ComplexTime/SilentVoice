
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

from ml.dynamic_alphabet_features import (
    extract_dynamic_features
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "dynamic_alphabet_model.joblib"
)


@lru_cache(maxsize=1)
def load_dynamic_model():
    return joblib.load(MODEL_PATH)


def predict_dynamic_letter(sequence):

    if not MODEL_PATH.exists():
        return None

    features = extract_dynamic_features(sequence)

    if features is None:
        return None

    artifact = load_dynamic_model()
    model = artifact["model"]

    threshold = artifact["confidence_threshold"]

    probabilities = model.predict_proba(
        features.reshape(1, -1)
    )[0]

    classes = list(model.classes_)

    j_prob = float(probabilities[classes.index("J")])
    z_prob = float(probabilities[classes.index("Z")])

    if j_prob >= z_prob:
        letter = "J"
        confidence = j_prob
    else:
        letter = "Z"
        confidence = z_prob

    print(
        f"[DYNAMIC] J={j_prob:.3f} "
        f"Z={z_prob:.3f}"
    )

    if confidence < threshold:
        return None

    return {
        "sign": letter,
        "confidence": confidence
    }
