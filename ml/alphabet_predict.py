from ml.dynamic_alphabet_predict import predict_dynamic_letter
import joblib
import numpy as np

from functools import lru_cache

from ml.alphabet_config import (
    MODEL_PATH,
    MIN_VALID_FRAMES
)

from vision.alphabet_features import (
    sequence_to_alphabet_samples
)


@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Alphabet model not found"
        )

    return joblib.load(MODEL_PATH)


def predict_letter(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (15, 128):
        raise ValueError(
            f"Invalid sequence shape: {sequence.shape}"
        )
    # Dynamic alphabet recognition (J and Z)
    dynamic_result = predict_dynamic_letter(sequence)

    if dynamic_result is not None:
        return dynamic_result
     # Static alphabet recognition (24 letters)
    samples = sequence_to_alphabet_samples(
        sequence
    )

    if len(samples) < MIN_VALID_FRAMES:
        return {
            "sign": "UNKNOWN",
            "confidence": 0.0
        }

    artifact = load_model()

    model = artifact["model"]

    probabilities = model.predict_proba(
        samples
    )

    average = np.mean(
        probabilities,
        axis=0
    )

    best = int(np.argmax(average))

    letter = str(model.classes_[best])

    confidence = float(average[best])

    threshold = artifact["confidence_threshold"]

    if confidence < threshold:

        return {
            "sign": "UNKNOWN",
            "confidence": 0.0
        }

    return {
        "sign": letter,
        "confidence": confidence
    }

