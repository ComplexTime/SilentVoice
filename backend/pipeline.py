import numpy as np

from backend.config import MOCK_MODE
from backend.utils import decode_frame


def run_prediction(frames: list[str]) -> dict:

    if MOCK_MODE:
        return {
            "sign": "WATER",
            "confidence": 0.94
        }

    from vision.hand_tracking import extract_landmarks
    from vision.preprocessing import preprocess_landmarks
    from ml.predict import predict_sign

    sequence = []

    for encoded_frame in frames:

        frame = decode_frame(encoded_frame)

        landmarks = extract_landmarks(frame)

        features = preprocess_landmarks(landmarks)

        sequence.append(features)

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (15, 128):
        raise ValueError(
            f"Invalid sequence shape: {sequence.shape}"
        )

    return predict_sign(sequence)