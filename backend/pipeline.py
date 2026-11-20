from backend.config import MOCK_MODE
from backend.utils import decode_frame


def run_prediction(frames: list[str]) -> dict:

    if MOCK_MODE:
        return {
            "sign": "WATER",
            "confidence": 0.94
        }

    from ml.features import extract_sequence
    from ml.predict import predict_sign

    decoded_frames = [
        decode_frame(frame)
        for frame in frames
    ]

    sequence = extract_sequence(decoded_frames)

    if sequence.shape != (15, 128):
        raise ValueError(
            f"Invalid sequence shape: {sequence.shape}"
        )

    return predict_sign(sequence)