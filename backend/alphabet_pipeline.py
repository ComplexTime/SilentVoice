
from backend.utils import decode_frame


def run_alphabet_prediction(frames: list[str]) -> dict:

    if len(frames) != 15:
        raise ValueError("Exactly 15 frames are required")

    from ml.features import extract_sequence
    from ml.alphabet_predict import predict_letter

    decoded_frames = [
        decode_frame(frame)
        for frame in frames
    ]

    sequence = extract_sequence(decoded_frames)

    if sequence.shape != (15, 128):
        raise ValueError(
            f"Invalid sequence shape: {sequence.shape}"
        )

    return predict_letter(sequence)
