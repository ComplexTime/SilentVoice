import numpy as np

from vision.hand_tracking import HandTracker
from ml.config import NUM_FRAMES, NUM_FEATURES


def extract_features(detected_hands):

    features = np.zeros(
        NUM_FEATURES,
        dtype=np.float32
    )

    best_hands = {}

    for hand in detected_hands:

        side = hand["handedness"]

        if side not in ("Left", "Right"):
            continue

        if (
            side not in best_hands
            or hand["score"] > best_hands[side]["score"]
        ):
            best_hands[side] = hand

    for side, offset, flag in [
        ("Left", 0, 126),
        ("Right", 63, 127)
    ]:

        if side not in best_hands:
            continue

        landmarks = np.asarray(
            best_hands[side]["landmarks"],
            dtype=np.float32
        )

        if landmarks.shape != (21, 3):
            continue

        if not np.isfinite(landmarks).all():
            continue

        features[offset:offset + 63] = (
            landmarks.flatten()
        )

        features[flag] = 1.0

    return features


def extract_sequence(frames):

    if len(frames) != NUM_FRAMES:
        raise ValueError(
            f"Expected {NUM_FRAMES} frames"
        )

    sequence = []

    with HandTracker() as tracker:

        for index, frame in enumerate(frames):

            hands = tracker.process_frame(
                frame,
                timestamp_ms=index * 100
            )

            features = extract_features(hands)

            sequence.append(features)

    return np.asarray(
        sequence,
        dtype=np.float32
    )