import numpy as np


def normalize_sequence(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    ).copy()

    if sequence.shape != (15, 128):
        raise ValueError("Invalid sequence shape")

    hands = sequence[:, :126].reshape(15, 2, 21, 3)

    detected = sequence[:, 126:128] > 0

    if not detected.any():
        return sequence

    # All valid hands throughout the recording
    valid_hands = hands[detected]

    # Common position reference for the sequence
    wrists = valid_hands[:, 0, :]

    center = np.mean(wrists[:, :2], axis=0)

    # Estimate palm size
    palm_indices = [5, 9, 13, 17]

    distances = np.linalg.norm(
        valid_hands[:, palm_indices, :2]
        - wrists[:, None, :2],
        axis=-1
    )

    scale = float(np.median(distances))

    if scale < 1e-6:
        scale = 1.0

    # Normalize both hands using the same
    # position reference and scale.
    for frame in range(15):

        for hand in range(2):

            if not detected[frame, hand]:
                hands[frame, hand] = 0
                continue

            hands[frame, hand, :, :2] -= center
            hands[frame, hand] /= scale

    return sequence