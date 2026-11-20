
import numpy as np

FRAME_COUNT = 15
MIN_VALID_FRAMES = 8

KEY_POINTS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]


def extract_dynamic_features(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (15, 128):
        raise ValueError("Expected shape (15, 128)")

    if not np.isfinite(sequence).all():
        return None

    left = sequence[:, 126] > 0.5
    right = sequence[:, 127] > 0.5

    left_valid = left & ~right
    right_valid = right & ~left

    use_left = left_valid.sum() >= right_valid.sum()

    if use_left:
        valid = left_valid
        landmarks = sequence[:, :63]
    else:
        valid = right_valid
        landmarks = sequence[:, 63:126]

    if valid.sum() < MIN_VALID_FRAMES:
        return None

    points = landmarks.reshape(15, 21, 3).copy()

    # Correct the camera aspect ratio.
    points[:, :, 0] *= 640 / 480

    # Canonicalize handedness.
    if use_left:
        points[:, :, 0] *= -1

    points = points[:, :, :2].copy()

    # Interpolate missing frames.
    times = np.arange(FRAME_COUNT)
    valid_times = times[valid]

    for joint in range(21):
        for axis in range(2):
            points[:, joint, axis] = np.interp(
                times,
                valid_times,
                points[valid, joint, axis]
            )

    # Normalize using a single scale for the entire clip.
    palm_sizes = np.linalg.norm(
        points[valid, 9] - points[valid, 0],
        axis=1
    )

    scale = np.median(palm_sizes)

    if scale < 1e-6:
        return None

    # Preserve movement relative to the initial wrist.
    origin = points[0, 0].copy()
    points = (points - origin) / scale

    # Wrist and fingertip trajectories.
    trajectories = points[:, KEY_POINTS, :]

    # Hand shape relative to the wrist.
    hand_shape = (
        trajectories - points[:, 0:1, :]
    )

    # Movement between consecutive frames.
    velocity = np.diff(trajectories, axis=0)

    # Total movement per landmark.
    path_length = np.linalg.norm(
        velocity, axis=2
    ).sum(axis=0)

    # Movement range.
    movement_range = np.ptp(
        trajectories, axis=0
    )

    # Final displacement.
    displacement = (
        trajectories[-1] - trajectories[0]
    )

    features = np.concatenate([
        trajectories.flatten(),
        hand_shape.flatten(),
        velocity.flatten(),
        path_length.flatten(),
        movement_range.flatten(),
        displacement.flatten()
    ])

    return features.astype(np.float32)
