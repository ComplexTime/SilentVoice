"""
Tests for vision/alphabet_motion.py (J and Z motion features)

Run from the project root:
    python -m pytest tests/test_alphabet_motion.py -v
"""

import numpy as np
import pytest

from tests.test_alphabet_features import (
    FIST,
    OPEN_HAND,
    make_frame,
    make_sequence,
    mirror_image,
    opposite,
    place_hand,
)
from vision.alphabet_motion import (
    MIN_VALID_FRAMES,
    MOTION_FEATURES,
    motion_amount,
    sequence_to_motion_features,
)

# Pinky-up hand (roughly I / J start): fist with the pinky extended
PINKY_UP = FIST.copy()
PINKY_UP[18:21] = [(-0.14, -0.36, -0.01), (-0.15, -0.43, -0.01), (-0.16, -0.50, -0.02)]

# Index-up hand (roughly the Z hand shape)
INDEX_UP = FIST.copy()
INDEX_UP[6:9] = OPEN_HAND[6:9]


def moving_sequence(shape, path, label="Right", size=0.4, angles=None, mirror=False):
    """Build a 15-frame sequence where the hand center follows `path`."""
    frames = []
    for t, center in enumerate(path):
        angle = 0.0 if angles is None else angles[t]
        hand = place_hand(shape, center=center, size=size, angle_deg=angle)
        side = label
        if mirror:
            hand, side = mirror_image(hand), opposite(label)
        frames.append(make_frame(**{side.lower(): hand}))
    return make_sequence(frames)


def j_path(start=(0.5, 0.45), scale=1.0):
    """Down, then hook to the left (as seen by the camera)."""
    t = np.linspace(0, 1, 15)
    x = start[0] - scale * 0.08 * np.clip((t - 0.6) / 0.4, 0, 1) ** 1.5
    y = start[1] + scale * 0.12 * np.sin(np.minimum(t, 0.8) / 0.8 * np.pi / 2)
    return list(zip(x, y))


def z_path(start=(0.4, 0.4), scale=1.0):
    """Right, diagonal down-left, right."""
    corners = np.array([(0, 0), (0.15, 0), (0, 0.12), (0.15, 0.12)]) * scale + start
    t = np.linspace(0, 3, 15)
    return [tuple(corners[int(min(s, 2.999))] + (s % 1 if s < 3 else 1)
                  * (corners[min(int(s) + 1, 3)] - corners[int(min(s, 2.999))]))
            for s in t]


STILL = [(0.5, 0.55)] * 15


# ---------- shape and validity ----------

def test_output_shape():
    features = sequence_to_motion_features(moving_sequence(PINKY_UP, j_path()))
    assert features.shape == (MOTION_FEATURES,)
    assert features.dtype == np.float32


def test_missing_hands_return_none():
    assert sequence_to_motion_features(np.zeros((15, 128))) is None
    assert motion_amount(np.zeros((15, 128))) == 0.0


def test_too_few_valid_frames_return_none():
    seq = moving_sequence(PINKY_UP, j_path())
    seq[: 15 - MIN_VALID_FRAMES + 1] = 0
    assert sequence_to_motion_features(seq) is None


def test_two_hand_frames_are_ignored():
    seq = moving_sequence(PINKY_UP, j_path())
    seq[3, 0:63] = seq[3, 63:126]
    seq[3, 126] = 1.0                        # frame 3 now has two hands
    assert sequence_to_motion_features(seq) is not None


def test_short_gap_is_filled():
    full = sequence_to_motion_features(moving_sequence(INDEX_UP, z_path()))
    seq = moving_sequence(INDEX_UP, z_path())
    seq[7] = 0
    gapped = sequence_to_motion_features(seq)
    assert gapped is not None
    assert np.abs(gapped - full).max() < 0.5


@pytest.mark.parametrize("shape", [(15, 127), (15,)])
def test_rejects_wrong_shape(shape):
    with pytest.raises(ValueError):
        sequence_to_motion_features(np.zeros(shape))


# ---------- invariances ----------

def test_position_invariance():
    a = sequence_to_motion_features(moving_sequence(PINKY_UP, j_path((0.3, 0.3))))
    b = sequence_to_motion_features(moving_sequence(PINKY_UP, j_path((0.6, 0.5))))
    assert np.allclose(a, b, atol=1e-4)


def test_distance_invariance():
    # Closer to the camera: the hand AND its movement get bigger together.
    a = sequence_to_motion_features(moving_sequence(INDEX_UP, z_path(scale=1.0), size=0.4))
    b = sequence_to_motion_features(moving_sequence(INDEX_UP, z_path(scale=1.5), size=0.6))
    assert np.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("label", ["Left", "Right"])
def test_webcam_mirroring_consistency(label):
    raw = sequence_to_motion_features(moving_sequence(PINKY_UP, j_path(), label=label))
    mirrored = sequence_to_motion_features(
        moving_sequence(PINKY_UP, j_path(), label=label, mirror=True)
    )
    assert np.allclose(raw, mirrored, atol=1e-5)


# ---------- separability ----------

def test_static_hand_has_little_motion():
    assert motion_amount(moving_sequence(FIST, STILL)) < 0.05


def test_j_and_z_have_clear_motion():
    assert motion_amount(moving_sequence(PINKY_UP, j_path())) > 0.3
    assert motion_amount(moving_sequence(INDEX_UP, z_path())) > 0.3


def test_j_and_z_features_differ():
    j = sequence_to_motion_features(moving_sequence(PINKY_UP, j_path()))
    z = sequence_to_motion_features(moving_sequence(INDEX_UP, z_path()))
    assert np.linalg.norm(j - z) > 1.0


def test_hand_rotation_is_kept():
    # J includes a wrist twist; the last frame's shape must reflect it.
    straight = sequence_to_motion_features(moving_sequence(PINKY_UP, j_path()))
    twisted = sequence_to_motion_features(
        moving_sequence(PINKY_UP, j_path(), angles=np.linspace(0, 60, 15))
    )
    assert np.abs(straight[63:126] - twisted[63:126]).max() > 0.3
