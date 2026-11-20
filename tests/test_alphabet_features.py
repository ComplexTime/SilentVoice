"""
Tests for vision/alphabet_features.py

Run from the project root:
    python -m pytest tests/test_alphabet_features.py -v

Tests that need MediaPipe/OpenCV are skipped automatically if those
packages are missing. The real-image lighting test runs only when
images exist in tests/fixtures/alphabet/ (see that test for details).
"""

from pathlib import Path

import numpy as np
import pytest

from vision.alphabet_features import (
    ALPHABET_FEATURES,
    DEFAULT_ASPECT_RATIO,
    InvalidHandError,
    apparent_palm_sign,
    frame_to_alphabet_sample,
    normalize_hand,
    sequence_to_alphabet_samples,
)

ASPECT = DEFAULT_ASPECT_RATIO
FIXTURES = Path(__file__).parent / "fixtures" / "alphabet"


# ==========================================
# Synthetic hands (MediaPipe image coordinates)
# ==========================================

# Open hand, fingers up, palm toward the camera, thumb on image-right.
# Units: fraction of the hand size; y points down like in images.
OPEN_HAND = np.array([
    (0.00, 0.00, 0.00),                                                  # wrist
    (0.12, -0.05, -0.02), (0.20, -0.12, -0.03), (0.26, -0.20, -0.04), (0.30, -0.27, -0.05),
    (0.09, -0.30, -0.01), (0.10, -0.42, -0.02), (0.11, -0.50, -0.03), (0.12, -0.57, -0.03),
    (0.02, -0.32, -0.01), (0.02, -0.46, -0.02), (0.02, -0.55, -0.03), (0.02, -0.63, -0.03),
    (-0.05, -0.30, -0.01), (-0.06, -0.42, -0.02), (-0.07, -0.50, -0.02), (-0.07, -0.57, -0.03),
    (-0.11, -0.26, 0.00), (-0.13, -0.35, -0.01), (-0.14, -0.41, -0.01), (-0.15, -0.47, -0.02),
])

# Closed fist with the thumb alongside (roughly letter A)
FIST = OPEN_HAND.copy()
FIST[6:9] = [(0.10, -0.36, -0.08), (0.10, -0.30, -0.10), (0.10, -0.26, -0.08)]
FIST[10:13] = [(0.02, -0.38, -0.08), (0.02, -0.31, -0.10), (0.02, -0.27, -0.08)]
FIST[14:17] = [(-0.06, -0.36, -0.08), (-0.06, -0.30, -0.09), (-0.06, -0.26, -0.07)]
FIST[18:21] = [(-0.13, -0.32, -0.06), (-0.13, -0.27, -0.07), (-0.13, -0.24, -0.06)]
FIST[2:5] = [(0.16, -0.14, -0.03), (0.17, -0.24, -0.04), (0.17, -0.31, -0.05)]


def place_hand(shape, center=(0.5, 0.6), size=0.4, angle_deg=0.0, aspect=ASPECT):
    """Put a hand shape into normalized image coordinates."""
    local = shape.copy()
    if angle_deg:
        a = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        local[:, :2] = local[:, :2] @ rot.T
    image = np.empty_like(local)
    image[:, 0] = center[0] + size * local[:, 0] / aspect   # x is divided by width
    image[:, 1] = center[1] + size * local[:, 1]            # y is divided by height
    image[:, 2] = size * local[:, 2] / aspect               # z uses the x scale
    return image


def mirror_image(landmarks):
    """Horizontal flip of the frame: x -> 1 - x."""
    mirrored = landmarks.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


def opposite(handedness):
    return {"Left": "Right", "Right": "Left"}[handedness]


def make_frame(left=None, right=None):
    """Build one raw 128-value frame like ml.features.extract_features()."""
    frame = np.zeros(128, dtype=np.float32)
    if left is not None:
        frame[0:63] = np.asarray(left).reshape(-1)
        frame[126] = 1.0
    if right is not None:
        frame[63:126] = np.asarray(right).reshape(-1)
        frame[127] = 1.0
    return frame


def make_sequence(frames):
    seq = np.stack(frames).astype(np.float32)
    assert seq.shape[1] == 128
    return seq


# ==========================================
# normalize_hand basics
# ==========================================

def test_output_shape_and_type():
    features = normalize_hand(place_hand(OPEN_HAND), "Right")
    assert features.shape == (ALPHABET_FEATURES,)
    assert features.dtype == np.float32


def test_wrist_at_origin_and_unit_scale():
    points = normalize_hand(place_hand(OPEN_HAND), "Right").reshape(21, 3)
    assert np.allclose(points[0], 0.0)
    scale = np.mean(np.linalg.norm(points[[5, 9, 13, 17]], axis=1))
    assert scale == pytest.approx(1.0, abs=1e-5)


def test_accepts_flat_63_values():
    hand = place_hand(OPEN_HAND)
    assert np.array_equal(
        normalize_hand(hand, "Right"),
        normalize_hand(hand.reshape(-1), "Right"),
    )


@pytest.mark.parametrize("bad", ["left", "RIGHT", "Unknown", None, 0])
def test_rejects_unknown_handedness(bad):
    with pytest.raises(InvalidHandError):
        normalize_hand(place_hand(OPEN_HAND), bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite(value):
    hand = place_hand(OPEN_HAND)
    hand[7, 1] = value
    with pytest.raises(InvalidHandError):
        normalize_hand(hand, "Right")


@pytest.mark.parametrize("shape", [(20, 3), (21, 2), (63, 1), (0,)])
def test_rejects_wrong_shape(shape):
    with pytest.raises(InvalidHandError):
        normalize_hand(np.zeros(shape), "Right")


def test_rejects_degenerate_hand():
    with pytest.raises(InvalidHandError):
        normalize_hand(np.zeros((21, 3)), "Right")
    with pytest.raises(InvalidHandError):
        normalize_hand(np.full((21, 3), 0.5), "Left")


def test_rejects_out_of_range_coordinates():
    hand = place_hand(OPEN_HAND)
    hand[:, 0] += 10.0
    with pytest.raises(InvalidHandError):
        normalize_hand(hand, "Right")


def test_does_not_modify_input():
    hand = place_hand(OPEN_HAND)
    before = hand.copy()
    normalize_hand(hand, "Left")
    assert np.array_equal(hand, before)


# ==========================================
# Invariances (position, distance) and orientation
# ==========================================

@pytest.mark.parametrize("center", [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7), (0.35, 0.9)])
def test_different_hand_positions(center):
    reference = normalize_hand(place_hand(OPEN_HAND), "Right")
    moved = normalize_hand(place_hand(OPEN_HAND, center=center), "Right")
    assert np.allclose(reference, moved, atol=1e-5)


@pytest.mark.parametrize("size", [0.15, 0.25, 0.6, 0.9])
def test_different_distances_from_camera(size):
    reference = normalize_hand(place_hand(OPEN_HAND, size=0.4), "Right")
    scaled = normalize_hand(place_hand(OPEN_HAND, size=size), "Right")
    assert np.allclose(reference, scaled, atol=1e-5)


def test_orientation_is_preserved():
    upright = normalize_hand(place_hand(OPEN_HAND), "Right")
    sideways = normalize_hand(place_hand(OPEN_HAND, angle_deg=90), "Right")
    # Rotation must NOT be normalized away (e.g. H vs U, G vs Q).
    assert np.abs(upright - sideways).max() > 0.5


def test_different_handshapes_stay_different():
    open_hand = normalize_hand(place_hand(OPEN_HAND), "Right")
    fist = normalize_hand(place_hand(FIST), "Right")
    assert np.linalg.norm(open_hand - fist) > 0.5


def test_aspect_ratio_correction():
    # Same physical hand captured at 640x480 and at a square 480x480.
    hand_43 = place_hand(OPEN_HAND, aspect=640 / 480)
    hand_11 = place_hand(OPEN_HAND, aspect=1.0)
    assert np.allclose(
        normalize_hand(hand_43, "Right", aspect_ratio=640 / 480),
        normalize_hand(hand_11, "Right", aspect_ratio=1.0),
        atol=1e-5,
    )


# ==========================================
# Handedness and webcam mirroring
# ==========================================

def test_left_hand_is_flipped_on_x_only():
    hand = place_hand(OPEN_HAND)
    right = normalize_hand(hand, "Right").reshape(21, 3)
    left = normalize_hand(hand, "Left").reshape(21, 3)
    assert np.allclose(left[:, 0], -right[:, 0])
    assert np.allclose(left[:, 1:], right[:, 1:])


@pytest.mark.parametrize("label", ["Left", "Right"])
@pytest.mark.parametrize("shape", [OPEN_HAND, FIST])
@pytest.mark.parametrize("angle", [0, 35, -60])
def test_webcam_mirroring_consistency(label, shape, angle):
    """
    Mirroring a frame flips the apparent chirality, so MediaPipe flips the
    label too (HandTracker's swap is constant). The features must not change.
    """
    raw = place_hand(shape, center=(0.4, 0.55), angle_deg=angle)
    mirrored = mirror_image(raw)
    assert np.allclose(
        normalize_hand(raw, label),
        normalize_hand(mirrored, opposite(label)),
        atol=1e-5,
    )


def test_left_and_right_handed_signers_match():
    # A left hand making the same letter is the mirror image of the right hand.
    right_hand = place_hand(FIST, center=(0.6, 0.6))
    left_hand = mirror_image(place_hand(FIST, center=(0.4, 0.6)))
    assert np.allclose(
        normalize_hand(right_hand, "Right"),
        normalize_hand(left_hand, "Left"),
        atol=1e-5,
    )


def test_apparent_palm_sign_flips_with_mirror():
    raw = place_hand(OPEN_HAND)
    assert apparent_palm_sign(raw) == -apparent_palm_sign(mirror_image(raw))
    assert apparent_palm_sign(raw) != 0


# ==========================================
# Raw 15x128 sequences: detection and rejection
# ==========================================

def test_left_hand_detection():
    hand = place_hand(OPEN_HAND)
    samples = sequence_to_alphabet_samples(make_sequence([make_frame(left=hand)] * 15))
    assert samples.shape == (15, 63)
    assert np.allclose(samples[0], normalize_hand(hand, "Left"))


def test_right_hand_detection():
    hand = place_hand(OPEN_HAND)
    samples = sequence_to_alphabet_samples(make_sequence([make_frame(right=hand)] * 15))
    assert samples.shape == (15, 63)
    assert np.allclose(samples[0], normalize_hand(hand, "Right"))


def test_missing_hands():
    samples = sequence_to_alphabet_samples(np.zeros((15, 128), dtype=np.float32))
    assert samples.shape == (0, 63)
    assert samples.dtype == np.float32


def test_two_hands_are_rejected():
    hand = place_hand(OPEN_HAND)
    frame = make_frame(left=hand, right=place_hand(FIST, center=(0.3, 0.6)))
    assert frame_to_alphabet_sample(frame) is None
    assert sequence_to_alphabet_samples(make_sequence([frame] * 15)).shape == (0, 63)


def test_mixed_sequence_keeps_only_valid_frames():
    hand = place_hand(OPEN_HAND)
    nan_hand = hand.copy()
    nan_hand[3, 0] = np.nan
    frames = [make_frame()] * 15
    frames[2] = make_frame(right=hand)
    frames[5] = make_frame(left=hand, right=hand)        # two hands
    frames[7] = make_frame(left=hand)
    frames[9] = make_frame(right=nan_hand)               # nonfinite
    frames[11] = make_frame(right=np.zeros((21, 3)))     # flag set, no data
    samples, indices = sequence_to_alphabet_samples(
        make_sequence(frames), return_indices=True
    )
    assert indices.tolist() == [2, 7]
    assert samples.shape == (2, 63)


def test_rejects_data_in_absent_slot():
    frame = make_frame(right=place_hand(OPEN_HAND))
    frame[0:63] = 0.3          # left slot has data but left flag is 0
    assert frame_to_alphabet_sample(frame) is None


def test_rejects_nonfinite_flags():
    frame = make_frame(right=place_hand(OPEN_HAND))
    frame[126] = np.nan
    assert frame_to_alphabet_sample(frame) is None


@pytest.mark.parametrize("shape", [(15, 127), (15,), (15, 2, 64)])
def test_rejects_wrong_sequence_shape(shape):
    with pytest.raises(ValueError):
        sequence_to_alphabet_samples(np.zeros(shape))


# ==========================================
# Lighting (simulated at landmark level)
# ==========================================

def test_lighting_noise_changes_features_only_slightly():
    """
    Lighting changes what MediaPipe sees, which shows up as small landmark
    jitter. Normalization must not amplify it.
    """
    rng = np.random.default_rng(0)
    hand = place_hand(OPEN_HAND)
    reference = normalize_hand(hand, "Right")
    for _ in range(50):
        noisy = hand + rng.normal(0.0, 0.002, size=hand.shape)
        assert np.abs(normalize_hand(noisy, "Right") - reference).max() < 0.15


def test_lighting_dropouts_are_rejected():
    """Dark/overexposed frames: MediaPipe finds no hand -> frame rejected."""
    hand = place_hand(OPEN_HAND)
    frames = [make_frame(right=hand) if i % 3 else make_frame() for i in range(15)]
    samples = sequence_to_alphabet_samples(make_sequence(frames))
    assert samples.shape == (10, 63)
    assert np.allclose(samples, samples[0])


# ==========================================
# Training vs inference: identical preprocessing
# ==========================================

def test_deterministic():
    seq = make_sequence([make_frame(right=place_hand(FIST))] * 15)
    a = sequence_to_alphabet_samples(seq)
    b = sequence_to_alphabet_samples(seq.copy())
    assert a.tobytes() == b.tobytes()


def test_float32_and_float64_inputs_match():
    seq = make_sequence([make_frame(left=place_hand(FIST))] * 15)
    assert np.allclose(
        sequence_to_alphabet_samples(seq),
        sequence_to_alphabet_samples(seq.astype(np.float64)),
        atol=1e-6,
    )


def test_word_pipeline_is_untouched_and_not_mixed_in():
    """
    The alphabet path must use the RAW sequence. Word-normalized frames are
    not image coordinates and must be rejected, not silently accepted.
    normalize_sequence() itself must keep working on the same raw input.
    """
    from ml.normalize import normalize_sequence

    frames = [make_frame()] * 15
    frames[3] = make_frame(right=place_hand(OPEN_HAND, center=(0.3, 0.5)))
    frames[8] = make_frame(left=place_hand(FIST, center=(0.7, 0.6)))
    raw = make_sequence(frames)
    raw_copy = raw.copy()

    assert sequence_to_alphabet_samples(raw).shape == (2, 63)
    word = normalize_sequence(raw)
    assert word.shape == (15, 128)
    assert np.array_equal(raw, raw_copy)                 # neither path mutates input
    assert sequence_to_alphabet_samples(word).shape == (0, 63)


def _fake_tracker_output(landmarks, handedness, score=0.95):
    return {
        "handedness": handedness,
        "score": score,
        "landmarks": [tuple(map(float, p)) for p in landmarks],
        "world_landmarks": [],
    }


def test_live_path_equals_training_path():
    """
    Live: HandTracker output -> hands_to_alphabet_sample()
    Training: HandTracker output -> extract_features() x15 -> sequence_to_alphabet_samples()
    Both must give exactly the same numbers.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from ml.features import extract_features
    from vision.alphabet_features import hands_to_alphabet_sample

    hand = place_hand(FIST, center=(0.45, 0.55), angle_deg=20)
    detections = [
        [_fake_tracker_output(hand, "Right")],
        [_fake_tracker_output(hand, "Left")],
        [],                                                        # missing
        [_fake_tracker_output(hand, "Left"), _fake_tracker_output(hand, "Right")],
    ]

    training = sequence_to_alphabet_samples(
        make_sequence([extract_features(d) for d in detections])
    )
    live = [hands_to_alphabet_sample(d) for d in detections]

    assert live[2] is None and live[3] is None
    assert training.shape == (2, 63)
    assert np.array_equal(training[0], live[0])
    assert np.array_equal(training[1], live[1])


# ==========================================
# Optional: real images with MediaPipe
# ==========================================

def _fixture_images():
    if not FIXTURES.is_dir():
        return []
    return sorted(
        p for p in FIXTURES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


@pytest.mark.skipif(not _fixture_images(), reason="no images in tests/fixtures/alphabet/")
@pytest.mark.parametrize("image_path", _fixture_images(), ids=lambda p: p.name)
def test_real_image_lighting_and_mirroring(image_path):
    """
    Put a few photos of ONE static letter per image (one hand visible) in
    tests/fixtures/alphabet/. Each image is run through HandTracker at
    several brightness levels and mirrored; features must stay close.
    Keep fixtures out of git if they show people (privacy).
    """
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from vision.hand_tracking import HandTracker
    from ml.features import extract_features

    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    aspect = image.shape[1] / image.shape[0]

    def features_of(img):
        # A fresh tracker per image so VIDEO-mode tracking does not leak.
        with HandTracker() as tracker:
            frame = extract_features(tracker.process_frame(img, timestamp_ms=0))
        return frame_to_alphabet_sample(frame, aspect_ratio=aspect)

    reference = features_of(image)
    if reference is None:
        pytest.skip("MediaPipe found no single hand in the original image")

    variants = {
        "darker": cv2.convertScaleAbs(image, alpha=0.6, beta=-20),
        "brighter": cv2.convertScaleAbs(image, alpha=1.2, beta=30),
        "mirrored": cv2.flip(image, 1),
    }
    for name, variant in variants.items():
        features = features_of(variant)
        if features is None:
            continue  # rejected is acceptable; a wrong vector is not
        assert np.abs(features - reference).max() < 0.35, name
