"""
SilentVoice - ASL Alphabet Feature Preprocessing

Turns single-hand MediaPipe landmarks into a 63-value feature vector
for static ASL alphabet recognition (24 letters; J and Z are dynamic
and not covered yet).

The SAME functions must be used for training and for live recognition.
Both paths start from the raw 15x128 sequence produced by
ml.features.extract_sequence(), so they go through identical code:

    training:  video/images -> extract_sequence -> sequence_to_alphabet_samples
    live:      webcam frames -> extract_sequence -> sequence_to_alphabet_samples
    per frame: HandTracker.process_frame -> hands_to_alphabet_sample

This module does NOT change the word recognition pipeline
(ml/features.py, ml/normalize.py stay untouched).

Normalization steps (normalize_hand):
    1. Validate: 21x3, finite, not degenerate.
    2. Aspect-ratio correction: x and z are multiplied by width/height so
       one unit of x has the same length as one unit of y
       (MediaPipe image landmarks are divided by width and height).
    3. Translation: wrist (landmark 0) moved to the origin.
    4. Scale: divide by the mean 3D wrist->MCP distance (landmarks
       5, 9, 13, 17), so distance from the camera does not matter.
    5. Handedness: hands in the "Left" slot are flipped on the x axis.
    6. Rotation is NOT normalized: orientation distinguishes letters
       such as H/U, G/Q, K/P and M/N/S/T variants.

Handedness convention (verified against the source, see README note
in the PR):
    * HandTracker swaps MediaPipe's labels (config.SWAP_HANDEDNESS) and
      extract_features() puts the hand into slot 0..62 ("Left") or
      63..125 ("Right") according to that swapped label.
    * MediaPipe decides the label from how the hand LOOKS in the image.
      A horizontally mirrored image flips the label as well. Because the
      swap is a constant, the slot always reflects the hand's apparent
      chirality in the image, whether or not the frame was mirrored.
    * Flipping the "Left" slot on x therefore gives one canonical hand
      shape for left- and right-handed signers AND for mirrored and
      unmirrored frames. No separate "is this dataset mirrored?" flag is
      needed as long as the landmarks came through HandTracker.
    * Landmarks from any other source (e.g. a pre-extracted CSV with its
      own handedness labels) must be checked with
      vision/alphabet_check.py before they are converted.
"""

from __future__ import annotations

import numpy as np

# ==========================================
# CONSTANTS (shared with the ML developer)
# ==========================================

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
ALPHABET_FEATURES = NUM_LANDMARKS * COORDS_PER_LANDMARK   # 63

# Raw sequence layout from ml/features.py (verified in ml/config.py)
RAW_FEATURES = 128
LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)
LEFT_FLAG = 126
RIGHT_FLAG = 127

HANDEDNESS_VALUES = ("Left", "Right")

# The slot that gets flipped on x. Do not change after data collection.
FLIPPED_HANDEDNESS = "Left"

# Frame width / height. The web frontend captures 640x480
# (frontend/src/config.js) and the vision camera uses 640x480.
DEFAULT_ASPECT_RATIO = 640 / 480

WRIST = 0
PALM_MCP_INDICES = (5, 9, 13, 17)

# Static letters supported in the first version (no J, no Z).
STATIC_LETTERS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")

# Rejection limits
MIN_HAND_SCALE = 1e-4        # in aspect-corrected image units
MAX_ABS_IMAGE_COORD = 2.0    # MediaPipe x/y are ~0..1; allow partial out-of-frame
FLAG_THRESHOLD = 0.5


class InvalidHandError(ValueError):
    """Raised when landmarks cannot be normalized."""


# ==========================================
# SINGLE HAND
# ==========================================

def _check_handedness(handedness):
    if handedness not in HANDEDNESS_VALUES:
        raise InvalidHandError(
            f"handedness must be 'Left' or 'Right', got {handedness!r}"
        )


def _as_landmarks(landmarks):
    try:
        points = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidHandError("Landmarks are not numeric.") from error

    if points.shape == (ALPHABET_FEATURES,):
        points = points.reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)

    if points.shape != (NUM_LANDMARKS, COORDS_PER_LANDMARK):
        raise InvalidHandError(
            f"Expected landmarks of shape (21, 3), got {points.shape}"
        )

    if not np.isfinite(points).all():
        raise InvalidHandError("Landmarks contain NaN or infinite values.")

    if np.abs(points[:, :2]).max() > MAX_ABS_IMAGE_COORD:
        raise InvalidHandError(
            "Landmark x/y are outside the expected image range."
        )

    return points


def normalize_hand(landmarks, handedness, *, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Normalize one hand for alphabet recognition.

    Args:
        landmarks: 21x3 array (or 63 flat values) of MediaPipe IMAGE
            landmarks (x, y in 0..1, z relative depth), exactly as stored
            in the raw extract_sequence() output.
        handedness: "Left" or "Right" - the slot label produced by
            HandTracker / extract_features().
        aspect_ratio: frame width / height of the images the landmarks
            came from. Keep the default (640/480) unless the dataset used
            a different resolution, and pass the same value at inference.

    Returns:
        np.ndarray of shape (63,), dtype float32:
        [x0, y0, z0, ..., x20, y20, z20], wrist at the origin,
        mean wrist->MCP distance = 1, "Left" slot flipped on x.

    Raises:
        InvalidHandError (a ValueError) for invalid input.
    """

    _check_handedness(handedness)

    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise InvalidHandError("aspect_ratio must be a positive number.")

    points = _as_landmarks(landmarks).copy()

    # 1. Aspect-ratio correction (units of frame height)
    points[:, 0] *= aspect_ratio
    points[:, 2] *= aspect_ratio

    # 2. Wrist at the origin
    points -= points[WRIST]

    # 3. Hand scale
    scale = float(np.mean(
        np.linalg.norm(points[list(PALM_MCP_INDICES)], axis=1)
    ))

    if not np.isfinite(scale) or scale < MIN_HAND_SCALE:
        raise InvalidHandError("Hand is too small or degenerate.")

    points /= scale

    # 4. One canonical chirality
    if handedness == FLIPPED_HANDEDNESS:
        points[:, 0] *= -1.0

    features = points.reshape(-1).astype(np.float32)

    if not np.isfinite(features).all():
        raise InvalidHandError("Normalization produced invalid values.")

    return features


# ==========================================
# RAW 15x128 SEQUENCES
# ==========================================

def frame_to_alphabet_sample(frame, *, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Convert one raw 128-value frame into a 63-value sample.

    Returns None when the frame must be rejected:
        * no hand, or two hands (ambiguous for a one-hand letter),
        * flags that are not 0/1-like or not finite,
        * an "absent" slot that still contains data,
        * invalid landmarks for the present hand.
    """

    frame = np.asarray(frame, dtype=np.float64)

    if frame.shape != (RAW_FEATURES,):
        raise ValueError(f"Expected a frame of 128 values, got {frame.shape}")

    flags = frame[[LEFT_FLAG, RIGHT_FLAG]]

    if not np.isfinite(flags).all():
        return None

    present = flags > FLAG_THRESHOLD

    if present.sum() != 1:
        return None

    if present[0]:
        handedness, hand, other = "Left", frame[LEFT_SLICE], frame[RIGHT_SLICE]
    else:
        handedness, hand, other = "Right", frame[RIGHT_SLICE], frame[LEFT_SLICE]

    # The absent slot must be empty, otherwise the frame is corrupt.
    if not np.isfinite(other).all() or np.any(other != 0):
        return None

    try:
        return normalize_hand(hand, handedness, aspect_ratio=aspect_ratio)
    except InvalidHandError:
        return None


def sequence_to_alphabet_samples(
    sequence,
    *,
    aspect_ratio=DEFAULT_ASPECT_RATIO,
    return_indices=False
):
    """
    Convert a raw extract_sequence() output into alphabet samples.

    Args:
        sequence: raw (15, 128) array from ml.features.extract_sequence().
            Any (T, 128) array is accepted. Pass the RAW sequence, NOT the
            output of ml.normalize.normalize_sequence(): that output is no
            longer in image coordinates, so its frames are rejected.
        aspect_ratio: see normalize_hand().
        return_indices: also return the frame index of every sample
            (useful to group samples by recording during training).

    Returns:
        (N, 63) float32 array of valid single-hand frames (N may be 0),
        or (samples, indices) when return_indices=True.
    """

    sequence = np.asarray(sequence, dtype=np.float64)

    if sequence.ndim != 2 or sequence.shape[1] != RAW_FEATURES:
        raise ValueError(
            f"Expected a sequence of shape (T, 128), got {sequence.shape}"
        )

    samples = []
    indices = []

    for index, frame in enumerate(sequence):
        sample = frame_to_alphabet_sample(frame, aspect_ratio=aspect_ratio)
        if sample is not None:
            samples.append(sample)
            indices.append(index)

    if samples:
        result = np.stack(samples).astype(np.float32)
    else:
        result = np.empty((0, ALPHABET_FEATURES), dtype=np.float32)

    if return_indices:
        return result, np.asarray(indices, dtype=np.int64)

    return result


def hands_to_alphabet_sample(detected_hands, *, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Live helper: HandTracker.process_frame() output -> 63 features or None.

    Goes through ml.features.extract_features() so that live frames are
    packed exactly like training sequences.
    """

    from ml.features import extract_features  # lazy: needs MediaPipe

    frame = extract_features(detected_hands)
    return frame_to_alphabet_sample(frame, aspect_ratio=aspect_ratio)


# ==========================================
# DATASET CHECK HELPER
# ==========================================

def apparent_palm_sign(landmarks, *, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Sign of the 2D cross product (index MCP - wrist) x (pinky MCP - wrist)
    in image coordinates (y points down).

    Mirroring the image flips this sign. Turning the hand around
    (palm vs back to the camera) also flips it, so use it only as a
    majority vote over many palm-forward samples, never per sample.
    Used by vision/alphabet_check.py to verify dataset conventions.
    """

    points = _as_landmarks(landmarks)
    wrist = points[WRIST, :2] * (aspect_ratio, 1.0)
    index_vec = points[5, :2] * (aspect_ratio, 1.0) - wrist
    pinky_vec = points[17, :2] * (aspect_ratio, 1.0) - wrist
    cross = index_vec[0] * pinky_vec[1] - index_vec[1] * pinky_vec[0]
    return int(np.sign(cross))
