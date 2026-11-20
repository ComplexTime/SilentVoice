"""
SilentVoice - Motion features for the dynamic ASL letters J and Z

J and Z cannot be recognized from one frame: J traces a hook with the
pinky, Z draws a "Z" with the index finger. This module turns a whole
raw 15x128 sequence (from ml.features.extract_sequence) into ONE
fixed-length feature vector that describes hand shape AND movement.

It builds on vision/alphabet_features.py and uses the same conventions:
    * the same frame validation (exactly one hand, finite, not degenerate),
    * the same aspect-ratio correction (default 640x480),
    * the same left/right handling: the "Left" slot is flipped on x, so
      features are identical for mirrored/unmirrored frames and for
      left-/right-handed signers.

Feature vector (MOTION_FEATURES = 222 values):
    [  0:63 ]  normalized hand shape of the first valid frame
    [ 63:126]  normalized hand shape of the last valid frame
    [126:216]  trajectories of wrist, index tip, pinky tip
               (3 points x 15 time steps x (x, y)), relative to the first
               wrist position and divided by the hand size
    [216:222]  path length of index tip and pinky tip, and x/y extent of
               both tips

The backend is unchanged: /api/predict/alphabet already sends 15 frames.
Suggested use in ml/alphabet_predict.py:
    if motion_amount(sequence) > threshold: use the J/Z model on
    sequence_to_motion_features(sequence); otherwise use the static model.
"""

from __future__ import annotations

import numpy as np

from vision.alphabet_features import (
    DEFAULT_ASPECT_RATIO,
    FLIPPED_HANDEDNESS,
    LEFT_FLAG,
    LEFT_SLICE,
    PALM_MCP_INDICES,
    RAW_FEATURES,
    RIGHT_FLAG,
    RIGHT_SLICE,
    WRIST,
    frame_to_alphabet_sample,
)

DYNAMIC_LETTERS = ("J", "Z")

NUM_STEPS = 15
MIN_VALID_FRAMES = 10
TRACKED_POINTS = (WRIST, 8, 20)          # wrist, index tip, pinky tip
INDEX_TIP_POS, PINKY_TIP_POS = 1, 2      # positions inside TRACKED_POINTS

SHAPE_SIZE = 63
TRAJ_SIZE = len(TRACKED_POINTS) * NUM_STEPS * 2
STATS_SIZE = 6
MOTION_FEATURES = 2 * SHAPE_SIZE + TRAJ_SIZE + STATS_SIZE   # 222


def _valid_frames(sequence, aspect_ratio):
    """Yield (index, slot, raw 21x3 points, normalized 63) for usable frames."""
    for index, frame in enumerate(sequence):
        sample = frame_to_alphabet_sample(frame, aspect_ratio=aspect_ratio)
        if sample is None:
            continue
        slot = "Left" if frame[LEFT_FLAG] > 0.5 else "Right"
        hand = frame[LEFT_SLICE if slot == "Left" else RIGHT_SLICE]
        yield index, slot, np.asarray(hand, dtype=np.float64).reshape(21, 3), sample


def _check_sequence(sequence):
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != RAW_FEATURES:
        raise ValueError(f"Expected a sequence of shape (T, 128), got {sequence.shape}")
    return sequence


def _collect(sequence, aspect_ratio):
    """Valid frames of the majority hand slot, or None if too few."""
    frames = list(_valid_frames(sequence, aspect_ratio))
    if not frames:
        return None
    slots = [slot for _, slot, _, _ in frames]
    slot = max(set(slots), key=slots.count)
    frames = [f for f in frames if f[1] == slot]
    if len(frames) < MIN_VALID_FRAMES:
        return None
    return slot, frames


def _trajectories(frames, slot, length, aspect_ratio):
    """Tracked points over time, relative to the first wrist, in hand units."""
    indices = np.array([i for i, _, _, _ in frames], dtype=np.float64)
    points = np.stack([p for _, _, p, _ in frames])            # (F, 21, 3)
    points[:, :, 0] *= aspect_ratio
    points[:, :, 2] *= aspect_ratio

    scales = np.linalg.norm(
        points[:, list(PALM_MCP_INDICES)] - points[:, [WRIST]], axis=2
    ).mean(axis=1)
    scale = float(np.median(scales))
    if not np.isfinite(scale) or scale <= 0:
        return None

    tracked = points[:, list(TRACKED_POINTS), :2] - points[0, WRIST, :2]
    tracked /= scale
    if slot == FLIPPED_HANDEDNESS:
        tracked[:, :, 0] *= -1.0

    # Resample to NUM_STEPS evenly spaced steps over the sequence
    # (fills short gaps where a frame was rejected).
    targets = np.linspace(0, length - 1, NUM_STEPS)
    resampled = np.empty((len(TRACKED_POINTS), NUM_STEPS, 2))
    for p in range(len(TRACKED_POINTS)):
        for axis in range(2):
            resampled[p, :, axis] = np.interp(targets, indices, tracked[:, p, axis])
    return resampled


def sequence_to_motion_features(sequence, *, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Raw (T, 128) sequence -> (222,) float32 motion features, or None.

    Returns None when fewer than MIN_VALID_FRAMES frames contain exactly
    one valid hand (missing hands, two hands, invalid landmarks).
    """
    sequence = _check_sequence(sequence)
    collected = _collect(sequence, aspect_ratio)
    if collected is None:
        return None
    slot, frames = collected

    traj = _trajectories(frames, slot, len(sequence), aspect_ratio)
    if traj is None:
        return None

    tips = traj[[INDEX_TIP_POS, PINKY_TIP_POS]]                  # (2, steps, 2)
    path = np.linalg.norm(np.diff(tips, axis=1), axis=2).sum(axis=1)
    extent = (tips.max(axis=1) - tips.min(axis=1)).reshape(-1)  # (4,)

    features = np.concatenate([
        frames[0][3], frames[-1][3], traj.reshape(-1), path, extent
    ]).astype(np.float32)

    if features.shape != (MOTION_FEATURES,) or not np.isfinite(features).all():
        return None
    return features


def motion_amount(sequence, *, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    How far the index/pinky tips travel over the sequence, in hand sizes.

    About 0 for a static letter held still; clearly larger for J and Z.
    Returns 0.0 when the sequence has too few valid frames. Pick the
    routing threshold from recorded data (see collect_alphabet_motion).
    """
    features = sequence_to_motion_features(sequence, aspect_ratio=aspect_ratio)
    if features is None:
        return 0.0
    return float(features[216:218].max())
