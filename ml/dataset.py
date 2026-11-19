
"""Dataset loading, validation, flattening and group-aware splitting.

Data contract (agreed with the team, do not change):
  X: (N, 15, 128) float32, y: (N,) sign labels, groups: (N,) participant/session ids.
  Per frame: 0-62 left hand XYZ, 63-125 right hand XYZ, 126/127 left/right detected.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from ml import config

_FLAG_IDX = [config.LEFT_DETECTED_IDX, config.RIGHT_DETECTED_IDX]


def _check_flags(flags):
    """Detection flags must be exactly 0 or 1."""
    if not np.isin(flags, (0.0, 1.0)).all():
        raise ValueError("Hand-detected flags (indices 126, 127) must be 0 or 1.")


def validate_sequence(sequence):
    """Validate ONE sequence (15, 128) and return it as float32. Used by predict.py."""
    try:
        seq = np.asarray(sequence, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sequence must be numeric: {exc}") from exc
    if seq.shape != (config.NUM_FRAMES, config.NUM_FEATURES):
        raise ValueError(
            f"Expected sequence shape {(config.NUM_FRAMES, config.NUM_FEATURES)}, got {seq.shape}"
        )
    if not np.isfinite(seq).all():
        raise ValueError("Sequence contains NaN or infinite values.")
    _check_flags(seq[:, _FLAG_IDX])
    return seq


def validate_dataset(X, y, groups):
    """Validate the full dataset; return (X float32, y str array, groups str array)."""
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    expected = (config.NUM_FRAMES, config.NUM_FEATURES)
    if X.ndim != 3 or X.shape[1:] != expected:
        raise ValueError(f"X must have shape (N, {expected[0]}, {expected[1]}), got {X.shape}")
    n = X.shape[0]
    if n == 0:
        raise ValueError("Dataset is empty.")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError(f"X must be floating point (float32), got {X.dtype}")
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {y.shape}")
    if groups.shape != (n,):
        raise ValueError(f"groups must have shape ({n},), got {groups.shape}")

    bad = ~np.isfinite(X).reshape(n, -1).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} recording(s) contain NaN or infinite values.")

    _check_flags(X[:, :, _FLAG_IDX])

    y = np.char.strip(y.astype(str))
    if (y == "").any():
        raise ValueError("y contains empty labels.")
    groups = groups.astype(str)

    # Contract says missing hands are zeros: warn (not fail) if a flag is 0 but coords are not.
    for name, sl, idx in (
        ("left", config.LEFT_HAND_SLICE, config.LEFT_DETECTED_IDX),
        ("right", config.RIGHT_HAND_SLICE, config.RIGHT_DETECTED_IDX),
    ):
        undetected = X[:, :, idx] == 0
        has_coords = np.any(X[:, :, sl] != 0, axis=2)
        n_bad = int((undetected & has_coords).sum())
        if n_bad:
            warnings.warn(f"{n_bad} {name}-hand frame(s) have detected=0 but non-zero coordinates.")

    if config.UNKNOWN_LABEL not in set(y):
        warnings.warn(f"No '{config.UNKNOWN_LABEL}' examples in the dataset.")

    return X.astype(np.float32, copy=False), y, groups


def load_npz(path=config.DATA_PATH):
    """Load and validate data/training.npz (never uses pickle)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. The Vision developer provides data/training.npz "
            "(it is not tracked by Git)."
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in set(data.files) & {"X", "y", "groups"}}
    except ValueError as exc:
        raise ValueError(
            f"Could not read {path} without pickle ({exc}). Ask the Vision developer to save "
            "y and groups as plain arrays, e.g. np.array(y, dtype=str)."
        ) from exc
    missing = {"X", "y", "groups"} - set(arrays)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    return validate_dataset(arrays["X"], arrays["y"], arrays["groups"])


def flatten_sequences(X):
    """(N, 15, 128) -> (N, 1920)."""
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1:] != (config.NUM_FRAMES, config.NUM_FEATURES):
        raise ValueError(f"Expected (N, {config.NUM_FRAMES}, {config.NUM_FEATURES}), got {X.shape}")
    return X.reshape(X.shape[0], config.FLAT_FEATURES)


def group_train_test_split(y, groups, test_size=config.TEST_SIZE,
                           random_state=config.RANDOM_STATE, max_attempts=500):
    """Split by group (participant/session), never by single recording or frame.

    Retries with different seeds until every class appears in BOTH train and test.
    Returns (train_idx, test_idx).
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        raise ValueError("Need at least 2 groups (participants/sessions) to split.")
    classes = np.unique(y)
    for c in classes:
        if len(np.unique(groups[y == c])) < 2:
            raise ValueError(
                f"Class '{c}' appears in only one group, so it cannot be in both train and "
                "test. Ask the Vision developer for samples from more participants."
            )
    indices = np.arange(len(y))
    for attempt in range(max_attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=random_state + attempt)
        train_idx, test_idx = next(splitter.split(indices, y, groups))
        if (set(np.unique(y[train_idx])) == set(classes)
                and set(np.unique(y[test_idx])) == set(classes)):
            assert not set(groups[train_idx]) & set(groups[test_idx])
            return train_idx, test_idx
    raise ValueError("Could not find a group split containing every class in train and test.")


def describe_dataset(X, y, groups):
    lines = [f"X shape: {X.shape}, dtype: {X.dtype}",
             f"Groups: {len(np.unique(groups))}", "Samples per class:"]
    for c in np.unique(y):
        lines.append(f"  {c}: {int((y == c).sum())} samples in {len(np.unique(groups[y == c]))} groups")
    return "\n".join(lines)


def make_synthetic_dataset(n_participants=8, samples_per_class=6, seed=config.RANDOM_STATE):
    """SYNTHETIC random data to test the pipeline only.

    It says NOTHING about real recognition accuracy.
    """
    rng = np.random.default_rng(seed)
    classes = list(config.VOCABULARY[config.DEFAULT_LANGUAGE]) + [config.UNKNOWN_LABEL]
    protos = {}
    for c in classes:
        left, right = bool(rng.random() < 0.6), bool(rng.random() < 0.6)
        if not (left or right):
            left = True
        protos[c] = (rng.uniform(0.0, 1.0, size=(config.NUM_FRAMES, 126)), left, right)

    X, y, g = [], [], []
    for p in range(n_participants):
        for c in classes:
            coords, left, right = protos[c]
            for _ in range(samples_per_class):
                seq = np.zeros((config.NUM_FRAMES, config.NUM_FEATURES), dtype=np.float32)
                noisy = coords + rng.normal(0, 0.03, size=coords.shape)
                l, r = left, right
                if c == config.UNKNOWN_LABEL and rng.random() < 0.3:
                    l = r = False  # simulate "no hands detected"
                if l:
                    seq[:, config.LEFT_HAND_SLICE] = noisy[:, config.LEFT_HAND_SLICE]
                    seq[:, config.LEFT_DETECTED_IDX] = 1
                if r:
                    seq[:, config.RIGHT_HAND_SLICE] = noisy[:, config.RIGHT_HAND_SLICE]
                    seq[:, config.RIGHT_DETECTED_IDX] = 1
                X.append(seq)
                y.append(c)
                g.append(f"P{p}")
    return np.stack(X), np.array(y), np.array(g)


def _selftest():
    print("SYNTHETIC DATA - pipeline test only, not evidence of recognition accuracy.\n")
    X, y, groups = validate_dataset(*make_synthetic_dataset())
    print(describe_dataset(X, y, groups))

    flat = flatten_sequences(X)
    assert flat.shape == (len(X), config.FLAT_FEATURES)
    print(f"\nFlattened shape: {flat.shape}")

    train_idx, test_idx = group_train_test_split(y, groups)
    print(f"Train: {len(train_idx)} samples, groups {sorted(set(groups[train_idx]))}")
    print(f"Test:  {len(test_idx)} samples, groups {sorted(set(groups[test_idx]))}")
    assert not set(groups[train_idx]) & set(groups[test_idx])
    assert set(y[train_idx]) == set(y[test_idx]) == set(y)

    # Negative tests: invalid input must be rejected.
    for label, bad in (
        ("wrong shape", X[:, :10, :]),
        ("NaN values", np.where(np.arange(X.size).reshape(X.shape) == 5, np.nan, X)),
    ):
        try:
            validate_dataset(bad, y, groups)
        except ValueError:
            print(f"Rejected as expected: {label}")
        else:
            raise AssertionError(f"{label} was not rejected")
    print("\nAll dataset checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="load and summarize the real data/training.npz")
    args = parser.parse_args()
    if args.real:
        Xr, yr, gr = load_npz()
        print(describe_dataset(Xr, yr, gr))
        tr, te = group_train_test_split(yr, gr)
        print(f"Train: {len(tr)}, Test: {len(te)}")
    else:
        _selftest()
        