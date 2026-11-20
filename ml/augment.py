import numpy as np

from ml import config
from ml.dataset import flatten_sequences


def augment_training(X, y, copies=5):

    rng = np.random.default_rng(
        config.RANDOM_STATE
    )

    # Apply our existing normalization first.
    X = flatten_sequences(X).reshape(
        -1, 15, 128
    )

    augmented_X = list(X)
    augmented_y = list(y)

    for sequence, label in zip(X, y):

        for _ in range(copies):

            sample = sequence.copy()

            # Small temporal shift.
            shift = rng.integers(-1, 2)

            indices = np.clip(
                np.arange(15) + shift,
                0,
                14
            )

            sample = sample[indices].copy()

            hands = sample[:, :126].reshape(
                15, 2, 21, 3
            )

            detected = sample[:, 126:128] > 0

            # Small image-plane rotation.
            angle = np.radians(
                rng.uniform(-8, 8)
            )

            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ], dtype=np.float32)

            valid = hands[detected].copy()

            valid[:, :, :2] = (
                valid[:, :, :2] @ rotation.T
            )

            # Simulate landmark detection noise.
            noise = rng.normal(
                0,
                0.01,
                size=valid.shape
            )

            valid += noise.astype(np.float32)

            hands[detected] = valid

            # Missing hands must remain zero.
            hands[~detected] = 0

            augmented_X.append(sample)
            augmented_y.append(label)

    return (
        np.asarray(
            augmented_X,
            dtype=np.float32
        ).reshape(-1, 1920),

        np.asarray(augmented_y)
    )