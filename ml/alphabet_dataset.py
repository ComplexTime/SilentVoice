
import argparse
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from vision.hand_tracking import HandTracker
from vision.alphabet_features import (
    STATIC_LETTERS,
    hands_to_alphabet_sample
)

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "alphabet"
OUTPUT_PATH = ROOT / "data" / "alphabet_training.npz"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

RANDOM_STATE = 42


def resize_frame(image):

    height, width = image.shape[:2]

    scale = min(640 / width, 480 / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    canvas = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    x = (640 - new_width) // 2
    y = (480 - new_height) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return canvas


def load_image(path):

    encoded = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )


def process_split(split, max_per_letter):

    X = []
    y = []
    sources = []

    accepted = Counter()
    rejected = Counter()

    rng = random.Random(RANDOM_STATE)

    for letter in sorted(STATIC_LETTERS):

        folder = DATA_DIR / split / letter

        if not folder.exists():
            raise FileNotFoundError(folder)

        images = sorted([
            path for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.name.startswith(".")
        ])

        rng.shuffle(images)

        print(
            f"\n{split} | Letter {letter} | "
            f"Available: {len(images)}"
        )

        with HandTracker() as tracker:

            for index, path in enumerate(tqdm(images)):

                if accepted[letter] >= max_per_letter:
                    break

                image = load_image(path)

                if image is None:
                    rejected[letter] += 1
                    continue

                frame = resize_frame(image)

                hands = tracker.process_frame(
                    frame,
                    timestamp_ms=index * 100
                )

                sample = hands_to_alphabet_sample(hands)

                if sample is None:
                    rejected[letter] += 1
                    continue

                sample = np.asarray(
                    sample,
                    dtype=np.float32
                )

                if (
                    sample.shape != (63,)
                    or not np.isfinite(sample).all()
                ):
                    rejected[letter] += 1
                    continue

                X.append(sample)
                y.append(letter)

                sources.append(
                    str(path.relative_to(DATA_DIR))
                )

                accepted[letter] += 1

    print(f"\n=== {split.upper()} RESULTS ===")

    for letter in sorted(STATIC_LETTERS):

        print(
            f"{letter}: "
            f"{accepted[letter]} accepted, "
            f"{rejected[letter]} rejected"
        )

    missing = set(STATIC_LETTERS) - set(y)

    if missing:
        raise RuntimeError(
            f"Missing letters: {sorted(missing)}"
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=str),
        np.asarray(sources, dtype=str)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-per-letter",
        type=int,
        default=150
    )

    parser.add_argument(
        "--test-per-letter",
        type=int,
        default=40
    )

    args = parser.parse_args()

    X_train, y_train, train_sources = process_split(
        "Train",
        args.train_per_letter
    )

    X_test, y_test, test_sources = process_split(
        "Test",
        args.test_per_letter
    )

    np.savez_compressed(
        OUTPUT_PATH,
        X_train=X_train,
        y_train=y_train,
        train_sources=train_sources,
        X_test=X_test,
        y_test=y_test,
        test_sources=test_sources
    )

    print("\n=== ALPHABET DATASET COMPLETE ===")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("Classes:", len(set(y_train)))

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
