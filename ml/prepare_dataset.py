import csv
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from ml import config
from ml.features import extract_sequence
from ml.dataset import validate_dataset


ROOT = config.PROJECT_ROOT
VIDEO_DIR = ROOT / "data" / "wlasl"

MANIFEST_PATH = VIDEO_DIR / "manifest.csv"
OUTPUT_PATH = ROOT / "data" / "training.npz"

MIN_VALID_FRAMES = 5
SAMPLE_INTERVAL = 0.1


def load_video_frames(path):

    cap = cv2.VideoCapture(str(path))

    try:

        if not cap.isOpened():
            raise ValueError("Cannot open video")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        frames = []

        while True:

            success, frame = cap.read()

            if not success:
                break

            frames.append(frame)

        if not frames:
            raise ValueError("Video is empty")

    finally:
        cap.release()

    # Select 15 frames approximately 100 ms apart,
    # centered around the middle of the video.

    center = (len(frames) - 1) / 2

    offsets = (
        np.arange(config.NUM_FRAMES)
        - (config.NUM_FRAMES - 1) / 2
    )

    indices = np.round(
        center + offsets * fps * SAMPLE_INTERVAL
    ).astype(int)

    # Repeat boundary frames for short videos.
    indices = np.clip(
        indices,
        0,
        len(frames) - 1
    )

    return [frames[i] for i in indices]


def main():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest missing: {MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        encoding="utf-8",
        newline=""
    ) as file:

        recordings = list(
            csv.DictReader(file)
        )

    X = []
    y = []
    groups = []

    accepted = Counter()
    rejected = Counter()

    print(
        f"Processing {len(recordings)} videos..."
    )

    for row in tqdm(recordings):

        label = row["label"]
        path = VIDEO_DIR / row["path"]

        try:

            frames = load_video_frames(path)

            sequence = extract_sequence(frames)

            if sequence.shape != (15, 128):
                raise ValueError(
                    f"Invalid shape: {sequence.shape}"
                )

            detected = np.any(
                sequence[:, 126:128] > 0,
                axis=1
            )

            valid_frames = int(detected.sum())

            if valid_frames < MIN_VALID_FRAMES:

                rejected[label] += 1
                continue

            X.append(sequence)
            y.append(label)
            groups.append(row["signer_id"])

            accepted[label] += 1

        except Exception as error:

            print(
                f"\nSkipped {row['video_id']}: "
                f"{error}"
            )

            rejected[label] += 1

    if not X:
        raise RuntimeError(
            "No valid training sequences generated"
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=str)
    groups = np.asarray(groups, dtype=str)

    X, y, groups = validate_dataset(
        X, y, groups
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        groups=groups
    )

    print("\n=== DATASET COMPLETE ===")

    print("Accepted recordings:", len(X))
    print("Rejected recordings:", sum(rejected.values()))

    print("\nAccepted per class:")

    for label, count in sorted(accepted.items()):
        print(f"{label}: {count}")

    print("\nRejected per class:")

    for label, count in sorted(rejected.items()):
        print(f"{label}: {count}")

    print("\nDataset dimensions:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("groups:", groups.shape)

    print("\nUnique signers:", len(np.unique(groups)))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()