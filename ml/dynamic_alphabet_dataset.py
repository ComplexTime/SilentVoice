
import re
from pathlib import Path

import cv2
import numpy as np

from ml.features import extract_sequence

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "alphabet_dynamic"
OUTPUT = ROOT / "data" / "dynamic_alphabet_training.npz"

FRAME_COUNT = 15
WIDTH = 640
HEIGHT = 480

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_sort(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


def load_frame(path):
    frame = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError(f"Cannot read {path}")

    h, w = frame.shape[:2]

    scale = min(WIDTH / w, HEIGHT / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    x = (WIDTH - new_w) // 2
    y = (HEIGHT - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def main():
    X = []
    y = []
    sources = []
    rejected = 0

    clips = sorted(
        DATA_DIR.rglob("*_frames")
    )

    clips = [p for p in clips if p.is_dir()]

    print(f"Found {len(clips)} clips")

    for i, clip in enumerate(clips, start=1):

        original_label = clip.parent.name.upper()

        label = (
            original_label
            if original_label in {"J", "Z"}
            else "OTHER"
        )

        files = sorted(
            [
                p for p in clip.iterdir()
                if p.suffix.lower() in EXTENSIONS
            ],
            key=natural_sort
        )

        if len(files) < FRAME_COUNT:
            print(f"Skipping {clip.name}: too few frames")
            rejected += 1
            continue

        indices = np.linspace(
            0, len(files) - 1, FRAME_COUNT
        ).round().astype(int)

        try:
            frames = [
                load_frame(files[index])
                for index in indices
            ]

            sequence = extract_sequence(frames)

            if sequence.shape != (15, 128):
                raise ValueError("Invalid sequence shape")

            if not np.isfinite(sequence).all():
                raise ValueError("Invalid landmarks")

            left = sequence[:, 126] > 0.5
            right = sequence[:, 127] > 0.5

            valid = np.count_nonzero(left ^ right)

            if valid < 12:
                raise ValueError(
                    f"Only {valid} single-hand frames"
                )

            X.append(sequence)
            y.append(label)
            sources.append(str(clip.relative_to(DATA_DIR)))

            print(
                f"[{i}/{len(clips)}] "
                f"{clip.name} -> {label} OK"
            )

        except Exception as e:
            rejected += 1
            print(f"Skipping {clip.name}: {e}")

    if not X:
        raise RuntimeError("No valid clips extracted")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    np.savez_compressed(
        OUTPUT,
        X=X,
        y=y,
        sources=np.asarray(sources)
    )

    print("\nDATASET SUMMARY")
    print("Shape:", X.shape)
    print("Rejected:", rejected)

    for label, count in zip(
        *np.unique(y, return_counts=True)
    ):
        print(f"{label}: {count}")

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
