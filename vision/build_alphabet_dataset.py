"""
SilentVoice - Build the ASL alphabet training set from image folders

Converts a folder-per-letter image dataset into features that use
EXACTLY the same preprocessing as live recognition:

    image -> HandTracker -> extract_features -> frame_to_alphabet_sample

Expected layout (any depth; the letter is taken from the folder name):

    data/asl_images/
        Train/A/*.jpg   Train/B/*.jpg ...
        Test/A/*.jpg    ...

Folders that are not one of the 24 static letters (J, Z, "nothing",
"space", "del", ...) are skipped. Images with no hand or two hands
are rejected.

Usage (from the project root):
    python -m vision.build_alphabet_dataset --images data/asl_images
    python -m vision.build_alphabet_dataset --images data/asl_images --max-per-letter 300

Output: data/alphabet_training.npz (data/ is git-ignored - never commit it)
    features  (N, 63) float32  -> train on THIS (already normalized)
    y         (N,)    str      letter
    groups    (N,)    str      top-level folder, e.g. "Train"/"Test" or the dataset name
    X         (N, 1, 128)      raw frame, for vision.alphabet_check --npz
    aspect    (N,)    float32  image width / height used for normalization
    paths     (N,)    str      image path relative to --images
"""

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ml.features import extract_features
from vision.alphabet_features import STATIC_LETTERS, frame_to_alphabet_sample
from vision.hand_tracking import HandTracker

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUTPUT = Path("data") / "alphabet_training.npz"
MIN_IMAGE_SIDE = 64   # MediaPipe cannot find hands in tiny images (e.g. 28x28 MNIST)


def find_images(root):
    """Return {letter: [paths]} using the nearest single-letter folder name."""
    by_letter = defaultdict(list)
    skipped = Counter()
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        letter = next(
            (p.name.upper() for p in path.parents
             if len(p.name) == 1 and p.name.upper() in STATIC_LETTERS),
            None,
        )
        if letter is None:
            skipped[path.parent.name] += 1
            continue
        by_letter[letter].append(path)
    return by_letter, skipped


def read_image(path):
    # np.fromfile + imdecode works with non-English paths on Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def process_image(image):
    """Same path as live: HandTracker -> extract_features -> normalize."""
    # A fresh tracker per image: HandTracker runs in VIDEO mode, and
    # tracking must not carry over from one unrelated photo to the next.
    with HandTracker() as tracker:
        hands = tracker.process_frame(image, timestamp_ms=0)
    frame = extract_features(hands)
    aspect = image.shape[1] / image.shape[0]
    return frame, frame_to_alphabet_sample(frame, aspect_ratio=aspect), aspect


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="root folder of the image dataset")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-per-letter", type=int, default=0,
                        help="random subset per letter (0 = all) - use for a quick first run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.images)
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    by_letter, skipped = find_images(root)
    missing = [l for l in STATIC_LETTERS if l not in by_letter]
    print(f"Letters found: {len(by_letter)}/24  missing: {missing or 'none'}")
    if skipped:
        print(f"Skipped non-letter folders: {dict(skipped.most_common(8))}")
    if not by_letter:
        raise SystemExit("No letter folders found. Expected e.g. <root>/A/*.jpg")

    rng = random.Random(args.seed)
    jobs = []
    for letter in sorted(by_letter):
        paths = sorted(by_letter[letter])
        if args.max_per_letter and len(paths) > args.max_per_letter:
            paths = rng.sample(paths, args.max_per_letter)
        jobs.extend((letter, p) for p in paths)

    features, raw, labels, groups, aspects, rel_paths = [], [], [], [], [], []
    kept, rejected = Counter(), Counter()
    start = time.monotonic()

    try:
        from tqdm import tqdm
        iterator = tqdm(jobs, unit="img")
    except ImportError:
        iterator = jobs

    for letter, path in iterator:
        image = read_image(path)
        if image is None or min(image.shape[:2]) < MIN_IMAGE_SIDE:
            rejected[letter] += 1
            continue
        frame, sample, aspect = process_image(image)
        if sample is None:
            rejected[letter] += 1
            continue
        rel = path.relative_to(root)
        features.append(sample)
        raw.append(frame)
        labels.append(letter)
        groups.append(rel.parts[0] if len(rel.parts) > 2 else root.name)
        aspects.append(aspect)
        rel_paths.append(rel.as_posix())
        kept[letter] += 1

    elapsed = time.monotonic() - start
    print(f"\nProcessed {len(jobs)} images in {elapsed:.0f}s")
    print("letter  kept  rejected")
    for letter in STATIC_LETTERS:
        if letter in by_letter:
            print(f"  {letter}     {kept[letter]:5d}  {rejected[letter]:5d}")

    if not features:
        raise SystemExit("No usable samples - nothing saved.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=np.stack(features).astype(np.float32),
        X=np.stack(raw).astype(np.float32)[:, None, :],
        y=np.array(labels, dtype=str),
        groups=np.array(groups, dtype=str),
        aspect=np.array(aspects, dtype=np.float32),
        paths=np.array(rel_paths, dtype=str),
    )
    print(f"\nSaved {len(features)} samples -> {out}")
    print(f"Next: python -m vision.alphabet_check --npz {out}")


if __name__ == "__main__":
    main()
