"""
SilentVoice - Alphabet handedness / mirroring check (standalone tool)

Verifies the handedness convention BEFORE alphabet data is converted,
so left hands are never mirrored blindly.

Usage (from the project root):

    # 1. Dataset: raw sequences saved as .npz with X of shape (N, 15, 128)
    python -m vision.alphabet_check --npz data/alphabet_training.npz

    # 2. Live webcam: shows the slot label and the palm side per hand.
    #    Press M to toggle mirroring of the frame, Q/ESC to quit.
    python -m vision.alphabet_check --live

How to read the dataset report:
    For mostly palm-forward letters, the "Right" slot and the "Left" slot
    should show OPPOSITE majority palm signs before the flip and the SAME
    sign after it. If both slots already share a sign before the flip,
    the labels do not follow MediaPipe's convention (e.g. landmarks from
    another tool) - do not use normalize_hand() on them as-is; tell the
    Vision developer.
"""

import argparse
from collections import Counter

import numpy as np

from vision.alphabet_features import (
    LEFT_FLAG,
    LEFT_SLICE,
    RIGHT_FLAG,
    RIGHT_SLICE,
    apparent_palm_sign,
    frame_to_alphabet_sample,
)


def _canonical_sign(sample):
    """Palm sign of a normalized 63-value sample (aspect already corrected)."""
    points = sample.reshape(21, 3)
    index_vec, pinky_vec = points[5, :2], points[17, :2]
    return int(np.sign(index_vec[0] * pinky_vec[1] - index_vec[1] * pinky_vec[0]))


def check_npz(path):
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
    if X.ndim != 3 or X.shape[2] != 128:
        raise SystemExit(f"Expected X of shape (N, T, 128), got {X.shape}")

    before = {"Left": Counter(), "Right": Counter()}
    after = Counter()
    frames = Counter()

    for frame in X.reshape(-1, 128):
        left, right = frame[LEFT_FLAG] > 0.5, frame[RIGHT_FLAG] > 0.5
        if left and right:
            frames["two hands (rejected)"] += 1
            continue
        if not (left or right):
            frames["no hand (rejected)"] += 1
            continue
        side = "Left" if left else "Right"
        sample = frame_to_alphabet_sample(frame)
        if sample is None:
            frames["invalid (rejected)"] += 1
            continue
        frames[f"{side} slot (kept)"] += 1
        hand = frame[LEFT_SLICE if left else RIGHT_SLICE].reshape(21, 3)
        before[side][apparent_palm_sign(hand)] += 1
        after[_canonical_sign(sample)] += 1

    print(f"Dataset: {path}  sequences={len(X)}")
    for name, count in sorted(frames.items()):
        print(f"  {name:22s} {count}")
    for side in ("Left", "Right"):
        print(f"  palm sign in {side:5s} slot before flip: {dict(before[side])}")
    print(f"  palm sign after normalize_hand():      {dict(after)}")

    majority = {s: before[s].most_common(1)[0][0] for s in before if before[s]}
    if len(majority) == 2 and majority["Left"] == majority["Right"]:
        print("\nWARNING: both slots look the same BEFORE the flip.")
        print("The handedness labels do not match the landmarks. Do not train on this yet.")
    elif len(majority) == 2:
        print("\nOK: slots are mirror images before the flip and agree after it.")
    else:
        print("\nOnly one slot is present; convention cannot be cross-checked here.")
        print("Run --live with both hands once to confirm.")


def run_live():
    import cv2
    from vision.camera import CameraManager
    from vision.hand_tracking import HandTracker
    from ml.features import extract_features

    extra_flip = False
    with CameraManager() as camera, HandTracker() as tracker:
        aspect = camera.width / camera.height
        print("M: toggle extra mirroring | Q/ESC: quit")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            if extra_flip:
                frame = cv2.flip(frame, 1)
            hands = tracker.process_frame(frame)
            sample = frame_to_alphabet_sample(extract_features(hands), aspect_ratio=aspect)
            tracker.draw_landmarks(frame, hands)

            lines = [
                f"camera mirror={camera.mirror} extra flip={extra_flip}",
                f"hands={len(hands)} " + " ".join(h["handedness"] for h in hands),
            ]
            if sample is None:
                lines.append("alphabet sample: rejected")
            else:
                lines.append(f"canonical palm sign: {_canonical_sign(sample):+d}")
            for row, text in enumerate(lines):
                cv2.putText(frame, text, (10, 30 + 28 * row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("SilentVoice - Alphabet check", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("m"), ord("M")):
                extra_flip = not extra_flip
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", help="raw sequences .npz with X (N, T, 128)")
    group.add_argument("--live", action="store_true", help="webcam check")
    args = parser.parse_args()
    if args.live:
        run_live()
    else:
        check_npz(args.npz)


if __name__ == "__main__":
    main()
