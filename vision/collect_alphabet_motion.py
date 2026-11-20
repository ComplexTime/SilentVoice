"""
SilentVoice - Record J / Z (and negative) samples from the webcam

Each sample is recorded exactly like the web frontend sends it:
15 frames, 100 ms apart, 640x480, NOT mirrored, JPEG-compressed, then
decode_frame -> extract_sequence (the same path as the backend).

Labels:
    J    draw J with the pinky
    Z    draw Z with the index finger
    NEG  any STATIC letter held still, or a random hand movement
         (needed so the model learns what is NOT J/Z)

Usage (from the project root):
    python -m vision.collect_alphabet_motion --signer avichay
    python -m vision.collect_alphabet_motion --signer naomi --reps 15 --labels J Z

Keys in the window: SPACE = record next sample, R = redo last sample,
Q/ESC = save and quit. Samples are appended to
data/alphabet_motion.npz (git-ignored) after every take:
    X (N, 15, 128) raw sequences, y (N,), groups (N,) signer,
    features (N, 222) from sequence_to_motion_features
"""

import argparse
import base64
import time
from pathlib import Path

import cv2
import numpy as np

from vision import config as vision_config
from vision.alphabet_motion import motion_amount, sequence_to_motion_features

OUTPUT = Path("data") / "alphabet_motion.npz"
FRAMES, INTERVAL_S, WIDTH, HEIGHT, JPEG_QUALITY = 15, 0.1, 640, 480, 80
HINTS = {
    "J": "Pinky up, draw a J (down, then hook)",
    "Z": "Index up, draw a Z in the air",
    "NEG": "Hold any static letter still, or move randomly",
}


def load_existing(path):
    if not path.exists():
        return [], [], []
    with np.load(path, allow_pickle=False) as data:
        return list(data["X"]), list(data["y"]), list(data["groups"])


def save(path, X, y, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = [sequence_to_motion_features(s) for s in X]
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=str),
        groups=np.asarray(groups, dtype=str),
        features=np.stack([f if f is not None else np.full(222, np.nan, np.float32)
                           for f in feats]),
    )


def show(camera, lines, color=(0, 255, 255)):
    ok, frame = camera.read()
    if not ok:
        raise SystemExit("Failed to read a camera frame.")
    preview = cv2.flip(cv2.resize(frame, (WIDTH, HEIGHT)), 1)   # preview only
    for row, text in enumerate(lines):
        cv2.putText(preview, text, (10, 30 + 30 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.imshow("SilentVoice - Record J/Z", preview)
    return cv2.waitKey(1) & 0xFF


def record_take(camera):
    """15 frames like the frontend -> raw (15, 128) via the backend path."""
    from backend.utils import decode_frame
    from ml.features import extract_sequence

    encoded = []
    for _ in range(FRAMES):
        start = time.monotonic()
        ok, frame = camera.read()
        if not ok:
            raise SystemExit("Failed to read a camera frame.")
        frame = cv2.resize(frame, (WIDTH, HEIGHT))                 # NOT mirrored
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        encoded.append(base64.b64encode(jpeg.tobytes()).decode("ascii"))
        cv2.imshow("SilentVoice - Record J/Z", cv2.flip(frame, 1))
        cv2.waitKey(1)
        time.sleep(max(0.0, INTERVAL_S - (time.monotonic() - start)))
    return extract_sequence([decode_frame(e) for e in encoded])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signer", required=True, help="name/id of the person recording")
    parser.add_argument("--labels", nargs="+", default=["J", "Z", "NEG"],
                        choices=sorted(HINTS))
    parser.add_argument("--reps", type=int, default=20, help="samples per label")
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()

    out = Path(args.out)
    X, y, groups = load_existing(out)
    print(f"Existing samples in {out}: {len(X)}")

    camera = cv2.VideoCapture(vision_config.CAMERA_INDEX)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    try:
        for label in args.labels:
            done, last_status = 0, ""
            while done < args.reps:
                key = show(camera, [
                    f"{label} ({done}/{args.reps}) signer={args.signer}",
                    HINTS[label],
                    "SPACE: record   R: redo last   Q: save+quit",
                    last_status,
                ])
                if key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt
                if key in (ord("r"), ord("R")) and done > 0 and y and y[-1] == label:
                    X.pop(); y.pop(); groups.pop(); done -= 1
                    save(out, X, y, groups)
                    last_status = "Last sample removed"
                    continue
                if key != ord(" "):
                    continue

                for count in (3, 2, 1):                        # short countdown
                    end = time.monotonic() + 0.4
                    while time.monotonic() < end:
                        show(camera, [f"{label}: get ready... {count}", HINTS[label]],
                             color=(0, 165, 255))

                seq = record_take(camera)
                feats = sequence_to_motion_features(seq)
                if feats is None:
                    last_status = "REJECTED: keep ONE hand fully in view - try again"
                    continue
                X.append(seq); y.append(label); groups.append(args.signer)
                done += 1
                save(out, X, y, groups)
                last_status = f"Saved. motion={motion_amount(seq):.2f}"
                print(f"{label} {done}/{args.reps}  motion={motion_amount(seq):.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if X:
        save(out, X, y, groups)
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        print(f"\nSaved {len(X)} samples -> {out}")
        print("Per label:", dict(zip(labels.tolist(), counts.tolist())))
        amounts = {}
        for seq, lab in zip(X, y):
            amounts.setdefault(lab, []).append(motion_amount(seq))
        for lab, vals in sorted(amounts.items()):
            print(f"  motion_amount {lab:4s} min={min(vals):.2f} "
                  f"median={np.median(vals):.2f} max={max(vals):.2f}")


if __name__ == "__main__":
    main()
