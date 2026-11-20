"""
SilentVoice - Alphabet end-to-end check (standalone tool)

Records 15 webcam frames exactly like the web frontend does (640x480,
100 ms apart, NOT mirrored, JPEG + Base64), then runs them through the
same path as POST /api/predict/alphabet:

    decode_frame -> extract_sequence -> sequence_to_alphabet_samples

If the backend is running, it also sends the frames to the real endpoint.
Until ml/alphabet_predict.py exists, HTTP 503 is the expected answer.

Usage (from the project root):
    python -m vision.alphabet_e2e_check
    python -m vision.alphabet_e2e_check --url http://localhost:8000
"""

import argparse
import base64
import time

import cv2
import numpy as np

from vision import config as vision_config
from vision.alphabet_features import sequence_to_alphabet_samples

FRAME_COUNT = 15
FRAME_INTERVAL_S = 0.1
WIDTH, HEIGHT = 640, 480
JPEG_QUALITY = 80


def record_frames(countdown_s=3):
    camera = cv2.VideoCapture(vision_config.CAMERA_INDEX)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera.")
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        end = time.monotonic() + countdown_s
        while time.monotonic() < end:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Failed to read a camera frame.")
            preview = cv2.flip(frame, 1)  # preview only, like the CSS mirror
            left = end - time.monotonic()
            cv2.putText(preview, f"Show ONE letter with ONE hand: {left:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("SilentVoice - Alphabet E2E", preview)
            cv2.waitKey(1)

        encoded = []
        for _ in range(FRAME_COUNT):
            start = time.monotonic()
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Failed to read a camera frame.")
            frame = cv2.resize(frame, (WIDTH, HEIGHT))  # raw, NOT mirrored
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise SystemExit("JPEG encoding failed.")
            encoded.append(base64.b64encode(jpeg.tobytes()).decode("ascii"))
            cv2.imshow("SilentVoice - Alphabet E2E", cv2.flip(frame, 1))
            cv2.waitKey(1)
            time.sleep(max(0.0, FRAME_INTERVAL_S - (time.monotonic() - start)))
        return encoded
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_local(frames):
    from backend.utils import decode_frame
    from ml.features import extract_sequence

    decoded = [decode_frame(f) for f in frames]
    sequence = extract_sequence(decoded)
    print(f"extract_sequence shape: {sequence.shape}")

    left = int((sequence[:, 126] > 0.5).sum())
    right = int((sequence[:, 127] > 0.5).sum())
    both = int(((sequence[:, 126] > 0.5) & (sequence[:, 127] > 0.5)).sum())
    print(f"frames with Left={left}  Right={right}  both={both}  none={FRAME_COUNT - left - right + both}")

    samples, indices = sequence_to_alphabet_samples(sequence, return_indices=True)
    print(f"alphabet samples: {samples.shape}  dtype={samples.dtype}  frames={indices.tolist()}")

    ok = (
        sequence.shape == (15, 128)
        and samples.ndim == 2 and samples.shape[1] == 63
        and samples.dtype == np.float32
        and np.isfinite(samples).all()
    )
    if len(samples) > 1:
        spread = float(np.abs(samples - samples.mean(axis=0)).max())
        print(f"max deviation between frames: {spread:.3f} (small = stable letter)")
    print("LOCAL PIPELINE:", "PASS" if ok else "FAIL")
    if len(samples) == 0:
        print("No valid single-hand frames: keep exactly one hand fully in view and retry.")
    return ok


def call_endpoint(frames, url):
    import requests

    endpoint = url.rstrip("/") + "/api/predict/alphabet"
    try:
        response = requests.post(endpoint, json={"frames": frames}, timeout=30)
    except requests.RequestException as error:
        print(f"ENDPOINT: backend not reachable at {endpoint} ({error.__class__.__name__}) - skipped")
        return
    print(f"ENDPOINT: HTTP {response.status_code} {response.text[:200]}")
    if response.status_code == 503:
        print("Expected until ml/alphabet_predict.py and the model exist.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--no-endpoint", action="store_true")
    args = parser.parse_args()

    frames = record_frames()
    run_local(frames)
    if not args.no_endpoint:
        call_endpoint(frames, args.url)


if __name__ == "__main__":
    main()
