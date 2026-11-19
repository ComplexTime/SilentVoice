"""
SilentVoice - Sign Language Data Collector

Collects labeled hand landmark samples for machine learning.

Features:
    1. Captures hand landmarks using MediaPipe.
    2. Preprocesses detected hands.
    3. Supports one or two hands per sample.
    4. Single-shot (SPACE) and continuous (R) recording.
    5. Saves labeled samples in JSONL format.
    6. Organizes samples into sessions and stores the settings
       that were used, so the data can be reproduced later.

Run from the project root:
    python -m vision.collect_data
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Works both as "python -m vision.collect_data" (from the project root)
# and as "python collect_data.py" (from inside the vision folder).
try:
    from . import config
    from .camera import CameraManager
    from .hand_tracking import HandTracker
    from .preprocessing import LandmarkPreprocessor
except ImportError:
    import config
    from camera import CameraManager
    from hand_tracking import HandTracker
    from preprocessing import LandmarkPreprocessor


# Labels: lowercase English letters, digits and "_" (e.g. "alef", "shalom").
# OpenCV cannot draw Hebrew text, and one fixed spelling per sign
# prevents the same sign from being saved under two different names.
LABEL_PATTERN = re.compile(r"^[a-z0-9_]+$")


def validate_label(label):
    """
    Clean and validate a sign label.
    """

    label = str(label).strip().lower()

    if not label:
        raise ValueError("Sign label cannot be empty.")

    if not LABEL_PATTERN.match(label):
        raise ValueError(
            "Sign label must contain only English letters, digits "
            "and '_' (for example: alef, bet, shalom)."
        )

    return label


def select_hands(processed_hands, expected_hands):
    """
    Keep only the hands that belong to the sign.

    Returns the `expected_hands` hands with the highest scores,
    or None if fewer hands were detected.
    """

    if len(processed_hands) < expected_hands:
        return None

    ranked = sorted(
        processed_hands,
        key=lambda hand: hand["score"],
        reverse=True
    )

    return ranked[:expected_hands]


class SignDataCollector:
    """
    Saves processed hand landmarks with their sign labels.
    """

    def __init__(self, output_path=None, metadata=None):
        """
        Initialize the data collector.

        Args:
            output_path: Optional path for the dataset file.
                Default: config.DATASET_FILE
            metadata: Settings saved with every sample
                (camera mirroring, preprocessing, signer, ...).
        """

        if output_path is None:
            output_path = config.DATASET_FILE

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.session_id = str(uuid.uuid4())
        self.metadata = dict(metadata or {})
        self.sample_count = 0

    def save_sample(self, label, processed_hands, timestamp_ms):
        """
        Save one labeled sign sample.

        All selected hands are saved together as one sample.

        Args:
            label: Name of the sign.
            processed_hands: Processed hand data.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            int: Number of samples saved in this session.
        """

        label = validate_label(label)

        if not processed_hands:
            raise ValueError("No detected hands to save.")

        record = {
            "schema_version": config.DATASET_SCHEMA_VERSION,
            "sample_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "sample_index": self.sample_count + 1,
            "label": label,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "timestamp_ms": int(timestamp_ms),
            "metadata": self.metadata,
            "hand_count": len(processed_hands),
            "hands": [
                {
                    "handedness": hand["handedness"],
                    "score": hand["score"],
                    "coordinate_type": hand["coordinate_type"],
                    "mirrored": hand.get("mirrored", False),
                    "raw_landmarks": hand["raw_landmarks"],
                    "normalized_landmarks": hand["normalized_landmarks"],
                    "features": hand["features"]
                }
                for hand in processed_hands
            ]
        }

        # Append one JSON object per line
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.sample_count += 1

        return self.sample_count


def ask_settings():
    """
    Ask the user for the collection settings.

    Returns:
        tuple: (label, expected_hands, signer) or None if invalid.
    """

    label = input(
        "\nSign name in English (e.g. alef, bet, shalom): "
    )

    try:
        label = validate_label(label)
    except ValueError as error:
        print(f"Error: {error}")
        return None

    hands_answer = input("Hands in this sign [1/2] (default 1): ").strip()

    if hands_answer in ("", "1"):
        expected_hands = 1
    elif hands_answer == "2":
        expected_hands = 2
    else:
        print("Error: Number of hands must be 1 or 2.")
        return None

    signer = input(
        "Signer name or ID (optional, helps split train/test): "
    ).strip()

    return label, expected_hands, signer or "unknown"


def draw_text(frame, text, position, color, scale=0.7):
    """
    Draw text with a dark outline so it is readable on any background.
    """

    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4
    )
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2
    )


def main():
    """
    Run the sign language data collection application.
    """

    logging.basicConfig(level=logging.INFO)

    print("=" * 45)
    print("       SilentVoice - Data Collection")
    print("=" * 45)

    settings = ask_settings()

    if settings is None:
        return

    label, expected_hands, signer = settings

    collector = None
    recording = False
    last_saved_ms = 0
    last_save_flash_ms = 0

    try:
        with CameraManager() as camera, HandTracker() as tracker:

            # Use the camera's actual resolution, not the requested one
            preprocessor = LandmarkPreprocessor(
                image_size=(camera.width, camera.height)
            )

            collector = SignDataCollector(metadata={
                "signer": signer,
                "expected_hands": expected_hands,
                "camera_mirrored": camera.mirror,
                "coordinate_type": preprocessor.coordinate_type,
                "mirror_left_hand": preprocessor.mirror_left_hand,
                "image_size": [camera.width, camera.height]
            })

            print(f"\nCurrent sign: {label} ({expected_hands} hand(s))")
            print("SPACE: save one sample")
            print("R:     start / stop continuous recording")
            print("Q/ESC: exit")
            print(f"\nDataset location: {collector.output_path}")

            while True:

                success, frame = camera.read()

                if not success:
                    print("Failed to capture frame.")
                    break

                timestamp_ms = time.monotonic_ns() // 1_000_000

                # Detect hands
                detected_hands = tracker.process_frame(
                    frame,
                    timestamp_ms=timestamp_ms
                )

                # Preprocess detected hands.
                # A bad frame must not stop the whole session.
                try:
                    processed_hands = preprocessor.process_hands(
                        detected_hands
                    )
                except (ValueError, KeyError) as error:
                    logging.debug("Skipped frame: %s", error)
                    processed_hands = []

                selected_hands = select_hands(
                    processed_hands,
                    expected_hands
                )

                # Draw landmarks
                display_frame = tracker.draw_landmarks(frame, detected_hands)

                # Status information
                ready = selected_hands is not None
                status_color = (0, 255, 0) if ready else (0, 0, 255)

                draw_text(display_frame, f"Sign: {label}", (10, 30),
                          (0, 255, 255), 0.8)
                draw_text(display_frame,
                          f"Hands: {len(detected_hands)}/{expected_hands}",
                          (10, 65), status_color)
                draw_text(display_frame,
                          f"Samples: {collector.sample_count}",
                          (10, 100), (255, 255, 0))
                draw_text(display_frame,
                          "SPACE: Save | R: Record | Q: Quit",
                          (10, 135), (255, 255, 255), 0.6)

                if recording:
                    draw_text(display_frame, "REC", (10, 175),
                              (0, 0, 255), 0.9)

                if timestamp_ms - last_save_flash_ms < 250:
                    draw_text(display_frame, "SAVED",
                              (display_frame.shape[1] - 130, 40),
                              (0, 255, 0), 1.0)

                cv2.imshow(config.COLLECTION_WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key in config.QUIT_KEYS:
                    break

                if key in config.RECORD_KEYS:
                    recording = not recording
                    print("Recording started." if recording
                          else "Recording stopped.")

                # Decide whether to save this frame
                should_save = key == config.SAVE_KEY or (
                    recording
                    and timestamp_ms - last_saved_ms >= config.CAPTURE_INTERVAL_MS
                )

                if not should_save:
                    continue

                if not ready:
                    if key == config.SAVE_KEY:
                        print(
                            f"Need {expected_hands} hand(s), detected "
                            f"{len(processed_hands)}. Sample not saved."
                        )
                    continue

                sample_number = collector.save_sample(
                    label=label,
                    processed_hands=selected_hands,
                    timestamp_ms=timestamp_ms
                )

                last_saved_ms = timestamp_ms
                last_save_flash_ms = timestamp_ms

                if key == config.SAVE_KEY:
                    print(f"Sample {sample_number} saved.")

    except (RuntimeError, FileNotFoundError, ValueError, OSError) as error:
        print(f"\nData Collection Error: {error}")

    finally:
        cv2.destroyAllWindows()

        print("\nData collection finished.")

        if collector is not None:
            print(f"Total samples saved: {collector.sample_count}")
            print(f"Dataset: {collector.output_path}")


if __name__ == "__main__":
    main()