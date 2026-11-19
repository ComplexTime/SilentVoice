
"""
SilentVoice - Sign Language Data Collector

Collects labeled hand landmark samples for machine learning.

Features:
    1. Captures hand landmarks using MediaPipe.
    2. Preprocesses detected hands.
    3. Supports one or two hands per sample.
    4. Saves labeled samples in JSONL format.
    5. Organizes samples into collection sessions.
"""

import cv2
import json
import uuid

from pathlib import Path
from datetime import datetime, timezone

from .camera import CameraManager
from .hand_tracking import HandTracker
from .preprocessing import LandmarkPreprocessor


class SignDataCollector:
    """
    Saves processed hand landmarks with their sign labels.
    """

    def __init__(self, output_path=None):
        """
        Initialize the data collector.

        Args:
            output_path: Optional path for the dataset file.
        """

        project_root = Path(__file__).resolve().parent.parent

        if output_path is None:
            output_path = (
                project_root / "data" / "sign_samples.jsonl"
            )

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.session_id = str(uuid.uuid4())
        self.sample_count = 0

    def save_sample(self, label, processed_hands, timestamp_ms):
        """
        Save one labeled sign sample.

        All detected hands are saved together as one sample.

        Args:
            label: Name of the sign.
            processed_hands: Processed hand data.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            int: Number of samples saved in this session.
        """

        label = str(label).strip()

        if not label:
            raise ValueError("Sign label cannot be empty.")

        if not processed_hands:
            raise ValueError("No detected hands to save.")

        record = {
            "schema_version": 1,
            "sample_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "sample_index": self.sample_count + 1,
            "label": label,
            "captured_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "timestamp_ms": int(timestamp_ms),
            "hand_count": len(processed_hands),
            "hands": []
        }

        for hand in processed_hands:

            hand_record = {
                "handedness": hand["handedness"],
                "score": hand["score"],
                "coordinate_type": hand["coordinate_type"],
                "raw_landmarks": hand["raw_landmarks"],
                "normalized_landmarks": hand[
                    "normalized_landmarks"
                ],
                "features": hand["features"]
            }

            record["hands"].append(hand_record)

        # Append one JSON object per line
        with self.output_path.open(
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

        self.sample_count += 1

        return self.sample_count


def main():
    """
    Run the sign language data collection application.
    """

    print("=" * 45)
    print("       SilentVoice - Data Collection")
    print("=" * 45)

    label = input(
        "\nEnter the sign name to collect: "
    ).strip()

    if not label:
        print("Error: Sign name cannot be empty.")
        return

    camera = CameraManager()
    tracker = HandTracker()
    preprocessor = LandmarkPreprocessor(
        coordinate_type="world"
    )
    collector = SignDataCollector()

    try:
        camera.start()
        tracker.start()

        print(f"\nCurrent sign: {label}")
        print("Press SPACE to save a sample.")
        print("Press Q to exit.")
        print(f"\nDataset location: {collector.output_path}")

        while True:

            success, frame = camera.read()

            if not success:
                print("Failed to capture frame.")
                break

            # Create a strictly increasing frame timestamp
            import time
            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Detect hands
            detected_hands = tracker.process_frame(
                frame,
                timestamp_ms=timestamp_ms
            )

            # Preprocess detected hands
            processed_hands = preprocessor.process_hands(
                detected_hands
            )

            # Draw landmarks
            display_frame = tracker.draw_landmarks(
                frame,
                detected_hands
            )

            # Display collection information
            cv2.putText(
                display_frame,
                f"Sign: {label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Hands: {len(detected_hands)}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Samples: {collector.sample_count}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                "SPACE: Save | Q: Quit",
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "SilentVoice - Data Collection",
                display_frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Save sample when SPACE is pressed
            if key == ord(" "):

                if not processed_hands:
                    print(
                        "No hands detected. Sample not saved."
                    )
                    continue

                sample_number = collector.save_sample(
                    label=label,
                    processed_hands=processed_hands,
                    timestamp_ms=timestamp_ms
                )

                print(
                    f"Sample {sample_number} saved "
                    f"successfully! "
                    f"Hands detected: {len(processed_hands)}"
                )

            # Exit when Q is pressed
            elif key == ord("q"):
                break

    except (RuntimeError, FileNotFoundError, ValueError, OSError) as error:
        print(f"\nData Collection Error: {error}")

    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

        print("\nData collection finished.")
        print(f"Total samples saved: {collector.sample_count}")
        print(f"Dataset: {collector.output_path}")


if __name__ == "__main__":
    main()
