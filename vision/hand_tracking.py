"""
SilentVoice - Hand Tracking

Detects and tracks hands using MediaPipe.
Extracts 21 landmarks from each detected hand
and identifies whether it is a left or right hand.

Note: MediaPipe's left/right labels assume a mirrored (selfie) image,
so use CameraManager(mirror=True).
"""

import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Works both as "python -m vision.<module>" and from inside the vision folder
try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


class HandTracker:
    """
    Detects hands and extracts their landmarks using MediaPipe.
    """

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    def __init__(
        self,
        model_path=None,
        max_hands=None,
        detection_confidence=None,
        presence_confidence=None,
        tracking_confidence=None
    ):
        """
        Initialize the hand tracker.

        Any argument left as None uses the value from config.py.

        Args:
            model_path: Path to the MediaPipe model.
            max_hands: Maximum number of hands to detect.
            detection_confidence: Minimum detection confidence.
            presence_confidence: Minimum hand presence confidence.
            tracking_confidence: Minimum tracking confidence.
        """

        if model_path is None:
            model_path = config.HAND_LANDMARKER_MODEL
        if max_hands is None:
            max_hands = config.MAX_HANDS
        if detection_confidence is None:
            detection_confidence = config.MIN_HAND_DETECTION_CONFIDENCE
        if presence_confidence is None:
            presence_confidence = config.MIN_HAND_PRESENCE_CONFIDENCE
        if tracking_confidence is None:
            tracking_confidence = config.MIN_TRACKING_CONFIDENCE

        if max_hands < 1:
            raise ValueError("max_hands must be at least 1.")

        confidences = [
            detection_confidence,
            presence_confidence,
            tracking_confidence
        ]

        if any(not 0.0 <= value <= 1.0 for value in confidences):
            raise ValueError("Confidence values must be between 0 and 1.")

        self.model_path = Path(model_path)
        self.max_hands = max_hands

        self.detection_confidence = detection_confidence
        self.presence_confidence = presence_confidence
        self.tracking_confidence = tracking_confidence

        self.landmarker = None
        self.last_timestamp_ms = -1

    def start(self):
        """
        Load the MediaPipe model and initialize hand tracking.
        """

        if self.landmarker is not None:
            return

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}\n"
                f"Download it from:\n{config.HAND_LANDMARKER_URL}"
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_hand_presence_confidence=self.presence_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )

        # A new landmarker starts a new video stream
        self.last_timestamp_ms = -1

        logger.info("Hand tracking initialized.")

    def process_frame(self, frame, timestamp_ms=None):
        """
        Detect hands and extract their landmarks from a frame.

        Args:
            frame: OpenCV image in BGR format.
            timestamp_ms: Frame timestamp in milliseconds.
                If omitted, the current time is used.

        Returns:
            list[dict]: One dict per detected hand with the keys
                "handedness", "score", "landmarks", "world_landmarks".
        """

        if self.landmarker is None:
            raise RuntimeError("HandTracker is not initialized.")

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        if timestamp_ms is None:
            timestamp_ms = time.monotonic_ns() // 1_000_000

        timestamp_ms = int(timestamp_ms)

        # MediaPipe requires strictly increasing timestamps.
        # Instead of crashing on a duplicate, move forward by 1 ms.
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        # Convert OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image to MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run hand detection and tracking
        results = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        detected_hands = []

        # Process every detected hand
        for index, hand_landmarks in enumerate(results.hand_landmarks):

            handedness = "Unknown"
            score = 0.0

            if index < len(results.handedness) and results.handedness[index]:
                category = results.handedness[index][0]
                handedness = category.category_name
                score = float(category.score)

            # 21 landmarks normalized to the image (0..1)
            landmarks = [
                (float(point.x), float(point.y), float(point.z))
                for point in hand_landmarks
            ]

            # 21 landmarks in meters, relative to the hand center
            world_landmarks = []
            if index < len(results.hand_world_landmarks):
                world_landmarks = [
                    (float(point.x), float(point.y), float(point.z))
                    for point in results.hand_world_landmarks[index]
                ]

            detected_hands.append({
                "handedness": handedness,
                "score": score,
                "landmarks": landmarks,
                "world_landmarks": world_landmarks
            })

        return detected_hands

    def draw_landmarks(self, frame, detected_hands):
        """
        Draw hand landmarks and connections on the image.

        Note: draws on the given frame (in place).
        Pass frame.copy() to keep the original image clean.

        Args:
            frame: OpenCV image.
            detected_hands: Output from process_frame().

        Returns:
            numpy.ndarray: Image with hand landmarks drawn.
        """

        height, width = frame.shape[:2]

        for hand in detected_hands:

            points = hand["landmarks"]

            pixel_points = [
                (
                    max(0, min(int(x * width), width - 1)),
                    max(0, min(int(y * height), height - 1))
                )
                for x, y, _ in points
            ]

            # Draw connections between landmarks
            for start, end in self.HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    pixel_points[start],
                    pixel_points[end],
                    (0, 255, 0),
                    2
                )

            # Draw each landmark
            for point in pixel_points:
                cv2.circle(frame, point, 5, (255, 0, 0), -1)

            # Display handedness near the wrist
            wrist_x, wrist_y = pixel_points[0]

            label = f"{hand['handedness']} ({hand['score']:.2f})"

            cv2.putText(
                frame,
                label,
                (wrist_x, max(25, wrist_y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        return frame

    def close(self):
        """
        Release MediaPipe resources.
        """

        if self.landmarker is None:
            return

        self.landmarker.close()
        self.landmarker = None
        logger.info("Hand tracker released.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# Standalone hand tracking test
if __name__ == "__main__":

    # Works both as "python -m vision.hand_tracking" (from the project root)
    # and as "python hand_tracking.py" (from inside the vision folder).
    try:
        from .camera import CameraManager
    except ImportError:
        from camera import CameraManager

    logging.basicConfig(level=logging.INFO)

    try:
        with CameraManager() as camera, HandTracker() as tracker:

            print("Hand tracking is running.")
            print("Press Q to exit.")

            while True:

                success, frame = camera.read()

                if not success:
                    print("Failed to read camera frame.")
                    break

                # Detect hands
                hands = tracker.process_frame(frame)

                # Draw detected landmarks
                frame = tracker.draw_landmarks(frame, hands)

                # Display number of detected hands
                cv2.putText(
                    frame,
                    f"Hands detected: {len(hands)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(config.TRACKING_WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in config.QUIT_KEYS:
                    break

    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"Hand Tracking Error: {error}")

    finally:
        cv2.destroyAllWindows()