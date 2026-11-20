"""
SilentVoice - Landmark Preprocessing

Processes MediaPipe hand landmarks for machine learning.

Features:
    1. Landmark validation.
    2. Aspect-ratio correction for image landmarks.
    3. Translation normalization (wrist at the origin).
    4. Hand scale normalization.
    5. Optional mirroring of left hands to look like right hands.
    6. Feature vector generation (one hand: 63 values,
       two hands: 126 values).
    7. Preservation of original landmark data.

Rotation is NOT normalized on purpose: in sign language the
orientation of the hand can change the meaning of a sign.
"""

import math

# Works both as "python -m vision.<module>" and from inside the vision folder
try:
    from . import config
except ImportError:
    import config


class LandmarkPreprocessor:
    """
    Prepares hand landmarks for machine learning.
    """

    LANDMARK_COUNT = config.HAND_LANDMARK_COUNT            # 21
    FEATURES_PER_HAND = config.FEATURE_VECTOR_SIZE         # 63

    # MCP landmarks used to estimate palm size
    PALM_REFERENCE_INDICES = (5, 9, 13, 17)

    WRIST_INDEX = 0

    # Order of the hands in the two-hand feature vector
    HAND_SLOTS = ("Left", "Right")

    def __init__(
        self,
        coordinate_type=None,
        mirror_left_hand=None,
        image_size=None
    ):
        """
        Initialize the preprocessor.

        Any argument left as None uses the value from config.py.

        Args:
            coordinate_type (str):
                "world" for 3D world landmarks (recommended).
                "image" for normalized image landmarks.
            mirror_left_hand (bool):
                Flip left hands on the x axis so they look like
                right hands. Lets one model work for right- and
                left-handed users. Must be the same for training
                and for the app.
            image_size (tuple):
                (width, height) of the camera frames. Only used
                when coordinate_type is "image". Pass the camera's
                actual resolution (camera.width, camera.height).
        """

        if coordinate_type is None:
            coordinate_type = config.COORDINATE_TYPE
        if mirror_left_hand is None:
            mirror_left_hand = config.MIRROR_LEFT_HAND
        if image_size is None:
            image_size = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        if coordinate_type not in ("world", "image"):
            raise ValueError(
                "coordinate_type must be 'world' or 'image'."
            )

        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size values must be positive.")

        self.coordinate_type = coordinate_type
        self.mirror_left_hand = mirror_left_hand
        self.image_size = (width, height)
        self.scale_epsilon = 1e-8

    def _validate_landmarks(self, landmarks):
        """
        Validate and convert landmarks into numeric tuples.

        Each hand must contain exactly 21 points.
        Each point must contain x, y and z.
        """

        if not isinstance(landmarks, (list, tuple)):
            raise TypeError("Landmarks must be a list or tuple.")

        if len(landmarks) != self.LANDMARK_COUNT:
            raise ValueError(
                f"Expected {self.LANDMARK_COUNT} landmarks, "
                f"received {len(landmarks)}."
            )

        validated = []

        for index, point in enumerate(landmarks):

            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError(
                    f"Landmark {index} must contain x, y and z."
                )

            coordinates = tuple(float(value) for value in point)

            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(
                    f"Landmark {index} contains invalid coordinates."
                )

            validated.append(coordinates)

        return validated

    def _correct_aspect_ratio(self, landmarks):
        """
        Image landmarks are divided by the frame width (x) and
        height (y), so on a 640x480 frame one unit of x is not
        the same length as one unit of y. Convert them to pixels
        so distances are correct. MediaPipe's z uses roughly the
        same scale as x.
        """

        width, height = self.image_size

        return [
            (x * width, y * height, z * width)
            for x, y, z in landmarks
        ]

    def _calculate_hand_scale(self, landmarks):
        """
        Estimate hand scale using the wrist and four palm landmarks.

        The average distance from the wrist to the MCP landmarks
        is used as the normalization factor.
        """

        wrist = landmarks[self.WRIST_INDEX]

        distances = [
            math.dist(wrist, landmarks[index])
            for index in self.PALM_REFERENCE_INDICES
        ]

        scale = sum(distances) / len(distances)

        if scale <= self.scale_epsilon:
            raise ValueError(
                "Cannot normalize landmarks: hand scale is too small."
            )

        return scale

    def normalize_landmarks(self, landmarks, handedness=None):
        """
        Normalize hand landmarks.

        Step 1: (image only) Correct the aspect ratio.
        Step 2: Move the wrist to the origin (0, 0, 0).
        Step 3: Divide coordinates by the estimated hand scale.
        Step 4: (optional) Mirror left hands on the x axis.

        Args:
            landmarks: 21 (x, y, z) points.
            handedness: "Left", "Right" or None.

        Returns:
            list: 21 normalized 3D landmark coordinates.
        """

        landmarks = self._validate_landmarks(landmarks)

        if self.coordinate_type == "image":
            landmarks = self._correct_aspect_ratio(landmarks)

        wrist = landmarks[self.WRIST_INDEX]
        hand_scale = self._calculate_hand_scale(landmarks)

        flip_x = self.mirror_left_hand and handedness == "Left"
        x_sign = -1.0 if flip_x else 1.0

        return [
            (
                x_sign * (x - wrist[0]) / hand_scale,
                (y - wrist[1]) / hand_scale,
                (z - wrist[2]) / hand_scale
            )
            for x, y, z in landmarks
        ]

    def flatten_landmarks(self, landmarks):
        """
        Convert 21 3D landmarks into a 63-value feature vector.

        Output:
            [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        """

        landmarks = self._validate_landmarks(landmarks)

        return [
            coordinate
            for point in landmarks
            for coordinate in point
        ]

    def process_hand(self, hand_data):
        """
        Process one detected hand.

        Args:
            hand_data (dict):
                Output dictionary from HandTracker.

        Returns:
            dict: Original and normalized landmarks,
                  together with the 63-value feature vector.
        """

        if not isinstance(hand_data, dict):
            raise TypeError("hand_data must be a dictionary.")

        source_key = (
            "world_landmarks"
            if self.coordinate_type == "world"
            else "landmarks"
        )

        if source_key not in hand_data:
            raise KeyError(
                f"Missing '{source_key}' in hand data."
            )

        handedness = hand_data.get("handedness", "Unknown")

        raw_landmarks = self._validate_landmarks(hand_data[source_key])
        normalized = self.normalize_landmarks(raw_landmarks, handedness)
        features = self.flatten_landmarks(normalized)

        return {
            "handedness": handedness,
            "score": float(hand_data.get("score", 0.0)),
            "coordinate_type": self.coordinate_type,
            "mirrored": self.mirror_left_hand and handedness == "Left",
            "raw_landmarks": raw_landmarks,
            "normalized_landmarks": normalized,
            "features": features
        }

    def process_hands(self, detected_hands):
        """
        Process multiple detected hands.

        Args:
            detected_hands (list):
                List of detected hands from HandTracker.

        Returns:
            list: Processed hand data.
        """

        if not isinstance(detected_hands, (list, tuple)):
            raise TypeError(
                "detected_hands must be a list or tuple."
            )

        return [
            self.process_hand(hand)
            for hand in detected_hands
        ]

    def combine_hands(self, processed_hands):
        """
        Build one fixed-size feature vector for two-handed signs.

        Output (126 values):
            [63 values of the left hand, 63 values of the right hand]

        A missing hand is filled with zeros. If MediaPipe reports
        the same hand twice, the one with the higher score is used.

        Args:
            processed_hands (list): Output from process_hands().

        Returns:
            list: 126 feature values.
        """

        best = {}

        for hand in processed_hands:
            slot = hand["handedness"]

            if slot not in self.HAND_SLOTS:
                continue

            if slot not in best or hand["score"] > best[slot]["score"]:
                best[slot] = hand

        features = []

        for slot in self.HAND_SLOTS:
            if slot in best:
                features.extend(best[slot]["features"])
            else:
                features.extend([0.0] * self.FEATURES_PER_HAND)

        return features


# Standalone preprocessing test
if __name__ == "__main__":

    import random

    random.seed(0)

    def make_hand(offset=(0.0, 0.0, 0.0), scale=1.0):
        """Create a random but repeatable fake hand."""
        rng = random.Random(42)
        return [
            (
                offset[0] + scale * rng.uniform(-0.05, 0.05),
                offset[1] + scale * rng.uniform(-0.05, 0.05),
                offset[2] + scale * rng.uniform(-0.05, 0.05)
            )
            for _ in range(21)
        ]

    def close_enough(a, b, tolerance=1e-9):
        return all(abs(x - y) < tolerance for x, y in zip(a, b))

    preprocessor = LandmarkPreprocessor(
        coordinate_type="world",
        mirror_left_hand=False
    )

    base = make_hand()
    sample_hand = {
        "handedness": "Right",
        "score": 0.98,
        "landmarks": base,
        "world_landmarks": base
    }

    result = preprocessor.process_hand(sample_hand)

    # 1. Shapes
    assert len(result["normalized_landmarks"]) == 21
    assert len(result["features"]) == 63
    assert result["normalized_landmarks"][0] == (0.0, 0.0, 0.0)

    # 2. Moving the hand must not change the features
    moved = dict(sample_hand, world_landmarks=make_hand(offset=(1, 2, 3)))
    assert close_enough(
        result["features"],
        preprocessor.process_hand(moved)["features"]
    )

    # 3. A bigger / closer hand must not change the features
    bigger = dict(sample_hand, world_landmarks=make_hand(scale=2.5))
    assert close_enough(
        result["features"],
        preprocessor.process_hand(bigger)["features"]
    )

    # 4. Mirroring: a left hand flips the x values only
    mirroring = LandmarkPreprocessor(
        coordinate_type="world",
        mirror_left_hand=True
    )
    left = mirroring.process_hand(dict(sample_hand, handedness="Left"))
    right = mirroring.process_hand(sample_hand)
    for (lx, ly, lz), (rx, ry, rz) in zip(
        left["normalized_landmarks"], right["normalized_landmarks"]
    ):
        assert abs(lx + rx) < 1e-9 and ly == ry and lz == rz

    # 5. Image coordinates: aspect ratio is corrected
    image_pre = LandmarkPreprocessor(
        coordinate_type="image",
        image_size=(640, 480)
    )
    image_result = image_pre.process_hand(sample_hand)
    assert len(image_result["features"]) == 63

    # 6. Two-hand vector: missing hand is zeros
    combined = preprocessor.combine_hands([result])
    assert len(combined) == 126
    assert combined[:63] == [0.0] * 63          # no left hand
    assert combined[63:] == result["features"]  # right hand

    # 7. Bad input is rejected
    try:
        preprocessor.normalize_landmarks(base[:20])
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    print("All preprocessing tests passed!")