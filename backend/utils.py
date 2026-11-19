import base64
import binascii
import cv2
import numpy as np


def decode_frame(encoded_frame: str) -> np.ndarray:

    try:
        image_bytes = base64.b64decode(
            encoded_frame,
            validate=True
        )
    except (ValueError, binascii.Error) as error:
        raise ValueError("Invalid Base64 image") from error

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    if image_array.size == 0:
        raise ValueError("Empty image")

    try:
        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )
    except cv2.error as error:
        raise ValueError("Invalid image") from error

    if frame is None:
        raise ValueError("Could not decode image")

    return frame