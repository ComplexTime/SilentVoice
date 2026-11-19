"""
SilentVoice - Camera Manager

Handles camera initialization, frame capture,
and camera resource management using OpenCV.
"""

import logging

import cv2

# Works both as "python -m vision.<module>" and from inside the vision folder
try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages the computer camera using OpenCV.

    Can be used with a "with" statement:

        with CameraManager() as camera:
            success, frame = camera.read()
    """

    def __init__(
        self,
        camera_index=None,
        width=None,
        height=None,
        mirror=None,
        backend=None
    ):
        """
        Initialize the camera manager.

        Args:
            Any argument left as None uses the value from config.py.

            camera_index (int): Camera device index.
            width (int): Requested frame width.
            height (int): Requested frame height.
            mirror (bool): Flip frames horizontally (selfie view).
                Must be the same during data collection and in the app.
            backend (int | None): Optional OpenCV backend,
                e.g. cv2.CAP_DSHOW on Windows if the camera opens slowly.
        """
        self.camera_index = (
            config.CAMERA_INDEX if camera_index is None else camera_index
        )
        self.width = config.CAMERA_WIDTH if width is None else width
        self.height = config.CAMERA_HEIGHT if height is None else height
        self.mirror = config.CAMERA_MIRROR if mirror is None else mirror
        self.backend = config.CAMERA_BACKEND if backend is None else backend
        self.camera = None

    def start(self):
        """
        Open the camera and prepare it for frame capture.
        """
        if self.is_opened():
            return

        if self.backend is None:
            self.camera = cv2.VideoCapture(self.camera_index)
        else:
            self.camera = cv2.VideoCapture(self.camera_index, self.backend)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            raise RuntimeError(
                f"Could not open camera {self.camera_index}"
            )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # The camera may not support the requested resolution,
        # so store the resolution it actually uses.
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Camera %s started at %dx%d.",
            self.camera_index, self.width, self.height
        )

    def read(self):
        """
        Capture a single frame from the camera.

        Returns:
            tuple: (success, frame). frame is None when success is False.
        """
        if not self.is_opened():
            raise RuntimeError("Camera is not started.")

        success, frame = self.camera.read()

        if not success or frame is None:
            return False, None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def is_opened(self):
        """
        Check whether the camera is currently open.
        """
        return self.camera is not None and self.camera.isOpened()

    def release(self):
        """
        Release the camera and free its resources.
        """
        if self.camera is None:
            return

        self.camera.release()
        self.camera = None
        logger.info("Camera released.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


# Camera testing
if __name__ == "__main__":

    try:
        from .display import DisplayWindow
    except ImportError:
        from display import DisplayWindow

    logging.basicConfig(level=logging.INFO)

    try:
        with CameraManager() as camera:

            window = DisplayWindow(config.CAMERA_WINDOW_NAME)

            print("F: full screen | Q/ESC: exit")

            while True:
                success, frame = camera.read()

                if not success:
                    print("Failed to capture frame.")
                    break

                window.show(frame)

                key = window.wait_key()
                if key in config.QUIT_KEYS or window.is_closed():
                    break

    except RuntimeError as error:
        print(f"Camera Error: {error}")

    finally:
        cv2.destroyAllWindows()