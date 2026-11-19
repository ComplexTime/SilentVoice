"""
SilentVoice - Display Window

A resizable OpenCV window that scales the camera image with it.

    - Drag the window edges: the image grows / shrinks with the window.
    - Press F: toggle full screen.
    - The image keeps its proportions (black bars are added
      instead of stretching the picture).
"""

import cv2

# Works both as "python -m vision.<module>" and from inside the vision folder
try:
    from . import config
except ImportError:
    import config


class DisplayWindow:
    """
    Resizable window for showing camera frames.

        window = DisplayWindow("SilentVoice")
        window.show(frame)
        key = window.wait_key()
    """

    def __init__(self, name, fullscreen=None):
        """
        Create the window.

        Args:
            name (str): Window title.
            fullscreen (bool | None): Start in full screen.
                None uses config.START_FULLSCREEN.
        """

        self.name = name
        self.fullscreen = (
            config.START_FULLSCREEN if fullscreen is None else fullscreen
        )

        # WINDOW_NORMAL lets the image scale with the window
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self._apply_fullscreen()

        self._sized = False

    def _apply_fullscreen(self):
        mode = (
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        )
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, mode)

    def toggle_fullscreen(self):
        """
        Switch between full screen and a normal window.
        """

        self.fullscreen = not self.fullscreen
        self._apply_fullscreen()

    def _window_size(self):
        """
        Return the current drawing area (width, height),
        or None if OpenCV cannot report it.
        """

        try:
            _, _, width, height = cv2.getWindowImageRect(self.name)
        except cv2.error:
            return None

        if width <= 0 or height <= 0:
            return None

        return width, height

    def _fit_to_window(self, frame):
        """
        Add black bars so the frame has the same shape as the window.
        This keeps the picture from being stretched.
        """

        size = self._window_size()

        if size is None:
            return frame

        window_width, window_height = size
        frame_height, frame_width = frame.shape[:2]

        window_ratio = window_width / window_height
        frame_ratio = frame_width / frame_height

        if abs(window_ratio - frame_ratio) < 0.01:
            return frame

        if window_ratio > frame_ratio:
            # Window is wider: add bars on the left and right
            new_width = int(round(frame_height * window_ratio))
            pad = new_width - frame_width
            left, right, top, bottom = pad // 2, pad - pad // 2, 0, 0
        else:
            # Window is taller: add bars on the top and bottom
            new_height = int(round(frame_width / window_ratio))
            pad = new_height - frame_height
            left, right, top, bottom = 0, 0, pad // 2, pad - pad // 2

        return cv2.copyMakeBorder(
            frame, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    def show(self, frame):
        """
        Show a frame, scaled to the window size.
        """

        if not self._sized and not self.fullscreen:
            # Start at a comfortable size: the frame size times the scale
            height, width = frame.shape[:2]
            cv2.resizeWindow(
                self.name,
                int(width * config.WINDOW_SCALE),
                int(height * config.WINDOW_SCALE)
            )
            self._sized = True

        cv2.imshow(self.name, self._fit_to_window(frame))

    def wait_key(self, delay=1):
        """
        Wait for a key press and handle the full screen key.

        Returns:
            int: The pressed key (0-255), or 255 if none.
        """

        key = cv2.waitKey(delay) & 0xFF

        if key in config.FULLSCREEN_KEYS:
            self.toggle_fullscreen()

        return key

    def is_closed(self):
        """
        True if the user closed the window with the X button.
        """

        try:
            return cv2.getWindowProperty(
                self.name, cv2.WND_PROP_VISIBLE
            ) < 1
        except cv2.error:
            return True

    def close(self):
        """
        Close this window.
        """

        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass
