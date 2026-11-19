"""
SilentVoice - Computer Vision Package

Handles camera testing, MediaPipe hand tracking,
landmark normalization, and Israeli Sign Language (ISL)
training data collection.

Usage:
    from vision import config
    from vision.camera import CameraManager
    from vision.hand_tracking import HandTracker
    from vision.preprocessing import LandmarkPreprocessor

Run from the project root:
    python -m vision.config          # check settings and model file
    python -m vision.camera
    python -m vision.hand_tracking
    python -m vision.preprocessing
    python -m vision.collect_data    # collect training samples
"""

__version__ = "0.1.0"
__author__ = "SilentVoice Team"