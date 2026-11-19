from backend.config import MOCK_MODE


def run_prediction(frames: list[str]) -> dict:

    if MOCK_MODE:
        return {
            "sign": "WATER",
            "confidence": 0.94
        }

    raise NotImplementedError(
        "Vision and ML are not connected yet"
    )