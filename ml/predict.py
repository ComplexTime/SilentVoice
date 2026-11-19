
"""Prediction API used by the backend.

    from ml.predict import predict_sign
    predict_sign(sequence)   # sequence shape (15, 128)
    # -> {"sign": "WATER", "confidence": 0.94}

Importing this module never trains and never loads the model: the model is loaded
lazily on the first prediction and then reused.
Only load joblib files produced by our own team: joblib uses pickle internally.
"""
import argparse
import threading
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

from ml import config
from ml.dataset import make_synthetic_dataset, validate_dataset, validate_sequence

_FLAG_IDX = [config.LEFT_DETECTED_IDX, config.RIGHT_DETECTED_IDX]
_lock = threading.Lock()
_artifact = None


def _check_artifact(artifact, path):
    required = {"model", "classes", "input_shape", "library_versions"}
    if not isinstance(artifact, dict) or not required <= set(artifact):
        raise ValueError(f"{path} is not a SilentVoice model file. Re-run: python -m ml.train")
    if tuple(artifact["input_shape"]) != (config.NUM_FRAMES, config.NUM_FEATURES):
        raise ValueError(
            f"Model expects input {artifact['input_shape']}, but the contract is "
            f"{(config.NUM_FRAMES, config.NUM_FEATURES)}."
        )
    if artifact.get("language", config.DEFAULT_LANGUAGE) != config.DEFAULT_LANGUAGE:
        raise ValueError(f"Model language {artifact['language']!r} != {config.DEFAULT_LANGUAGE!r}")
    if not hasattr(artifact["model"], "predict_proba"):
        raise ValueError("Model has no predict_proba.")
    saved = artifact["library_versions"].get("scikit-learn")
    if saved != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {saved} but {sklearn.__version__} is "
            "installed. Install ml/requirements.txt to match."
        )
    if artifact.get("synthetic"):
        warnings.warn("Loaded a SYNTHETIC model (pipeline test only). "
                      "It must not be used for real recognition.")


def load_model(path=None, force=False):
    """Load the model artifact once and cache it. Only load trusted team-made files."""
    global _artifact
    with _lock:
        if _artifact is not None and not force:
            return _artifact
        path = Path(path) if path else config.MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Train it first with: python -m ml.train"
            )
        artifact = joblib.load(path)
        _check_artifact(artifact, path)
        # One sample per call: a single thread is faster than spinning up a thread pool.
        artifact["model"].n_jobs = 1
        _artifact = artifact
        return artifact


def predict_sign(sequence):
    """Classify one sequence of shape (15, 128).

    Returns {"sign": str, "confidence": float in [0, 1]}.
    Raises ValueError for invalid input (wrong shape, NaN/Inf, non-numeric, bad flags).
    Returns UNKNOWN when no hands are detected (confidence 0.0) or when the model's
    top probability is below the confidence threshold.
    Note: predict_proba is the model's estimated probability, not a guarantee of correctness.
    """
    seq = validate_sequence(sequence)

    if not seq[:, _FLAG_IDX].any():
        return {"sign": config.UNKNOWN_LABEL, "confidence": 0.0}

    artifact = _artifact if _artifact is not None else load_model()
    model = artifact["model"]
    probs = model.predict_proba(seq.reshape(1, -1))[0]
    best = int(np.argmax(probs))
    sign = str(model.classes_[best])
    confidence = float(min(max(probs[best], 0.0), 1.0))
    if confidence < config.CONFIDENCE_THRESHOLD:
        sign = config.UNKNOWN_LABEL
    return {"sign": sign, "confidence": confidence}


# ---------------------------------------------------------------------------
# Self-test: checks the interface and latency only (NOT recognition accuracy).
# ---------------------------------------------------------------------------
def _check_result(result):
    assert set(result) == {"sign", "confidence"}, result
    assert type(result["sign"]) is str, type(result["sign"])
    assert type(result["confidence"]) is float, type(result["confidence"])
    assert 0.0 <= result["confidence"] <= 1.0, result["confidence"]


def _selftest(model_path):
    print("SYNTHETIC INPUTS - checks the interface and latency only, "
          "not recognition accuracy.\n")
    first = load_model(model_path, force=True)

    X, _, _ = validate_dataset(*make_synthetic_dataset(seed=123))
    with_hands = [s for s in X if s[:, _FLAG_IDX].any()]

    result = predict_sign(with_hands[0])
    _check_result(result)
    print(f"Result format OK: {result}")

    no_hands = np.zeros((config.NUM_FRAMES, config.NUM_FEATURES), dtype=np.float32)
    result = predict_sign(no_hands)
    assert result == {"sign": config.UNKNOWN_LABEL, "confidence": 0.0}, result
    print(f"No hands -> {result}")

    nan_seq = with_hands[0].copy()
    nan_seq[0, 0] = np.nan
    inf_seq = with_hands[0].copy()
    inf_seq[0, 0] = np.inf
    bad_flag = with_hands[0].copy()
    bad_flag[0, config.LEFT_DETECTED_IDX] = 2
    bad_inputs = {
        "wrong shape": np.zeros((10, config.NUM_FEATURES)),
        "NaN value": nan_seq,
        "infinite value": inf_seq,
        "invalid hand flag": bad_flag,
        "non-numeric": [["a"] * config.NUM_FEATURES] * config.NUM_FRAMES,
        "None": None,
    }
    for label, bad in bad_inputs.items():
        try:
            predict_sign(bad)
        except ValueError:
            print(f"Rejected as expected: {label}")
        else:
            raise AssertionError(f"{label} was not rejected")

    predict_sign(with_hands[1])
    assert load_model() is first, "model was reloaded"
    print("Model loaded once and reused: OK")

    times = []
    for i in range(210):
        seq = with_hands[i % len(with_hands)]
        start = time.perf_counter()
        predict_sign(seq)
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times[10:])  # skip warm-up calls
    print(f"Latency over {len(times)} predictions on this machine: "
          f"median {np.median(times):.1f} ms, p95 {np.percentile(times, 95):.1f} ms")

    print("\nAll predict checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(config.MODELS_DIR / "asl_model_synthetic.joblib"),
                        help="model file for the self-test (default: the synthetic test model)")
    args = parser.parse_args()
    _selftest(args.model)
    