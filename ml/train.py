
"""Train the ASL sign classifier (RandomForest) and export it with joblib.

Usage (from the repo root, venv active):
    python -m ml.train --synthetic   # pipeline test only (saves a *_synthetic model)
    python -m ml.train               # real data/training.npz -> ml/models/asl_model.joblib

Only load joblib files produced by our own team: joblib uses pickle internally.
"""
from ml.augment import augment_training
import argparse
import platform
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from ml import config
from ml.dataset import (
    describe_dataset,
    flatten_sequences,
    group_train_test_split,
    load_npz,
    make_synthetic_dataset,
    validate_dataset,
)

def build_model():
    return RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        class_weight="balanced"
    )


def library_versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }


def train(X, y, groups, synthetic=False):
    """Split by group, fit on the TRAIN groups only, return (artifact, train_idx, test_idx)."""
    train_idx, test_idx = group_train_test_split(y, groups)
    X_flat = flatten_sequences(X)

    model = build_model()
    start = time.perf_counter()
    X_train, y_train = augment_training(X[train_idx],y[train_idx],copies=5)
    print("Augmented training shape:", X_train.shape)
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    print(f"Training took {seconds:.2f} s")

    artifact = {
        "model": model,
        "language": config.DEFAULT_LANGUAGE,
        "classes": [str(c) for c in model.classes_],
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "input_shape": (config.NUM_FRAMES, config.NUM_FEATURES),
        "train_groups": sorted({str(g) for g in groups[train_idx]}),
        "test_groups": sorted({str(g) for g in groups[test_idx]}),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "synthetic": bool(synthetic),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": library_versions(),
    }
    return artifact, train_idx, test_idx


def save_artifact(artifact, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    print(f"Saved model to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="train on random synthetic data (pipeline test only)")
    parser.add_argument("--data", default=str(config.DATA_PATH),
                        help="path to the training .npz (default: data/training.npz)")
    args = parser.parse_args()

    if args.synthetic:
        print("SYNTHETIC DATA - pipeline test only, not evidence of recognition accuracy.\n")
        X, y, groups = validate_dataset(*make_synthetic_dataset())
        out_path = config.MODELS_DIR / "asl_model_synthetic.joblib"
    else:
        X, y, groups = load_npz(args.data)
        out_path = config.MODEL_PATH
        expected = set(config.VOCABULARY[config.DEFAULT_LANGUAGE]) | {config.UNKNOWN_LABEL}
        if set(y) != expected:
            print(f"WARNING: dataset labels {sorted(set(y))} differ from config vocabulary "
                  f"{sorted(expected)}. Update VOCABULARY in ml/config.py once the team agrees.")

    print(describe_dataset(X, y, groups))
    artifact, _, _ = train(X, y, groups, synthetic=args.synthetic)
    print(f"Train: {artifact['n_train']} samples, groups {artifact['train_groups']}")
    print(f"Test:  {artifact['n_test']} samples, groups {artifact['test_groups']}")
    print(f"Library versions: {artifact['library_versions']}")
    save_artifact(artifact, out_path)
    print("Next: measure real performance on the TEST groups with ml/evaluate.py.")


if __name__ == "__main__":
    main()