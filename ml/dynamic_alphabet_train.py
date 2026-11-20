
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from ml.dynamic_alphabet_features import (
    extract_dynamic_features
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "dynamic_alphabet_training.npz"

MODEL_PATH = (
    ROOT / "ml" / "models" /
    "dynamic_alphabet_model.joblib"
)

CONFIDENCE_THRESHOLD = 0.35
RANDOM_STATE = 42


def main():

    data = np.load(DATA_PATH)

    X = []
    y = []

    for sequence, label in zip(data["X"], data["y"]):

        features = extract_dynamic_features(sequence)

        if features is not None:
            X.append(features)
            y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    print("Training shape:", X.shape)

    classes, counts = np.unique(
        y, return_counts=True
    )

    for label, count in zip(classes, counts):
        print(f"{label}: {count}")

    if set(classes) != {"J", "Z", "OTHER"}:
        raise RuntimeError("Missing training class")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=1
    )

    folds = min(5, int(counts.min()))

    if folds < 2:
        raise RuntimeError("Insufficient samples")

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    print("\nEvaluating classifier...")

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )

    predictions = classes[
        np.argmax(probabilities, axis=1)
    ]

    confidence = probabilities.max(axis=1)


    print("\nDYNAMIC THRESHOLD TEST")

    dynamic_labels = np.array(["J", "Z"])

    dynamic_probs = probabilities[:, [
        list(classes).index("J"),
        list(classes).index("Z")
    ]]

    best_label = dynamic_labels[
        dynamic_probs.argmax(axis=1)
    ]

    best_conf = dynamic_probs.max(axis=1)

    for threshold in np.arange(0.10, 0.65, 0.05):

        pred = np.where(
            best_conf >= threshold,
            best_label,
            "OTHER"
        )

        j = np.sum((y == "J") & (pred == "J"))
        z = np.sum((y == "Z") & (pred == "Z"))

        fp = np.sum(
            (y == "OTHER") & (pred != "OTHER")
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"J: {j}/10 | "
            f"Z: {z}/7 | "
            f"False positives: {fp}/237"
        )


    print("\nDYNAMIC THRESHOLD TEST")

    j_idx = list(classes).index("J")
    z_idx = list(classes).index("Z")

    dynamic_probs = probabilities[:, [j_idx, z_idx]]
    dynamic_labels = np.array(["J", "Z"])

    best_idx = dynamic_probs.argmax(axis=1)
    best_conf = dynamic_probs.max(axis=1)
    best_label = dynamic_labels[best_idx]

    for threshold in [0.10, 0.15, 0.20, 0.25,
                      0.30, 0.35, 0.40, 0.50, 0.60]:

        pred = np.where(
            best_conf >= threshold,
            best_label,
            "OTHER"
        )

        j_correct = np.sum((y == "J") & (pred == "J"))
        z_correct = np.sum((y == "Z") & (pred == "Z"))
        false_positive = np.sum(
            (y == "OTHER") & (pred != "OTHER")
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"J: {j_correct}/{np.sum(y == 'J')} | "
            f"Z: {z_correct}/{np.sum(y == 'Z')} | "
            f"False positives: {false_positive}"
        )


    print("\nRAW CLASSIFICATION REPORT")

    print(classification_report(
        y,
        predictions,
        labels=["J", "Z", "OTHER"],
        digits=4,
        zero_division=0
    ))

    # Reject uncertain dynamic predictions.
    predictions = np.where(
        best_conf >= CONFIDENCE_THRESHOLD,
        best_label,
        "OTHER"
    )

    print("\nTHRESHOLD CLASSIFICATION REPORT")

    print(classification_report(
        y,
        predictions,
        labels=["J", "Z", "OTHER"],
        digits=4,
        zero_division=0
    ))

    print("CONFUSION MATRIX")
    print("Order: J, Z, OTHER")

    print(confusion_matrix(
        y,
        predictions,
        labels=["J", "Z", "OTHER"]
    ))

    print("\nTraining final model...")

    model.fit(X, y)
    model.n_jobs = 1

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump({
        "model": model,
        "classes": model.classes_,
        "num_features": X.shape[1],
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }, MODEL_PATH, compress=3)

    print("\nMODEL SAVED:", MODEL_PATH)


if __name__ == "__main__":
    main()
