
import time
import joblib
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from ml.alphabet_config import (
    DATA_PATH,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    RANDOM_STATE,
    LETTERS
)


def main():

    with np.load(DATA_PATH, allow_pickle=False) as data:

        X_train = data["X_train"]
        y_train = data["y_train"]

        X_test = data["X_test"]
        y_test = data["y_test"]

    if set(y_train) != set(LETTERS):
        raise ValueError("Missing training classes")

    if set(y_test) != set(LETTERS):
        raise ValueError("Missing testing classes")

    print("Training samples:", X_train.shape)
    print("Testing samples:", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    start = time.perf_counter()

    print("\nTraining alphabet model...")
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    probabilities = model.predict_proba(X_test)

    best = np.argmax(probabilities, axis=1)

    confidence = np.max(probabilities, axis=1)

    predictions = model.classes_[best]

    rejected = np.where(
        confidence >= CONFIDENCE_THRESHOLD,
        predictions,
        "UNKNOWN"
    )

    accuracy = accuracy_score(
        y_test, predictions
    )

    threshold_accuracy = accuracy_score(
        y_test, rejected
    )

    macro_f1 = f1_score(
        y_test,
        rejected,
        labels=list(LETTERS),
        average="macro",
        zero_division=0
    )

    print("\n=== ALPHABET MODEL RESULTS ===")

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Accuracy: {accuracy:.2%}")

    print(
        f"Accuracy with rejection: "
        f"{threshold_accuracy:.2%}"
    )

    print(f"Macro F1: {macro_f1:.4f}")

    print(
        f"Rejected samples: "
        f"{np.sum(rejected == 'UNKNOWN')}"
    )

    print("\nClassification report:")

    print(classification_report(
        y_test,
        rejected,
        labels=list(LETTERS),
        zero_division=0
    ))

    artifact = {
        "model": model,
        "classes": list(model.classes_),
        "num_features": 63,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "accuracy": accuracy,
        "threshold_accuracy": threshold_accuracy,
        "macro_f1": macro_f1,
        "sklearn_version": sklearn.__version__
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.n_jobs = 1

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3
    )

    print("\nModel saved:", MODEL_PATH)


if __name__ == "__main__":
    main()

