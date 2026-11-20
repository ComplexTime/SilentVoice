import time

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from ml import config
from ml.dataset import load_npz
from ml.predict import load_model, predict_sign


def evaluate():

    X, y, groups = load_npz()

    artifact = load_model(force=True)

    if artifact.get("synthetic"):
        raise ValueError(
            "Cannot evaluate real accuracy using "
            "a synthetic model."
        )

    train_groups = set(artifact["train_groups"])
    test_groups = set(artifact["test_groups"])

    if train_groups & test_groups:
        raise ValueError("Training and testing overlap")

    test_mask = np.isin(groups, list(test_groups))

    X_test = X[test_mask]
    y_test = y[test_mask]

    if len(X_test) == 0:
        raise ValueError("No testing samples found")

    if set(groups[test_mask]) != test_groups:
        raise ValueError("Missing test groups")

    if len(X_test) != artifact["n_test"]:
        raise ValueError(
            "Dataset differs from the original "
            "training dataset"
        )

    predictions = []
    latencies = []

    for sequence in X_test:

        start = time.perf_counter()

        result = predict_sign(sequence)

        elapsed = time.perf_counter() - start

        predictions.append(result["sign"])
        latencies.append(elapsed * 1000)

    predictions = np.asarray(predictions)

    accuracy = accuracy_score(
        y_test, predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    labels = sorted(
        set(y_test) |
        set(predictions) |
        {config.UNKNOWN_LABEL}
    )

    print("\n=== SILENTVOICE ML EVALUATION ===")

    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {f1:.4f}")

    print(
        f"Median latency: "
        f"{np.median(latencies):.2f} ms"
    )

    print(
        f"P95 latency: "
        f"{np.percentile(latencies, 95):.2f} ms"
    )

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    ))

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45
    )

    plt.title("SilentVoice ASL Confusion Matrix")
    plt.tight_layout()

    output_path = (
        config.MODELS_DIR / "confusion_matrix.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"\nConfusion matrix saved: {output_path}")


if __name__ == "__main__":
    evaluate()