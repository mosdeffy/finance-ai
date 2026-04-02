"""
evaluate.py
Evaluates the trained LSTM model using ROC curve analysis,
optimal threshold selection via Youden's J statistic,
and generates classification metrics and a saved ROC plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    classification_report, confusion_matrix,
)
import tensorflow as tf


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray):
    """Select threshold that maximizes Youden's J statistic (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], fpr, tpr


def plot_roc_curve(fpr, tpr, auc_score: float, threshold: float, save_path: str = "model/roc_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Classifier")
    plt.scatter([], [], color="green", label=f"Optimal Threshold: {threshold:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LSTM Stock Prediction — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"ROC curve saved to {save_path}")


def main():
    model = tf.keras.models.load_model("model/saved_model.keras")
    X_test = np.load("model/X_test.npy")
    y_test = np.load("model/y_test.npy")

    y_scores = model.predict(X_test).flatten()

    auc = roc_auc_score(y_test, y_scores)
    print(f"ROC AUC Score: {auc:.4f}")

    threshold, fpr, tpr = find_optimal_threshold(y_test, y_scores)
    print(f"Optimal Threshold: {threshold:.4f}")

    y_pred = (y_scores >= threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Down (0)", "Up (1)"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    plot_roc_curve(fpr, tpr, auc, threshold)


if __name__ == "__main__":
    main()
