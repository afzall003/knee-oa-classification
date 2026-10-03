"""Evaluation harness: metrics, confusion matrix, balanced-accuracy callback, results log."""
import csv
import json
from datetime import datetime
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, recall_score)

GRADES = [0, 1, 2, 3, 4]
RESULTS_CSV = Path.home() / "projects/knee-oa-classification/reports/results.csv"


def set_seed(seed=42):
    """Seed Python, NumPy and TensorFlow for reproducible runs."""
    keras.utils.set_random_seed(seed)


def compute_metrics(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, labels=GRADES, average=None, zero_division=0)
    m = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "macro_f1": f1_score(y_true, y_pred, labels=GRADES, average="macro", zero_division=0),
    }
    m.update({f"recall_g{g}": r for g, r in zip(GRADES, recalls)})
    return {k: float(v) for k, v in m.items()}


def plot_confusion_matrix(y_true, y_pred, title="", save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=GRADES)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(GRADES)):
        for j in range(len(GRADES)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(GRADES)
    ax.set_yticks(GRADES)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def evaluate_predictions(y_true, y_pred, title="", save_path=None):
    """Print headline metrics and per-class report, plot the confusion matrix."""
    m = compute_metrics(y_true, y_pred)
    print(f"Balanced accuracy: {m['balanced_accuracy']:.4f} | Accuracy: {m['accuracy']:.4f} | "
          f"QWK: {m['qwk']:.4f} | Macro F1: {m['macro_f1']:.4f}\n")
    print(classification_report(y_true, y_pred, labels=GRADES, digits=3, zero_division=0))
    plot_confusion_matrix(y_true, y_pred, title, save_path)
    plt.show()
    return m


class BalancedAccuracyCallback(keras.callbacks.Callback):
    """Adds 'val_balanced_accuracy' and 'val_qwk' to the epoch logs.

    Must be listed BEFORE ModelCheckpoint / EarlyStopping in the callbacks list,
    so they can monitor 'val_balanced_accuracy'. val_ds must NOT be shuffled.
    """

    def __init__(self, val_ds, y_val):
        super().__init__()
        self.val_ds = val_ds
        self.y_val = np.asarray(y_val)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.val_ds, verbose=0).argmax(axis=1)
        logs = logs if logs is not None else {}
        logs["val_balanced_accuracy"] = float(balanced_accuracy_score(self.y_val, y_pred))
        logs["val_qwk"] = float(cohen_kappa_score(self.y_val, y_pred, weights="quadratic"))
        print(f"  val_balanced_accuracy: {logs['val_balanced_accuracy']:.4f}"
              f" | val_qwk: {logs['val_qwk']:.4f}")


def log_run(run_name, config, metrics, split="val", path=RESULTS_CSV):
    """Append one experiment's config and metrics to reports/results.csv."""
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "run": run_name, "split": split,
           **{k: round(v, 4) for k, v in metrics.items()},
           "config": json.dumps(config)}
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row
