"""Reusable training + evaluation loop so every experiment runs identically.

Supports softmax models (default) and other output formats via `loss` and `decode_fn`.
"""
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

import evaluation as ev

PROJECT = Path.home() / "projects/knee-oa-classification"
MODELS = PROJECT / "models"
REPORTS = PROJECT / "reports"


def argmax_decode(p):
    return p.argmax(axis=1)


class DecodedBalancedAccuracyCallback(keras.callbacks.Callback):
    """Adds val_balanced_accuracy / val_qwk to the logs, decoding outputs with decode_fn.
    Must come BEFORE ModelCheckpoint / EarlyStopping. val_ds must NOT be shuffled."""

    def __init__(self, val_ds, y_val, decode_fn=argmax_decode):
        super().__init__()
        self.val_ds, self.y_val, self.decode_fn = val_ds, np.asarray(y_val), decode_fn

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.decode_fn(self.model.predict(self.val_ds, verbose=0))
        logs = logs if logs is not None else {}
        logs["val_balanced_accuracy"] = float(balanced_accuracy_score(self.y_val, y_pred))
        logs["val_qwk"] = float(cohen_kappa_score(self.y_val, y_pred, weights="quadratic"))
        print(f"  val_balanced_accuracy: {logs['val_balanced_accuracy']:.4f}"
              f" | val_qwk: {logs['val_qwk']:.4f}")


def plot_history(history, run_name):
    h = history.history
    metric = next((k for k in h if not k.startswith("val_") and k not in ("loss", "learning_rate")), None)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].plot(h["loss"], label="train"); axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title(f"{run_name}: loss")
    if metric:
        axes[1].plot(h[metric], label="train")
        if f"val_{metric}" in h:
            axes[1].plot(h[f"val_{metric}"], label="val")
        axes[1].set_title(metric)
    axes[2].plot(h["val_balanced_accuracy"], color="C2", label="val balanced accuracy")
    axes[2].plot(h["val_qwk"], color="C3", alpha=0.6, label="val QWK")
    axes[2].axhline(0.2, ls="--", color="gray", label="always-grade-0")
    axes[2].set_title("Validation balanced accuracy and QWK")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / f"{run_name}_curves.png", dpi=150)
    plt.show()


def train_and_evaluate(run_name, model, config, train_ds, val_ds, y_val,
                       class_weight=None, max_epochs=80, patience=12,
                       loss="sparse_categorical_crossentropy", metrics=("accuracy",),
                       decode_fn=argmax_decode):
    """Train with balanced-accuracy checkpointing, then evaluate and log the best model."""
    MODELS.mkdir(exist_ok=True)
    model.compile(optimizer=keras.optimizers.Adam(config.get("lr", 1e-3)),
                  loss=loss, metrics=list(metrics))
    ckpt = MODELS / f"{run_name}.keras"
    callbacks = [
        DecodedBalancedAccuracyCallback(val_ds, y_val, decode_fn),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_balanced_accuracy", mode="max",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_balanced_accuracy", mode="max",
                                      patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(REPORTS / f"{run_name}_history.csv"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs,
                        class_weight=class_weight, callbacks=callbacks, verbose=2)

    plot_history(history, run_name)
    best = keras.models.load_model(ckpt)
    y_pred = decode_fn(best.predict(val_ds, verbose=0))
    metrics_out = ev.evaluate_predictions(y_val, y_pred, title=f"{run_name} (validation)",
                                          save_path=REPORTS / f"cm_{run_name}.png")
    h = history.history
    ev.log_run(run_name, {**config,
                          "best_epoch": int(np.argmax(h["val_balanced_accuracy"])) + 1,
                          "epochs_run": len(h["loss"]),
                          "max_epochs": max_epochs, "patience": patience}, metrics_out)
    return best, history, metrics_out
