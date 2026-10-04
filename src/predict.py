"""Final inference pipeline: preprocess -> ensemble of two custom CNNs -> prior-adjusted decision.

Frozen before test-set evaluation. Members and decision rule chosen on validation data only.
"""
from pathlib import Path

import keras
import numpy as np

import data
import preprocessing as pp

PROJECT = Path.home() / "projects/knee-oa-classification"
MODELS = PROJECT / "models"
MEMBERS = ["exp1_augmentation", "exp3_aug_wider"]

TRAIN_COUNTS = np.array([2286, 1046, 1516, 757, 173])      # training images per KL grade
TRAIN_PRIOR = TRAIN_COUNTS / TRAIN_COUNTS.sum()


class KneeOAPipeline:
    def __init__(self, members=MEMBERS):
        self.members = list(members)
        self.models = [keras.models.load_model(MODELS / f"{m}.keras") for m in self.members]

    def predict_proba(self, X):
        """X: preprocessed uint8 images, shape (N, 224, 224). Returns averaged softmax (N, 5)."""
        ds = data.make_dataset(X, np.zeros(len(X), dtype=np.int64), batch_size=64)
        return np.mean([m.predict(ds, verbose=0) for m in self.models], axis=0)

    def predict(self, X, prior_adjust=True):
        p = self.predict_proba(X)
        return (p / TRAIN_PRIOR).argmax(axis=1) if prior_adjust else p.argmax(axis=1)

    def predict_image(self, path):
        """Full pipeline for one raw X-ray file: preprocess, predict, return grade + probabilities."""
        img = pp.load_and_preprocess(path)
        p = self.predict_proba(img[None])[0]
        adjusted = p / TRAIN_PRIOR
        return {"kl_grade": int(adjusted.argmax()),
                "model_probabilities": {g: round(float(v), 4) for g, v in enumerate(p)},
                "adjusted_scores": {g: round(float(v), 4) for g, v in enumerate(adjusted / adjusted.sum())}}
