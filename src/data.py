"""Data loading: processed .npz arrays -> tf.data pipelines."""
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT = Path.home() / "projects/knee-oa-classification"
PROCESSED = PROJECT / "data/processed"
NUM_CLASSES = 5


def load_split(split):
    """Return (X, y) for 'train', 'val' or 'test'. X is uint8 (N, 224, 224)."""
    with np.load(PROCESSED / f"{split}.npz") as d:
        return d["X"], d["y"]


def make_dataset(X, y, batch_size=32, shuffle=False, augment=None, seed=42):
    """Build a tf.data pipeline: scale to [0, 1], add channel dim, batch, prefetch.

    augment: optional Keras layer/model applied to each training batch.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, t: (tf.cast(x[..., None], tf.float32) / 255.0, t),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment is not None:
        ds = ds.map(lambda x, t: (augment(x, training=True), t),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
