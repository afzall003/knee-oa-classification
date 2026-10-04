"""Ordinal (CORAL-style) classification: P(grade > k) for k = 0..3 on a shared severity score."""
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from models import conv_block

NUM_CLASSES = 5


@keras.saving.register_keras_serializable(package="knee_oa")
class CoralHead(layers.Layer):
    """One shared weight vector (the severity score) + one learned threshold per cut-point."""

    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform", name="w")
        self.b = self.add_weight(shape=(self.num_classes - 1,), initializer="zeros", name="b")

    def call(self, x):
        return keras.ops.sigmoid(keras.ops.matmul(x, self.w) + self.b)

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}


def build_ordinal_cnn(input_shape=(224, 224, 1), filters=(32, 64, 128, 256), dropout=0.3):
    inputs = keras.Input(shape=input_shape, name="xray")
    x = inputs
    for i, f in enumerate(filters, start=1):
        x = conv_block(x, f, name=f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = CoralHead(NUM_CLASSES, name="p_grade_above")(x)
    return keras.Model(inputs, outputs, name="ordinal_cnn")


def to_ordinal(ds):
    """Convert integer labels to cumulative targets: grade 2 -> [1, 1, 0, 0]."""
    cuts = tf.range(NUM_CLASSES - 1, dtype=tf.int64)
    return ds.map(lambda x, y: (x, tf.cast(tf.expand_dims(y, -1) > cuts, tf.float32)),
                  num_parallel_calls=tf.data.AUTOTUNE)


def decode_count(p, threshold=0.5):
    """Predicted grade = number of 'grade > k' questions answered yes."""
    return (p > threshold).sum(axis=1)


def to_class_probs(p):
    """Turn P(grade > k) into a probability for each of the 5 grades."""
    p = np.minimum.accumulate(np.clip(p, 0.0, 1.0), axis=1)     # enforce P(>0) >= P(>1) >= ...
    probs = np.concatenate([1 - p[:, :1], p[:, :-1] - p[:, 1:], p[:, -1:]], axis=1)
    return probs / probs.sum(axis=1, keepdims=True)
