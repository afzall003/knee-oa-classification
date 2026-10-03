"""CNN architectures for knee OA KL-grade classification (all trained from scratch)."""
import keras
from keras import layers


def conv_block(x, filters, name):
    """Two 3x3 conv -> BatchNorm -> ReLU layers, then 2x2 max pooling."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.ReLU(name=f"{name}_relu{i}")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_baseline_cnn(input_shape=(224, 224, 1), num_classes=5,
                       filters=(32, 64, 128, 256), dropout=0.3):
    inputs = keras.Input(shape=input_shape, name="xray")
    x = inputs
    for i, f in enumerate(filters, start=1):
        x = conv_block(x, f, name=f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="kl_grade")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")
