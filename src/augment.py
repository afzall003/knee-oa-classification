"""Training-time augmentation. Anatomically sensible: no vertical flips."""
import keras
from keras import layers


def build_augmentation(rotation_deg=10, translate=0.05, zoom=0.10,
                       brightness=0.10, contrast=0.10, seed=42):
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(rotation_deg / 360, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomTranslation(translate, translate, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomZoom((-zoom, zoom), fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomBrightness(brightness, value_range=(0.0, 1.0), seed=seed),
        layers.RandomContrast(contrast, value_range=(0.0, 1.0), seed=seed),
    ], name="augmentation")
