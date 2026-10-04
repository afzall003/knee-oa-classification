"""Explainability: Grad-CAM (single, multi-layer, ensemble), occlusion sensitivity, deletion test."""
import cv2
import keras
import numpy as np
import tensorflow as tf


def make_gradcam_model(model, layer_name="block4_relu2"):
    """Model returning (feature maps of layer_name, predictions)."""
    return keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])


def gradcam(grad_model, img_uint8, class_idx=None):
    """Heatmap (224x224, values 0-1) of evidence for class_idx in one preprocessed image."""
    x = tf.convert_to_tensor(img_uint8[None, ..., None].astype("float32") / 255.0)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cv2.resize(cam, (img_uint8.shape[1], img_uint8.shape[0]))
    return (cam / cam.max() if cam.max() > 0 else cam), class_idx


def build_gradcam_models(models, layers=("block3_relu2", "block4_relu2")):
    """One Grad-CAM model per (network, layer) pair."""
    return [make_gradcam_model(m, layer) for m in models for layer in layers]


def ensemble_gradcam(grad_models, img_uint8, class_idx):
    """Average of normalised Grad-CAM maps across networks and layers."""
    cams = [gradcam(gm, img_uint8, class_idx)[0] for gm in grad_models]
    cam = np.mean(cams, axis=0)
    return cam / cam.max() if cam.max() > 0 else cam


def occlusion_map(predict_fn, img_uint8, class_idx, patch=24, stride=8):
    """Confidence drop for class_idx when each patch is hidden. predict_fn: uint8 (N,224,224) -> probs."""
    H, W = img_uint8.shape
    fill = np.uint8(img_uint8.mean())
    base = predict_fn(img_uint8[None])[0, class_idx]
    coords = [(y, x) for y in range(0, H - patch + 1, stride) for x in range(0, W - patch + 1, stride)]
    batch = np.repeat(img_uint8[None], len(coords), axis=0)
    for k, (y, x) in enumerate(coords):
        batch[k, y:y + patch, x:x + patch] = fill
    drops = base - predict_fn(batch)[:, class_idx]
    heat, count = np.zeros((H, W)), np.zeros((H, W))
    for d, (y, x) in zip(drops, coords):
        heat[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1
    heat = np.maximum(heat / np.maximum(count, 1), 0)
    return heat / heat.max() if heat.max() > 0 else heat


def random_blob_map(rng, shape=(224, 224), grid=7):
    """Smooth random 'explanation' used as a baseline in the deletion test."""
    return cv2.resize(rng.random((grid, grid)).astype("float32"), (shape[1], shape[0]))


def deletion_curve(predict_fn, img_uint8, saliency, class_idx, fractions=np.linspace(0, 0.5, 11)):
    """Confidence in class_idx as the top-ranked fraction of pixels is erased."""
    order = np.argsort(saliency.ravel())[::-1]
    fill = np.uint8(img_uint8.mean())
    batch = []
    for f in fractions:
        x = img_uint8.copy().ravel()
        x[order[:int(f * x.size)]] = fill
        batch.append(x.reshape(img_uint8.shape))
    return predict_fn(np.stack(batch))[:, class_idx]


def overlay(img_uint8, cam, alpha=0.4):
    """Blend a jet-coloured heatmap over the grayscale X-ray."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    base = np.repeat(img_uint8[..., None], 3, axis=-1)
    return np.uint8((1 - alpha) * base + alpha * heat)
