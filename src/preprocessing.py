"""Knee X-ray preprocessing: polarity fix -> ROI percentile stretch -> CLAHE."""
import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def polarity_score(img):
    """Centre minus edge brightness. Negative means the image is inverted."""
    center = img[40:184, 62:162].mean()
    border = np.concatenate([img[:, :20].ravel(), img[:, -20:].ravel()]).mean()
    return center - border


def fix_polarity(img):
    return 255 - img if polarity_score(img) < 0 else img


def stretch(img, lo=1, hi=99):
    """Percentile contrast stretch, measured on the central joint region."""
    roi = img[40:184, 40:184]
    p_lo, p_hi = np.percentile(roi, (lo, hi))
    if p_hi <= p_lo:
        return img
    out = (img.astype(np.float32) - p_lo) * 255.0 / (p_hi - p_lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(img):
    return _CLAHE.apply(stretch(fix_polarity(img)))


def load_and_preprocess(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return preprocess(img)
