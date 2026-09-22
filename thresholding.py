"""Adaptive (Sauvola-style) thresholding using integral images, plus a
heuristic to decide whether binarization is likely to help or hurt."""

import numpy as np
from PIL import Image

import config
from utils import to_array, from_array


def _integral_images(arr: np.ndarray):
    """Return integral image and integral image of squares, padded by one
    row/col of zeros so window sums can be computed without edge checks."""
    padded = np.pad(arr, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral_sq = np.cumsum(np.cumsum(padded ** 2, axis=0), axis=1)
    return integral, integral_sq


def _window_sum(integral: np.ndarray, y0, y1, x0, x1):
    """Sum over arr[y0:y1, x0:x1] using an integral image."""
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def sauvola_threshold(gray_img: Image.Image, window=None, k=None, r=None) -> Image.Image:
    """
    Local adaptive threshold (Sauvola's method), computed efficiently with
    integral images so pure NumPy stays fast enough for full-size photos.
    Handles uneven lighting far better than a single global threshold,
    which is the main reason document-scanner apps use a local method.
    """
    window = window or config.THRESHOLD_WINDOW
    k = k if k is not None else config.THRESHOLD_K
    r = r if r is not None else config.THRESHOLD_R

    arr = to_array(gray_img)
    h, w = arr.shape
    half = window // 2

    integral, integral_sq = _integral_images(arr)

    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - half, 0, h)
    y1 = np.clip(ys + half + 1, 0, h)
    x0 = np.clip(xs - half, 0, w)
    x1 = np.clip(xs + half + 1, 0, w)

    y0g, x0g = np.meshgrid(y0, x0, indexing="ij")
    y1g, x1g = np.meshgrid(y1, x1, indexing="ij")

    counts = (y1g - y0g) * (x1g - x0g)
    sums = _window_sum(integral, y0g, y1g, x0g, x1g)
    sums_sq = _window_sum(integral_sq, y0g, y1g, x0g, x1g)

    means = sums / counts
    variances = np.maximum(sums_sq / counts - means ** 2, 0)
    stds = np.sqrt(variances)

    threshold = means * (1 + k * (stds / r - 1))
    binary = np.where(arr > threshold, 255, 0)

    return from_array(binary, mode="L")


def should_binarize(binary_img: Image.Image) -> bool:
    """
    Heuristic guard: if binarization produced an almost entirely white or
    almost entirely black image, it likely wiped out the text rather than
    isolating it. In that case the grayscale/contrast versions should be
    preferred over the binary one.
    """
    arr = to_array(binary_img)
    white_ratio = np.mean(arr > 200)
    black_ratio = np.mean(arr < 55)
    return not (white_ratio > 0.98 or black_ratio > 0.98)
