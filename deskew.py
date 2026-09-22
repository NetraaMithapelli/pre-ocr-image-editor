"""Deskew via rotation search over a horizontal projection profile.

This is a lightweight, dependency-free analogue of Hough-line-based skew
detection: for each candidate angle we rotate the image, sum "ink" pixels
per row, and measure how peaky that row-sum profile is. Text lines that
are perfectly horizontal produce sharp peaks/valleys (rows on a text line
vs. rows in the gap between lines); a skewed image blurs those peaks
together, lowering the profile's variance. The angle with the highest
variance is taken as the correct rotation.
"""

import numpy as np
from PIL import Image

import config
from utils import to_array


def _projection_variance(gray_img: Image.Image, angle: float) -> float:
    rotated = gray_img.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
    arr = to_array(rotated)
    ink = 255.0 - arr  # dark pixels count as "ink"
    row_sums = ink.sum(axis=1)
    return float(np.var(row_sums))


def estimate_skew_angle(gray_img: Image.Image) -> float:
    """Search a range of angles and return the one with the highest
    projection-profile variance (best line alignment)."""
    w, h = gray_img.size
    if w > config.DESKEW_DOWNSCALE_WIDTH:
        scale = config.DESKEW_DOWNSCALE_WIDTH / w
        search_img = gray_img.resize(
            (config.DESKEW_DOWNSCALE_WIDTH, max(1, int(h * scale))),
            Image.BILINEAR,
        )
    else:
        search_img = gray_img

    angles = np.arange(
        -config.DESKEW_ANGLE_RANGE,
        config.DESKEW_ANGLE_RANGE + config.DESKEW_ANGLE_STEP,
        config.DESKEW_ANGLE_STEP,
    )
    best_angle, best_score = 0.0, -1.0
    for angle in angles:
        score = _projection_variance(search_img, float(angle))
        if score > best_score:
            best_score, best_angle = score, float(angle)

    return best_angle


def deskew(gray_img: Image.Image):
    angle = estimate_skew_angle(gray_img)

    if abs(angle) < config.DESKEW_MIN_ANGLE_TO_CORRECT:
        return gray_img, 0.0, f"Deskew skipped: image already aligned ({angle:+.2f}\u00b0 detected)"

    corrected = gray_img.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
    return corrected, angle, f"Deskew applied: {angle:+.2f}\u00b0"
