"""Shared helper utilities: array conversion, image statistics, timing."""

import time
import numpy as np
from PIL import Image, ImageFilter

_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    (3, 3),
    [0, -1, 0, -1, 4, -1, 0, -1, 0],
    scale=1,
)


def to_array(img: Image.Image) -> np.ndarray:
    """Convert a Pillow image to a float64 numpy array."""
    return np.asarray(img, dtype=np.float64)


def from_array(arr: np.ndarray, mode: str = "L") -> Image.Image:
    """Convert a numpy array back into a Pillow image, clipping to valid range."""
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode=mode)


def image_std(img: Image.Image) -> float:
    """Standard deviation of pixel intensities - a rough contrast measure."""
    return float(np.std(to_array(img.convert("L"))))


def image_mean(img: Image.Image) -> float:
    return float(np.mean(to_array(img.convert("L"))))


def estimate_blur_score(gray_img: Image.Image) -> float:
    """
    Variance of a Laplacian-filtered image.
    Low variance -> flat/blurry image, high variance -> sharp edges present.
    This mirrors the common `cv2.Laplacian(...).var()` blur-detection trick,
    reimplemented with a Pillow convolution kernel.
    """
    edges = gray_img.filter(_LAPLACIAN_KERNEL)
    return float(np.var(to_array(edges)))


def estimate_noise_score(gray_img: Image.Image) -> float:
    """
    Rough noise estimate: difference between the image and a lightly
    median-filtered version of itself. Genuine noise/speckle causes larger
    differences than real edges, since median filtering removes speckle
    while mostly preserving edges.
    """
    denoised = gray_img.filter(ImageFilter.MedianFilter(size=3))
    diff = to_array(gray_img) - to_array(denoised)
    return float(np.std(diff))


class StepTimer:
    """Small context manager for timing pipeline stages."""

    def __init__(self):
        self.start = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False


def save_debug_comparison(original_img: Image.Image, enhanced_img: Image.Image, out_path) -> None:
    """
    Save a side-by-side "Original | Enhanced" image so corner/edge text can
    be visually spot-checked. Never touches the required output/ files -
    purely an optional debugging aid (enabled via --debug).
    """
    orig = original_img.convert("L") if original_img.mode != "L" else original_img
    enh = enhanced_img.convert("L") if enhanced_img.mode != "L" else enhanced_img

    target_h = max(orig.height, enh.height)

    def _scale_to_height(img, h):
        if img.height == h:
            return img
        w = max(1, int(img.width * (h / img.height)))
        return img.resize((w, h), Image.LANCZOS)

    orig_r = _scale_to_height(orig, target_h)
    enh_r = _scale_to_height(enh, target_h)

    gap = 8
    canvas = Image.new("L", (orig_r.width + gap + enh_r.width, target_h), color=255)
    canvas.paste(orig_r, (0, 0))
    canvas.paste(enh_r, (orig_r.width + gap, 0))
    canvas.save(out_path)


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"
