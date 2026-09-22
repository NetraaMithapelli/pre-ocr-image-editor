"""Core enhancement stages: resize, grayscale, illumination correction,
contrast, noise reduction, and sharpening. Each function returns
(image, log_message) so the pipeline can report exactly what happened."""

import numpy as np
from PIL import Image, ImageFilter

import config
from utils import to_array, from_array, image_std, estimate_blur_score, estimate_noise_score

_RESAMPLE = Image.LANCZOS


def intelligent_resize(img: Image.Image):
    w, h = img.size
    longer_side = max(w, h)

    if longer_side > config.MAX_IMAGE_SIZE:
        scale = config.MAX_IMAGE_SIZE / longer_side
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = img.resize(new_size, _RESAMPLE)
        return resized, f"Resized (capped): {w}x{h} -> {new_size[0]}x{new_size[1]}"

    if w < config.MIN_OCR_WIDTH:
        scale = config.MIN_OCR_WIDTH / w
        new_size = (config.MIN_OCR_WIDTH, max(1, int(h * scale)))
        resized = img.resize(new_size, _RESAMPLE)
        return resized, f"Resized (upscaled): {w}x{h} -> {new_size[0]}x{new_size[1]}"

    return img, f"Resize skipped: {w}x{h} already adequate"


def to_grayscale(img: Image.Image):
    if img.mode == "L":
        return img, "Grayscale skipped: already single-channel"
    return img.convert("L"), "Converted to grayscale"


def correct_illumination(gray_img: Image.Image):
    """
    Flatten uneven lighting (shadows, dark corners, vignetting) by estimating
    a smooth background with a large-radius blur, then dividing the original
    by that background. This is a lightweight, division-based analogue of
    the 'rolling ball' background-subtraction technique, implementable with
    Pillow + NumPy alone (no OpenCV needed).

    The blur radius scales with the image's own dimensions rather than using
    a fixed pixel value: on a large photo a small fixed radius just tracks
    individual text strokes instead of the slow page-lighting gradient, so
    it fails to flatten real shadows near the edges/corners.
    """
    w, h = gray_img.size
    radius = int(np.clip(
        max(w, h) * config.ILLUMINATION_BLUR_FRACTION,
        config.ILLUMINATION_BLUR_MIN,
        config.ILLUMINATION_BLUR_MAX,
    ))

    background = gray_img.filter(ImageFilter.GaussianBlur(radius=radius))
    fg = to_array(gray_img)
    bg = to_array(background)
    bg = np.maximum(bg, 1.0)  # avoid divide-by-zero

    # Rescale relative to the background's own average brightness (rather
    # than forcing it to pure white/255). This is gentler: it equalizes the
    # shading without blowing out images whose true background isn't white,
    # and without over-brightening regions that were only mildly shadowed.
    corrected = (fg / bg) * float(np.mean(bg))
    corrected = np.clip(corrected, 0, 255)

    if np.std(corrected - fg) < config.ILLUMINATION_SKIP_STD:
        return gray_img, f"Illumination correction skipped: lighting already even (radius={radius})"

    return from_array(corrected, mode="L"), f"Illumination corrected (background flattened, radius={radius}px)"


def _clahe(gray_arr: np.ndarray, grid_size=(8, 8), clip_limit=2.0, bins=256) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization, implemented directly
    in NumPy (no OpenCV).

    Why this replaces a single global contrast stretch: a global operation
    (autocontrast / ImageEnhance.Contrast) must pick ONE mapping for the
    entire image. If the center is bright and corners are dim, any mapping
    that makes the center pop will crush or blow out the corners, and vice
    versa - that's exactly the "corner text disappears" failure mode.

    CLAHE instead divides the image into a grid of tiles, builds a clipped
    (noise-limited) histogram-equalization mapping PER TILE, and blends
    neighboring tiles' mappings with bilinear interpolation so there are no
    hard seams at tile boundaries. Every region - including corners - gets
    contrast enhancement tuned to its own local content.
    """
    h, w = gray_arr.shape
    grid_h, grid_w = grid_size
    tile_h = max(1, h // grid_h)
    tile_w = max(1, w // grid_w)
    # Recompute actual tile counts (last tile absorbs any remainder pixels,
    # so we never crop/drop rows or columns).
    grid_h = max(1, h // tile_h)
    grid_w = max(1, w // tile_w)

    arr_u8 = np.clip(gray_arr, 0, 255).astype(np.uint8)
    clip_px = max(1.0, clip_limit * tile_h * tile_w / bins)

    # One LUT per tile, padded by one "phantom" tile of margin on each side
    # (filled by replicating the nearest real tile) so that pixels in the
    # outermost half-tile - including every corner - still interpolate
    # smoothly instead of needing special-cased edge handling.
    luts = np.zeros((grid_h + 2, grid_w + 2, bins), dtype=np.float32)

    for gy in range(grid_h):
        y0 = gy * tile_h
        y1 = h if gy == grid_h - 1 else y0 + tile_h
        for gx in range(grid_w):
            x0 = gx * tile_w
            x1 = w if gx == grid_w - 1 else x0 + tile_w
            tile = arr_u8[y0:y1, x0:x1]
            hist = np.bincount(tile.ravel(), minlength=bins).astype(np.float64)

            # Clip the histogram (this is the "contrast limited" part): any
            # bin taller than clip_px has its excess redistributed evenly,
            # which caps how aggressively a tile can be stretched and keeps
            # flat, noisy, or near-empty tiles (e.g. blank margins) from
            # amplifying noise into fake "contrast".
            excess = np.sum(np.maximum(hist - clip_px, 0))
            hist = np.minimum(hist, clip_px)
            hist += excess / bins

            cdf = np.cumsum(hist)
            nonzero = np.nonzero(cdf)[0]
            cdf_min = cdf[nonzero[0]] if len(nonzero) else 0.0
            denom = max(cdf[-1] - cdf_min, 1e-6)
            lut = np.clip((cdf - cdf_min) / denom * 255.0, 0, 255)
            luts[gy + 1, gx + 1] = lut

    luts[0, :] = luts[1, :]
    luts[-1, :] = luts[-2, :]
    luts[:, 0] = luts[:, 1]
    luts[:, -1] = luts[:, -2]

    center_y = (np.arange(grid_h + 2) - 0.5) * tile_h
    center_x = (np.arange(grid_w + 2) - 0.5) * tile_w

    ys = np.arange(h)
    xs = np.arange(w)

    y_pos = np.clip(np.searchsorted(center_y, ys, side="right") - 1, 0, grid_h)
    x_pos = np.clip(np.searchsorted(center_x, xs, side="right") - 1, 0, grid_w)

    y_frac = np.clip((ys - center_y[y_pos]) / (center_y[y_pos + 1] - center_y[y_pos]), 0, 1)
    x_frac = np.clip((xs - center_x[x_pos]) / (center_x[x_pos + 1] - center_x[x_pos]), 0, 1)

    y0_idx = y_pos[:, None]
    y1_idx = (y_pos + 1)[:, None]
    x0_idx = x_pos[None, :]
    x1_idx = (x_pos + 1)[None, :]

    top_left = luts[y0_idx, x0_idx, arr_u8]
    top_right = luts[y0_idx, x1_idx, arr_u8]
    bottom_left = luts[y1_idx, x0_idx, arr_u8]
    bottom_right = luts[y1_idx, x1_idx, arr_u8]

    yf = y_frac[:, None]
    xf = x_frac[None, :]

    top = top_left * (1 - xf) + top_right * xf
    bottom = bottom_left * (1 - xf) + bottom_right * xf
    return top * (1 - yf) + bottom * yf


def enhance_contrast(gray_img: Image.Image):
    """
    Gentle, LOCAL contrast enhancement (CLAHE) applied after illumination
    correction. Deliberately does not fall back to a global autocontrast or
    ImageEnhance.Contrast boost for "already fine" images - a mild CLAHE
    pass is safe (clip-limited) across the board and keeps behavior
    predictable and corner-safe regardless of the image's global stats.
    """
    std = image_std(gray_img)
    arr = to_array(gray_img)
    result = _clahe(arr, grid_size=config.CLAHE_GRID_SIZE, clip_limit=config.CLAHE_CLIP_LIMIT)
    gh, gw = config.CLAHE_GRID_SIZE
    return (
        from_array(result, mode="L"),
        f"Contrast enhanced (adaptive local CLAHE, grid={gh}x{gw}, "
        f"clip={config.CLAHE_CLIP_LIMIT}, pre-std={std:.1f})",
    )


def reduce_noise(gray_img: Image.Image):
    noise_score = estimate_noise_score(gray_img)
    if noise_score < config.NOISE_ESTIMATE_THRESHOLD:
        return gray_img, f"Noise reduction skipped (noise score {noise_score:.2f} low)"

    denoised = gray_img.filter(ImageFilter.MedianFilter(size=config.MEDIAN_FILTER_SIZE))
    return denoised, f"Noise reduced (median filter, noise score {noise_score:.2f})"


def sharpen(gray_img: Image.Image):
    blur_score = estimate_blur_score(gray_img)
    if blur_score > config.BLUR_VARIANCE_THRESHOLD * 3:
        return gray_img, f"Sharpening skipped (already sharp, blur score {blur_score:.1f})"

    percent = config.SHARPEN_PERCENT
    if blur_score < config.BLUR_VARIANCE_THRESHOLD:
        percent = int(percent * 1.3)  # blurrier images get a bit more sharpening

    sharpened = gray_img.filter(ImageFilter.UnsharpMask(
        radius=config.SHARPEN_RADIUS,
        percent=percent,
        threshold=config.SHARPEN_THRESHOLD,
    ))
    return sharpened, f"Sharpened (UnsharpMask, blur score {blur_score:.1f}, percent={percent})"
