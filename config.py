"""
Configuration for the OCR preprocessing pipeline.
Tune these values to adjust behavior without touching pipeline logic.
"""

# --- Resize ---
MIN_OCR_WIDTH = 1600            # if image is narrower than this, upscale
MAX_IMAGE_SIZE = 4000           # cap on the longer side, to avoid huge files

# --- Contrast (local adaptive, CLAHE-style) ---
# Global autocontrast/ImageEnhance.Contrast is intentionally NOT used as the
# main contrast step: any single global stretch is forced to pick one mapping
# for the whole image, so a mapping that makes bright center text pop will
# over-expose it while leaving darker corner text under-stretched (or vice
# versa). CLAHE_GRID_SIZE divides the image into tiles and equalizes each
# tile against its own local histogram, so corners get their own contrast
# treatment independent of the center. Because it's grid-based, tile pixel
# size automatically scales with the image's actual dimensions.
CLAHE_GRID_SIZE = (8, 8)        # (rows, cols) of tiles; tile size = image_size / grid
CLAHE_CLIP_LIMIT = 2.0          # higher = more contrast but more noise/halo risk; keep gentle

# --- Noise reduction ---
MEDIAN_FILTER_SIZE = 3          # must be odd
NOISE_ESTIMATE_THRESHOLD = 6.0  # heuristic noise score above which we denoise

# --- Sharpening ---
SHARPEN_RADIUS = 1.5
SHARPEN_PERCENT = 130
SHARPEN_THRESHOLD = 3
BLUR_VARIANCE_THRESHOLD = 150.0 # below this, image is considered blurry -> sharpen more

# --- Illumination correction ---
# Radius is now a FRACTION of the image's longer side (clamped to sensible
# bounds) instead of a fixed pixel value. A fixed radius (e.g. 35px) is far
# too small on a 4000px photo: the "background" blur ends up following text
# strokes instead of the slow-varying page lighting, so it fails to flatten
# genuine shadows/vignetting near corners. Scaling with image size keeps the
# background estimate meaningfully "smooth" at any resolution.
ILLUMINATION_BLUR_FRACTION = 0.10   # blur radius = 10% of the longer image side
ILLUMINATION_BLUR_MIN = 25          # px, floor for small images
ILLUMINATION_BLUR_MAX = 150         # px, ceiling for very large images
ILLUMINATION_SKIP_STD = 1.5         # skip correction if it barely changes anything

# --- Deskew ---
ENABLE_DESKEW = True               # set False to disable if you don't want rotation applied
DESKEW_ANGLE_RANGE = 15            # search +/- degrees
DESKEW_ANGLE_STEP = 0.5
DESKEW_MIN_ANGLE_TO_CORRECT = 0.3  # ignore smaller angles (not worth rotating)
DESKEW_DOWNSCALE_WIDTH = 800        # speed up angle search on a smaller copy

# --- Thresholding (Sauvola adaptive) ---
# Off by default: the required output preserves grayscale information.
# Pass --binarize on the CLI to opt in; even then, a guard (should_binarize)
# refuses to use the binary result if it looks like it wiped out the image.
ENABLE_THRESHOLDING_BY_DEFAULT = False
THRESHOLD_WINDOW = 25           # local window size (pixels) for adaptive threshold
THRESHOLD_K = 0.2               # Sauvola sensitivity parameter
THRESHOLD_R = 128               # Sauvola dynamic range of std dev

# --- Batch / IO ---
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
ENHANCED_SUFFIX = "_enhanced"   # output/<stem>_enhanced.png
OUTPUT_FORMAT = "PNG"
