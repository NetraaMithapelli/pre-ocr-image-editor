"""Safe image loading: EXIF correction, format handling, corruption guards."""

from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

from config import SUPPORTED_EXTENSIONS


class ImageLoadError(Exception):
    """Raised when an image cannot be safely loaded."""


def discover_images(input_dir: Path):
    """Return a sorted list of supported image file paths in input_dir."""
    return [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_image(path: Path) -> Image.Image:
    """
    Load an image safely:
      - verify it can be opened and is not corrupted
      - apply EXIF orientation correction
      - convert to a consistent color mode (RGB, or keep L if already grayscale)
    Raises ImageLoadError on any failure, so the caller can skip this file
    without crashing the batch.
    """
    try:
        img = Image.open(path)
        img.verify()  # cheap structural check; does not decode pixel data
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Cannot identify or corrupted file: {exc}") from exc

    try:
        # verify() leaves the file handle unusable for further operations,
        # so the image must be reopened for actual processing.
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # correct rotation from EXIF tag
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.load()
    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"Failed to decode image data: {exc}") from exc

    return img
