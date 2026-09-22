"""Per-image preprocessing pipeline: chains all stages and produces a
single, fully-processed OCR-ready image."""

from pathlib import Path

import config
from image_loader import load_image, ImageLoadError
from enhancement import (
    intelligent_resize, to_grayscale, correct_illumination,
    enhance_contrast, reduce_noise, sharpen,
)
from deskew import deskew
from thresholding import sauvola_threshold, should_binarize
from utils import StepTimer


class PipelineResult:
    def __init__(self, name):
        self.name = name
        self.steps = []       # list of (ok: bool, message: str)
        self.success = True
        self.error = None
        self.elapsed = 0.0
        self.outputs = {}     # version name -> Path

    def log(self, message, ok=True):
        self.steps.append((ok, message))

    def fail(self, error):
        self.success = False
        self.error = str(error)


def process_image(path: Path, output_root: Path, binarize: bool = False) -> PipelineResult:
    """Run the full pipeline on one image. Never raises - failures are
    captured on the result object so a batch run can continue past them.

    The input folder is assumed to already contain only accepted/validated
    images (no accept/reject validation happens here). Output is written
    directly as `output_root/<stem>_enhanced.png` - no per-image subfolder.
    """
    result = PipelineResult(path.name)
    timer = StepTimer()

    with timer:
        try:
            img = load_image(path)
            result.log("Loaded and EXIF orientation corrected")
        except ImageLoadError as exc:
            result.fail(exc)
            return result

        try:
            img, msg = intelligent_resize(img)
            result.log(msg)

            gray, msg = to_grayscale(img)
            result.log(msg)

            gray, msg = correct_illumination(gray)
            result.log(msg)

            contrasted, msg = enhance_contrast(gray)
            result.log(msg)

            denoised, msg = reduce_noise(contrasted)
            result.log(msg)

            final, msg = sharpen(denoised)
            result.log(msg)

            if config.ENABLE_DESKEW:
                final, angle, msg = deskew(final)
                result.log(msg)

            if binarize:
                candidate = sauvola_threshold(final)
                if should_binarize(candidate):
                    final = candidate
                    result.log("Thresholding applied (Sauvola adaptive, passed information-loss guard)")
                else:
                    result.log(
                        "Thresholding skipped: binary version looked like it lost too much "
                        "information (guard rejected it); kept grayscale/contrast version instead",
                        ok=False,
                    )

            output_root.mkdir(parents=True, exist_ok=True)
            out_path = output_root / f"{path.stem}{config.ENHANCED_SUFFIX}.png"
            final.save(out_path, format=config.OUTPUT_FORMAT)
            result.outputs["enhanced"] = out_path

            result.log(f"Saved to {out_path}")

        except Exception as exc:  # noqa: BLE001 - one bad image must not crash the batch
            result.fail(exc)

    result.elapsed = timer.elapsed
    return result
