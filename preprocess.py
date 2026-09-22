#!/usr/bin/env python3
"""
OCR image preprocessing pipeline - CLI entrypoint.

Assumes --input already contains only accepted/validated images; this
script does not re-validate or reject anything, it only enhances.

Usage:
    python preprocess.py --input input --output output               # whole folder
    python preprocess.py --input input/receipt.jpg --output output   # one file
    python preprocess.py --input input --output output --binarize    # also try adaptive threshold
    python preprocess.py --input input --output output --debug       # + debug/ comparison images
"""

import argparse
import sys
import time
from pathlib import Path

from config import SUPPORTED_EXTENSIONS
from image_loader import discover_images, load_image
from pipeline import process_image
from utils import save_debug_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="OCR image preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Folder of accepted/validated images, or a single image file")
    parser.add_argument("--output", required=True, help="Folder to write OCR-ready outputs")
    parser.add_argument(
        "--binarize", action="store_true",
        help="Also attempt adaptive (Sauvola) thresholding; only used if it doesn't lose information",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Write side-by-side original|enhanced comparison images to a debug/ folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Error: unsupported file type: {input_path.suffix}")
            sys.exit(1)
        images = [input_path]
    elif input_path.is_dir():
        images = discover_images(input_path)
    else:
        print(f"Error: input path not found: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not images:
        print(f"No supported images found in {input_path}")
        sys.exit(0)

    print(f"Found {len(images)} image(s) to process.\n")

    debug_dir = None
    if args.debug:
        debug_dir = (output_dir.parent / "debug") if output_dir.parent != Path(".") else Path("debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

    successes, failures = [], []
    batch_start = time.perf_counter()

    for i, path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] Processing {path.name}")
        result = process_image(path, output_dir, binarize=args.binarize)

        for ok, message in result.steps:
            marker = "\u2713" if ok else "-"
            print(f"    {marker} {message}")

        if result.success:
            print(f"    \u2713 Done in {result.elapsed:.2f}s")
            print(f"    \u2192 {result.outputs.get('enhanced')}\n")
            successes.append(path.name)

            if debug_dir is not None:
                try:
                    original = load_image(path)
                    enhanced_path = result.outputs["enhanced"]
                    from PIL import Image
                    enhanced = Image.open(enhanced_path)
                    debug_path = debug_dir / f"{path.stem}_compare.png"
                    save_debug_comparison(original, enhanced, debug_path)
                except Exception as exc:  # noqa: BLE001 - debug output is best-effort
                    print(f"    (debug comparison failed: {exc})")
        else:
            print(f"    \u2717 Failed: {path.name}")
            print(f"      Reason: {result.error}\n")
            failures.append((path.name, result.error))

    total_time = time.perf_counter() - batch_start

    print("=" * 50)
    print(f"Batch complete in {total_time:.2f}s")
    print(f"  Succeeded: {len(successes)}")
    print(f"  Failed:    {len(failures)}")
    if failures:
        print("\nFailed files:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
