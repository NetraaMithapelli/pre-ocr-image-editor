"""
Tests for the OCR preprocessing pipeline.

Run with:
    pytest tests/test_pipeline.py -v

These tests generate synthetic "text-like" images on the fly (no external
fixture files required), covering: valid JPG/PNG, rotated text, low
resolution, dark image, low contrast, noisy image, already-good image,
corrupted file, unsupported file, and batch processing that must survive
one bad file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_loader import load_image, discover_images, ImageLoadError  # noqa: E402
from pipeline import process_image  # noqa: E402


def make_text_image(size=(600, 200), angle=0, noise=False, dark=False, low_contrast=False):
    """Draw simple horizontal stripes to stand in for lines of text."""
    img = Image.new("L", size, color=235)
    draw = ImageDraw.Draw(img)
    for y in range(20, size[1] - 20, 30):
        draw.line([(20, y), (size[0] - 20, y)], fill=30, width=4)
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=235)

    arr = np.array(img).astype(np.float64)
    if noise:
        arr += np.random.normal(0, 45, arr.shape)
    if dark:
        arr *= 0.35
    if low_contrast:
        arr = arr * 0.3 + 128
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def tmp_input(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    return input_dir


def test_valid_jpg(tmp_input):
    path = tmp_input / "valid.jpg"
    make_text_image().convert("RGB").save(path, "JPEG")
    img = load_image(path)
    assert img.size[0] > 0


def test_valid_png(tmp_input):
    path = tmp_input / "valid.png"
    make_text_image().save(path, "PNG")
    img = load_image(path)
    assert img.size[0] > 0


def test_rotated_image_deskew(tmp_input, tmp_path):
    path = tmp_input / "rotated.png"
    make_text_image(angle=8).save(path, "PNG")
    result = process_image(path, tmp_path / "output")
    assert result.success
    assert any("Deskew" in msg for _, msg in result.steps)


def test_low_resolution_upscaled(tmp_input, tmp_path):
    path = tmp_input / "small.png"
    make_text_image(size=(200, 80)).save(path, "PNG")
    result = process_image(path, tmp_path / "output")
    assert result.success
    assert any("Resized" in msg for _, msg in result.steps)


def test_dark_image(tmp_input, tmp_path):
    path = tmp_input / "dark.png"
    make_text_image(dark=True).save(path, "PNG")
    result = process_image(path, tmp_path / "output")
    assert result.success


def test_low_contrast_image(tmp_input, tmp_path):
    path = tmp_input / "low_contrast.png"
    make_text_image(low_contrast=True).save(path, "PNG")
    result = process_image(path, tmp_path / "output")
    assert result.success
    assert any("boosted" in msg or "enhanced" in msg for _, msg in result.steps)


def test_noisy_image(tmp_input, tmp_path):
    path = tmp_input / "noisy.png"
    make_text_image(noise=True).save(path, "PNG")
    result = process_image(path, tmp_path / "output")
    assert result.success
    assert any("Noise reduced" in msg for _, msg in result.steps)


def test_already_good_image(tmp_input, tmp_path):
    path = tmp_input / "good.png"
    make_text_image(size=(2000, 800)).save(path, "PNG")
    result = process_image(path, tmp_path / "output")
    assert result.success
    assert any("skipped" in msg.lower() for _, msg in result.steps)


def test_corrupted_image(tmp_input):
    path = tmp_input / "corrupted.jpg"
    path.write_bytes(b"not a real image file")
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_unsupported_file_ignored(tmp_input):
    (tmp_input / "notes.txt").write_text("hello")
    make_text_image().save(tmp_input / "ok.png")
    images = discover_images(tmp_input)
    assert len(images) == 1
    assert images[0].suffix == ".png"


def test_batch_processing_continues_after_failure(tmp_input, tmp_path):
    make_text_image().save(tmp_input / "ok1.png")
    (tmp_input / "bad.jpg").write_bytes(b"garbage")
    make_text_image().save(tmp_input / "ok2.png")

    images = discover_images(tmp_input)
    results = [process_image(p, tmp_path / "output") for p in images]

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    assert len(successes) == 2
    assert len(failures) == 1


def test_outputs_written_to_disk(tmp_input, tmp_path):
    path = tmp_input / "sample.png"
    make_text_image().save(path, "PNG")
    output_dir = tmp_path / "output"
    result = process_image(path, output_dir)
    assert result.success
    for out_path in result.outputs.values():
        assert out_path.exists()
