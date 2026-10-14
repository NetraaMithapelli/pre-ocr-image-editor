# OCR Image Preprocessor 

## What This Project Is

This is an image preprocessing tool that prepares photos for OCR (Optical
Character Recognition). It takes already-accepted images and enhances them
so text is easier for an OCR engine to read — correcting orientation,
resizing, fixing lighting, boosting local contrast, reducing noise,
sharpening edges, straightening rotation, and optionally converting to
black-and-white.

It is built only with **Pillow + NumPy** (no OpenCV), which keeps the
dependency list small and lets it run in environments where OpenCV isn't
available.

It does **not** decide whether an image is good enough to use — that
validation is assumed to happen before this tool runs. It only improves
images it's given, and it never modifies or deletes the original files.

Each image goes through the same sequence of steps: orientation fix →
resize → grayscale → lighting correction → contrast enhancement → noise
reduction → sharpening → deskew → (optional) black-and-white conversion.
Most of these steps check the image first and skip themselves (or scale
their strength) if they're not needed, so an already-good image comes out
barely changed.

---

## File Structure

```
ocr-preprocessor/
├── preprocess.py       # Run this file to start the tool (command line)
├── pipeline.py         # Runs all the steps above, in order, per image
├── image_loader.py     # Loads images safely, fixes orientation, finds files
├── enhancement.py      # Resize, grayscale, lighting fix, contrast, denoise, sharpen
├── thresholding.py     # Optional black-and-white conversion (used with --binarize)
├── deskew.py           # Detects and corrects rotated/tilted text
├── config.py           # All the adjustable settings in one place
├── utils.py            # Helper functions (image stats, timing, etc.)
├── requirements.txt    # List of Python packages needed
├── README.md
├── input/              # Put the images you want processed here
├── output/             # Processed images are saved here
└── tests/
    └── test_pipeline.py
```

---

## Requirements

- **Python 3.9 or newer**
- Packages listed in `requirements.txt` (installed via `pip`)
- (Optional, for testing) `pytest`

---

## How to Run It (Windows)

1. **Install Python 3.9+** from [python.org](https://www.python.org/downloads/)
   — make sure to check "Add Python to PATH" during setup.

2. **Open PowerShell** and go to the project folder:
   ```powershell
   cd C:\path\to\ocr-preprocessor
   ```

3. **(Recommended) Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks this, run once as Administrator:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

4. **Install the required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Add your images** to the `input/` folder.

6. **Run the tool:**
   ```powershell
   python preprocess.py --input input --output output
   ```
   Processed images will appear in `output/` as `<name>_enhanced.png`.

   Optional flags:
   ```powershell
   python preprocess.py --input input --output output --binarize   # also apply black-and-white conversion
   python preprocess.py --input input --output output --debug      # save before/after comparison images
   ```

---

## Important Notes

- **No perspective correction**: it can straighten a rotated image, but it
  can't fix a photo taken at an angle (like a phone held sideways to a
  page). Only in-plane rotation is corrected.
- **Deskew is approximate**: rotation correction works in coarse 0.5° steps
  and is more reliable on multi-line text than a single short line.
- **Denoising is basic**: it uses a median filter, so it won't remove noise
  as well as OpenCV-based tools on very noisy images.
- **Black-and-white output is optional and off by default**: use the
  `--binarize` flag to try it. The tool checks first to make sure this
  doesn't accidentally wipe out text.
- **Settings can be changed** in `config.py` without touching the main
  code — things like minimum output size, contrast strength, and whether
  deskew is enabled at all.
- **Testing**: run `pytest tests/test_pipeline.py -v` to verify everything
  works. All test images are generated automatically, so no extra files
  are needed.
