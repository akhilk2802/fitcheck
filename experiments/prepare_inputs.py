"""Normalize raw photos into model-ready inputs.

Handles two real problems found in the source files:

  1. The garment images are AVIF wearing a .jpeg extension. Pillow 11.1 on this
     machine reports avif support: False, so decoding goes through `sips`, which
     ships with macOS and reads them fine. Pillow is tried first so this keeps
     working if pillow-avif-plugin shows up later.
  2. The identity photos are 4536x8064 phone shots. Try-on models want 768x1024.
     Sending 36MP through the pipeline wastes upload time and GPU memory for
     detail the model throws away.

Run: python3 prepare_inputs.py
Writes: inputs/person/*.png, inputs/garment/*.png
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "inputs"

# Garments go to the model at its native 768x1024.
TARGET = (768, 1024)

# Person references keep their resolution. The render crops to the person and
# pastes the result back into the original, so every pixel thrown away here is
# detail the face and garment never get back. 2048 is a size cap for upload,
# not a model input size.
PERSON_MAX = 2048

# Photo 7 is the best reference in the set: standing, full body, single layer,
# even light. Photo 3 is the backup — square to camera but shorts and harsh sun,
# which is a genuinely different hard case worth measuring.
PERSON_SOURCES = {
    "v0-crosswalk": "identity-kit/Fitcheck - 7 of 7.jpeg",
    "v0-tahoe": "identity-kit/Fitcheck - 3 of 7.jpeg",
}

GARMENT_DIR = PROJECT / "garments"


def decode(src: Path, dst_png: Path) -> None:
    """Decode any input image to PNG, whatever container it is really in."""
    try:
        with Image.open(src) as im:
            im.convert("RGB").save(dst_png)
            return
    except Exception:
        pass  # falls through to sips, which reads the AVIF files

    if not shutil.which("sips"):
        raise RuntimeError(
            f"cannot decode {src.name}: Pillow failed and sips is unavailable. "
            "Install pillow-avif-plugin or convert the file by hand."
        )
    subprocess.run(
        ["sips", "-s", "format", "png", str(src), "--out", str(dst_png)],
        check=True,
        capture_output=True,
    )


def fit(png: Path, target: tuple[int, int]) -> None:
    """Resize onto a target canvas, preserving aspect, padding with white.

    Padding rather than cropping: cropping a garment flat-lay risks clipping a
    sleeve. White matches the product-photo backgrounds these came from.
    """
    with Image.open(png) as im:
        im = im.convert("RGB")
        im.thumbnail(target, Image.LANCZOS)
        canvas = Image.new("RGB", target, (255, 255, 255))
        canvas.paste(im, ((target[0] - im.width) // 2, (target[1] - im.height) // 2))
        canvas.save(png)


def cap(png: Path, longest: int) -> None:
    """Cap the longest edge, keeping aspect and full detail otherwise."""
    with Image.open(png) as im:
        im = im.convert("RGB")
        if max(im.size) > longest:
            im.thumbnail((longest, longest), Image.LANCZOS)
        im.save(png)


def main() -> int:
    (OUT / "person").mkdir(parents=True, exist_ok=True)
    (OUT / "garment").mkdir(parents=True, exist_ok=True)

    print("person references")
    for name, rel in PERSON_SOURCES.items():
        src = PROJECT / rel
        if not src.exists():
            print(f"  MISSING  {rel}")
            continue
        dst = OUT / "person" / f"{name}.png"
        decode(src, dst)
        cap(dst, PERSON_MAX)
        with Image.open(dst) as im:
            size = im.size
        print(f"  {name:16s} <- {src.name}  {size[0]}x{size[1]}")

    print("garments")
    small = []
    for src in sorted(GARMENT_DIR.glob("*")):
        if src.name.startswith("."):
            continue
        dst = OUT / "garment" / f"{src.stem}.png"
        decode(src, dst)
        with Image.open(dst) as im:
            w, h = im.size
        # Flag anything that was a thumbnail before we upscale it into the
        # target canvas — upscaling invents no detail and the model will smear
        # what it cannot see. Better to know now than to blame the model later.
        if w < 900 or h < 1200:
            small.append((src.name, f"{w}x{h}"))
        fit(dst, TARGET)
        print(f"  {src.stem:24s} {w}x{h}")

    if small:
        print("\nlow resolution, re-download at full size before trusting a render:")
        for name, size in small:
            print(f"  {name}  ({size})")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
