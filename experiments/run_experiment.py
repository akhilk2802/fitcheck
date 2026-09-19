"""Run the comparison matrix and build a contact sheet to judge by eye.

Answers Open Question 5 from docs/design.md — does one identity reference
generalize — by rendering the same garments against two very different
reference photos and putting the results side by side.

Run after `modal deploy modal_tryon.py`:
    python3 run_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
INPUTS = HERE / "inputs"
OUTPUTS = HERE / "outputs"

# Two references, deliberately different failure modes:
#   crosswalk — single layer, even light, busy background, hand in pocket
#   tahoe     — square to camera, harsh sun, shorts (bare legs)
PERSONS = ["v0-crosswalk", "v0-tahoe"]

# One per garment type we actually own, with the description the model is
# prompted on. Jeans are included knowing the source image is a 400x533
# thumbnail — that IS the test of whether low-res input ruins a render.
GARMENTS = [
    ("red-shirt-1", "a dark red plaid flannel shirt", "upper_body"),
    ("black-white-shirt-1", "a black and white checked flannel shirt", "upper_body"),
    ("blue-jeans-1", "blue straight leg denim jeans", "lower_body"),
]


def main() -> int:
    OUTPUTS.mkdir(exist_ok=True)
    TryOn = modal.Cls.from_name("fitcheck-tryon", "TryOn")
    tryon = TryOn()

    results = []
    for person in PERSONS:
        p_path = INPUTS / "person" / f"{person}.png"
        if not p_path.exists():
            print(f"missing {p_path} — run prepare_inputs.py")
            return 1

        for garment, description, category in GARMENTS:
            g_path = INPUTS / "garment" / f"{garment}.png"
            if not g_path.exists():
                print(f"skip {garment}: no input")
                continue

            label = f"{person}__{garment}"
            print(f"rendering {label} ... ", end="", flush=True)
            started = time.time()
            try:
                out = tryon.render.remote(
                    p_path.read_bytes(),
                    g_path.read_bytes(),
                    garment_description=description,
                    category=category,
                )
            except Exception as exc:  # one failure must not lose the batch
                print(f"FAILED ({exc.__class__.__name__}: {exc})")
                results.append({"label": label, "error": str(exc)})
                continue

            (OUTPUTS / f"{label}.png").write_bytes(out["image"])
            (OUTPUTS / f"{label}.mask.png").write_bytes(out["mask"])
            wall = time.time() - started
            print(f"{out['render_seconds']}s gpu / {wall:.0f}s wall")
            results.append(
                {
                    "label": label,
                    "person": person,
                    "garment": garment,
                    "gpu_seconds": out["render_seconds"],
                    "wall_seconds": round(wall, 1),
                    "cold_load_seconds": out["load_seconds"],
                    "gpu": out["gpu"],
                }
            )

    (OUTPUTS / "results.json").write_text(json.dumps(results, indent=2))
    contact_sheet(results)

    ok = [r for r in results if "gpu_seconds" in r]
    if ok:
        slowest = max(r["gpu_seconds"] for r in ok)
        print(f"\nslowest render: {slowest}s on {ok[0]['gpu']}")
        print("compare against the design doc's 60s / ~$0.02 estimate")
    print(f"contact sheet: {OUTPUTS/'contact-sheet.png'}")
    return 0


def contact_sheet(results: list[dict]) -> None:
    """Grid of every render, labelled. Judging happens here, not in a file list."""
    tiles = [r for r in results if "gpu_seconds" in r]
    if not tiles:
        return

    cols, tw, th, pad, cap = len(GARMENTS), 384, 512, 12, 28
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * (tw + pad) + pad, rows * (th + cap + pad) + pad),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(sheet)

    for i, r in enumerate(tiles):
        x = pad + (i % cols) * (tw + pad)
        y = pad + (i // cols) * (th + cap + pad)
        with Image.open(OUTPUTS / f"{r['label']}.png") as im:
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        draw.text(
            (x, y + th + 6),
            f"{r['label']}  {r['gpu_seconds']}s",
            fill=(40, 40, 40),
        )

    sheet.save(OUTPUTS / "contact-sheet.png")


if __name__ == "__main__":
    sys.exit(main())
