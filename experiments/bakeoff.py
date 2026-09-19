"""Three try-on backends, identical inputs, one contact sheet.

Answers Open Question 1 from ../docs/design.md by eye rather than by benchmark:

  idm         IDM-VTON, open weights on your own Modal GPU (~$0.006/render)
  fashn       FASHN v1.6 via fal, purpose-built and maskless (~$0.075/render)
  nanobanana  Gemini 3 Pro Image, a general editor told what to do (~$0.13/render)

Keys come from ~/.fitcheck.env (0600, outside the repo) and are never printed.

Run: python3 bakeoff.py [backend ...]
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
INPUTS = HERE / "inputs"
OUTPUTS = HERE / "outputs" / "bakeoff"

PERSON = INPUTS / "person/v0-crosswalk.png"
GARMENT = INPUTS / "garment/red-shirt-1.png"
DESCRIPTION = "a dark red and cream plaid flannel shirt"


def load_env() -> dict[str, str]:
    path = Path.home() / ".fitcheck.env"
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


ENV = load_env()


def run_idm() -> bytes:
    """Open weights on our own Modal GPU."""
    import modal

    tryon = modal.Cls.from_name("fitcheck-tryon", "TryOn")()
    out = tryon.render.remote(
        PERSON.read_bytes(),
        GARMENT.read_bytes(),
        garment_description=DESCRIPTION,
        category="upper_body",
    )
    print(f"    gpu {out['render_seconds']}s")
    return out["image"]


def run_fashn() -> bytes:
    """FASHN v1.6 on fal. Maskless, native 864x1296."""
    import fal_client
    import requests

    os.environ["FAL_KEY"] = ENV["FAL_KEY"]

    result = fal_client.subscribe(
        "fal-ai/fashn/tryon/v1.6",
        arguments={
            "model_image": fal_client.upload_file(str(PERSON)),
            "garment_image": fal_client.upload_file(str(GARMENT)),
            "category": "tops",
        },
    )
    url = result["images"][0]["url"]
    return requests.get(url, timeout=120).content


SOFT_PROMPT = (
    "Replace only the shirt worn by the person in the first image with the "
    "garment shown in the second image. Keep the person's face, hair, "
    "sunglasses, skin tone, body shape and pose exactly as they are. Keep "
    "the trousers, shoes, background, lighting and camera angle completely "
    "unchanged. Match the garment's colour and plaid pattern precisely."
)

# The soft prompt produced a beautiful image that quietly regenerated the
# street, moved the car, resized the photo and changed the garment from
# oversized to fitted. This version names each of those failures as a
# prohibition, because the model has no mask to enforce them structurally.
STRICT_PROMPT = (
    "This is a photo editing task, not image generation. You are editing the "
    "first image in place.\n\n"
    "EDIT: replace the shirt the person is wearing with the exact garment in "
    "the second image.\n\n"
    "The garment in image two is an OVERSIZED, LOOSE-FITTING flannel shirt "
    "with FULL-LENGTH sleeves worn down to the wrist. Reproduce that silhouette "
    "and that exact colour: dark red and burgundy plaid with cream lines. Do "
    "not slim it. Do not roll the sleeves. Do not restyle it.\n\n"
    "DO NOT CHANGE ANYTHING ELSE. Specifically, you must not alter:\n"
    "- the person's face, hair, facial hair, sunglasses, skin tone, height, "
    "build or pose\n"
    "- the trousers, the shoes, or the hand in the pocket\n"
    "- the road, crosswalk markings, kerb or pavement texture\n"
    "- the car, its position, angle and colour\n"
    "- the palm trees, their number, placement and height\n"
    "- the buildings, street furniture, signage and lamp posts\n"
    "- the sky, the clouds, the sunset colours and the direction of the light\n"
    "- the framing, crop, aspect ratio and output resolution\n\n"
    "Every pixel outside the shirt must be identical to the input. Return the "
    "edited image at the same dimensions as the first image."
)


def _gemini(prompt: str) -> bytes:
    """Gemini 3 Pro Image. A general editor, steered by prompt alone.

    No mask and no pose control, so the instruction has to carry every
    constraint the other two get structurally.
    """
    from google import genai

    client = genai.Client(api_key=ENV["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-3-pro-image",
        contents=[Image.open(PERSON), Image.open(GARMENT), prompt],
    )

    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            return part.inline_data.data

    # A refusal or a text-only answer is a result worth seeing, not a crash.
    text = " ".join(
        p.text for p in response.candidates[0].content.parts if getattr(p, "text", None)
    )
    raise RuntimeError(f"no image returned: {text[:300]}")


def run_nanobanana() -> bytes:
    return _gemini(SOFT_PROMPT)


def run_nanobanana_strict() -> bytes:
    return _gemini(STRICT_PROMPT)


BACKENDS = {
    "idm": run_idm,
    "fashn": run_fashn,
    "nanobanana": run_nanobanana,
    "nanobanana-strict": run_nanobanana_strict,
}


def main(argv: list[str]) -> int:
    wanted = argv or list(BACKENDS)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    results = []
    for name in wanted:
        if name not in BACKENDS:
            print(f"unknown backend: {name}")
            return 1
        print(f"  {name} ...", flush=True)
        started = time.time()
        try:
            data = BACKENDS[name]()
        except Exception as exc:  # one backend failing must not lose the rest
            print(f"    FAILED {exc.__class__.__name__}: {exc}")
            continue
        path = OUTPUTS / f"{name}.png"
        Image.open(io.BytesIO(data)).convert("RGB").save(path)
        wall = time.time() - started
        print(f"    {wall:.0f}s wall -> {path.name}")
        results.append((name, path, wall))

    if len(results) > 1:
        contact_sheet(results)
    return 0


def contact_sheet(results: list[tuple[str, Path, float]]) -> None:
    """Side by side with the source garment, which is what you judge against."""
    tw, th, pad, cap = 420, 560, 16, 30
    cols = len(results) + 1
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, th + cap + 2 * pad), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    panels = [("source garment", GARMENT, None)] + results
    for i, (name, path, wall) in enumerate(panels):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((tw, th), Image.LANCZOS)
            x = pad + i * (tw + pad) + (tw - im.width) // 2
            sheet.paste(im, (x, pad))
        label = name if wall is None else f"{name}  {wall:.0f}s"
        draw.text((pad + i * (tw + pad), pad + th + 8), label, fill=(30, 30, 30))

    out = OUTPUTS / "contact-sheet.png"
    sheet.save(out)
    print(f"\ncontact sheet: {out}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
