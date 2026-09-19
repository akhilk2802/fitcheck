"""Gemini 3 Pro Image ("Nano Banana Pro"). Long-tail fallback, never the default.

Measured 2026-09-18, twice, with a deliberately strict second prompt: it gets
garment colour and cut right and is fast (~18s), but it regenerates the scene.
The car moves, palm trees change, storefronts become different buildings, and
output comes back at 896x1195 regardless of a 1152x2048 input and an explicit
instruction to preserve resolution.

That is architectural, not a prompting failure — no mask, no inpainting, so
nothing enforces "leave this alone." Hence preserves_original = False, which is
the app's signal to warn before using it.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from .base import Availability, Category, Renderer, RenderResult

MODEL = "gemini-3-pro-image"
ENV_FILE = Path.home() / ".fitcheck.env"

GARMENT_WORDS = {
    "upper_body": "shirt or top",
    "lower_body": "trousers or lower garment",
    "dresses": "dress or one-piece outfit",
}


def read_env(name: str) -> str | None:
    """Read a key from the environment, falling back to ~/.fitcheck.env.

    The file is 0600 and lives outside the repo, so keys never reach git.
    """
    if os.environ.get(name):
        return os.environ[name]
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip()
    return None


class GeminiRenderer(Renderer):
    name = "gemini"
    preserves_original = False
    cost_per_render_usd = 0.13

    def available(self) -> Availability:
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return Availability(False, "pip install google-genai")
        if not read_env("GEMINI_API_KEY"):
            return Availability(False, f"GEMINI_API_KEY not set (looked in {ENV_FILE})")
        return Availability(True)

    def _prompt(self, description: str, category: Category) -> str:
        what = GARMENT_WORDS[category]
        return (
            "This is a photo editing task, not image generation. You are editing "
            "the first image in place.\n\n"
            f"EDIT: replace the {what} the person is wearing with the exact "
            "garment shown in the second image, which is "
            f"{description}. Reproduce its exact colour, pattern, cut and "
            "sleeve length. Do not restyle it.\n\n"
            "DO NOT CHANGE ANYTHING ELSE. Not the face, hair, facial hair, "
            "eyewear, skin tone, build or pose. Not the other garments or the "
            "shoes. Not the background, its objects, or their positions. Not "
            "the lighting, the framing, the crop, or the output resolution.\n\n"
            "Every pixel outside the replaced garment must be identical to the "
            "input. Return the edited image at the input's dimensions."
        )

    def _render(
        self,
        person: bytes,
        garment: bytes,
        *,
        description: str,
        category: Category,
        seed: int,
        steps: int,
    ) -> RenderResult:
        from google import genai
        from PIL import Image

        client = genai.Client(api_key=read_env("GEMINI_API_KEY"))
        prompt = self._prompt(description, category)
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                Image.open(io.BytesIO(person)),
                Image.open(io.BytesIO(garment)),
                prompt,
            ],
        )

        parts = response.candidates[0].content.parts
        for part in parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                return RenderResult(
                    image=part.inline_data.data,
                    backend=self.name,
                    seconds=0.0,  # filled in by Renderer.render
                    cost_usd=self.cost_per_render_usd,
                    params={"model": MODEL, "prompt": prompt, "category": category},
                )

        # A refusal or a text-only reply is a result worth surfacing, not a crash.
        text = " ".join(p.text for p in parts if getattr(p, "text", None))
        raise RuntimeError(f"{MODEL} returned no image: {text[:300]}")
