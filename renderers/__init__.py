"""Render backend registry.

The app asks for a renderer by name and never imports a backend directly:

    from renderers import get_renderer
    result = get_renderer().render(person_bytes, garment_bytes,
                                   description="a red plaid flannel shirt")

Switching engines is FITCHECK_RENDERER=gemini, or a value in config. Adding one
is a new module and a line in BACKENDS.
"""

from __future__ import annotations

import os

from .base import Availability, Category, Renderer, RenderResult
from .fashn import FashnRenderer
from .gemini import GeminiRenderer
from .idm_modal import IdmModalRenderer

BACKENDS: dict[str, type[Renderer]] = {
    IdmModalRenderer.name: IdmModalRenderer,
    GeminiRenderer.name: GeminiRenderer,
    FashnRenderer.name: FashnRenderer,
}

# IDM-VTON on Modal: the only backend measured to return the user's real
# photograph with just the garment changed, which is the whole product.
DEFAULT = IdmModalRenderer.name


def get_renderer(name: str | None = None) -> Renderer:
    """Return a renderer by name, by env var, or the default."""
    key = name or os.environ.get("FITCHECK_RENDERER") or DEFAULT
    if key not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown renderer {key!r}; known: {known}")
    return BACKENDS[key]()


def survey() -> list[tuple[Renderer, Availability]]:
    """Every backend with whether it can run right now, and why not.

    Lets the UI grey out an engine with a real reason instead of failing at
    click time — an exhausted fal balance returns a 403 that otherwise reads
    like a bad key.
    """
    return [(cls(), cls().available()) for cls in BACKENDS.values()]


__all__ = [
    "Availability",
    "Category",
    "Renderer",
    "RenderResult",
    "BACKENDS",
    "DEFAULT",
    "get_renderer",
    "survey",
]
