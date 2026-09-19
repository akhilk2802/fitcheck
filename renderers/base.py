"""The render interface every backend implements.

The app talks to this and never to a model. Switching engines is a config
change, not a rewrite — which matters because the engine choice was made on one
evening's evidence and the field moves every few months.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

Category = Literal["upper_body", "lower_body", "dresses"]


@dataclass
class RenderResult:
    """One rendered image plus everything worth recording about how it got made."""

    image: bytes
    backend: str
    seconds: float
    # What the model was allowed to repaint, when the backend has a mask.
    # Returned because a bad mask explains most bad renders, and re-rendering
    # to find out wastes GPU time.
    mask: bytes | None = None
    cost_usd: float | None = None
    # Backend-specific settings that affect output: seed, steps, prompt, model
    # id. Part of the cache key, so two renders that differ here are two
    # different renders.
    params: dict = field(default_factory=dict)


@dataclass
class Availability:
    ready: bool
    detail: str = ""


class Renderer(ABC):
    """A try-on engine.

    Subclasses declare their real-world characteristics as class attributes so
    the app can choose sensibly and warn honestly without hardcoding knowledge
    of any particular backend.
    """

    name: str

    #: Does this return the user's actual photograph with only the garment
    #: changed? False means the backend regenerates the scene — fine for a
    #: styling suggestion, wrong for a mirror. Measured, not claimed: Gemini 3
    #: Pro Image moved the car and rebuilt the street even when told not to.
    preserves_original: bool

    #: Rough cost of one warm render, for showing the user before they click.
    cost_per_render_usd: float

    #: Seconds to first render after idle. Serverless pays this; APIs do not.
    cold_start_seconds: float = 0.0

    @abstractmethod
    def available(self) -> Availability:
        """Can this backend actually run right now?

        Covers missing credentials, an exhausted account balance, an
        undeployed function. The app uses this to grey out a backend with a
        reason rather than failing at click time.
        """

    @abstractmethod
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
        """Backend-specific work. Use render() instead."""

    def render(
        self,
        person: bytes,
        garment: bytes,
        *,
        description: str = "a shirt",
        category: Category = "upper_body",
        seed: int = 0,
        steps: int = 30,
    ) -> RenderResult:
        """Render one person wearing one garment, with timing filled in."""
        started = time.time()
        result = self._render(
            person,
            garment,
            description=description,
            category=category,
            seed=seed,
            steps=steps,
        )
        if not result.seconds:
            result.seconds = round(time.time() - started, 1)
        return result
