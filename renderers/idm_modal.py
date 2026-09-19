"""IDM-VTON on Modal serverless GPU. The default engine.

Chosen because it returns the user's actual photograph with only the garment
region changed. Measured 2026-09-18: ~28s and ~$0.006 warm on an L4, ~39s cold.

Its known weakness is colour contamination from whatever the person is already
wearing in the reference photo. That is an input problem, fixed by a neutral
base layer, not a reason to swap engines.
"""

from __future__ import annotations

from .base import Availability, Category, Renderer, RenderResult

APP = "fitcheck-tryon"
CLASS = "TryOn"

# Modal L4, derived from the published A100 rate of $0.000694/sec with L4 about
# 3.1x cheaper. Used for the pre-click estimate, not for billing.
L4_USD_PER_SECOND = 0.000224


class IdmModalRenderer(Renderer):
    name = "idm-modal"
    preserves_original = True
    cost_per_render_usd = 0.006
    cold_start_seconds = 39.0

    def __init__(self, app: str = APP, cls: str = CLASS) -> None:
        self._app, self._cls = app, cls
        self._handle = None

    def _remote(self):
        if self._handle is None:
            import modal

            self._handle = modal.Cls.from_name(self._app, self._cls)()
        return self._handle

    def available(self) -> Availability:
        try:
            import modal  # noqa: F401
        except ImportError:
            return Availability(False, "modal not installed: pip install modal")
        try:
            self._remote()
        except Exception as exc:
            return Availability(
                False,
                f"{self._app} not reachable ({exc.__class__.__name__}). "
                f"Deploy it: modal deploy experiments/modal_tryon.py",
            )
        return Availability(True)

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
        out = self._remote().render.remote(
            person,
            garment,
            garment_description=description,
            category=category,
            seed=seed,
            steps=steps,
        )
        seconds = out["render_seconds"]
        return RenderResult(
            image=out["image"],
            mask=out.get("mask"),
            backend=self.name,
            seconds=seconds,
            cost_usd=round(seconds * L4_USD_PER_SECOND, 5),
            params={
                "model": "yisol/IDM-VTON",
                "gpu": out.get("gpu"),
                "steps": out.get("steps", steps),
                "seed": out.get("seed", seed),
                "category": category,
            },
        )
