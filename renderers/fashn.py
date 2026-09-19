"""FASHN v1.6 via fal. Untested — the account had no balance on 2026-09-18.

Kept because it is the strongest commercial option on paper: purpose-built,
maskless, native 864x1296. preserves_original is set True on the vendor's
claim rather than on measurement, and that stays a claim until a render proves
it. Verify before trusting it for anything.
"""

from __future__ import annotations

from .base import Availability, Category, Renderer, RenderResult
from .gemini import ENV_FILE, read_env

ENDPOINT = "fal-ai/fashn/tryon/v1.6"

# FASHN categories, which are not the same vocabulary the open models use.
CATEGORY_MAP = {
    "upper_body": "tops",
    "lower_body": "bottoms",
    "dresses": "one-pieces",
}


class FashnRenderer(Renderer):
    name = "fashn"
    preserves_original = True  # vendor claim, NOT yet measured
    cost_per_render_usd = 0.075

    def available(self) -> Availability:
        try:
            import fal_client  # noqa: F401
        except ImportError:
            return Availability(False, "pip install fal-client")
        if not read_env("FAL_KEY"):
            return Availability(False, f"FAL_KEY not set (looked in {ENV_FILE})")
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
        import os
        import tempfile

        import fal_client
        import requests

        os.environ["FAL_KEY"] = read_env("FAL_KEY")

        # fal wants URLs, so the bytes have to land on disk briefly.
        urls = []
        for data, suffix in ((person, "-person.png"), (garment, "-garment.png")):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(data)
                urls.append(fal_client.upload_file(handle.name))

        try:
            result = fal_client.subscribe(
                ENDPOINT,
                arguments={
                    "model_image": urls[0],
                    "garment_image": urls[1],
                    "category": CATEGORY_MAP[category],
                },
            )
        except Exception as exc:
            # The failure worth naming: an exhausted balance reads as a 403 and
            # looks like an auth problem, which sends you chasing the key.
            if "403" in str(exc) or "locked" in str(exc).lower():
                raise RuntimeError(
                    "fal rejected the request. If it mentions balance, the "
                    "account is out of credit: fal.ai/dashboard/billing"
                ) from exc
            raise

        return RenderResult(
            image=requests.get(result["images"][0]["url"], timeout=120).content,
            backend=self.name,
            seconds=0.0,
            cost_usd=self.cost_per_render_usd,
            params={"endpoint": ENDPOINT, "category": CATEGORY_MAP[category]},
        )
