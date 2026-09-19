# fitcheck

Compose outfits from clothes you own and render a photoreal image of yourself
wearing them.

A single-user desktop app. Photos of your wardrobe and reference photos of you
stay on your machine; only the render call leaves it.

## Status

Render pipeline works end to end. The app itself is not built yet.

## How it works

```
  Local                                Cloud (on demand)
  ┌─────────────────────────┐          ┌──────────────────────┐
  │ Desktop app (localhost) │          │ Serverless GPU fn    │
  │  • closet grid          │          │  • try-on model      │
  │  • outfit composer      │ ───────► │  • weights in Volume │
  │  • render viewport      │  render  │  • scale to zero     │
  ├─────────────────────────┤ ◄─────── └──────────────────────┘
  │ SQLite  items, outfits, │  image
  │         renders         │
  └─────────────────────────┘
```

IDM-VTON runs on Modal serverless GPU. Measured on an L4: about 28s and
$0.006 per warm render, ~39s cold start. Renders happen on demand and are
cached, so a repeated outfit is free.

## Swappable render backends

The app talks to an interface, never to a model:

```python
from renderers import get_renderer

result = get_renderer().render(person_bytes, garment_bytes,
                               description="a red plaid flannel shirt")
```

Switch with `FITCHECK_RENDERER=<name>`. Each backend declares what it actually
is, so the app can choose sensibly:

| backend | preserves your photo | $/render | notes |
|---|---|---|---|
| `idm-modal` | yes | ~0.006 | default; open weights on your own GPU |
| `fashn` | claimed, unverified | 0.075 | commercial, maskless, 864x1296 |
| `gemini` | **no** | ~0.13 | regenerates the scene; long-tail fallback only |

`preserves_original` is measured, not taken from marketing. Gemini 3 Pro Image
produces a good-looking picture but rewrites the background, moves objects and
changes the output resolution, even when instructed not to. That is fine for a
styling suggestion and wrong for a mirror, which is why it is not the default.

Adding a backend is a module in `renderers/` and a line in `BACKENDS`.

## Layout

```
renderers/      render backends behind one interface
experiments/    the prove-the-render harness (Modal function, input prep, bakeoff)
```

## Setup

```bash
pip install modal pillow
modal setup
modal deploy experiments/modal_tryon.py
modal run experiments/modal_tryon.py::fetch_weights   # one time, ~3 min
```

Optional backends need API keys in `~/.fitcheck.env` (mode 0600, never in the
repo):

```
FAL_KEY=...
GEMINI_API_KEY=...
```

## Running the render experiment

```bash
python3 experiments/prepare_inputs.py   # normalize your own photos
modal run experiments/modal_tryon.py    # one render
python3 experiments/bakeoff.py          # compare backends
```

Bring your own images: a full-body reference photo and garment photos. Put the
reference in `identity-kit/` and garments in `garments/`. Both are gitignored.

A reference shot in a plain, neutral base layer matters more than the
background. Models in this family are influenced by the garment you are already
wearing, and a strongly coloured shirt bleeds into the rendered one.
