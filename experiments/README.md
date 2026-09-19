# The prove-the-render experiment

Answer one question before building any app: does a try-on model produce
something that actually looks like you wearing your own clothes? No UI, no
database. Everything downstream depends on the answer.

Inputs live outside this repo. Put a full-body reference photo in
`identity-kit/` and garment photos in `garments/`; both are gitignored.

## What it answers

| Question | How |
|---|---|
| Does a casual photo work as a reference, or is a studio shot needed? | Render against two very different reference photos |
| Cold start and GPU tier | Timings returned with every render |
| Which backend? | `bakeoff.py` runs the same inputs through each one |
| Real cost per render | Measured, then compared against the estimate |

## Run it

```bash
python3 prepare_inputs.py          # local, free
modal deploy modal_tryon.py        # builds the image, ~5-10 min first time
modal run modal_tryon.py::fetch_weights   # one time, ~3 min
modal run modal_tryon.py           # smoke test: one render
python3 bakeoff.py                 # compare backends
```

## Notes from getting this working

**Weights go in a Modal Volume, not the image.** Downloading them takes about
three minutes; mounting them from the Volume takes under a second. That is the
difference between a usable cold start and an unusable one. `fetch_weights`
runs without a GPU on purpose: downloading is network and disk, and renting an
accelerator to watch a progress bar costs money for nothing.

**The repo's `ckpt/` files are 25-byte placeholders**, not weights. The real
ones come from the Hugging Face repo under different folder names, so
`modal_tryon.py` wires them in at container start. Any code that skips a
checkpoint path because the file "already exists" silently does nothing.

**IDM-VTON's pipeline is not a normal diffusers call.** It wants precomputed
prompt embeddings from two `encode_prompt` passes, a DensePose segmentation
image, and the garment as a normalized tensor. Follow `gradio_demo/app.py`
rather than the diffusers API it resembles, or the output is quietly wrong.

**Pin `scipy<1.15`.** OpenPose imports `scipy.ndimage.filters`, removed in 1.15.

**Crop to the person, then composite through the mask.** Rendering a full frame
where the subject occupies a fraction of it puts the face through a VAE
round-trip at low resolution and returns a blurred stranger. IDM-VTON's own
centre 3:4 crop is a no-op on input that is already 3:4. Pasting the crop box
back leaves a visible rectangular seam, so composite through the feathered
garment mask instead: everything outside it keeps its original pixels.

## Reading the output

Judge by eye. Does it look like you, or like someone with your build? Is the
pattern and colour right, or approximated? Where a render is wrong, open the
matching `*.mask.png` — the mask is what the model was allowed to repaint, and
a bad mask explains most bad renders.
