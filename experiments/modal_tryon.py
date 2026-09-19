"""IDM-VTON on Modal serverless GPU.

This is the prove-the-render experiment from docs/design.md, not app code. Its
only job is to answer three questions before a line of the real app is written:

  1. Does a try-on model survive a real-world reference photo, or does it need
     a studio-grade identity kit?
  2. What does a cold start actually cost, with multi-GB weights?
  3. Is an L4 enough, or does this need an A10G / A100?

The render path mirrors IDM-VTON's own gradio_demo/app.py rather than the
simpler diffusers-style call it superficially resembles. That pipeline wants
precomputed prompt embeddings, a DensePose segmentation image, and the garment
as a normalized tensor. Deviating from it produces silent garbage, not errors.

Weights live in a Modal Volume so the first run downloads once and later cold
starts mount. API verified against modal 1.5.5.

Deploy:  modal deploy modal_tryon.py
Smoke:   modal run modal_tryon.py
"""

from __future__ import annotations

import io
import time

import modal

APP_NAME = "fitcheck-tryon"
MODEL_REPO = "yisol/IDM-VTON"
REPO = "/idm"

# GPU is a variable of the experiment, not a setting. Start on L4 because the
# design doc's cost model assumes it; if it OOMs or crawls, that estimate is
# wrong and this is how we find out.
GPU = "L4"

volume = modal.Volume.from_name("fitcheck-weights", create_if_missing=True)
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "wget")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "diffusers==0.25.0",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "huggingface_hub==0.25.2",
        "pillow",
        "numpy<2",
        # OpenPose's body.py and hand.py import scipy.ndimage.filters, which
        # SciPy removed in 1.15. Unpinned, pip installs 1.17 and the import
        # dies before a single keypoint is found.
        "scipy<1.15",
        "scikit-image==0.24.0",  # openpose hand.py: from skimage.measure import label
        "opencv-python-headless",
        "einops",
        "onnxruntime",
        "av",
        "sentencepiece",
        # detectron2 is vendored inside gradio_demo/, so it is not pip
        # installed, but it still imports these at module load.
        "fvcore",
        "iopath",
        "portalocker",
        "cloudpickle",
        "omegaconf",
        "yacs",
        "termcolor",
        "tabulate",
        "matplotlib",
        "tqdm",
        "pycocotools",
    )
    .run_commands(f"git clone --depth 1 https://github.com/yisol/IDM-VTON.git {REPO}")
    .env(
        {
            "HF_HOME": CACHE,
            # gradio_demo holds utils_mask, apply_net, detectron2 and densepose;
            # the repo root holds src/ and preprocess/. Both must be importable.
            "PYTHONPATH": f"{REPO}:{REPO}/gradio_demo",
        }
    )
    .workdir(REPO)  # './configs/...' and './ckpt/...' resolve from the repo root
)

app = modal.App(APP_NAME, image=image)

# The repo ships 25-byte placeholder files at these paths that literally say
# "put <name> here". The real weights come from the HF repo, under different
# folder names, so they are wired in at container start.
CHECKPOINTS = {
    "parsing_atr.onnx": "ckpt/humanparsing/parsing_atr.onnx",
    "parsing_lip.onnx": "ckpt/humanparsing/parsing_lip.onnx",
    "body_pose_model.pth": "ckpt/openpose/ckpts/body_pose_model.pth",
    "model_final_162be9.pkl": "ckpt/densepose/model_final_162be9.pkl",
}

# Anything at or below this is a placeholder, not a model.
PLACEHOLDER_BYTES = 1024


def wire_checkpoints(snapshot: str) -> None:
    """Point the repo's checkpoint paths at the real weights in the Volume.

    Replaces the shipped placeholders. An earlier version of this skipped any
    path that already existed, which silently did nothing at all, since the
    placeholders exist. Size is checked so a real file is never clobbered.
    """
    from pathlib import Path

    snap = Path(snapshot)
    found = {p.name: p for p in snap.rglob("*") if p.name in CHECKPOINTS and p.is_file()}

    for filename, rel in CHECKPOINTS.items():
        target = Path(REPO) / rel
        source = found.get(filename)
        if source is None:
            print(f"  MISSING in snapshot: {filename}")
            continue

        if target.exists() or target.is_symlink():
            size = target.stat().st_size if target.exists() else 0
            if size > PLACEHOLDER_BYTES and not target.is_symlink():
                print(f"  {rel} already real ({size} bytes), left alone")
                continue
            target.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        print(f"  {rel} -> {source.name} ({source.stat().st_size / 1e6:.0f} MB)")


@app.function(timeout=60 * 5)  # CPU only, seconds, costs almost nothing
def inspect_repo() -> str:
    """List the repo layout. One listing beats three rounds of guessing."""
    from pathlib import Path

    lines = [
        str(p.relative_to(REPO))
        for p in sorted(Path(REPO).rglob("*.py"))
        if len(p.relative_to(REPO).parts) <= 3
    ]
    out = "\n".join(lines)
    print(out)  # modal run does not echo return values
    return out


@app.function(
    # No GPU on purpose: this is network and disk. Renting an L4 to watch a
    # download finish costs real money for zero benefit.
    volumes={CACHE: volume},
    timeout=60 * 60,
)
def fetch_weights() -> str:
    """Download weights into the Volume once, so cold starts only mount."""
    from huggingface_hub import snapshot_download

    started = time.time()
    path = snapshot_download(repo_id=MODEL_REPO, cache_dir=CACHE)
    volume.commit()
    msg = f"{path} in {time.time() - started:.0f}s"
    print(msg)
    return msg


@app.cls(
    gpu=GPU,
    volumes={CACHE: volume},
    timeout=60 * 15,
    scaledown_window=120,  # stay warm between runs of a comparison batch
)
class TryOn:
    @modal.enter()
    def load(self) -> None:
        """Runs once per container. Everything here is cold-start cost."""
        import torch
        from diffusers import AutoencoderKL, DDPMScheduler
        from huggingface_hub import snapshot_download
        from torchvision import transforms
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection,
        )

        self.t0 = time.time()
        self.device = "cuda"
        dtype = torch.float16

        print("wiring preprocessing checkpoints")
        wire_checkpoints(snapshot_download(MODEL_REPO, cache_dir=CACHE))

        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet
        from src.unet_hacked_tryon import UNet2DConditionModel as TryonUNet
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose

        common = dict(torch_dtype=dtype, cache_dir=CACHE)

        unet = TryonUNet.from_pretrained(MODEL_REPO, subfolder="unet", **common)
        garment_unet = GarmentUNet.from_pretrained(
            MODEL_REPO, subfolder="unet_encoder", **common
        )
        vae = AutoencoderKL.from_pretrained(MODEL_REPO, subfolder="vae", **common)
        text_encoder = CLIPTextModel.from_pretrained(
            MODEL_REPO, subfolder="text_encoder", **common
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            MODEL_REPO, subfolder="text_encoder_2", **common
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            MODEL_REPO, subfolder="image_encoder", **common
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO, subfolder="tokenizer", use_fast=False, cache_dir=CACHE
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            MODEL_REPO, subfolder="tokenizer_2", use_fast=False, cache_dir=CACHE
        )

        self.pipe = TryonPipeline.from_pretrained(
            MODEL_REPO,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            image_encoder=image_encoder,
            # The repo ships no feature_extractor folder, so diffusers would
            # hunt for a preprocessor_config.json that does not exist. Building
            # the default and passing it in is what IDM-VTON's own app does.
            feature_extractor=CLIPImageProcessor(),
            scheduler=DDPMScheduler.from_pretrained(
                MODEL_REPO, subfolder="scheduler", cache_dir=CACHE
            ),
            **common,
        )
        self.pipe.unet_encoder = garment_unet
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)

        self.parsing = Parsing(0)
        self.openpose = OpenPose(0)
        self.openpose.preprocessor.body_estimation.model.to(self.device)

        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.load_seconds = round(time.time() - self.t0, 1)
        print(f"loaded in {self.load_seconds}s")

    def _person_box(self, image, margin: float = 0.12):
        """Find the person and return a 3:4 crop box around them.

        IDM-VTON's own app takes a centre 3:4 crop, which does nothing at all
        when the input is already 3:4 — the exact case here. What actually
        costs quality is the person occupying a fraction of the frame, so the
        crop follows the body instead of the frame.
        """
        import numpy as np

        probe = image.resize((384, 512))
        parsed, _ = self.parsing(probe)
        body = np.array(parsed) > 0  # any parsed body part, background is 0
        if not body.any():
            return None

        ys, xs = np.where(body)
        sx, sy = image.width / 384, image.height / 512
        x0, x1 = xs.min() * sx, xs.max() * sx
        y0, y1 = ys.min() * sy, ys.max() * sy

        pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
        x0, x1 = x0 - pad_x, x1 + pad_x
        y0, y1 = y0 - pad_y, y1 + pad_y

        # Grow to 3:4 around the body's centre so the model gets the aspect
        # ratio it was trained on without squashing anyone.
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        if w / h > 0.75:
            h = w / 0.75
        else:
            w = h * 0.75

        x0, y0 = cx - w / 2, cy - h / 2
        # Clamp inside the image, shifting rather than shrinking, so a person
        # standing near an edge does not get cropped through the shoes.
        x0 = max(0, min(x0, image.width - w))
        y0 = max(0, min(y0, image.height - h))
        w, h = min(w, image.width), min(h, image.height)
        return (int(x0), int(y0), int(x0 + w), int(y0 + h))

    @modal.method()
    def render(
        self,
        person_png: bytes,
        garment_png: bytes,
        garment_description: str = "a shirt",
        steps: int = 30,
        seed: int = 0,
        category: str = "upper_body",
        crop_to_person: bool = True,
    ) -> dict:
        """Render one person wearing one garment.

        With crop_to_person, the model sees a tight 3:4 crop at its native
        768x1024 and the result is pasted back into the full-resolution
        original. The face never goes through the VAE, and the garment gets
        several times the pixels it would otherwise.
        """
        import apply_net
        import torch
        from detectron2.data.detection_utils import (
            _apply_exif_orientation,
            convert_PIL_to_numpy,
        )
        from PIL import Image
        from utils_mask import get_mask_location

        started = time.time()
        original = Image.open(io.BytesIO(person_png)).convert("RGB")
        garment = Image.open(io.BytesIO(garment_png)).convert("RGB").resize((768, 1024))

        box = self._person_box(original) if crop_to_person else None
        if box:
            crop = original.crop(box)
            coverage = ((box[2] - box[0]) * (box[3] - box[1])) / (
                original.width * original.height
            )
            print(f"crop {box} ({coverage:.0%} of frame)")
        else:
            crop = original
        crop_size = crop.size
        human = crop.resize((768, 1024))

        # Mask: which pixels the model may repaint. OpenPose finds keypoints,
        # human parsing segments the body, and together they decide. This is
        # the step most likely to struggle with a busy street background.
        keypoints = self.openpose(human.resize((384, 512)))
        parsed, _ = self.parsing(human.resize((384, 512)))
        mask, _ = get_mask_location("hd", category, parsed, keypoints)
        mask = mask.resize((768, 1024))

        # DensePose gives the pipeline a body-part segmentation to condition on.
        # Paths here are relative to the repo root, which is why the image sets
        # workdir to it.
        arg_img = convert_PIL_to_numpy(
            _apply_exif_orientation(human.resize((384, 512))), format="BGR"
        )
        args = apply_net.create_argument_parser().parse_args(
            (
                "show",
                "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                "./ckpt/densepose/model_final_162be9.pkl",
                "dp_segm",
                "-v",
                "--opts",
                "MODEL.DEVICE",
                "cuda",
            )
        )
        pose = args.func(args, arg_img)[:, :, ::-1]
        pose = Image.fromarray(pose).resize((768, 1024))

        dtype = torch.float16
        with torch.inference_mode():
            prompt = f"model is wearing {garment_description}"
            negative = "monochrome, lowres, bad anatomy, worst quality, low quality"
            (
                prompt_embeds,
                negative_embeds,
                pooled,
                negative_pooled,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative,
            )
            # The garment gets its own conditioning pass, describing the cloth
            # itself rather than the person wearing it.
            cloth_embeds, _, _, _ = self.pipe.encode_prompt(
                [f"a photo of {garment_description}"],
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=[negative],
            )

            pose_tensor = self.to_tensor(pose).unsqueeze(0).to(self.device, dtype)
            garment_tensor = self.to_tensor(garment).unsqueeze(0).to(self.device, dtype)

            result = self.pipe(
                prompt_embeds=prompt_embeds.to(self.device, dtype),
                negative_prompt_embeds=negative_embeds.to(self.device, dtype),
                pooled_prompt_embeds=pooled.to(self.device, dtype),
                negative_pooled_prompt_embeds=negative_pooled.to(self.device, dtype),
                num_inference_steps=steps,
                generator=torch.Generator(self.device).manual_seed(seed),
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=cloth_embeds.to(self.device, dtype),
                cloth=garment_tensor,
                mask_image=mask,
                image=human,
                height=1024,
                width=768,
                ip_adapter_image=garment,
                guidance_scale=2.0,
            )[0][0]

        if box:
            # Paste back only the pixels the model was allowed to change.
            #
            # Pasting the whole crop box leaves a visible rectangular seam:
            # the re-encoded region comes back at slightly different contrast
            # than the untouched original, and the boundary cuts straight
            # across the background. Compositing through the mask instead
            # means the face, hair, hands and background keep their original
            # pixels exactly, at full resolution.
            from PIL import ImageFilter

            # Feathered so the garment edge blends instead of cutting hard.
            blend = mask.convert("L").resize(crop_size).filter(
                ImageFilter.GaussianBlur(radius=max(2, crop_size[0] // 150))
            )
            composite = original.copy()
            composite.paste(result.resize(crop_size), (box[0], box[1]), blend)
            result = composite

        out, mask_buf = io.BytesIO(), io.BytesIO()
        result.save(out, format="PNG")
        mask.save(mask_buf, format="PNG")

        return {
            "image": out.getvalue(),
            # The mask comes back too: when a render looks wrong the mask is
            # almost always why, and guessing wastes a GPU minute per attempt.
            "mask": mask_buf.getvalue(),
            "load_seconds": self.load_seconds,
            "render_seconds": round(time.time() - started, 1),
            "gpu": GPU,
            "steps": steps,
            "seed": seed,
        }


@app.local_entrypoint()
def main() -> None:
    """Smoke test: one person, one garment, printed timings."""
    from pathlib import Path

    here = Path(__file__).parent
    person = here / "inputs/person/v0-crosswalk.png"
    garment = here / "inputs/garment/red-shirt-1.png"

    for p in (person, garment):
        if not p.exists():
            raise SystemExit(f"missing {p} — run prepare_inputs.py first")

    out = TryOn().render.remote(
        person.read_bytes(),
        garment.read_bytes(),
        garment_description="a dark red plaid flannel shirt",
    )

    dest = here / "outputs"
    dest.mkdir(exist_ok=True)
    (dest / "smoke.png").write_bytes(out["image"])
    (dest / "smoke-mask.png").write_bytes(out["mask"])

    print(f"cold load : {out['load_seconds']}s")
    print(f"render    : {out['render_seconds']}s on {out['gpu']}")
    print(f"wrote     : {dest / 'smoke.png'}")
