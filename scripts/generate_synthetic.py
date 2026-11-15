#!/usr/bin/env python3
"""Generate synthetic training images with a pretrained latent-diffusion model.

Phase 2, spec section 12.2. Run this only after the go/no-go gate: the real-data
learning curve (2A-1) must still be rising from 100 to 200 images and the DIV2K
control must help. The output is *raw*: ``scripts/check_synthetic.py`` then
applies the BSDS500 preprocessing and the quality and leakage checks, and
writes the frozen set the training scripts load.

Every image has its own fixed seed and a prompt from a deterministic prompt
bank mirroring BSDS500 content; ``generation.json`` records the model id and
revision, prompts, seeds, steps, guidance and scheduler, so the set can be
regenerated exactly.

Image-to-image variations (``--img2img-from``) start from BSDS500 *train*
images only and are capped at 25% of the set by the spec.

Requires ``pip install diffusers transformers accelerate`` and a one-time model
download (2-7 GB); cite the model licence (CreativeML OpenRAIL-M / SDXL).

Examples:
    PYTHONPATH=src python3 scripts/generate_synthetic.py --out data/synthetic/raw-v1 --count 2600
    PYTHONPATH=src python3 scripts/generate_synthetic.py --out data/synthetic/raw-v1 \\
        --model runwayml/stable-diffusion-v1-5 --steps 30 --guidance 7.0 --count 2600
    # dry run: print the first prompts without loading a model
    PYTHONPATH=src python3 scripts/generate_synthetic.py --out /tmp/x --count 5 --dry-run
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

SUBJECTS = {
    "animals": ["a deer in a meadow", "a zebra on the savanna", "a bird perched on a branch",
                "a herd of elephants", "a tiger resting in grass", "a horse in a field",
                "a close-up of a cat", "fish in clear water", "a bear by a river"],
    "people": ["a street market with people", "a person walking in a park", "a fisherman on a boat",
               "children playing on a beach", "a farmer in a field", "a crowd at a festival"],
    "landscapes": ["a mountain range", "a rocky desert canyon", "a forest trail", "rolling hills",
                   "a snowy valley", "a rugged coastline with cliffs", "a waterfall in a gorge"],
    "architecture": ["an old stone church", "a city skyline", "a wooden barn", "a stone bridge",
                     "a lighthouse", "a narrow European street", "ancient ruins"],
    "plants": ["wildflowers in a field", "a large oak tree", "a cactus garden", "autumn leaves",
               "tall grass", "a vineyard"],
    "water": ["a calm lake with reflections", "ocean waves on rocks", "a river with rapids",
              "a harbour with boats", "a frozen pond"],
    "vehicles": ["an old car on a road", "a train on a bridge", "sailboats on a bay",
                 "a bicycle against a wall", "an airplane on a runway"],
    "objects": ["fruit on a wooden table", "a stack of books", "pottery on a shelf",
                "tools in a workshop", "a fence and a gate"],
    "textures": ["tree bark", "a brick wall", "pebbles on a beach", "woven fabric",
                 "rock strata", "a field of sand dunes"],
}
MODIFIERS = ["in daylight", "on an overcast day", "in the late afternoon", "seen from eye level",
             "seen from a distance", "in close-up"]
SUFFIX = "sharp, in-focus photograph, everything in focus, deep depth of field, natural colours"
NEGATIVE = ("blur, blurry, bokeh, shallow depth of field, motion blur, out of focus, text, watermark, "
            "logo, signature, illustration, painting, drawing, cartoon, 3d render, cgi, frame, border")


def prompt_bank() -> list[tuple[str, str]]:
    """Deterministic (category, prompt) list: every subject x modifier."""
    return [
        (category, f"{subject} {modifier}, {SUFFIX}")
        for (category, subjects), modifier in itertools.product(SUBJECTS.items(), MODIFIERS)
        for subject in subjects
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory for raw images")
    parser.add_argument("--count", type=int, default=2600, help="images to generate (QC rejects 15-30%%)")
    parser.add_argument("--model", default="stabilityai/sdxl-turbo")
    parser.add_argument("--revision", default=None, help="model revision (recorded)")
    parser.add_argument("--steps", type=int, default=4, help="4 for SDXL-Turbo; ~30 for SD 1.5")
    parser.add_argument("--guidance", type=float, default=0.0, help="0 for SDXL-Turbo; ~7 for SD 1.5")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--img2img-from", default=None,
                        help="BSDS500 root: make image-to-image variations of *train* images")
    parser.add_argument("--img2img-fraction", type=float, default=0.0,
                        help="share of images made by img2img (at most 0.25)")
    parser.add_argument("--strength", type=float, default=0.4, help="img2img strength (0.3-0.5)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without a model")
    args = parser.parse_args()

    if not 0.0 <= args.img2img_fraction <= 0.25:
        raise SystemExit("--img2img-fraction must be in [0, 0.25] (spec 12.2)")
    if args.img2img_fraction > 0 and not args.img2img_from:
        raise SystemExit("--img2img-fraction needs --img2img-from (BSDS500 root)")

    bank = prompt_bank()
    out = Path(args.out)
    n_img2img = round(args.count * args.img2img_fraction)
    plan = []
    for i in range(args.count):
        category, prompt = bank[i % len(bank)]
        plan.append({
            "file": f"syn_{args.seed_start + i:06d}.png",
            "seed": args.seed_start + i,
            "category": category,
            "prompt": prompt,
            "mode": "img2img" if i >= args.count - n_img2img else "txt2img",
        })
    if args.dry_run:
        print(f"{len(bank)} distinct prompts; {args.count} images ({n_img2img} img2img)")
        for item in plan[:5]:
            print(f"  {item['file']} seed {item['seed']} [{item['mode']}] {item['prompt']}")
        return

    try:
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error}; install with: pip install diffusers transformers accelerate"
        ) from None

    sources = []
    if n_img2img:
        from deblur.data import BSDS500  # train split only: val/test must never seed generation

        sources = BSDS500(args.img2img_from, split="train", luminance=False).paths
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    text2img = AutoPipelineForText2Image.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype
    ).to(args.device)
    img2img = AutoPipelineForImage2Image.from_pipe(text2img) if n_img2img else None
    records = []
    for item in plan:
        path = out / item["file"]
        generator = torch.Generator(device=args.device).manual_seed(item["seed"])
        common = dict(prompt=item["prompt"], negative_prompt=NEGATIVE, num_inference_steps=args.steps,
                      guidance_scale=args.guidance, generator=generator)
        if not path.exists():
            if item["mode"] == "img2img":
                from PIL import Image

                source = sources[item["seed"] % len(sources)]
                init = Image.open(source).convert("RGB").resize((args.width, args.height))
                image = img2img(image=init, strength=args.strength, **common).images[0]
                item["source"] = source.name
            else:
                image = text2img(width=args.width, height=args.height, **common).images[0]
            image.save(path)
        records.append(item)
        if len(records) % 50 == 0:
            print(f"  {len(records)}/{len(plan)}")
    (out / "generation.json").write_text(json.dumps({
        "model": args.model,
        "revision": args.revision,
        "scheduler": type(text2img.scheduler).__name__,
        "steps": args.steps,
        "guidance": args.guidance,
        "size": [args.width, args.height],
        "negative_prompt": NEGATIVE,
        "strength": args.strength if n_img2img else None,
        "images": records,
    }, indent=1))
    print(f"wrote {len(records)} raw images and generation.json to {out}")


if __name__ == "__main__":
    main()
