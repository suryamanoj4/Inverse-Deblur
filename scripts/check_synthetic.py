#!/usr/bin/env python3
"""Quality control and freezing of a synthetic training-image set (spec 12.2).

Reads raw generated images, applies the BSDS500 preprocessing (Lanczos resize
and centre crop to 481x321 / 321x481, Rec.601 luminance), rejects images that
fail any check, and writes the accepted images as lossless PNG together with
``manifest.json`` (SHA-256 per accepted file, every rejection with its reason,
thresholds, and a statistics report). Training scripts load the set through
that manifest, so it cannot change silently.

Checks:
- sharpness: variance of the Laplacian below the 5th percentile of BSDS500
  train (a "sharp" target must be sharp; diffusion images often have bokeh);
- degenerate content: intensity entropy below the 5th percentile of BSDS train;
- near-duplicates inside the set: perceptual-hash Hamming distance <= --dup-bits;
- leakage: nearest BSDS500 val/test image by perceptual hash within
  --leak-bits, and, with ``--embedding clip`` (needs open_clip_torch), cosine
  similarity of CLIP embeddings above --leak-cosine.

Text/watermark detection needs OCR and is not automated; the negative prompt
and a manual spot check of the contact sheet cover it.

Example:
    PYTHONPATH=src python3 scripts/check_synthetic.py --raw data/synthetic/raw-v1 \\
        --out data/synthetic/syn-v1 --name syn-v1 --data-root data --keep 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from deblur.data import BSDS500, load_image, list_images
from deblur.data.synthetic import (
    MANIFEST,
    entropy,
    gradient_histogram,
    laplacian_variance,
    load_extra_image,
    nearest_hash,
    perceptual_hash,
    sha256_file,
    spectrum_slope,
)

GRAD_BINS = np.linspace(0.0, 0.5, 51)


def stats(images: list[np.ndarray]) -> dict:
    texture = [float(np.mean(np.hypot(*np.gradient(x)))) for x in images]
    return {
        "count": len(images),
        "gradient_histogram": np.mean([gradient_histogram(x, GRAD_BINS) for x in images], axis=0).tolist(),
        "spectrum_slope": {"mean": float(np.mean([spectrum_slope(x) for x in images])),
                           "std": float(np.std([spectrum_slope(x) for x in images]))},
        "texture_percentiles": dict(zip(("p5", "p25", "p50", "p75", "p95"),
                                        np.percentile(texture, [5, 25, 50, 75, 95]).round(4).tolist())),
    }


def clip_embedder(device: str):
    try:
        import open_clip
        import torch
    except ModuleNotFoundError as error:
        raise SystemExit(f"{error}; install open_clip_torch or drop --embedding") from None
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()

    def embed(gray: np.ndarray) -> np.ndarray:
        rgb = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).convert("RGB")
        with torch.no_grad():
            v = model.encode_image(preprocess(rgb)[None].to(device)).float()
        v = v / v.norm(dim=-1, keepdim=True)
        return v[0].cpu().numpy()

    return embed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="directory of raw generated images")
    parser.add_argument("--out", required=True, help="output directory for the frozen set")
    parser.add_argument("--name", required=True, help="dataset name, e.g. syn-v1")
    parser.add_argument("--data-root", required=True, help="BSDS500 root (thresholds and leakage refs)")
    parser.add_argument("--keep", type=int, default=None, help="keep at most N accepted images")
    parser.add_argument("--sharpness-percentile", type=float, default=5.0)
    parser.add_argument("--entropy-percentile", type=float, default=5.0)
    parser.add_argument("--dup-bits", type=int, default=4, help="pHash distance for duplicates (of 64)")
    parser.add_argument("--leak-bits", type=int, default=8, help="pHash distance for leakage (of 64)")
    parser.add_argument("--embedding", choices=["none", "clip"], default="none")
    parser.add_argument("--leak-cosine", type=float, default=0.92)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.out)
    if (out / MANIFEST).exists():
        raise SystemExit(f"{out / MANIFEST} exists: a frozen set is never overwritten")

    train = [load_image(p) for p in BSDS500(args.data_root, split="train").paths]
    sharp_min = float(np.percentile([laplacian_variance(x) for x in train], args.sharpness_percentile))
    entropy_min = float(np.percentile([entropy(x) for x in train], args.entropy_percentile))
    references = []
    for split in ("val", "test"):
        dataset = BSDS500(args.data_root, split=split)
        references += [(p.relative_to(dataset.image_root).as_posix(), load_image(p)) for p in dataset.paths]
    ref_hashes = [perceptual_hash(x) for _, x in references]
    embed = clip_embedder(args.device) if args.embedding == "clip" else None
    ref_embeddings = np.stack([embed(x) for _, x in references]) if embed else None

    raw_paths = list_images(args.raw)
    print(f"{len(raw_paths)} raw images; sharpness >= {sharp_min:.2e}, entropy >= {entropy_min:.2f} bits")
    accepted, rejected, kept_images, kept_hashes = [], [], [], []
    out.mkdir(parents=True, exist_ok=True)
    for path in raw_paths:
        name = path.relative_to(args.raw).as_posix()
        try:
            image = load_extra_image(path)
        except ValueError as error:
            rejected.append({"file": name, "reason": f"size: {error}"})
            continue
        reasons = []
        if laplacian_variance(image) < sharp_min:
            reasons.append("not sharp")
        if entropy(image) < entropy_min:
            reasons.append("low entropy")
        h = perceptual_hash(image)
        if kept_hashes and nearest_hash(h, kept_hashes)[1] <= args.dup_bits:
            reasons.append("near-duplicate")
        index, distance = nearest_hash(h, ref_hashes)
        if distance <= args.leak_bits:
            reasons.append(f"leakage: phash {distance} bits from {references[index][0]}")
        if embed is not None:
            cosine = ref_embeddings @ embed(image)
            if cosine.max() >= args.leak_cosine:
                reasons.append(f"leakage: clip cosine {cosine.max():.3f} to {references[int(cosine.argmax())][0]}")
        if reasons:
            rejected.append({"file": name, "reason": "; ".join(reasons)})
            continue
        if args.keep is not None and len(accepted) >= args.keep:
            rejected.append({"file": name, "reason": "over --keep"})
            continue
        target = out / (Path(name).stem + ".png")
        Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(target)
        accepted.append({"file": target.name, "source": name, "sha256": sha256_file(target)})
        kept_images.append(image)
        kept_hashes.append(h)

    generation = Path(args.raw) / "generation.json"
    manifest = {
        "name": args.name,
        "preprocessing": "Lanczos resize + centre crop to 481x321 / 321x481, Rec.601 luminance, 8-bit PNG",
        "generation_sha256": sha256_file(generation) if generation.exists() else None,
        "thresholds": {"laplacian_variance_min": sharp_min, "entropy_min": entropy_min,
                       "dup_bits": args.dup_bits, "leak_bits": args.leak_bits,
                       "embedding": args.embedding, "leak_cosine": args.leak_cosine if embed else None},
        "accepted": accepted,
        "rejected": rejected,
        "report": {"bsds_train": stats(train), args.name: stats(kept_images) if kept_images else None},
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))
    reasons = {}
    for item in rejected:
        for reason in item["reason"].split("; "):
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    print(f"accepted {len(accepted)}, rejected {len(rejected)} {reasons}")
    print(f"frozen set and manifest written to {out}")


if __name__ == "__main__":
    main()
