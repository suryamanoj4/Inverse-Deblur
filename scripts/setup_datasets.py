#!/usr/bin/env python3
"""Single entry point for dataset setup and protocol manifests.

Datasets are opt-in, so only what an experiment needs is downloaded:

    bsds500    core, Tier 1 (default)          official Berkeley archive
    gopro      Tier 2, stretch                 manual archive or direct URL
    levin      real kernels (v2 P2-extrap)     manual archive or direct URL
    sun2013    real kernels (v2 P2-extrap)     manual archive or direct URL
    kohler2012 real kernels (v2 P2-extrap)     manual archive or direct URL
    div2k      Phase 2 real-data control       official ETH archive (800 train images, ~3.5 GB)

Examples:
    python scripts/setup_datasets.py --datasets bsds500
    python scripts/setup_datasets.py --datasets bsds500 --write-protocol
    python scripts/setup_datasets.py --datasets gopro --gopro-archive /path/GoPro.zip
    python scripts/setup_datasets.py --datasets levin --levin-archive /path/kernels.zip
    python scripts/setup_datasets.py --datasets div2k
    # Phase 2: freeze protocol v2 (kernels as arrays + SHA-256) in artifacts/protocol/v2
    python scripts/setup_datasets.py --write-protocol --version v2 \
        --real-kernels levin=data/levin/kernels
    python scripts/setup_datasets.py --check

Nothing is downloaded by the test suite. ``--check`` only inspects the local
layout; ``--dry-run`` prints the plan without downloading.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BSDS500_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/"
    "BSR/BSR_bsds500.tgz"
)
GOPRO_PAGE = "https://seungjunnah.github.io/Datasets/gopro.html"
DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DATASETS = ("bsds500", "gopro", "levin", "sun2013", "kohler2012", "div2k")
KERNEL_SETS = ("levin", "sun2013", "kohler2012")
MANUAL_HINTS = {
    "gopro": GOPRO_PAGE,
    "levin": "the Levin et al. (2009) project page",
    "sun2013": "the Sun et al. (ICCP 2013) project page",
    "kohler2012": "the Kohler et al. (ECCV 2012) benchmark page",
}


def _progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        percent = min(100, blocks * block_size * 100 // total_size)
        print(f"\r  {percent:3d}%", end="", flush=True)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, destination, reporthook=_progress)
    print()


def _safe_target(destination: Path, name: str) -> Path:
    target = (destination / name).resolve()
    base = destination.resolve()
    if target != base and base not in target.parents:
        raise RuntimeError(f"unsafe path in archive: {name}")
    return target


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {destination}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                _safe_target(destination, name)
            bundle.extractall(destination)
        return
    with tarfile.open(archive) as bundle:
        members = []
        for member in bundle.getmembers():
            _safe_target(destination, member.name)
            members.append(member)
        bundle.extractall(destination, members=members)


def find_bsds_images(destination: Path) -> Path | None:
    candidates = [
        destination / "BSR" / "BSDS500" / "data" / "images",
        destination / "BSDS500" / "data" / "images",
        destination / "data" / "images",
        destination,
    ]
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ("train", "val", "test")):
            return candidate
    return None


def find_gopro_root(destination: Path) -> Path | None:
    for candidate in (destination / "gopro", destination):
        if all(
            (candidate / split / folder).is_dir()
            for split in ("train", "test")
            for folder in ("blur", "sharp")
        ):
            return candidate
    return None


def kernel_files(directory: Path) -> list[Path]:
    suffixes = {".npy", ".mat", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in suffixes)


def find_div2k(destination: Path) -> Path | None:
    for candidate in (destination / "div2k" / "DIV2K_train_HR", destination / "DIV2K_train_HR"):
        if candidate.is_dir() and any(candidate.glob("*.png")):
            return candidate
    return None


def status(datasets: list[str], destination: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    if "bsds500" in datasets:
        images = find_bsds_images(destination)
        result["bsds500"] = images is not None
        print(f"  bsds500: {'ready at ' + str(images) if images else 'missing'}")
    if "gopro" in datasets:
        gopro = find_gopro_root(destination)
        result["gopro"] = gopro is not None
        print(f"  gopro:    {'ready at ' + str(gopro) if gopro else 'missing'}")
    for name in KERNEL_SETS:
        if name in datasets:
            files = kernel_files(destination / name)
            result[name] = len(files) > 0
            print(f"  {name}: {len(files)} kernel files" if files else f"  {name}: missing")
    if "div2k" in datasets:
        div2k = find_div2k(destination)
        result["div2k"] = div2k is not None
        count = len(list(div2k.glob("*.png"))) if div2k else 0
        print(f"  div2k:    {count} images at {div2k}" if div2k else "  div2k:    missing")
    return result


def setup_bsds500(destination: Path, force: bool, keep_archive: bool, dry_run: bool) -> None:
    if find_bsds_images(destination) is not None and not force:
        print("  bsds500 already present, skipping")
        return
    archive = destination / "BSR_bsds500.tgz"
    if dry_run:
        print(f"  would download {BSDS500_URL} -> {archive}")
        return
    download(BSDS500_URL, archive)
    extract(archive, destination)
    if not keep_archive:
        archive.unlink(missing_ok=True)
    if find_bsds_images(destination) is None:
        raise RuntimeError(
            "BSDS500 archive unpacked, but train/val/test were not found; "
            "expected BSR/BSDS500/data/images"
        )


def setup_div2k(destination: Path, force: bool, keep_archive: bool, dry_run: bool) -> None:
    if find_div2k(destination) is not None and not force:
        print("  div2k already present, skipping")
        return
    archive = destination / "DIV2K_train_HR.zip"
    if dry_run:
        print(f"  would download {DIV2K_URL} -> {archive}")
        return
    download(DIV2K_URL, archive)
    extract(archive, destination / "div2k")
    if not keep_archive:
        archive.unlink(missing_ok=True)
    if find_div2k(destination) is None:
        raise RuntimeError("DIV2K archive unpacked, but DIV2K_train_HR/*.png was not found")


def setup_manual(
    name: str,
    destination: Path,
    archive: Path | None,
    url: str | None,
    force: bool,
    keep_archive: bool,
    dry_run: bool,
    verify,
) -> None:
    if verify(destination) is not None and not force:
        print(f"  {name} already present, skipping")
        return
    if archive is None and url is None:
        hint = MANUAL_HINTS.get(name, "the dataset's project page")
        raise SystemExit(
            f"{name}: pass --{name}-archive or --{name}-url "
            f"(see {hint}); nothing was downloaded"
        )
    target_archive = archive
    if target_archive is None:
        target_archive = destination / f"{name}_archive"
        if dry_run:
            print(f"  would download {url} -> {target_archive}")
            return
        download(str(url), target_archive)
    if dry_run:
        print(f"  would extract {target_archive} -> {destination / name}")
        return
    extract(Path(target_archive), destination / name)
    if archive is None and not keep_archive:
        Path(target_archive).unlink(missing_ok=True)
    if verify(destination) is None:
        raise RuntimeError(f"{name}: archive unpacked but the expected layout was not found")


def write_protocol(image_root: Path, destination: Path, args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from deblur.protocol import (
        ProtocolConfig,
        check_disjoint,
        manifest,
        standard_splits,
        validation_subset,
    )
    import hashlib

    import numpy as np

    splits = standard_splits(image_root)
    check_disjoint(splits)
    counts = {split: len(paths) for split, paths in splits.items()}
    levin_dir = destination / "levin"
    real_kernel_dirs = tuple(tuple(item.split("=", 1)) for item in args.real_kernels)
    config = ProtocolConfig(
        data_root=str(image_root),
        protocol_version=args.version,
        real_kernel_dirs=real_kernel_dirs,
        patch_size=args.patch_size,
        levin_dir=str(levin_dir) if kernel_files(levin_dir) and args.version == "v1" else None,
        seed=args.seed,
        eval_crop=args.eval_crop,
        val_subset_size=args.val_subset_size,
        val_subset_seed=args.val_subset_seed,
    )
    val_subset = validation_subset(
        splits["val"], size=config.val_subset_size, seed=config.val_subset_seed
    )
    val_entries = [
        path.resolve().relative_to(image_root.resolve()).as_posix() for path in val_subset
    ]
    psf_sets = config.psf_sets()
    psf_entries = {
        name: [
            {
                "name": psf.name,
                "kind": psf.kind,
                "shape": list(psf.kernel.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(psf.kernel).tobytes()).hexdigest(),
            }
            for psf in kernels
        ]
        for name, kernels in psf_sets.items()
    }

    out_dir = Path(args.protocol_out)
    if args.version != "v1" and args.protocol_out == "artifacts/protocol":
        out_dir = out_dir / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    # the kernels themselves, so a version can be checked or reloaded exactly
    np.savez_compressed(
        out_dir / "psfs.npz",
        **{f"{set_name}/{psf.name}": psf.kernel for set_name, kernels in psf_sets.items() for psf in kernels},
    )
    split_entries = manifest(splits, root=image_root)
    combined = {
        "image_root": str(image_root),
        "protocol": config.manifest(),
        "splits": split_entries,
        "validation_subset": val_entries,
        "psf_sets": psf_entries,
    }
    (out_dir / "splits.json").write_text(json.dumps(split_entries, indent=2))
    (out_dir / "protocol.json").write_text(json.dumps(config.manifest(), indent=2))
    (out_dir / "val_subset.json").write_text(json.dumps(val_entries, indent=2))
    (out_dir / "psfs.json").write_text(json.dumps(psf_entries, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(combined, indent=2))

    print(f"protocol {args.version} written (kernel sets sha256 {config.manifest()['psf_sets_sha256'][:16]}...):")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {counts[split]:4d} images")
    print(f"  validation subset: {len(val_entries)} images (seed {config.val_subset_seed})")
    for name, entries in psf_entries.items():
        print(f"  {name}: {len(entries)} kernels")
    print(f"  noise levels: {config.noise_levels_over_255} / 255")
    print(f"  M1 regime: {config.m1_regime}; checkpoint rule: {config.checkpoint_rule}")
    print(f"  output: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", default="bsds500",
                        help="comma-separated selection from: " + ", ".join(DATASETS))
    parser.add_argument("--dest", default="data", help="dataset root (default: data)")
    parser.add_argument("--check", action="store_true", help="only inspect the local layout")
    parser.add_argument("--dry-run", action="store_true", help="print actions without downloading")
    parser.add_argument("--force", action="store_true", help="re-download/re-extract even if present")
    parser.add_argument("--keep-archive", action="store_true", help="keep downloaded archives")
    parser.add_argument("--gopro-archive", default=None, help="local GoPro archive (zip)")
    parser.add_argument("--gopro-url", default=None, help="direct URL for a GoPro archive")
    parser.add_argument("--levin-archive", default=None, help="local Levin kernel archive")
    parser.add_argument("--levin-url", default=None, help="direct URL for a Levin kernel archive")
    for name in ("sun2013", "kohler2012"):
        parser.add_argument(f"--{name}-archive", default=None, help=f"local {name} kernel archive")
        parser.add_argument(f"--{name}-url", default=None, help=f"direct URL for a {name} kernel archive")
    parser.add_argument("--write-protocol", action="store_true",
                        help="write protocol manifests after checking BSDS500")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"],
                        help="protocol version to write (v2 goes to <protocol-out>/v2)")
    parser.add_argument("--real-kernels", action="append", default=[], metavar="PREFIX=DIR",
                        help="v2: directory of real recorded kernel files to add to P2_extrap (repeatable)")
    parser.add_argument("--protocol-out", default="artifacts/protocol")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--eval-crop", type=int, default=256)
    parser.add_argument("--val-subset-size", type=int, default=10)
    parser.add_argument("--val-subset-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")
    destination = Path(args.dest)

    print(f"dataset root: {destination}")
    if args.check:
        ready = status(datasets, destination)
        return 0 if all(ready.values()) else 1

    if "bsds500" in datasets:
        setup_bsds500(destination, args.force, args.keep_archive, args.dry_run)
    if "gopro" in datasets:
        setup_manual(
            "gopro",
            destination,
            Path(args.gopro_archive) if args.gopro_archive else None,
            args.gopro_url,
            args.force,
            args.keep_archive,
            args.dry_run,
            find_gopro_root,
        )
    for name in KERNEL_SETS:
        if name in datasets:
            archive = getattr(args, f"{name}_archive")
            setup_manual(
                name,
                destination,
                Path(archive) if archive else None,
                getattr(args, f"{name}_url"),
                args.force,
                args.keep_archive,
                args.dry_run,
                lambda root, name=name: (root / name) if kernel_files(root / name) else None,
            )
    if "div2k" in datasets:
        setup_div2k(destination, args.force, args.keep_archive, args.dry_run)

    print("status:")
    status(datasets, destination)

    if args.write_protocol:
        image_root = find_bsds_images(destination)
        if image_root is None:
            raise SystemExit("cannot write the protocol: BSDS500 is not set up")
        write_protocol(image_root, destination, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
