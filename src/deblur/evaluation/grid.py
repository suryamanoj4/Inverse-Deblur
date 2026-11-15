"""Grid evaluation, baselines and aggregation (NumPy only)."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from ..metrics import data_consistency, psnr, ssim
from ..protocol.observations import observe
from ..psf.library import PSF

Restorer = Callable[[np.ndarray, PSF, float], np.ndarray]


def evaluate_grid(
    restore_fn: Restorer,
    images: list[tuple[str, np.ndarray]],
    psfs: list[PSF],
    noise_levels: tuple[float, ...] | list[float],
    *,
    seed: int,
    full_metrics: bool = False,
    lpips_metric=None,
    quantize: bool = False,
) -> list[dict]:
    """Score ``restore_fn(y, psf, sigma)`` on every (image, PSF, sigma) cell.

    ``restore_fn`` receives the PSF and sigma so the same harness serves the
    physics-aware arms; M1 restorers ignore them. ``quantize`` rounds every
    observation to 8 bits (protocol v2).

    A restorer may also expose ``batch(ys, psfs, sigma)`` (and optionally
    ``batch_size``) to restore several PSFs of one image and noise level in one
    call; the cells and the row order are the same either way.
    """
    batch_fn = getattr(restore_fn, "batch", None)
    rows = []
    for image_id, x in images:
        estimates = {}
        if batch_fn is not None:
            # one call per (image, sigma) and chunk of PSFs; same cells, same order
            chunk = max(1, int(getattr(restore_fn, "batch_size", len(psfs))))
            for sigma in noise_levels:
                for start in range(0, len(psfs), chunk):
                    group = psfs[start : start + chunk]
                    ys = [
                        observe(x, psf.kernel, sigma, seed=seed, image_id=image_id,
                                psf_name=psf.name, quantize=quantize)
                        for psf in group
                    ]
                    for psf, y, x_hat in zip(group, ys, batch_fn(ys, group, sigma)):
                        estimates[(psf.name, sigma)] = (y, x_hat)
        for psf in psfs:
            for sigma in noise_levels:
                if batch_fn is not None:
                    y, x_hat = estimates[(psf.name, sigma)]
                else:
                    y = observe(
                        x, psf.kernel, sigma, seed=seed, image_id=image_id, psf_name=psf.name,
                        quantize=quantize,
                    )
                    x_hat = restore_fn(y, psf, sigma)
                row = {
                    "image": image_id,
                    "psf": psf.name,
                    "kind": psf.kind,
                    "sigma_255": round(sigma * 255),
                    "psnr": psnr(x, x_hat),
                }
                if full_metrics:
                    row["ssim"] = ssim(x, x_hat)
                    row["residual_rmse"] = data_consistency(x_hat, y, psf.kernel).rmse
                if lpips_metric is not None:
                    row["lpips"] = lpips_metric(x, x_hat)
                rows.append(row)
    return rows


def mean_psnr(rows: list[dict]) -> float:
    """Checkpoint score: mean over cells of the per-cell mean PSNR."""
    cells: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        cells[(row["psf"], row["sigma_255"])].append(row["psnr"])
    if not cells:
        raise ValueError("no rows to score")
    return float(np.mean([np.mean(values) for values in cells.values()]))


def identity_restorer(y: np.ndarray, psf: PSF, sigma: float) -> np.ndarray:
    """No restoration; used for the per-cell blurry baseline."""
    del psf, sigma
    return np.asarray(y, dtype=np.float64)


def merge_baseline(rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    """Attach ``baseline_psnr`` and ``gain`` to each row from identity rows."""
    key = lambda row: (row["image"], row["psf"], row["sigma_255"])
    baseline = {key(row): row["psnr"] for row in baseline_rows}
    for row in rows:
        value = baseline.get(key(row))
        row["baseline_psnr"] = value
        row["gain"] = None if value is None else row["psnr"] - value
    return rows


def summarize(
    rows: list[dict], keys: tuple[str, ...] = ("psf_set", "kind", "sigma_255")
) -> list[dict]:
    """Aggregate rows by ``keys``, averaging every metric that is present."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)

    def sort_key(group_key: tuple):
        # numeric keys (sigma) must sort numerically, not lexicographically
        return tuple(
            (0, float(value), "") if isinstance(value, (int, float)) else (1, 0.0, str(value))
            for value in group_key
        )

    summary = []
    for group_key, items in sorted(groups.items(), key=lambda item: sort_key(item[0])):
        entry = dict(zip(keys, group_key))
        entry["n"] = len(items)
        for metric in ("psnr", "ssim", "baseline_psnr", "gain", "residual_rmse", "lpips"):
            values = [row[metric] for row in items if row.get(metric) is not None]
            entry[metric] = float(np.mean(values)) if values else None
        summary.append(entry)
    return summary
