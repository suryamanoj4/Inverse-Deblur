"""Grid evaluation for the learned arms.

Observations come from :func:`deblur.protocol.observe`, so every arm sees the
same noise draw for the same (image, PSF, sigma) cell. Validation uses PSNR
only (it runs every ``val_interval`` iterations); test evaluation can add SSIM,
the data-consistency residual and LPIPS.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch

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
) -> list[dict]:
    """Score ``restore_fn(y, psf, sigma)`` on every (image, PSF, sigma) cell.

    ``restore_fn`` receives the PSF and sigma so the same harness serves the
    physics-aware arms; M1 restorers must ignore them.
    """
    rows = []
    for image_id, x in images:
        for psf in psfs:
            for sigma in noise_levels:
                y = observe(x, psf.kernel, sigma, seed=seed, image_id=image_id, psf_name=psf.name)
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
    """Checkpoint score: mean over cells of the per-cell mean PSNR.

    Averaging per cell first keeps every (PSF, sigma) cell at equal weight,
    which is what the locked ``mean_val_psnr`` rule specifies.
    """
    cells: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        cells[(row["psf"], row["sigma_255"])].append(row["psnr"])
    if not cells:
        raise ValueError("no rows to score")
    return float(np.mean([np.mean(values) for values in cells.values()]))


def model_restorer(
    model: torch.nn.Module,
    device: str | torch.device,
    amp: bool = False,
) -> Restorer:
    """Wrap an M1 network as a restorer; it ignores the PSF and sigma it is given."""
    from ..methods.direct.common import restore

    def restore_fn(y: np.ndarray, psf: PSF, sigma: float) -> np.ndarray:
        return restore(model, y, device=device, amp=amp)

    return restore_fn
