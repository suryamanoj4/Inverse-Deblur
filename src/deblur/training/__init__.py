"""Training machinery for the learned arms (M1, M3). Requires PyTorch."""
from ..evaluation import identity_restorer, merge_baseline, summarize
from .evaluation import evaluate_grid, mean_psnr, model_restorer, physics_restorer
from .sampler import Batch, PatchSampler
from .trainer import Criterion, ForwardFn, TrainSettings, Trainer

__all__ = [
    "Batch",
    "Criterion",
    "ForwardFn",
    "PatchSampler",
    "TrainSettings",
    "Trainer",
    "evaluate_grid",
    "identity_restorer",
    "mean_psnr",
    "merge_baseline",
    "model_restorer",
    "physics_restorer",
    "summarize",
]
