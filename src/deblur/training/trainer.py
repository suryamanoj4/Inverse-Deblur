"""Training loop shared by the learned arms.

Optimiser, schedule, loss, clipping, validation cadence and early stopping all
come from the locked protocol (:class:`deblur.protocol.ProtocolConfig`):
AdamW, linear warmup then cosine decay, L2 loss, gradient-norm clipping,
validation every ``val_interval`` iterations, and early stopping after
``early_stop_patience`` validations without improvement. The checkpoint kept as
``best.pt`` is the one with the highest validation score (``mean_val_psnr``);
``last.pt`` holds full state for resuming.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..protocol.config import ProtocolConfig
from .sampler import Batch, PatchSampler

Validator = Callable[[nn.Module], tuple[float, list[dict]]]
ForwardFn = Callable[[nn.Module, Batch], torch.Tensor]
Criterion = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainSettings:
    iterations: int
    learning_rate: float
    weight_decay: float
    warmup_iterations: int
    gradient_clip: float
    val_interval: int
    early_stop_patience: int
    betas: tuple[float, float] = (0.9, 0.999)
    min_learning_rate: float = 1e-7
    amp: bool = False
    log_interval: int = 100

    @classmethod
    def from_protocol(cls, config: ProtocolConfig, arm: str = "m1", **overrides) -> "TrainSettings":
        if arm not in ("m1", "m3"):
            raise ValueError("arm must be 'm1' or 'm3'")
        values = {
            "iterations": getattr(config, f"{arm}_iterations"),
            "learning_rate": getattr(config, f"{arm}_learning_rate"),
            "weight_decay": config.weight_decay,
            "warmup_iterations": config.warmup_iterations,
            "gradient_clip": config.gradient_clip,
            "val_interval": config.val_interval,
            "early_stop_patience": config.early_stop_patience,
        }
        values.update(overrides)
        return cls(**values)

    def lr_factor(self, iteration: int) -> float:
        """Multiplier on ``learning_rate`` at ``iteration`` (0-based)."""
        if iteration < self.warmup_iterations:
            return (iteration + 1) / self.warmup_iterations
        span = max(1, self.iterations - self.warmup_iterations)
        progress = min(1.0, (iteration - self.warmup_iterations) / span)
        floor = self.min_learning_rate / self.learning_rate
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        sampler: PatchSampler,
        validate: Validator,
        settings: TrainSettings,
        out_dir: str | Path,
        metadata: dict | None = None,
        forward_fn: ForwardFn | None = None,
        criterion: Criterion | None = None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.validate = validate
        self.settings = settings
        self.out_dir = Path(out_dir)
        self.metadata = metadata or {}
        self.device = sampler.device
        self.forward_fn = forward_fn or (lambda model, batch: model(batch.observed))
        self.criterion = criterion or (
            lambda model, output, clean: F.mse_loss(output.float(), clean)
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            betas=settings.betas,
            weight_decay=settings.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, settings.lr_factor)
        self.iteration = 0
        self.best_score = -math.inf
        self.best_iteration = -1
        self.bad_checks = 0

    # -- checkpoints ---------------------------------------------------------
    def _payload(self, full: bool) -> dict:
        payload = {
            "model": self.model.state_dict(),
            "iteration": self.iteration,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "settings": asdict(self.settings),
            "metadata": self.metadata,
        }
        if full:
            payload.update(
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                sampler=self.sampler.state_dict(),
                bad_checks=self.bad_checks,
            )
        return payload

    def save(self, name: str, full: bool) -> Path:
        path = self.out_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(self._payload(full), tmp)
        tmp.replace(path)
        return path

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        current = asdict(self.settings)
        changed = {
            key: (state["settings"].get(key), value)
            for key, value in current.items()
            if state["settings"].get(key) != value
        }
        if changed:
            # a resumed run must be the same run: a new iteration count would reshape the schedule
            raise ValueError(f"settings differ from the checkpoint (saved, current): {changed}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.sampler.load_state_dict(state["sampler"])
        self.iteration = state["iteration"]
        self.best_score = state["best_score"]
        self.best_iteration = state["best_iteration"]
        self.bad_checks = state["bad_checks"]

    def _log(self, event: dict) -> None:
        with open(self.out_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")

    # -- loop ----------------------------------------------------------------
    def _validate(self) -> bool:
        """Run validation; returns True when early stopping should trigger."""
        self.model.eval()
        start = time.time()
        score, rows = self.validate(self.model)
        self.model.train()
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_iteration = self.iteration
            self.bad_checks = 0
            self.save("best.pt", full=False)
            (self.out_dir / "best_val.json").write_text(json.dumps(rows, indent=1))
        else:
            self.bad_checks += 1
        self._log(
            {
                "event": "val",
                "iteration": self.iteration,
                "mean_psnr": score,
                "best": self.best_score,
                "best_iteration": self.best_iteration,
                "bad_checks": self.bad_checks,
                "seconds": round(time.time() - start, 2),
            }
        )
        print(
            f"  val @ {self.iteration}: mean PSNR {score:.3f} dB "
            f"(best {self.best_score:.3f} @ {self.best_iteration}, "
            f"{self.bad_checks}/{self.settings.early_stop_patience} without improvement)"
        )
        return self.bad_checks >= self.settings.early_stop_patience

    def run(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        s = self.settings
        device_type = self.device.type
        self.model.train()
        running, count, tick = 0.0, 0, time.time()
        stopped_early = False
        while self.iteration < s.iterations:
            batch = self.sampler.sample()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=s.amp):
                output = self.forward_fn(self.model, batch)
            loss = self.criterion(self.model, output, batch.clean)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), s.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.iteration += 1
            running += float(loss.detach())
            count += 1

            if self.iteration % s.log_interval == 0 or self.iteration == s.iterations:
                elapsed = time.time() - tick
                event = {
                    "event": "train",
                    "iteration": self.iteration,
                    "loss": running / count,
                    "lr": self.scheduler.get_last_lr()[0],
                    "it_per_s": count / elapsed if elapsed > 0 else None,
                }
                self._log(event)
                print(
                    f"  it {self.iteration}/{s.iterations}  loss {event['loss']:.3e}  "
                    f"lr {event['lr']:.2e}  {event['it_per_s'] or 0:.1f} it/s"
                )
                running, count, tick = 0.0, 0, time.time()

            if self.iteration % s.val_interval == 0 or self.iteration == s.iterations:
                stopped_early = self._validate()
                self.save("last.pt", full=True)
                if stopped_early:
                    break

        summary = {
            "iterations_run": self.iteration,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "stopped_early": stopped_early,
        }
        self._log({"event": "done", **summary})
        return summary
