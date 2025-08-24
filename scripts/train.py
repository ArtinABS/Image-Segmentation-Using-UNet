import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import MetricTracker
from config.config import TrainConfig

from tqdm.auto import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[TrainConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or TrainConfig()

        self.model.to(self.device)
        if isinstance(self.criterion, nn.Module):
            self.criterion.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp and self.device.type == "cuda")
        os.makedirs(self.config.ckpt_dir, exist_ok=True)

        self.best_score = -math.inf if self.config.best_metric == "miou" else math.inf
        self.best_path = os.path.join(self.config.ckpt_dir, "best.pt")

    def _step_scheduler(self, val_metric: Optional[float] = None):
        if self.scheduler is None:
            return
        if hasattr(self.scheduler, "step"):
            if self.config.scheduler_step_on == "val" and val_metric is not None:
                try:
                    self.scheduler.step(val_metric)
                except TypeError:
                    self.scheduler.step()
            else:
                self.scheduler.step()

    def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        mt = MetricTracker(self.config.num_classes, self.config.ignore_index, self.device)

        for step, (images, targets) in enumerate(self.train_loader, 1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()

            if self.config.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            mt.update(logits.detach(), targets, float(loss.detach().item()))

            if step % self.config.log_interval == 0 or step == len(self.train_loader):
                stats = mt.compute()
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} | Step {step}/{len(self.train_loader)} | "
                      f"loss {stats['loss']:.4f} | mIoU {stats['miou']:.4f} | pixAcc {stats['pixel_acc']:.4f} | lr {lr:.3e}")

        return mt.compute()

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        if self.val_loader is None:
            return {"loss": float("nan"), "miou": float("nan"), "pixel_acc": float("nan")}
        self.model.eval()
        mt = MetricTracker(self.config.num_classes, self.config.ignore_index, self.device)

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            mt.update(logits, targets, float(loss.item()))

        stats = mt.compute()
        print(f"Val | loss {stats['loss']:.4f} | mIoU {stats['miou']:.4f} | pixAcc {stats['pixel_acc']:.4f}")
        return stats

    def _is_improved(self, metric_value: float) -> bool:
        if self.config.best_metric == "miou":
            return metric_value > self.best_score
        else:  # val_loss
            return metric_value < self.best_score

    def _save_checkpoint(self, path: str, epoch: int, extra: Optional[Dict[str, Any]] = None):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": asdict(self.config),
        }
        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        try:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            pass
        print(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch','?')})")
        return int(ckpt.get("epoch", 0))

    def fit(self) -> Dict[str, Any]:
        best_epoch = -1
        no_improve = 0
        history = {"train": [], "val": []}

        pbar = tqdm(total=self.config.epochs, desc="Epochs", leave=True)

        for epoch in range(1, self.config.epochs + 1):
            train_stats = self.train_one_epoch(epoch)
            history["train"].append(train_stats)

            val_stats = self.validate()
            history["val"].append(val_stats)

            pbar.set_postfix({
                "loss": f"{val_stats['loss']:.4f}",
                "mIoU": f"{val_stats['miou']:.4f}",
                "pixAcc": f"{val_stats['pixel_acc']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.3e}",
            })
            pbar.update(1)

            # Scheduler step
            if self.config.scheduler_step_on == "val":
                key = "miou" if self.config.best_metric == "miou" else "loss"
                self._step_scheduler(val_stats[key])
            else:
                self._step_scheduler()

            # Checkpointing
            score = val_stats["miou"] if self.config.best_metric == "miou" else val_stats["loss"]
            improved = self._is_improved(score)
            if improved:
                self.best_score = score
                best_epoch = epoch
                self._save_checkpoint(self.best_path, epoch, extra={"best_score": self.best_score})
                no_improve = 0
            else:
                no_improve += 1

            if self.config.ckpt_every and (epoch % self.config.ckpt_every == 0):
                last_path = os.path.join(self.config.ckpt_dir, f"epoch_{epoch}.pt")
                self._save_checkpoint(last_path, epoch, extra={"best_score": self.best_score})

            if self.config.early_stop_patience is not None and no_improve >= self.config.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {no_improve} epochs). Best at epoch {best_epoch}.")
                break

        print(f"Training done. Best {self.config.best_metric}: {self.best_score:.4f} at epoch {best_epoch}.")
        return {"history": history, "best": {"epoch": best_epoch, self.config.best_metric: self.best_score}}


