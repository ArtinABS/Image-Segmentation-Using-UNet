import yaml
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class TrainConfig:
    epochs: int = 40
    log_interval: int = 50                # steps
    grad_clip_norm: Optional[float] = 1.0 # None to disable
    amp: bool = True                      # mixed precision
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 1                   # save every N epochs
    best_metric: str = "miou"             # or "val_loss"
    scheduler_step_on: str = "epoch"      # or "val"
    early_stop_patience: Optional[int] = None
    num_classes: int = 9
    batch_size: int = 100

    # --- Save config to JSON file ---
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"[TrainConfig] Saved config to {path}")

    # --- Load config from JSON file ---
    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        print(f"[TrainConfig] Loaded config from {path}")
        return cls(**data)