import yaml
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

class Config:
    def __init__(self, config_path=None):
        if config_path:
            self.load_config(config_path)
        else:
            self.model_params = {
                'input_dim': 21,
                'hidden_dims': [16],
                'output_dim': 1
            }
            
            self.training_params = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 1000,
                'momentum': 0.9,
                'lambda': 0.1,
                'early_stopping': False,
                'patience': 10
            }

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            self.model_params = config["model_parameters"]
            self.training_params = config["training_parameters"]

    def save_config(self, config_path):
        config = {
            "model_parameters": self.model_params,
            "training_parameters": self.training_params
        }
        with open(config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
            print(f"Configuration saved to {config_path}")





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
    ignore_index: Optional[int] = None    # set to e.g. 255 if masks have void label

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