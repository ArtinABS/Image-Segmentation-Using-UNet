import numpy as np
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

class MSE:

    def __init__(self):
        self.input = None
        self.target = None
        self.output = None

    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.forward(input, target)

    def forward(self, input: np.ndarray, target: np.ndarray, need_to_onehot : bool = True) -> float:
        self.input = input
        if need_to_onehot:
            self.target = np.eye(input.shape[1])[target.astype(int)]
        else:
            self.target = target
        self.output = np.mean((self.input - self.target) ** 2)
        return self.output

    def backward(self) -> np.ndarray:
        return 2 * (self.input - self.target) / self.input.size
    

class CrossEntropy:
    def __init__(self):
        self.input = None
        self.target = None
        self.output = None

    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.forward(input, target)

    def forward(self, input: np.ndarray, target: np.ndarray, need_to_onehot : bool = True) -> float:
        self.input = input
        if need_to_onehot:
            self.target = np.eye(10)[target]
        else:
            self.target = target

        eps = 1e-10
        self.input = np.clip(self.input, eps, 1 - eps)

        self.output = -np.sum(self.target * np.log(self.input)) / input.shape[0]
        return self.output  
    
    def backward(self) -> np.ndarray:
        return (self.input - self.target) / self.input.shape[0]
    


class MetricTracker:
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_examples = 0
        self.cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_val: float):
        # logits: (N,C,H,W); targets: (N,H,W)
        self.total_loss += float(loss_val) * targets.numel()
        self.total_examples += targets.numel()

        preds = logits.argmax(dim=1)  # (N,H,W)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]
        # Confusion matrix via bincount
        k = (targets * self.num_classes + preds).view(-1)
        binc = torch.bincount(k, minlength=self.num_classes ** 2)
        self.cm += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, Any]:
        eps = 1e-7
        loss = self.total_loss / max(self.total_examples, 1)
        tp = torch.diag(self.cm).float()
        pos = self.cm.sum(1).float()
        pred_pos = self.cm.sum(0).float()
        union = pos + pred_pos - tp

        iou = tp / (union + eps)
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            valid = torch.ones(self.num_classes, dtype=torch.bool, device=self.cm.device)
            valid[self.ignore_index] = False
            miou = iou[valid].mean().item()
            pixel_acc = tp[valid].sum().item() / (pos[valid].sum().item() + eps)
        else:
            miou = iou.mean().item()
            pixel_acc = tp.sum().item() / (pos.sum().item() + eps)
        return {
            "loss": loss,
            "miou": miou,
            "pixel_acc": pixel_acc,
            "per_class_iou": iou.detach().cpu().tolist(),
        }


