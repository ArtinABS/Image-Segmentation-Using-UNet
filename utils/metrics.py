# utils/metrics.py
from typing import Dict, Any
import torch

class MetricTracker:
    def __init__(self, num_classes: int, device: torch.device | None = None):
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_examples = 0
        self.cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_val: float):
        # logits: (N,C,H,W)
        self.total_loss += float(loss_val) * logits.shape[0] * logits.shape[2] * logits.shape[3]

        preds = torch.argmax(logits, dim=1)  # (N,H,W)

        # --- sanitize targets to (N,H,W) of class ids ---
        if targets.ndim == 4:
            # (N,1,H,W) -> squeeze
            if targets.size(1) == 1:
                targets = targets[:, 0, ...]
            # (N,C,H,W) one-hot/prob -> argmax to ids
            elif targets.size(1) == self.num_classes:
                targets = torch.argmax(targets, dim=1)
            else:
                raise ValueError(f"Unexpected targets shape {tuple(targets.shape)}. "
                                f"Expected (N,1,H,W) or (N,{self.num_classes},H,W) or (N,H,W).")
        elif targets.ndim != 3:
            raise ValueError(f"Unexpected targets dim {targets.ndim}. Expected 3 or 4.")

        targets = targets.to(torch.long)          # (N,H,W)

        # flatten
        t = targets.reshape(-1)
        p = preds.reshape(-1)

        k = t * self.num_classes + p
        binc = torch.bincount(k, minlength=self.num_classes ** 2)
        self.cm += binc.view(self.num_classes, self.num_classes)


    def compute(self) -> Dict[str, Any]:
        eps = 1e-7
        loss = self.total_loss / max(self.total_examples, 1)

        tp = torch.diag(self.cm).float()
        pos = self.cm.sum(1).float()
        pred_pos = self.cm.sum(0).float()
        union = pos + pred_pos - tp

        iou = tp / (union + eps)
        miou = iou.mean().item()
        pixel_acc = tp.sum().item() / (pos.sum().item() + eps)

        return {
            "loss": loss,
            "miou": miou,
            "pixel_acc": pixel_acc,
            "per_class_iou": iou.detach().cpu().tolist(),
        }
