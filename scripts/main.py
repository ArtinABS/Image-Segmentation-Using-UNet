from models.model import UNet
from data.data_loader import get_dataloader
from scripts.train import Trainer
from config.config import TrainConfig
import torch.nn as nn
import torch

def main():
    cfg = TrainConfig(epochs=40, num_classes=9, ignore_index=255)

    train_loader, val_loader = get_dataloader(
        root="/path/to/EasyPortrait",
        batch_size=8,
        img_size=256,
        num_workers=4,
        ignore_index=cfg.ignore_index
    )

    model = UNet(in_channels=3, num_classes=cfg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=cfg,
    )
    trainer.fit()

if __name__ == "__main__":
    main()