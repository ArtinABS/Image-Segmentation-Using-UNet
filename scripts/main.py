from models.model import UNet
from data.data_loader import get_dataloader
from scripts.train import Trainer
from config.config import TrainConfig
import torch.nn as nn
import torch

def main():
    cfg = TrainConfig(epochs=40, num_classes=9, batch_size=32)

    train_loader, val_loader, test_loader = get_dataloader(
        root="C:/Users/pouya/Desktop/resized",
        batch_size=cfg.batch_size,
        input_size=256,
    )

    model = UNet(in_channels=3, num_classes=cfg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=None,
        config=cfg,
    )
    trainer.fit()


if __name__ == "__main__":
    main()