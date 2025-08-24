import numpy as np
import yaml
import logging
import logging
import yaml
from datetime import datetime
from models.model import MLP, linear, relu, softmax
from utils.metrics import MSE, CrossEntropy
from data.data_loader import split_data, preprocess_data, data_loader_MNIST, data_loader_SVHN, preprocess_data_SVHN, data_loader_LE, preprocess_data_split_LE
from scripts.train import train, train_LE
from utils.visualization import plot, plot_predictions, plot_confusion_matrix, plot_loss
from config.config import Config
from scripts.evaluate import evaluate_loss, evaluate_accuracy

np.random.seed(42)



log_data = {
   "logs": []
}

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    log_data["logs"].append(log_entry)
    logging.info(message)

log_file = "D:/Deep/Assignment 1/config/logging.yaml"

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