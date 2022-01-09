import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
import wandb
from dataset import ChessPiecesDataset


CHESSBOARDS_DIR = "./dataset/chessboards"
TILES_DIR = "./dataset/tiles"
FEN_CHARS = "1RNBQKPrnbqkp"

wandb.init(project="chess_scanner", entity="sriram13")


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(val_imgs, preds, self.val_labels)
                ],
                "global_step": trainer.global_step,
            }
        )


class ImageTestingLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=64):
        super().__init__()
        self.val_imgs = val_samples
        self.val_imgs = self.val_imgs[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}")
                    for x, pred in zip(val_imgs, preds)
                ],
                "global_step": trainer.global_step,
            }
        )


class ChessPiecesClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-2):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=13)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.chess_pieces_dataset = ChessPiecesDataset()
        train_split = int(0.8 * len(self.chess_pieces_dataset))
        self.val_losses = []
        test_split = len(self.chess_pieces_dataset) - train_split
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.chess_pieces_dataset, [train_split, test_split]
        )

        self.save_hyperparameters()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = self.fc1(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log("train/loss", loss, on_epoch=True)
        self.train_acc(output, y)
        self.log("train/acc", self.train_acc, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        val_loss = self.criterion(output, y)
        self.log("train/val_loss", val_loss, on_epoch=True)
        self.val_acc(output, y)
        self.log("train/val_acc", self.val_acc, on_epoch=True)
        self.val_losses.append(val_loss.item())
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        print("average val loss", sum(self.val_losses) / len(self.val_losses))
        self.val_losses = []

    def train_dataloader(self):
        class_weights = []
        sample_weights = []
        count = [0] * 13
        for idx, (data, label) in enumerate(self.train_dataset):
            count[label] += 1

        for i in range(13):
            class_weights.append(1 / count[i])

        for idx, (data, label) in enumerate(self.train_dataset):
            class_weight = class_weights[label]
            sample_weights.append(class_weight)

        chess_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(self.train_dataset)
        )
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=64, sampler=chess_sampler
        )
        return train_dataloader

    def val_dataloader(self):
        class_weights = []
        sample_weights = []
        count = [0] * 13
        for idx, (data, label) in enumerate(self.test_dataset):
            count[label] += 1

        for i in range(13):
            class_weights.append(1 / count[i])

        for idx, (data, label) in enumerate(self.test_dataset):
            class_weight = class_weights[label]
            sample_weights.append(class_weight)

        chess_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(self.test_dataset)
        )
        val_dataloader = DataLoader(self.test_dataset, batch_size=64)
        return val_dataloader


if __name__ == "__main__":
    model = ChessPiecesClassifier()
    samples = next(iter(model.val_dataloader()))
    wandb_logger = WandbLogger(project="chess_scanner")
    trainer = pl.Trainer(
        logger=wandb_logger, max_epochs=10, callbacks=[ImagePredictionLogger(samples)]
    )
    trainer.fit(model)
    torch.save(model.state_dict(), "model.pth")
