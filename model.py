import pytorch_lightning as pl
import torch
from torch import optim 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from dataset import ChessPiecesDataset
from pytorch_lightning.callbacks import ModelCheckpoint

CHESSBOARDS_DIR = './dataset/chessboards'
TILES_DIR = './dataset/tiles'
USE_GRAYSCALE = True
FEN_CHARS = '1RNBQKPrnbqkp'
    

class ChessPiecesClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=13)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.chess_pieces_dataset = ChessPiecesDataset()
        train_split = int(0.7 * len(self.chess_pieces_dataset))
        self.val_losses = []
        test_split = len(self.chess_pieces_dataset) - train_split
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.chess_pieces_dataset, [train_split, test_split])

    
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
        x,y = batch
        output = self(x)
        loss = self.criterion(output, y)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        output = self(x)
        #print(output,y)
        val_loss = self.criterion(output, y)
        self.val_losses.append(val_loss.item())
        return {'val_loss': val_loss}
    
    def on_validation_epoch_end(self):
        print("average val loss", sum(self.val_losses) / len(self.val_losses))
        self.val_losses = []
    
    def train_dataloader(self):
        class_weights = []
        sample_weights = []
        #self.train_dataset = self.chess_pieces_dataset
        count = [0]*13
        for idx, (data, label) in enumerate(self.train_dataset):
            count[label] += 1
        
        for i in range(13):
            class_weights.append(1/count[i])
        
        for idx, (data, label) in enumerate(self.train_dataset):
            class_weight = class_weights[label]
            sample_weights.append(class_weight)

        chess_sampler = WeightedRandomSampler(sample_weights, num_samples=len(self.chess_pieces_dataset))
        train_dataloader = DataLoader(self.train_dataset, batch_size=64, sampler = chess_sampler)
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(self.test_dataset, batch_size=64)
        return val_dataloader




if __name__ == '__main__':
    model = ChessPiecesClassifier()
    #print(model(x).shape)
    #print(model(x))
    #checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model)
