import pytorch_lightning as pl
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import cv2

CHESSBOARDS_DIR = './dataset/chessboards'
TILES_DIR = './dataset/tiles'
USE_GRAYSCALE = True
FEN_CHARS = '1RNBQKPrnbqkp'
    

class ChessPiecesDataset(Dataset):
    def __init__(self):
        self.chessboard_img_paths = np.array(glob('{}/*/*/*.png'.format(TILES_DIR)))
        self.transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
                              
    def __len__(self):
        return len(self.chessboard_img_paths)
    
    def __getitem__(self, idx):
        path =  self.chessboard_img_paths[idx]
        piece_type = path[-5]
        assert piece_type in FEN_CHARS
        image = cv2.imread(path)
        gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = FEN_CHARS.index(piece_type)
        return self.transform(gray_scaled_image), label





"""
Test the dataset

train_data = ChessPiecesDataset()
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

"""