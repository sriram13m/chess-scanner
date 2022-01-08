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



if __name__ == '__main__':
    from torch.utils.data.sampler import WeightedRandomSampler
    train_data = ChessPiecesDataset()
    train_split = int(0.82 * len(train_data))
    test_split = len(train_data) - train_split
    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_split, test_split])
    class_weights = []
    sample_weights = []
    #self.train_dataset = self.chess_pieces_dataset
    count = [0]*13
    for idx, (data, label) in enumerate(train_dataset):
        count[label] += 1

    for i in range(13):
        class_weights.append(1/count[i])

    for idx, (data, label) in enumerate(train_dataset):
        class_weight = class_weights[label]
        sample_weights.append(class_weight)

    chess_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=64, sampler = chess_sampler)

    for data, labels in train_dataloader:
        print(labels)
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