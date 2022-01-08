from model import ChessPiecesClassifier
from generate_tiles import USE_GRAYSCALE, get_chessboard_tiles
import numpy as np
from glob import glob
import torch
import pytorch_lightning as pl
from torchvision import transforms

NN_MODEL_PATH = '/Users/fevenz/Sriram/Projects/chess-scanner/.checkpoints/model.pth'
FEN_CHARS = '1RNBQKPrnbqkp'
IMAGE_PATH = '/Users/fevenz/Sriram/Projects/chess-scanner/test_images/classic.png'
USE_GRAYSCALE = True

def _chessboard_tiles_img_data(chessboard_img_path, options={}):
    """ Given a file path to a chessboard PNG image, returns a
        size-64 array of 32x32 tiles representing each square of a chessboard
    """
    n_channels = 1 if USE_GRAYSCALE else 3
    tiles = get_chessboard_tiles(chessboard_img_path, use_grayscale=USE_GRAYSCALE)
    img_data_list = []
    for i in range(64):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
        img_data = transform(tiles[i])
        #img_data.unsqueeze_(0)
        #tiles[i].show()
        #print(img_data.shape)
        img_data_list.append(img_data)
    return torch.stack(img_data_list)

def compressed_fen(fen):
    """ From: 11111q1k/1111r111/111p1pQP/111P1P11/11prn1R1/11111111/111111P1/R11111K1
        To: 5q1k/4r3/3p1pQP/3P1P2/2prn1R1/8/6P1/R5K1
    """
    for length in reversed(range(2,9)):
        fen = fen.replace(length * '1', str(length))
    return fen


def predict_chessboard(chessboard_img_path):
    """ Given a file path to a chessboard PNG image,
        Returns a FEN string representation of the chessboard
    """
    img_data_list = _chessboard_tiles_img_data(chessboard_img_path)
    print(img_data_list.shape)
    predictions = model(img_data_list).argmax(dim=1)
    fen_predictions = [FEN_CHARS[prediction] for prediction in predictions]
    print(predictions)

    predicted_fen = compressed_fen(
        '/'.join(
            [''.join(r) for r in np.reshape([p[0] for p in fen_predictions], [8, 8])]
        )
    )
    return predicted_fen


if __name__ == '__main__':
    model = ChessPiecesClassifier()
    model.load_state_dict(torch.load(NN_MODEL_PATH))
    print(predict_chessboard(IMAGE_PATH))
