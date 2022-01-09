import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from model import ChessPiecesClassifier
from generate_tiles import get_chessboard_tiles
import numpy as np
import torch
from glob import glob
from torchvision import transforms
import time

NN_MODEL_PATH = "/Users/fevenz/Sriram/Projects/chess-scanner/model/model.pth"
FEN_CHARS = "1RNBQKPrnbqkp"
IMAGE_PATH = "/Users/fevenz/Sriram/Projects/chess-scanner/tests/support/"


def _chessboard_tiles_img_data(chessboard_img_path):
    """Given a file path to a chessboard PNG image, returns a
    size-64 array of 32x32 tiles representing each square of a chessboard
    """
    tiles = get_chessboard_tiles(chessboard_img_path)
    img_data_list = []
    for i in range(64):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)]
        )
        img_data = transform(tiles[i])
        img_data_list.append(img_data)
    return torch.stack(img_data_list)


def compressed_fen(fen):
    """From: 11111q1k/1111r111/111p1pQP/111P1P11/11prn1R1/11111111/111111P1/R11111K1
    To: 5q1k/4r3/3p1pQP/3P1P2/2prn1R1/8/6P1/R5K1
    """
    for length in reversed(range(2, 9)):
        fen = fen.replace(length * "1", str(length))
    return fen


def predict_chessboard(chessboard_img_path, model):
    """Given a file path to a chessboard PNG image,
    Returns a FEN string representation of the chessboard
    """
    img_data_list = _chessboard_tiles_img_data(chessboard_img_path)
    predictions = model(img_data_list).argmax(dim=1)
    fen_predictions = [FEN_CHARS[prediction] for prediction in predictions]

    predicted_fen = compressed_fen(
        "/".join(
            ["".join(r) for r in np.reshape([p[0] for p in fen_predictions], [8, 8])]
        )
    )
    return predicted_fen

def test_chess_scanner():
    support_file_names = glob("{}/*.png".format(IMAGE_PATH))
    start_time = time.time()
    model = ChessPiecesClassifier()
    model.load_state_dict(torch.load(NN_MODEL_PATH))
    scripted_model = model.to_torchscript(method="script", file_path=None)
    end_time = time.time()
    print(f"Time taken to initialize ChessPiecesClassifier: {round(end_time - start_time, 2)}s")
    predictions = list()
    output = ["R2KQB1R/1B1N1P2/PP1PP1PP/2P5/4pp2/2npbn2/ppp1b1pp/1kr1q2r","1nR5/5kpp/r7/q6r/6Q1/4B3/PP3PPb/3R3k","2R1Q2R/PP2KPPP/2P1PN2/3Pn3/pp1p4/2nqp3/5ppp/r1r2k2","4R2R/KBP3P1/1P1B3P/PnPp4/p3p1b1/5n2/1pp3pp/1kr1r3"]
    for chessboard_image_path in support_file_names:
        predictions.append(predict_chessboard(chessboard_image_path, scripted_model))
    assert predictions == output
    return predictions


if __name__ == "__main__":
    print(test_chess_scanner())
