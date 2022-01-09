from model import ChessPiecesClassifier
from generate_tiles import get_chessboard_tiles
import numpy as np
import torch
from torchvision import transforms

NN_MODEL_PATH = "artifacts/model.pth"
FEN_CHARS = "1RNBQKPrnbqkp"
IMAGE_PATH = "tests/support/chessbase.png"


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


if __name__ == "__main__":
    model = ChessPiecesClassifier()
    model.load_state_dict(torch.load(NN_MODEL_PATH))
    scripted_model = model.to_torchscript(method="script", file_path=None)
    print(predict_chessboard(IMAGE_PATH, scripted_model))
