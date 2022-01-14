from model import ChessPiecesClassifier
import numpy as np
import torch
from torchvision import transforms
import PIL.Image

NN_MODEL_PATH = "artifacts/model.pth"
FEN_CHARS = "1RNBQKPrnbqkp"
IMAGE_PATH = "tests/support/chessbase.png"


class ChessScanner:
    """Class to recogonize position of a chessboard from image"""

    def __init__(self):
        self.model = ChessPiecesClassifier()
        self.model.load_state_dict(torch.load(NN_MODEL_PATH))
        self.scripted_model = self.model.to_torchscript(method="script", file_path=None)
    
    def _get_image(self, chessboard_img_path, use_grayscale=True):
        img_data = self._get_resized_chessboard(chessboard_img_path)
        if use_grayscale:
            img_data = img_data.convert("L", (0.2989, 0.5870, 0.1140, 0))
        image = np.asarray(img_data, dtype=np.uint8)
        return image

    def _predict_chessboard(self, image):
        """Given a file path to a chessboard PNG image,
        Returns a FEN string representation of the chessboard
        """
        img_data_list = self._chessboard_tiles_img_data(image)
        predictions = self.scripted_model(img_data_list).argmax(dim=1)
        fen_predictions = [FEN_CHARS[prediction] for prediction in predictions]

        predicted_fen = self._compressed_fen(
            "/".join(
                [
                    "".join(r)
                    for r in np.reshape([p[0] for p in fen_predictions], [8, 8])
                ]
            )
        )
        return predicted_fen

    def _compressed_fen(self, fen):
        """From: 11111q1k/1111r111/111p1pQP/111P1P11/11prn1R1/11111111/111111P1/R11111K1
        To: 5q1k/4r3/3p1pQP/3P1P2/2prn1R1/8/6P1/R5K1
        """
        for length in reversed(range(2, 9)):
            fen = fen.replace(length * "1", str(length))
        return fen

    def _chessboard_tiles_img_data(self, image):
        """Given a file path to a chessboard PNG image, returns a
        size-64 array of 32x32 tiles representing each square of a chessboard
        """
        tiles = self._get_chessboard_tiles(image)
        img_data_list = []
        for i in range(64):
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)]
            )
            img_data = transform(tiles[i])
            img_data_list.append(img_data)
        return torch.stack(img_data_list)

    def _get_resized_chessboard(self, chessboard_img_path):
        """chessboard_img_path = path to a chessboard image
        Returns a 256x256 image of a chessboard (32x32 per tile)
        """
        img_data = PIL.Image.open(chessboard_img_path).convert("RGB")
        return img_data.resize([256, 256], PIL.Image.BILINEAR)

    def _get_chessboard_tiles(self, image, use_grayscale=True):
        """chessboard_img_path = path to a chessboard image
        use_grayscale = true/false for whether to return tiles in grayscale
        Returns a list (length 64) of 32x32 image data
        """
        # 64 tiles in order from top-left to bottom-right (A8, B8, ..., G1, H1)
        tiles = [None] * 64
        for rank in range(8):  # rows/ranks (numbers)
            for file in range(8):  # columns/files (letters)
                sq_i = rank * 8 + file
                tile = np.zeros([32, 32, 3], dtype=np.uint8)
                for i in range(32):
                    for j in range(32):
                        if use_grayscale:
                            tile[i, j] = image[
                                rank * 32 + i,
                                file * 32 + j,
                            ]
                        else:
                            tile[i, j] = image[
                                rank * 32 + i,
                                file * 32 + j,
                                :,
                            ]
                tiles[sq_i] = PIL.Image.fromarray(tile, "RGB")
        return tiles

    def predict(self, image):
        return self._predict_chessboard(image)

    def predict_image_path(self, image_path):
        image = self._get_image(image_path)
        return self.predict(image)


if __name__ == "__main__":
    chess_scanner = ChessScanner()
    predicted_fen = chess_scanner.predict_image_path(IMAGE_PATH)
    print(predicted_fen)
