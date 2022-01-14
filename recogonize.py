from chessscanner import IMAGE_PATH, ChessScanner


if __name__ == "__main__":
    chess_scanner = ChessScanner()
    print(chess_scanner.predict_image_path(IMAGE_PATH))