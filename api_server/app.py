from fastapi import FastAPI, File, UploadFile
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from chessscanner import ChessScanner


app = FastAPI()
chess_scanner = ChessScanner()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    fen = chess_scanner.predict_bytes(await file.read())
    print(fen)
    return{"fen": fen }
