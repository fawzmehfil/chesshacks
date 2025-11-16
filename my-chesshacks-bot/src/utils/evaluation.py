# evaluation.py
import torch
import numpy as np
from chessCNN import ChessCNN
from data_ingestion import board_to_tensor_coords


class CNNEvaluator:
    """Wraps your CNN for evaluation of python-chess Board objects."""
    def __init__(self, model_path="final_cnn.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Using device: {self.device}")

    def evaluate_board(self, board) -> float:
        tensor = board_to_tensor_coords(board).to(self.device)
        with torch.no_grad():
            return self.model(tensor).item()
