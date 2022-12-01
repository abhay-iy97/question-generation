import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, X: np.array, Y: np.array) -> None:
        # self.dataset = dataset
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
