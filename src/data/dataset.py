# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    X: (n_windows, seq_len, n_features)
    y: (n_windows,)
    meta: list of dicts or None
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, meta=None):
        assert X.dtype == np.float32
        self.X = X
        self.y = y.astype(np.int64)
        self.meta = meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            "x": torch.from_numpy(self.X[idx]),   # (seq_len, nfeat)
            "y": torch.tensor(int(self.y[idx]), dtype=torch.long)
        }
        if self.meta:
            item["meta"] = self.meta[idx]
        return item
