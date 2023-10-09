import numpy as np
from datasets import Dataset


class UDAODataset(Dataset):
    def __init__(self, array: np.ndarray) -> None:
        """UDAO dataset class"""
        self.array: np.ndarray = array
