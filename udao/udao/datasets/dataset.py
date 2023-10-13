from pathlib import Path

from datasets import Dataset


class UdaoDataset:
    def __init__(self, path: Path) -> None:
        self.data = Dataset.from_csv(path)
