from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataHandler(ABC):
    @abstractmethod
    def split_data(self) -> "BaseDataHandler":
        """Split data into train, test, val, or other custom splits."""
        pass

    @abstractmethod
    def extract_features(self) -> "BaseDataHandler":
        """Extract and store features for later usage."""
        pass

    @abstractmethod
    def get_dataset_iterator(self, split: str) -> Dataset:
        """Return the DatasetIterator for a given split."""
        pass
