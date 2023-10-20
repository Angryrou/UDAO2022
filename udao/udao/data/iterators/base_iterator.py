from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDatasetIterator(Dataset):
    """Dummy Base class for all dataset iterators.
    Inherits from torch.utils.data.Dataset.
    Defined to allow for type hinting and having an __init__ method.
    """

    @abstractmethod
    def __init__(self, keys, *args, **kwargs):  # type: ignore
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
