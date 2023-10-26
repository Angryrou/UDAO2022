from abc import abstractmethod
from inspect import signature
from typing import Any, List, Sequence, Type

from torch.utils.data import DataLoader, Dataset

from ..containers import BaseContainer


class BaseDatasetIterator(Dataset):
    """Base class for all dataset iterators.
    Inherits from torch.utils.data.Dataset.
    """

    @abstractmethod
    def __init__(self, keys: Sequence[str], *args: Type[BaseContainer]) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int, /) -> Any:
        pass

    @staticmethod
    def collate(items: List[Any]) -> Any:
        """Collates the items into a batch.
        Used in the dataloader."""
        return items

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> DataLoader:
        """Returns a torch dataloader for the iterator,
        that can be used for training.
        This will use the collate static method
        to collate the items into a batch.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def get_parameter_names(cls) -> List[str]:
        """Returns the names of the container parameters
        of the iterator.
        Useful to create dynamic parameters for related parts
        of the pipeline (feature extractors, preprocessors)
        """
        init_signature = signature(cls.__init__)
        params_list = list(init_signature.parameters.keys())
        # remove self, args and kwargs
        params_list = [
            param
            for param in params_list
            if param not in ["self", "keys", "args", "kwargs"]
        ]
        return params_list
