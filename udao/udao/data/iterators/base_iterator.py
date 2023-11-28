from abc import abstractmethod
from inspect import signature
from typing import Any, Generic, List, Sequence, Type, TypeVar

import torch as th
from torch.utils.data import DataLoader, Dataset

from ..containers import BaseContainer

T = TypeVar("T")


class BaseDatasetIterator(Dataset, Generic[T]):
    """Base class for all dataset iterators.
    Inherits from torch.utils.data.Dataset.
    """

    def __init__(self, keys: Sequence[str], *args: Type[BaseContainer]) -> None:
        self.keys = keys
        self.tensors_dtype = th.float32
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int, /) -> T:
        pass

    @abstractmethod
    def get_iterator_shape(self) -> Any:
        """Returns the shape of the iterator output."""

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

    def _get_sample(self) -> T:
        """Returns a random sample from the iterator."""
        return self[0]

    def set_tensors_dtype(self, dtype: th.dtype) -> None:
        """Sets the dtype of the iterator.
        Useful for mixed precision training.
        """
        self.tensors_dtype = dtype
