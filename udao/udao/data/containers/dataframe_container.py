import numpy as np
import pandas as pd
from attr import dataclass

from .base_container import BaseContainer


@dataclass
class DataFrameContainer(BaseContainer):
    """Container for a pandas DataFrame"""

    df: pd.DataFrame
    """Embeddings for each operation.
    MultiIndex (plan, operation)"""

    def get(self, key: str) -> np.ndarray:
        return self.df.loc[key].values  # type: ignore
