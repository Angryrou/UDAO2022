from typing import Callable, Dict, List, Sequence, Union

import dgl
import pandas as pd
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

from ..data.utils.utils import DatasetType, train_test_val_split_on_column
from .utils.query_plan_utils import QueryPlanStructure

tqdm.pandas()


class DatasetIterator(Dataset):
    def __init__(
        self,
        keys: Sequence[str],
        graph_features: pd.DataFrame,
        embedding_features: pd.DataFrame,
        template_plans: Dict[int, QueryPlanStructure],
        key_to_template: Dict[str, int],
    ):
        self.keys = keys
        self.key_to_template = key_to_template
        self.graph_features = graph_features
        self.embedding_features = embedding_features
        self.template_plans = template_plans

    def __len__(self) -> int:
        return len(self.keys)

    def _get_graph(self, key: str) -> dgl.DGLGraph:
        return self.template_plans[self.key_to_template[key]].graph.clone()

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        key = self.keys[idx]
        graph = self._get_graph(key)
        print(self.graph_features.loc[key].dtypes)
        graph.ndata["cbo"] = th.tensor(self.graph_features.loc[key].values)
        graph.ndata["op_encs"] = th.tensor(self.embedding_features.loc[key].values)
        return graph


class DataHandler:
    def __init__(
        self, path: str, index_column: str = "id", dryrun: bool = False
    ) -> None:
        self.dryrun = dryrun

        self.full_df = self._load_data(path, index_column)
        self.data_splits: Dict[DatasetType, List[str]] = {}
        self.features: Dict = {}
        self.graph_features: Dict[DatasetType, pd.DataFrame] = {}
        self.embedding_features: Dict[DatasetType, pd.DataFrame] = {}
        self.template_plans: Dict[DatasetType, Dict[int, QueryPlanStructure]] = {}
        self.id_to_template: Dict[DatasetType, Dict[str, int]] = {}

    def _load_data(self, path: str, index_column: str = "id") -> pd.DataFrame:
        full_df = pd.read_csv(path)
        if self.dryrun:
            full_df = full_df.sample(5000)
        full_df.set_index(index_column, inplace=True, drop=False)
        """
        #brief_df = pd.read_csv(str(self.data_dir / "brief.csv"))
        #cols_to_use = query_plan_df.columns.difference(brief_df.columns)
        self.full_df = pd.merge(
            brief_df,
            query_plan_df[["id", *cols_to_use]],
            how="inner",
            on="id",
        )
        """

        return full_df

    def split_data(
        self, stratify_on: str = "tid", val_frac: float = 0.2, test_frac: float = 0.1
    ) -> "DataHandler":
        if self.full_df is None:
            raise ValueError("Data not loaded yet.")
        df_splits = train_test_val_split_on_column(
            self.full_df, stratify_on, val_frac=val_frac, test_frac=test_frac
        )
        self.data_splits = {
            split: df.index.to_list() for split, df in df_splits.items()
        }
        return self

    def extract_features(
        self,
        features_funcs: Dict[str, Union[Callable, Dict[DatasetType, Callable]]],
    ) -> "DataHandler":
        if self.full_df is None:
            raise ValueError("Data not loaded yet.")
        # extractors = {split: StructureExtractor() for
        # split in self.data_splits.keys()}
        # embeddings_extractor = EmbeddingExtractor(Word2VecEmbedder())
        for name, func in features_funcs.items():
            if isinstance(func, dict):
                for split, f in func.items():
                    keys = self.data_splits[split]
                    self.graph_features[split][name] = f(self.full_df.loc[keys])
            else:
                for split, keys in self.data_splits.items():
                    self.graph_features[split][name] = func(self.full_df.loc[keys])
        """
        for split, keys in self.data_splits.items():
            df_split = self.full_df.loc[keys]
            self.graph_features[split] = extractors[split].extract_features(df_split)
            self.template_plans[split] = extractors[split].template_plans
            self.id_to_template[split] = extractors[split].id_template_dict
            self.embedding_features[split] = embeddings_extractor.extract_features(
                df_split, split
            )
        """
        return self

    def get_dataset_iterator(self, split: DatasetType) -> DatasetIterator:
        return DatasetIterator(
            self.data_splits[split],
            self.graph_features[split],
            self.embedding_features[split],
            self.template_plans[split],
            self.id_to_template[split],
        )

    def get_all_iterators(self) -> Dict[DatasetType, DatasetIterator]:
        return {split: self.get_dataset_iterator(split) for split in self.data_splits}
