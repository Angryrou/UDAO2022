from typing import Callable, Dict

import pandas as pd
from udao.data.embedders import BaseEmbedder
from udao.data.embedders.utils import extract_operations, prepare_operation

from .base_extractors import TrainedFeatureExtractor


class QueryEmbeddingExtractor(TrainedFeatureExtractor):
    """Class to extract embeddings from a DataFrame of query plans.

    Parameters
    ----------
    embedder : BaseEmbedder
        Embedder to use to extract the embeddings,
        e.g. an instance of Word2Vecembedder.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        op_preprocessing: Callable[[str], str] = prepare_operation,
    ) -> None:
        self.embedder = embedder
        self.op_preprocessing = op_preprocessing

    def extract_features(self, df: pd.DataFrame, split: str) -> Dict[str, pd.DataFrame]:
        """Extract embeddings from a DataFrame of query plans.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the query plans and their ids.
        split : str
            Split of the dataset, either "train", "test" or "validation".
            Will fit the embedder if "train" and transform otherwise.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the embeddings of each operation of the query plans.
        """

        plan_to_operations, operations_list = extract_operations(
            df, self.op_preprocessing
        )
        if split == "train":
            embeddings_list = self.embedder.fit_transform(operations_list)
        else:
            embeddings_list = self.embedder.transform(operations_list)
        emb_series = df["id"].apply(
            lambda idx: [embeddings_list[op_id] for op_id in plan_to_operations[idx]]
        )
        emb_df = emb_series.to_frame("embeddings")
        emb_df["plan_id"] = df["id"]
        emb_df = emb_df.explode("embeddings", ignore_index=True)
        emb_df[[f"emb_{i}" for i in range(32)]] = pd.DataFrame(
            emb_df.embeddings.tolist(),
            index=emb_df.index,
        )

        emb_df = emb_df.drop(columns=["embeddings"])
        emb_df["operation_id"] = emb_df.groupby("plan_id").cumcount()
        emb_df = emb_df.set_index(["plan_id", "operation_id"])
        return {"embeddings": emb_df}
