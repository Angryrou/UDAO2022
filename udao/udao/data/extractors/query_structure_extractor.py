from typing import Any, Dict

import pandas as pd
from udao.data.utils.query_plan import (
    QueryPlanOperationFeatures,
    QueryPlanStructure,
    extract_query_plan_features,
)
from udao.data.utils.utils import PandasTypes

from .base_extractors import StaticFeatureExtractor


class QueryStructureExtractor(StaticFeatureExtractor):
    """
    Extracts the features of the operations in the logical plan,
    and the tree structure of the logical plan.
    Keep track of the different query plans seen so far, and their template id.
    """

    def __init__(self) -> None:
        self.template_plans: Dict[int, QueryPlanStructure] = {}
        self.feature_types: Dict[
            str, type
        ] = QueryPlanOperationFeatures.get_feature_names_and_types()
        self.id_template_dict: Dict[str, int] = {}

    def _extract_structure_and_features(self, idx: str, query_plan: str) -> Dict:
        """Extract the features of the operations in the logical plan,
        and the tree structure of the logical plan.

        Parameters
        ----------
        query_plan : str
            A query plan string.

        Returns
        -------
        Dict
            - template_id: id of the template of the query plan
            - operation_id: list of operation ids in the query plan
            - one key per feature for features of the operations
            in the query plan

        """
        structure, features = extract_query_plan_features(query_plan)
        tid = None

        for template_id, template_structure in self.template_plans.items():
            if structure.graph_match(template_structure):
                tid = template_id
                break

        if tid is None:
            tid = len(self.template_plans) + 1
            self.template_plans[tid] = structure
        self.id_template_dict[idx] = tid
        return {
            "operation_id": features.operation_ids,
            **features.features_dict,
        }

    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract the features of the operations in the logical plan,
        and the tree structure of the logical plan for each query plan
        in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with a column "plan" containing the query plans.

        Returns
        -------
        pd.DataFrame
            Dataframe with one row per operation in the query plans,
            and one column per feature of the operations.
        """
        df_features: pd.DataFrame = df.apply(
            lambda row: self._extract_structure_and_features(row.id, row.plan),
            axis=1,
        ).apply(pd.Series)
        df_features["plan_id"] = df["id"]

        expanded_df = df_features.explode("operation_id", ignore_index=True)
        for feature_name in self.feature_types.keys():
            expanded_df[feature_name] = (
                expanded_df[feature_name]
                .explode(ignore_index=True)  # type: ignore
                .astype(PandasTypes[self.feature_types[feature_name]])
            )  # convert to pandas type
        expanded_df = expanded_df.set_index(["plan_id", "operation_id"])

        return {
            "graph_features": expanded_df,
            "template_plans": self.template_plans,
            "key_to_template": self.id_template_dict,
        }
