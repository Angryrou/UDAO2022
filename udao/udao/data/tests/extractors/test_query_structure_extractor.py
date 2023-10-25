import pandas as pd
import pytest

from ...extractors import QueryStructureExtractor
from ...utils.query_plan import QueryPlanStructure
from ..conftest import QueryPlanElements


@pytest.fixture
def df_fixture(
    sample_plan_1: QueryPlanElements, sample_plan_2: QueryPlanElements
) -> pd.DataFrame:
    input_df = pd.DataFrame.from_dict(
        {
            "id": [1, 2, 3],
            "tid": [1, 2, 1],
            "plan": [
                sample_plan_1.query_plan,
                sample_plan_2.query_plan,
                sample_plan_1.query_plan,
            ],
        }
    )
    return input_df


class TestStructureExtractor:
    def test_structure_extractor_has_feature_types(self) -> None:
        extractor = QueryStructureExtractor()
        assert extractor.feature_types == {"rows_count": float, "size": float}

    def test_structures_match_templates(self, df_fixture: pd.DataFrame) -> None:
        extractor = QueryStructureExtractor()
        for row in df_fixture.itertuples():
            s_dict = extractor._extract_structure_and_features(row.id, row.plan)
            assert set(s_dict.keys()) == {
                "operation_id",
                *extractor.feature_types.keys(),
            }
        for plan in extractor.template_plans.values():
            assert type(plan) == QueryPlanStructure
        assert len(extractor.template_plans) == 2
        assert extractor.id_template_dict == {1: 1, 2: 2, 3: 1}

    def test_extract_structure_from_df_returns_correct_shape(
        self, df_fixture: pd.DataFrame
    ) -> None:
        extractor = QueryStructureExtractor()
        structure_container = extractor.extract_features(df_fixture)

        multi_index = pd.MultiIndex.from_tuples(
            [
                [row.id, i]
                for row in df_fixture.itertuples()
                for i, _ in enumerate(row.plan.splitlines())
            ],
            names=["plan_id", "operation_id"],
        )
        assert (multi_index == structure_container.graph_features.index).all()
