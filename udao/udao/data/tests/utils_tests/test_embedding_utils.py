import numpy as np
import pandas as pd
import pytest

from ...utils.embedding_utils import (
    Doc2VecEmbedder,
    Doc2VecParams,
    Word2VecEmbedder,
    Word2VecParams,
    extract_operations,
    prepare_operation,
)


@pytest.fixture
def word2vec_embedder() -> Word2VecEmbedder:
    return Word2VecEmbedder(Word2VecParams())


class TestWord2Vec:
    def test_init(self, word2vec_embedder: Word2VecEmbedder) -> None:
        assert word2vec_embedder.w2v_model is not None
        assert word2vec_embedder._bigram_model is None

    def test_fit_transform(self, word2vec_embedder: Word2VecEmbedder) -> None:
        training_plans = ["a b c", "a b d"]
        training_encodings = word2vec_embedder.fit_transform(training_plans)
        assert word2vec_embedder._bigram_model is not None
        # 4 words
        assert word2vec_embedder.w2v_model.wv.vectors.shape[0] == 4
        # 32 dimensions - corresponds to param vec_size
        assert word2vec_embedder.w2v_model.wv.vectors.shape[1] == 32
        # 2 training plans with dimension 32
        assert training_encodings.shape == (2, 32)

    def test_transform_not_trained(self, word2vec_embedder: Word2VecEmbedder) -> None:
        with pytest.raises(ValueError):
            word2vec_embedder.transform(["a b c"])

    def test_transform_trained(self, word2vec_embedder: Word2VecEmbedder) -> None:
        training_plans = ["a b c", "a b d"]
        training_encodings = word2vec_embedder.fit_transform(training_plans)
        encoding = word2vec_embedder.transform(["a b c", "a b x"])
        assert np.array_equal(training_encodings[0], encoding[0])
        assert encoding.shape == (2, 32)


@pytest.fixture
def doc2vec_embedder() -> Doc2VecEmbedder:
    return Doc2VecEmbedder(Doc2VecParams())


class TestDoc2Vec:
    def test_init(self, doc2vec_embedder: Doc2VecEmbedder) -> None:
        assert doc2vec_embedder.d2v_model is not None
        assert doc2vec_embedder._is_trained is False

    def test_fit_sets_is_trained(self, doc2vec_embedder: Doc2VecEmbedder) -> None:
        training_plans = ["a b c", "a b d"]
        doc2vec_embedder.fit(training_plans)
        assert doc2vec_embedder._is_trained is True

    def test_transform_not_trained_raises_error(
        self, doc2vec_embedder: Doc2VecEmbedder
    ) -> None:
        with pytest.raises(ValueError):
            doc2vec_embedder.transform(["a b c"])

    def test_transform_trained_output_values(
        self, doc2vec_embedder: Doc2VecEmbedder
    ) -> None:
        training_plans = ["a b c", "a b d"]
        doc2vec_embedder.fit(training_plans)
        training_encodings = doc2vec_embedder.transform(training_plans)
        encoding = doc2vec_embedder.transform(["a b c", "a b x"])
        dot = np.dot(training_encodings[0], encoding[0])
        norm_a = np.linalg.norm(training_encodings[0])
        norm_b = np.linalg.norm(encoding[0])
        cosine_similarity = dot / (norm_a * norm_b)
        # similary superior to 0.999
        assert cosine_similarity > 0.999


@pytest.mark.parametrize(
    "op,expected",
    [
        (
            "GlobalLimit 10, Statistics(sizeInBytes=400.0 B, rowCount=10)",
            "globallimit 10",
        ),
        (
            "+- LocalLimit 10, Statistics(sizeInBytes=6.9 GiB, rowCount=1.86E+8)",
            "locallimit 10",
        ),
        (
            "   +- Sort [revenue#47608 DESC NULLS LAST, "
            "o_orderdate#22 ASC NULLS FIRST], true, Statistics(sizeInBytes=6.9 GiB,"
            " rowCount=1.86E+8)",
            "sort revenue desc nulls last o_orderdate asc nulls first true",
        ),
        (
            "      +- Aggregate [l_orderkey#23L, o_orderdate#22,"
            " o_shippriority#20], [l_orderkey#23L, sum(CheckOverflow"
            "((promote_precision(cast(l_extendedprice#28 as "
            "decimal(13,2))) * promote_precision(CheckOverflow((1.00"
            " - promote_precision(cast(l_discount#29 as decimal(13,2))))"
            ", DecimalType(13,2), true))), DecimalType(26,4), true)) AS"
            " revenue#47608, o_orderdate#22, o_shippriority#20],"
            " Statistics(sizeInBytes=6.9 GiB, rowCount=1.86E+8)",
            "aggregate l_orderkey o_orderdate o_shippriority l_orderkey"
            " sum checkoverflow promote_precision cast l_extendedprice as"
            " decimal 13 2 rmul promote_precision checkoverflow 1.00 rminus"
            " promote_precision cast l_discount as decimal 13 2 decimaltype 13 2"
            " true decimaltype 26 4 true as revenue o_orderdate o_shippriority",
        ),
        (
            "               +- Relation tpch_100.part[p_partkey#59113L,p_name#59114"
            ",p_mfgr#59115,p_type#59116,p_size#59117,p_container#59118,p_retailprice"
            "#59119,p_comment#59120,p_brand#59121] parquet, Statistics(sizeInBytes=3.5"
            " GiB, rowCount=1.92E+7)",
            "relation tpch_100.part p_partkey p_name p_mfgr p_type p_size p_container"
            " p_retailprice p_comment p_brand parquet",
        ),
    ],
)
def test_prepare_operation(op: str, expected: str) -> None:
    prepared = prepare_operation(op)
    assert prepared == expected


@pytest.fixture
def df_fixture() -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "plan": [
                "a b c\na b d",
                "a b d\na b x",
            ],
        }
    )


def test_extract_operations(df_fixture: pd.DataFrame) -> None:
    plan_to_op, operations = extract_operations(df_fixture)
    assert plan_to_op == {1: [0, 1], 2: [1, 2]}
    assert operations == ["a b c", "a b d", "a b x"]


def test_extract_operations_processing_is_applied(df_fixture: pd.DataFrame) -> None:
    plan_to_op, operations = extract_operations(
        df_fixture, operation_processing=lambda s: s.replace("x", "c")
    )
    print(operations)
    assert plan_to_op == {1: [0, 1], 2: [1, 0]}
    assert operations == ["a b c", "a b d"]
