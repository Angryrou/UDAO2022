import numpy as np
import pytest

from ...utils.embedding_utils import (
    Doc2VecEmbedder,
    Doc2VecParams,
    Word2VecEmbedder,
    Word2VecParams,
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
