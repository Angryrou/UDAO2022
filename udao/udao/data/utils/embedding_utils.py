import multiprocessing as mp
from typing import List, Optional, Sequence, Tuple

import numpy as np
from attr import dataclass
from gensim.corpora import Dictionary
from gensim.models import Doc2Vec, TfidfModel, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phraser, Phrases


@dataclass
class Word2VecParams:
    min_count: int = 1
    window: int = 3
    vec_size: int = 32
    alpha: float = 0.025
    sample: float = 0.1
    min_alpha: float = 0.0007
    workers: int = mp.cpu_count() - 1
    seed: int = 42


@dataclass
class Doc2VecParams(Word2VecParams):
    pass


class Word2VecEmbedder:
    def __init__(self, w2v_params: Word2VecParams):
        self.w2v_model = Word2Vec(
            min_count=w2v_params.min_count,
            window=w2v_params.window,  # 3
            vector_size=w2v_params.vec_size,  # 32
            sample=w2v_params.sample,
            alpha=w2v_params.alpha,  # 0.025
            min_alpha=w2v_params.min_alpha,
            workers=w2v_params.workers,
            seed=w2v_params.seed,
        )
        self._bigram_model: Optional[Phraser] = None
        self.dictionary = Dictionary()
        self.tfidf_model = TfidfModel(smartirs="ntc")

    def _get_encodings_from_corpus(
        self, bow_corpus: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        return np.array(
            [
                np.mean(
                    [
                        self.w2v_model.wv.get_vector(self.dictionary[wid], norm=True)
                        * freq
                        for wid, freq in wt
                    ],
                    0,
                )
                for wt in self.tfidf_model[bow_corpus]
            ]
        )

    def fit_transform(
        self, training_plans: Sequence[str], epochs: int = 10
    ) -> np.ndarray:
        training_sentences = [row.split() for row in training_plans]
        phrases = Phrases(training_sentences, min_count=30)  # Extract parameter
        self._bigram_model = Phraser(phrases)
        training_descriptions = self._bigram_model[training_sentences]
        self.w2v_model.build_vocab(training_descriptions)
        self.w2v_model.train(
            training_descriptions,
            total_examples=self.w2v_model.corpus_count,
            epochs=epochs,
            report_delay=1,
        )
        # print(f"get {self.w2v_model.wv.vectors.shape[0]} words from word2vec")
        bow_corpus: List[List[Tuple[int, int]]] = [
            self.dictionary.doc2bow(doc, allow_update=True, return_missing=False)  # type: ignore
            for doc in training_descriptions
        ]
        self.tfidf_model.initialize(bow_corpus)
        # print(f"get {len(self.tfidf.idfs)} words from tfidf")
        return self._get_encodings_from_corpus(bow_corpus=bow_corpus)

    def transform(self, plans: Sequence[str]) -> np.ndarray:
        sentences = [row.split() for row in plans]
        if self._bigram_model is None:
            raise ValueError("Must call fit_transform before calling transform")
        descriptions = self._bigram_model[sentences]

        bow_corpus: List[List[Tuple[int, int]]] = [
            self.dictionary.doc2bow(doc, allow_update=False, return_missing=False)  # type: ignore
            for doc in descriptions
        ]

        return self._get_encodings_from_corpus(bow_corpus=bow_corpus)


class Doc2VecEmbedder:
    def __init__(self, d2v_params: Doc2VecParams) -> None:
        self.d2v_model = Doc2Vec(
            min_count=d2v_params.min_count,
            window=d2v_params.window,
            vector_size=d2v_params.vec_size,
            sample=d2v_params.sample,
            alpha=d2v_params.alpha,
            min_alpha=d2v_params.min_alpha,
            workers=d2v_params.workers,
            seed=d2v_params.seed,
        )
        self._is_trained = False

    def _prepare_corpus(self, plans: Sequence[str]) -> List[TaggedDocument]:
        tokens_list = list(map(lambda x: x.split(), plans))
        corpus = [TaggedDocument(d, [i]) for i, d in enumerate(tokens_list)]
        return corpus

    def fit_transform(
        self, training_plans: Sequence[str], epochs: int = 10
    ) -> np.ndarray:
        corpus = self._prepare_corpus(training_plans)

        self.d2v_model.build_vocab(corpus)
        self.d2v_model.train(
            corpus, total_examples=self.d2v_model.corpus_count, epochs=epochs
        )
        self.d2v_model.dv.get_normed_vectors()
        self._is_trained = True
        return self.d2v_model.dv.get_normed_vectors()

    def transform(self, plans: Sequence[str]) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Must call fit_transform before calling transform")
        corpus = self._prepare_corpus(plans)
        op_encs = self.d2v_model.infer_vector(corpus)
        # normalize the embeddings
        return op_encs / np.linalg.norm(op_encs, axis=1, keepdims=True)
