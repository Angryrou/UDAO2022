import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from attr import dataclass
from gensim.corpora import Dictionary
from gensim.models import Doc2Vec, TfidfModel, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phraser, Phrases


@dataclass
class Word2VecParams:
    """
    Parameters to pass to the gensim.Word2Vec model
    (ref: https://radimrehurek.com/gensim/models/word2vec.html)
    """

    min_count: int = 1
    window: int = 3
    vec_size: int = 32
    alpha: float = 0.025
    sample: float = 0.1
    min_alpha: float = 0.0007
    workers: int = 1  # mp.cpu_count() - 1
    seed: int = 42
    epochs: int = 10


@dataclass
class Doc2VecParams(Word2VecParams):
    pass


class Word2VecEmbedder:
    """
    A class to embed query plans using Word2Vec.
    The embedding is computed as the average of the word
    embeddings of the words in the query plan.

    To use it:
    - first call fit_transform on a list of training query plans,
    - then call transform on a list of query plans.

    N.B. To ensure reproducibility, several things need to be done:
    - set the seed in the Word2VecParams
    - set the PYTHONHASHSEED
    - set the number of workers to 1
    """

    def __init__(self, w2v_params: Optional[Word2VecParams] = None) -> None:
        """Initialize the Word2VecEmbedder

        Parameters
        ----------
        w2v_params : Word2VecParams
            Parameters to pass to the gensim.Word2Vec model
            (ref: https://radimrehurek.com/gensim/models/word2vec.html)
        """
        if w2v_params is None:
            w2v_params = Word2VecParams()
        self.w2v_params = w2v_params
        self.w2v_model = Word2Vec(
            min_count=w2v_params.min_count,
            window=w2v_params.window,  # 3
            vector_size=w2v_params.vec_size,  # 32
            sample=w2v_params.sample,
            alpha=w2v_params.alpha,  # 0.025
            min_alpha=w2v_params.min_alpha,
            workers=w2v_params.workers,
            seed=w2v_params.seed,
            epochs=w2v_params.epochs,
        )
        self._bigram_model: Optional[Phraser] = None
        self.dictionary = Dictionary()
        self.tfidf_model = TfidfModel(smartirs="ntc")

    def _get_encodings_from_corpus(
        self, bow_corpus: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        """Compute the encodings from a bow_corpus"""
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
        self, training_plans: Sequence[str], epochs: Optional[int] = None
    ) -> np.ndarray:
        """Train the Word2Vec model on the training plans and return the embeddings.

        Parameters
        ----------
        training_plans : Sequence[str]
            list of training plans
        epochs : int, optional
            number of epochs for training the model, by default will use the value
            in the Word2VecParams.

        Returns
        -------
        np.ndarray
            Embeddings of the training plans
        """
        if epochs is None:
            epochs = self.w2v_params.epochs
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
        """Transform a list of query plans into embeddings.

        Parameters
        ----------
        plans : Sequence[str]
            list of query plans
        epochs : int, optional
            number of epochs for infering a document's embedding,
            by default will use the value in the Word2VecParams.

        Returns
        -------
        np.ndarray
            Embeddings of the query plans

        Raises
        ------
        ValueError
            If the model has not been trained
        """
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
    """A class to embed query plans using Doc2Vec.
    To use it:
    - first call fit_transform on a list of training query plans,
    - then call transform on a list of query plans.

    N.B. To ensure reproducibility, several things need to be done:
    - set the seed in the Doc2VecParams
    - set the PYTHONHASHSEED
    - set the number of workers to 1
    """

    def __init__(self, d2v_params: Optional[Doc2VecParams] = None) -> None:
        """Initialize the Doc2VecEmbedder

        Parameters
        ----------
        d2v_params : Doc2VecParams
            Parameters to pass to the gensim.Doc2Vec model
            (ref: https://radimrehurek.com/gensim/models/doc2vec.html)
        """
        if d2v_params is None:
            d2v_params = Doc2VecParams()
        self.d2v_params = d2v_params
        self.d2v_model = Doc2Vec(
            min_count=d2v_params.min_count,
            window=d2v_params.window,
            vector_size=d2v_params.vec_size,
            sample=d2v_params.sample,
            alpha=d2v_params.alpha,
            min_alpha=d2v_params.min_alpha,
            workers=d2v_params.workers,
            seed=d2v_params.seed,
            epochs=d2v_params.epochs,
        )
        self._is_trained = False

    def _prepare_corpus(self, plans: Sequence[str]) -> List[TaggedDocument]:
        """Transform strings into a list of TaggedDocument

        a TaggedDocument consists in a list of tokens and a tag
        (here the index of the plan)
        """
        tokens_list = list(map(lambda x: x.split(), plans))
        corpus = [TaggedDocument(d, [i]) for i, d in enumerate(tokens_list)]
        return corpus

    def fit(self, training_plans: Sequence[str], epochs: Optional[int] = None) -> None:
        """Train the Doc2Vec model on the training plans

        Parameters
        ----------
        training_plans : Sequence[str]
            list of training plans
        epochs : int, optional
            number of epochs for training the model, by default
            will use the value in the Doc2VecParams.

        Returns
        -------
        np.ndarray
            Normalized (L2) embeddings of the training plans
        """
        if epochs is None:
            epochs = self.d2v_params.epochs
        corpus = self._prepare_corpus(training_plans)
        self.d2v_model.build_vocab(corpus)
        self.d2v_model.train(
            corpus, total_examples=self.d2v_model.corpus_count, epochs=epochs
        )
        self._is_trained = True

    def transform(
        self, plans: Sequence[str], epochs: Optional[int] = None
    ) -> np.ndarray:
        """Transform a list of query plans into normalized embeddings.

        Parameters
        ----------
        plans : Sequence[str]
            list of query plans
        epochs: int, optional
            number of epochs for infering a document's embedding,
            by default will use the value in the Doc2VecParams.

        Returns
        -------
        np.ndarray
            Normalized (L2) embeddings of the query plans

        Raises
        ------
        ValueError
            If the model has not been trained
        """
        if epochs is None:
            epochs = self.d2v_params.epochs
        if not self._is_trained:
            raise ValueError("Must call fit_transform before calling transform")
        encodings = [
            self.d2v_model.infer_vector(doc.split(), epochs=epochs) for doc in plans
        ]
        norms = np.linalg.norm(encodings, axis=1)
        # normalize the embeddings
        return encodings / norms[..., np.newaxis]

    def fit_transform(
        self, training_plans: Sequence[str], epochs: Optional[int] = None
    ) -> np.ndarray:
        self.fit(training_plans, epochs)
        return self.transform(training_plans, epochs)


def remove_statistics(s: str) -> str:
    """Remove statistical information from a query plan
    (in the form of Statistics(...)
    """
    pattern = r"\bStatistics\([^)]+\)"
    # Remove statistical information
    s = re.sub(pattern, "", s)
    return s


def remove_hashes(s: str) -> str:
    """Remove hashes from a query plan, e.g. #1234L"""
    # Replace hashes with a placeholder or remove them
    return re.sub(r"#[0-9]+[L]*", "", s)


def replace_list_elements_with_token(s: str, token: str = "ELT_TOKEN") -> str:
    pattern = r"(?<=IN \()([^)]+)(?=\))"
    match = re.search(pattern, s)
    if match:
        replacement = re.sub(r"[^,]+", token, match.group())
        return re.sub(pattern, replacement, s)
    return s


def brief_clean(s: str) -> str:
    """Remove special characters from a string and convert to lower case"""
    return re.sub(r"[^0-9A-Za-z\'_.]+", " ", s).lower()


def replace_symbols(s: str) -> str:
    """Replace symbols with tokens"""
    return (
        s.replace(" >= ", " GE ")
        .replace(" <= ", " LE ")
        .replace(" == ", " EQ")
        .replace(" = ", " EQ ")
        .replace(" > ", " GT ")
        .replace(" < ", " LT ")
        .replace(" != ", " NEQ ")
        .replace(" + ", " rADD ")
        .replace(" - ", " rMINUS ")
        .replace(" / ", " rDIV ")
        .replace(" * ", " rMUL ")
    )


def remove_duplicate_spaces(s: str) -> str:
    return " ".join(s.split())


def prepare_operation(operation: str) -> str:
    """Prepare an operation for embedding by keeping only
    relevant semantic information"""
    processings: List[Callable[[str], str]] = [
        remove_statistics,
        remove_hashes,
        replace_symbols,
        replace_list_elements_with_token,
        brief_clean,
        remove_duplicate_spaces,
    ]
    for processing in processings:
        operation = processing(operation)
    return operation


def extract_operations(plan_df: pd.DataFrame) -> Tuple[Dict[int, List[int]], List[str]]:
    """Extract unique operations from a DataFrame of
    query plans and links them to query plans.
    Operations are transformed using prepare_operation
    to remove statistical information and hashes.

    Parameters
    ----------
    plan_df : pd.DataFrame
        _description_

    Returns
    -------
    Tuple[Dict[int, List[int]], List[str]]
        plan_to_ops: Dict[int, List[int]]
            Links a query plan ID to a list of operation IDs in the operations list
        operations_list: List[str]
            List of unique operations in the dataset
    """
    df = plan_df[["id", "plan"]].copy()

    df["plan"] = df["plan"].apply(
        lambda plan: [prepare_operation(op) for op in plan.splitlines()]
    )
    df = df.explode("plan", ignore_index=True)
    df.rename(columns={"plan": "operation"}, inplace=True)

    # Build a dictionary of unique operations and their IDs
    unique_ops: Dict[str, int] = defaultdict(lambda: len(unique_ops))
    plan_to_ops: Dict[int, List[int]] = defaultdict(list)
    for row in df.itertuples():
        plan_to_ops[row.id].append(unique_ops[row.operation])

    operations_list = list(unique_ops.keys())

    return plan_to_ops, operations_list
