import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from attr import dataclass
from gensim.corpora import Dictionary
from gensim.models import Doc2Vec, TfidfModel, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phraser, Phrases
from udao.data.utils.utils import TrainedFeatureExtractor


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


class BaseEmbedder(ABC):
    @abstractmethod
    def fit_transform(
        self, training_texts: Sequence[str], epochs: Optional[int] = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        pass


class Word2VecEmbedder(BaseEmbedder):
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

    Parameters
    ----------
    w2v_params : Word2VecParams
        Parameters to pass to the gensim.Word2Vec model
        (ref: https://radimrehurek.com/gensim/models/word2vec.html)
    """

    def __init__(self, w2v_params: Optional[Word2VecParams] = None) -> None:
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
        self, training_texts: Sequence[str], epochs: Optional[int] = None
    ) -> np.ndarray:
        """Train the Word2Vec model on the training texts and return the embeddings.

        Parameters
        ----------
        training_texts : Sequence[str]
            list of training texts
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
        training_sentences = [row.split() for row in training_texts]
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

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Transform a list of query plans into embeddings.

        Parameters
        ----------
        texts : Sequence[str]
            list of texts to transform
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
        sentences = [row.split() for row in texts]
        if self._bigram_model is None:
            raise ValueError("Must call fit_transform before calling transform")
        descriptions = self._bigram_model[sentences]

        bow_corpus: List[List[Tuple[int, int]]] = [
            self.dictionary.doc2bow(doc, allow_update=False, return_missing=False)  # type: ignore
            for doc in descriptions
        ]

        return self._get_encodings_from_corpus(bow_corpus=bow_corpus)


class Doc2VecEmbedder(BaseEmbedder):
    """A class to embed query plans using Doc2Vec.
    To use it:
    - first call fit_transform on a list of training query plans,
    - then call transform on a list of query plans.

    N.B. To ensure reproducibility, several things need to be done:
    - set the seed in the Doc2VecParams
    - set the PYTHONHASHSEED
    - set the number of workers to 1

    Parameters
    ----------
    d2v_params : Doc2VecParams
        Parameters to pass to the gensim.Doc2Vec model
        (ref: https://radimrehurek.com/gensim/models/doc2vec.html)
    """

    def __init__(self, d2v_params: Optional[Doc2VecParams] = None) -> None:
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

    def _prepare_corpus(self, texts: Sequence[str], /) -> List[TaggedDocument]:
        """Transform strings into a list of TaggedDocument

        a TaggedDocument consists in a list of tokens and a tag
        (here the index of the plan)
        """
        tokens_list = list(map(lambda x: x.split(), texts))
        corpus = [TaggedDocument(d, [i]) for i, d in enumerate(tokens_list)]
        return corpus

    def fit(
        self, training_texts: Sequence[str], /, epochs: Optional[int] = None
    ) -> None:
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
        corpus = self._prepare_corpus(training_texts)
        self.d2v_model.build_vocab(corpus)
        self.d2v_model.train(
            corpus, total_examples=self.d2v_model.corpus_count, epochs=epochs
        )
        self._is_trained = True

    def transform(self, texts: Sequence[str]) -> np.ndarray:
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
        epochs = self.d2v_params.epochs
        if not self._is_trained:
            raise ValueError("Must call fit_transform before calling transform")
        encodings = [
            self.d2v_model.infer_vector(doc.split(), epochs=epochs) for doc in texts
        ]
        norms = np.linalg.norm(encodings, axis=1)
        # normalize the embeddings
        return encodings / norms[..., np.newaxis]

    def fit_transform(
        self, training_texts: Sequence[str], /, epochs: Optional[int] = None
    ) -> np.ndarray:
        self.fit(training_texts, epochs)
        return self.transform(training_texts)


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
        brief_clean,
        remove_duplicate_spaces,
    ]
    for processing in processings:
        operation = processing(operation)
    return operation


def extract_operations(
    plan_df: pd.DataFrame, operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    """Extract unique operations from a DataFrame of
    query plans and links them to query plans.
    Operations are transformed using prepare_operation
    to remove statistical information and hashes.

    Parameters
    ----------
    plan_df : pd.DataFrame
        DataFrame containing the query plans and their ids.

    operation_processing : Callable[[str], str]
        Function to process the operations, by default no processing will be applied
        and the raw operations will be used.

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
        lambda plan: [operation_processing(op) for op in plan.splitlines()]  # type: ignore
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
