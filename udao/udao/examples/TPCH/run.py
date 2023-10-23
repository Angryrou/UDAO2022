from pathlib import Path

from udao.data.embedders import Word2VecEmbedder
from udao.data.extractors import QueryEmbeddingExtractor, QueryStructureExtractor
from udao.data.handler.data_handler import DataHandler, DataHandlerParams
from udao.data.iterators import QueryPlanIterator
from udao.utils.logging import logger

if __name__ == "__main__":
    queryPlanDataHandlerParams = DataHandlerParams(
        index_column="id",
        feature_extractors={
            "query_structure_container": (QueryStructureExtractor, []),
            "query_embeddings_container": (
                QueryEmbeddingExtractor,
                [Word2VecEmbedder()],
            ),
        },
        Iterator=QueryPlanIterator,
        stratify_on="tid",
        dryrun=True,
    )
    base_dir = Path(__file__).parent
    data_handler = DataHandler.from_csv(
        str(base_dir / "data/LQP.csv"), queryPlanDataHandlerParams
    )
    split_iterators = data_handler.split_data().extract_features().get_iterators()
    logger.info(split_iterators["train"][0])