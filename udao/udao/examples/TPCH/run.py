from pathlib import Path

from ...data.embedders import Word2VecEmbedder
from ...data.extractors import QueryEmbeddingExtractor, QueryStructureExtractor
from ...data.handler.data_handler import DataHandler, DataHandlerParams
from ...data.iterators import QueryPlanIterator
from ...utils.logging import logger

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
