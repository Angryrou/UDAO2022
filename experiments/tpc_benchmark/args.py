from argparse import ArgumentParser, Namespace


def _get_base_parser() -> ArgumentParser:
    # fmt: off
    parser = ArgumentParser(description="Udao Script with Input Arguments")
    # Data-related arguments
    parser.add_argument("--benchmark", type=str, default="tpch",
                        help="Benchmark name")
    parser.add_argument("--q_type", type=str, default="q_compile",
                        help="graph type, q_compile or q or qs")
    # Learning parameters
    parser.add_argument("--init_lr", type=float, default=1e-1,
                        help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    # Others
    parser.add_argument("--num_workers", type=int, default=15,
                        help="non-debug only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    # fmt: on
    return parser


def get_graph_avg_args() -> Namespace:
    parser = _get_base_parser()
    # fmt: off
    # Embedder parameters
    parser.add_argument("--lpe_size", type=int, default=8,
                        help="Laplacian Positional encoding size - for GTN only")
    parser.add_argument("--output_size", type=int, default=32,
                        help="Embedder output size")
    parser.add_argument("--op_groups", nargs="+", default=["type", "cbo", "op_enc"],
                        help="List of operation groups")
    parser.add_argument("--type_embedding_dim", type=int, default=8,
                        help="Type embedding dimension")
    parser.add_argument("--vec_size", type=int, default=16,
                        help="Word2Vec embedding size")
    parser.add_argument("--embedding_normalizer", type=str, default=None,
                        help="Embedding normalizer")
    # Regressor parameters
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers in the regressor")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    # fmt: on
    return parser.parse_args()
