export PYTHONPATH="$PWD"

# tuning (get 41203 TR, 5090 EVAL1, and 4586 EVAL2 rows)

# round1, fix (vec_size, epochs) = (100, 50), alpha=(0.01, 0.025, 0.05, 0.1)
python examples/data/spark/1.operator_encoding.py -b TPCH --scale-factor 100 \
--src-path-header resources/dataset/tpch_100_query_traces --cache-header examples/data/spark/cache \
--workers 24 --debug 0 --mode d2v --frac-per-struct 0.02 --tuning 1 --vec-size 100 --epochs 50 --alpha 0.005

python examples/data/spark/1.operator_encoding.py -b TPCH --scale-factor 100 \
--src-path-header resources/dataset/tpch_100_query_traces --cache-header examples/data/spark/cache \
--workers 24 --debug 0 --mode d2v --frac-per-struct 0.02 --tuning 1 --vec-size 100 --epochs 50 --alpha 0.01

python examples/data/spark/1.operator_encoding.py -b TPCH --scale-factor 100 \
--src-path-header resources/dataset/tpch_100_query_traces --cache-header examples/data/spark/cache \
--workers 24 --debug 0 --mode d2v --frac-per-struct 0.02 --tuning 1 --vec-size 100 --epochs 50 --alpha 0.025

python examples/data/spark/1.operator_encoding.py -b TPCH --scale-factor 100 \
--src-path-header resources/dataset/tpch_100_query_traces --cache-header examples/data/spark/cache \
--workers 24 --debug 0 --mode d2v --frac-per-struct 0.02 --tuning 1 --vec-size 100 --epochs 50 --alpha 0.05

python examples/data/spark/1.operator_encoding.py -b TPCH --scale-factor 100 \
--src-path-header resources/dataset/tpch_100_query_traces --cache-header examples/data/spark/cache \
--workers 24 --debug 0 --mode d2v --frac-per-struct 0.02 --tuning 1 --vec-size 100 --epochs 50 --alpha 0.1