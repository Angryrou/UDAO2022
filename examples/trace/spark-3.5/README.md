## Steps 

### TPCH 

#### Per-workload run

To compare the 4 SQL knobs that could potentially change the query plan structure.

```bash
python examples/trace/spark-3.5/1.per_tpch_workload_run.py --local 1
```