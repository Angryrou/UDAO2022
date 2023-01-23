model
=====


GTN training
------------
```bash
# MLP example
export PYTHONPATH="$PWD"
python examples/model/spark/1.train_gtn.py --ch1-type off --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
--obj latency --model-name GTN --nworkers 16 --bs 1024 --epochs 200 --init-lr 3e-2 --min-lr 1e-5 --weight-decay 1e-2 \
--loss-type wmape --L-mlp 5 --hidden-dim 512 --dropout 0.2

# GTN example
python examples/model/spark/1.train_gtn.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on --gpu 0 \
--obj latency --model-name GTN --nworkers 16 --bs 1024 --epochs 200 --init-lr 3e-3 --min-lr 1e-5 --weight-decay 1e-2 \
--loss-type wmape --L-gtn 3 --L-mlp 3 --hidden-dim 128 --out-dim 128 --ch1-type-dim 8 --n-heads 4 --dropout 0.2 --ped 8
```


Q-level tuning and benchmarking
--------------
```bash
export PYTHONPATH="$PWD"
# generating reco po points.
for qid in {1..22}; do
  for alpha in -3.0 -2.0 2.0 3.0 ; do
    python -u examples/model/spark/2.q_level_conf_reco.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
    --model-name GTN --template $qid --template-query 1 --ckp-sign b7698e80492e5d72 --n-samples 10000 --moo ws --n-weights 1000 \
    --query-header resources/tpch-kit/spark-sqls --worker tpch --gpu 3 --run 0 --alpha $alpha --if-robust 1
  done
  python -u examples/model/spark/2.q_level_conf_reco.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
    --model-name GTN --template $qid --template-query 1 --ckp-sign b7698e80492e5d72 --n-samples 10000 --moo ws --n-weights 1000 \
    --query-header resources/tpch-kit/spark-sqls --worker tpch --gpu 3 --run 0 --alpha $alpha --if-robust 0
done


export PYTHONPATH="$PWD"
for qid in {1..22}; do
  for aqe in 0 1; do
    python -u examples/model/spark/2.q_level_conf_reco.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
    --model-name GTN --template $qid --template-query 1 --ckp-sign b7698e80492e5d72 --n-samples 10000 --moo ws --n-weights 1000 \
    --query-header resources/tpch-kit/spark-sqls --worker tpch --gpu -1 --run 1 --if-aqe $aqe --if-robust 0
    for alpha in -3.0 -2.0 2.0 3.0 ; do
      python -u examples/model/spark/2.q_level_conf_reco.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
      --model-name GTN --template $qid --template-query 1 --ckp-sign b7698e80492e5d72 --n-samples 10000 --moo ws --n-weights 1000 \
      --query-header resources/tpch-kit/spark-sqls --worker tpch --gpu -1 --run 1 --if-aqe $aqe --if-robust 1 --alpha $alpha
    done
  done    
  python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query $qid --knob-type res --num-conf-lhs 40 --if-aqe 0 --worker tpch
  python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query $qid --knob-type sql --num-conf-lhs 40 --if-aqe 0 --worker tpch
done

for qid in {1..22}; do
  python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query $qid --knob-type res --num-conf-lhs 40 --if-aqe 1 --worker tpch
  python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query $qid --knob-type sql --num-conf-lhs 40 --if-aqe 1 --worker tpch  
done



# analyzing
python examples/analyze/1.heuristic.vs.q_level_tuning.py 

```


