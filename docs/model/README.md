model
=====


GTN training
------------
```bash
# MLP example
export PYTHONPATH="$PWD"
python examples/model/spark/1.train.py --ch1-type off --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
--obj latency --model-name GTN --nworkers 16 --bs 1024 --epochs 200 --init-lr 3e-2 --min-lr 1e-5 --weight-decay 1e-2 \
--loss-type wmape --L-mlp 5 --hidden-dim 512 --dropout 0.2

# GTN example
python examples/model/spark/1.train.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on --gpu 0 \
--obj latency --model-name GTN --nworkers 16 --bs 1024 --epochs 200 --init-lr 3e-3 --min-lr 1e-5 --weight-decay 1e-2 \
--loss-type wmape --L-gtn 3 --L-mlp 3 --hidden-dim 128 --out-dim 128 --ch1-type-dim 8 --n-heads 4 --dropout 0.2 --ped 8

# GTN local debug
python examples/model/spark/1.train.py --ch1-type on --ch1-cbo off --ch1-enc on --ch2 on --ch3 on --ch4 on --gpu -1 \
--obj latency --model-name GTN --nworkers 4 --bs 10 --epochs 2 --init-lr 3e-3 --min-lr 1e-5 --weight-decay 1e-2 \
--loss-type wmape --L-gtn 3 --L-mlp 3 --hidden-dim 128 --out-dim 128 --ch1-type-dim 8 --ch1-cbo-dim 4 --n-heads 4 \
--dropout 0.2 --ped 8

```


Q-level tuning and benchmarking
--------------
```bash
export PYTHONPATH="$PWD"
# run default
python examples/trace/spark/internal/1.run_default.py --out-header examples/trace/spark/internal/2.knob_hp_tuning --if-aqe 0 --worker hex1
python examples/trace/spark/internal/1.run_default.py --out-header examples/trace/spark/internal/2.knob_hp_tuning --if-aqe 1 --worker hex1

# run recommended configurations (by default run run 22 sampled queries)
python -u examples/model/spark/q_level_conf_reco.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
--ckp-sign b7698e80492e5d72 --n-samples 10000 --n-weights 5000
    
#for qid in {1..22}; do
for qid in 1 18 {2..17} {19..22}; do
  for aqe in 0 1; do
    # run manual tuning
    python examples/trace/spark/internal/2.knob_hp_tuning.py --q-sign $qid --knob-type res --num-conf-lhs 40 --if-aqe $aqe --worker tpch
    python examples/trace/spark/internal/2.knob_hp_tuning.py --q-sign $qid --knob-type sql --num-conf-lhs 40 --if-aqe $aqe --worker tpch
    # mapping to augment traces
    python examples/trace/spark/internal/3.trace_augment.py -b TPCH --q-sign $qid --if-aqe $aqe
        
    # run model-based tuning
    # (1) RS+VC+WS, alpha = 0  
    python -u examples/model/spark/q_level_conf_reco_run.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
    --ckp-sign b7698e80492e5d72 --n-samples 10000 --n-weights 5000 --q-signs $qid --if-aqe $aqe --debug 0 --worker tpch \
    --moo ws --algo vc --alpha 0
        
    # (2) RS+RB+WS, alpha = -3, -2, 0, 2, 3
    for a in -3 -2 0 2 3 ; do 
        python -u examples/model/spark/q_level_conf_reco_run.py --ch1-type on --ch1-cbo off --ch1-enc off --ch2 on --ch3 on --ch4 on \
        --ckp-sign b7698e80492e5d72 --n-samples 10000 --n-weights 5000 --q-signs $qid --if-aqe $aqe --debug 0 --worker tpch \
        --moo ws --algo robust --alpha $a
    done
    # (3) RS+VC+BF, alpha = 0    
    # (4) RS+RB+BF, alpha = -3, -2, 0, 2, 3
  done
done

# analyzing
python examples/experiments/1.tpch_benchmarking.py 

```


