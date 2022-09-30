Model
=====

Architecture 
------------

[comment]: # (all the models should inherent from pytorch.nn.model)

We have consolidated all the models generated with different architectures (GPR, GTN, MLP etc.) across repositories. They have been built for postgresql or spark batch/streaming data. The source repositories are [UDAO](https://github.com/shenoy1/UDAO), [udao-model-training](https://github.com/Angryrou/udao-model-training) 

The modelling code is organized as follows. The model architecture and the working internals are kept [here](../../model) and examples to demonstrate the models are located [here](../../examples/model).


Steps:
-----
1. Copy the dump of pickle files to `UDAO2022/examples/model/spark/batch_confs` from `node19:/mnt/disk8/repo_cache_files/batch_confs/*.pkl`

```bash
scp node19:/mnt/disk8/repo_cache_files/batch_confs/*.pkl UDAO2022/examples/model/spark/batch_confs
```

2. Copy files into [model](../../model)
```bash
scp -r node19:/mnt/disk8/repo_cache_files/data UDAO2022/examples/model/spark/
``` 

3. The [examples](../../examples/model/spark) directory contains executables that demonstrate the batch and streaming models taken from the UDAO repo. Brief description of code:

    - `batch_predict_test.py` and `streaming_predict_test.py`: the first calls internally `batch_models_eliminate.py` and uses mini-batch to find soo problem; while the second calls `stream_models_eliminate.py` still uses the one-conf-at-a-time approach. -- To make the Utopia point results more stable, it is a todo item for adjusting streaming to use mini-batch for soo as well.
    
    - `gpr_batch_predict_test.py` and `gpr_streaming_predict_test.py`: these are examples to demonstrate GPR models for batch and streaming data respectively.
    
    More details about the internal architecture of the models can be referred [here](./README_batch.md).