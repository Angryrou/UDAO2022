
[comment]: # (Taken from UDAO repository. Path: UDAO/snapshot/pythonZMQServer/README_batch.md)

Batch Objective Prediction

1. Dependency installation
    ```bash
    # python 3.6.9
    pip install numpy
    pip install torch torchvision
    ```

2. Data. `batch_confs/wrapped_model.pkl` provide all the needed information to restore the trained models 

3. Brief description of code
    - `batch_models.py` considers with the 17 combinations of `(k2,k3,k4)`; `batch_models_eliminate.py` does not have constraints
    on combinations of `(k2,k3,k4)`; `batch_models_eliminate_n_confs.py` use mini-batch when doing backprop.
    - `stream_models*.py` has same name policy as `batch_models*.py`
    - `batch_models_eliminate.py` uses mini-batch to find soo problem; while `stream_models_eliminate.py` still uses the one-conf-at-a-time approach. -- To make the Utopia point
    results more stable, it is a todo item for adjusting streaming to use mini-batch for soo as well.
    - `batch_models_eliminate.py` has a class called `Batch_model`, which includes three objective models for `latency`, `cpu` 
    and `network` respectively. 
    - `Batch_model` has a `predict` method which can take `zmesg` as input and return a predicted objective value
    - The `predict` method will check whether the `zmesg` is legal. If `zmesg` is illegal, it will return `-1`. 
    
    
4. Run model prediction test
    ```bash
    python batch_predict_test.py
    ``` 

5. Knob format. Please refer to [batch_knob_format](./batch_knob_format.md)