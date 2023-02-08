# Note for MOO with model uncertainty 


## Step-by-step Instruction

1. Fetch the repo and set up the python environment.
    ```bash
    # git clone the project of the specific branch `moo-with-model-uncertainty`
    git clone --depth 1 -b moo-with-model-uncertainty git@github.com:Angryrou/UDAO2022.git
    
    cd UDAO2022 
    # set up the python virtual environment (cpu-only)
    # make sure conda has been installed in advance
    conda create -n udao python=3.9
    conda activate udao
    conda install pytorch torchvision torchaudio cpuonly -c pytorch # version 1.13.1
    pip install -r requirements.txt
    ```

2. Run the code for MOO with model uncertainty.
    ```bash
    # be sure to run under `UDAO2022` to enable other packages in the repository. 
    export PYTHONPATH=$PWD
    python biao/main.py
    ```

## Brief Intro.

- Objectives: (latency, cost)
- Variables: 12 knobs (check details at resources/knob-meta/spark.json)

## Code Example

`biao/main.py` is the entry point for your MOO with model uncertainty, including
- `data_preparation()` 
- `sample_knobs()` to sample configurations as a DataFrame of knobs
- `pred_latency_and_cost()` to give the prediction of mu and std for both objectives (latency, cost)
- `reco_configurations()` to recommend configuration via MOO
- a validation for the generated results 

During your implementation, it is worth noting that
1. Note that you shall mainly focus on `reco_configurations` to finish your MOO approaches.
2. All your edits should be under your name directory `biao/`
3. you can modify code in `main.py` at your own convenience. 


## Submit

When you finish the tasks, please email me the generated pickle file. Good luck!

---
Chenghao