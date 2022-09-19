# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: an example of getting configurations via LHS and BO
#
# Created at 9/17/22
import pandas as pd

from trace.collect.sampler import LHSSampler, BOSampler
from utils.data.configurations import SparkKnobs, knob_normalize, knob_denormalize
import numpy as np

SEED = 42
N_SAMPLES_LHS = 16
N_SAMPLES_BO = 2


def fake_exec(query, conf):
    assert len(query) == len(conf)
    return np.random.rand(len(conf)).reshape(-1, 1) * 100


print(f"1. Be sure to prepare the meta file for the spark knobs under resources/knob-meta")
spark_knobs = SparkKnobs(meta_file="resources/knob-meta/spark.json")
knobs = spark_knobs.knobs

np.random.seed(SEED)
print(f"1. get {N_SAMPLES_LHS} configurations via LHS")
lhs_sampler = LHSSampler(knobs, seed=SEED)
samples, knob_df = lhs_sampler.get_samples(N_SAMPLES_LHS, debug=True)
conf_df = spark_knobs.df_knob2conf(knob_df)
print(knob_df.to_string())
print(conf_df.to_string())
print()

print(f"2. simulate {N_SAMPLES_LHS} objective values corresponding to the configurations")
objs = fake_exec(["q1"] * N_SAMPLES_LHS, conf_df)
print(objs)
print()

print(f"3. get {N_SAMPLES_BO} configurations via BO...")
print(f"3.1 parse and normalize all parameters to 0-1")
knob_df2 = spark_knobs.df_conf2knob(conf_df)
samples2 = knob_normalize(knob_df2, knobs)
assert (knob_df2 == knob_df).all().all()
assert (knob_df2 == knob_denormalize(samples2, knobs)).all().all()

bo_sampler = BOSampler(knobs, seed=SEED)
print(f"3.2 iteratively get the configurations via BO...")
bo_trial = 0
bo_samples = 0
bo_objs = objs.copy()  # maintain the obj values including the duplicated configurations due to the rounding issue.
while True:
    bo_trial += 1
    print(f"trial {bo_trial} starts...")
    reco_sample = bo_sampler.get_samples(1, observed_inputs=samples2, observed_outputs=bo_objs)
    # get the recommended knob values based on reco_samples by denormalizing and rounding
    reco_knob_df = knob_denormalize(reco_sample, knobs)
    assert reco_knob_df.shape == (1, len(knobs))
    # get the signature of the recommended knob values
    reco_knob_sign = reco_knob_df.index.values[0]

    # add the recommended knob values
    # to check whether the recommended knob values have already appeared in previous observation due to rounding.
    if reco_knob_sign in knob_df2.index:
        # map the new_ojb to the existed one if the reco have already been observed
        new_obj = objs[knob_df2.index.to_list().index(reco_knob_sign)].reshape(-1, 1)
        # update observed results
        samples2 = np.vstack([samples2, reco_sample])
        bo_objs = np.vstack([bo_objs, new_obj])
        print(f"trial {bo_trial} recommended an observed configuration, skip running.")
    else:
        # run sql to get the new_obj otherwise
        new_obj = fake_exec(["q1"], [reco_sample])
        # update observed results
        knob_df2 = pd.concat([knob_df2, reco_knob_df])
        samples2 = np.vstack([samples2, reco_sample])
        bo_objs = np.vstack([bo_objs, new_obj])
        objs = np.vstack([objs, new_obj])
        # update the bo samples to check the end signal
        bo_samples += 1
        print(f"trial {bo_trial} recommended BO sample {bo_samples}: {reco_knob_sign}")
        print(reco_knob_df.to_string())
        print(spark_knobs.df_knob2conf(reco_knob_df).to_string())
        if bo_samples == N_SAMPLES_BO:
            break

print()
print(f"we got {N_SAMPLES_LHS} samples from LHS and {N_SAMPLES_BO} samples from BO")
print(f"The objective values are:")
print(objs)