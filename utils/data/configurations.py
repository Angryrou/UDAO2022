# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Created at 9/16/22

from utils.parameters import VarTypes
from utils.common import JsonUtils
import numpy as np
import pandas as pd


class KnobUtils(object):

    @staticmethod
    def knob_normalize(df, knobs):
        """
        map the knob values in DataFrame to 0-1 range 2d np.array
        :param df: a DataFrame of knob values
        :param knobs: a list of Knob objects
        :return: a 2D-array of all knobs within 0-1.
        """
        df = df.copy()
        knobs_min = np.array([k.min for k in knobs])
        knobs_max = np.array([k.max for k in knobs])

        for k in knobs:
            if k.type == VarTypes.INTEGER:
                df[k.id] = df[k.id].round().astype(float)
            elif k.type == VarTypes.ENUM:
                df[k.id] = [k.enum.index(k_) for k_ in df[k.id]]
            elif k.type == VarTypes.BOOL:
                df[k.id] = [k.bools.index(k_) for k_ in df[k.id]]
            elif k.type == VarTypes.FLOAT:
                df[k.id] = df[k.id].asfloat(float)
            else:
                raise Exception(f"unsupported VarType {k.type}")
        samples = (df.values - knobs_min) / (knobs_max - knobs_min)
        return samples

    @staticmethod
    def knob_denormalize(samples, knobs):
        """
        map the 0-1 range 2d np.array to its knob values in DataFrame
        :param samples: a 2D-array of all knobs within 0-1
        :param knobs: a list of Knob objects
        :return: a DataFrame of dict for raw knob values
        """
        knob_names = [k.id for k in knobs]
        knob_min = np.array([k.min for k in knobs])
        knob_max = np.array([k.max for k in knobs])

        # map to the min-max space
        samples = samples * (knob_max - knob_min) + knob_min
        # convert to the raw knob values
        df = pd.DataFrame(samples, columns=knob_names)

        for k in knobs:
            if k.type == VarTypes.INTEGER:
                df[k.id] = df[k.id].round().astype(int)
            elif k.type == VarTypes.ENUM:
                df[k.id] = np.array(k.enum)[df[k.id].round().astype(int)]
            elif k.type == VarTypes.BOOL:
                df[k.id] = np.array(k.bools)[df[k.id].round().astype(int)]
            elif k.type == VarTypes.FLOAT:
                df[k.id] = df[k.id].astype(float)
            else:
                raise Exception(f"unsupported VarType {k.type}")

        df.index = df.apply(lambda x: KnobUtils.knobs2sign(x, knobs), axis=1)
        return df

    @staticmethod
    def knobs2sign(knob_values, knobs):
        """
        serialize a sequence of knob values into a string
        :param knob_values: a sequence of knob values (e.g., a list of knobs)
        :param knobs: a list of Knob objects
        :return: the sign of the knob values
        """
        assert len(knob_values) == len(knobs), "dismatch the number of knobs"
        return ",".join([str(v) for v in knob_values])

    @staticmethod
    def sign2knobs(knob_sign, knobs):
        """
        deserialize the sign of knob values to their knob values
        :param knob_sign: the sign of the knob values
        :param knobs: a list of Knob objects
        :return:
        """
        knob_values_str = knob_sign.split(",")
        knob_values = [
            int(v_str) if k.type == VarTypes.INTEGER else (
                float(v_str) if k.type == VarTypes.FLOAT else (
                    (v_str.lower() in ["true", "1"]) if k.type == VarTypes.BOOL else v_str
                )
            )
            for v_str, k in zip(knob_values_str, knobs)
        ]
        return knob_values


class Knob:
    def __init__(self, kid, name, ktype, unit, kmin, kmax, scale, factor, base, enum, bools, default, desc):
        assert ktype in VarTypes.ALL_TYPES
        self.id = kid
        self.name = name
        self.type = ktype
        self.unit = unit
        self.min = kmin
        self.max = kmax
        self.scale = scale
        self.factor = factor
        self.base = base
        self.enum = enum
        self.bools = bools
        self.default = default
        self.desc = desc

        if self.type == VarTypes.BOOL:
            self.min = 0
            self.max = 1
        elif self.type == VarTypes.ENUM:
            self.min = 0
            self.max = len(enum) - 1


class SparkKnobs:
    def __init__(self, meta_file="resources/knob-meta/spark.json"):
        meta = JsonUtils.load_json(meta_file)
        knobs = [
            Knob(
                k["id"],
                k["name"],
                VarTypes.str2type(k["type"]),
                k["unit"],
                k["min"],
                k["max"],
                k["scale"],
                k["factor"],
                k["base"],
                k["enum"],
                k["bools"],
                k["default"],
                k["desc"]
            )
            for k in meta
        ]
        self.knobs = knobs
        self.knob_names = [k.id for k in knobs]

    def df_knob2conf(self, knob_df):
        """
        convert the knob values to the parameter values in the configuration
        :param knob_df: a DataFrame of all knob values
        :return: a DataFrame of all parameter values in the knob
        """
        # convert Integer Knobs to the parameter values
        df = knob_df.copy()
        for k in self.knobs:
            if k.type == VarTypes.INTEGER:
                if k.scale == "linear":
                    df[k.id] = k.factor * df[k.id]
                elif k.scale == "log":
                    df[k.id] = k.factor * (k.base ** df[k.id])
                else:
                    raise Exception(f"unsupported scale attribute {k.scale}")

        # tailored for Spark Knobs
        pc = df["k2"] * df["k3"] * df["k4"]  # get the partition count
        # k4: set spark.default.parallelism = k2 * k3 * k4
        df["k4"] = pc
        # k6: set spark.shuffle.sort.bypassMergeThreshold < (k2 * k3 * k4) when True, > (k2 * k3 * k4) when False
        df["k6"] = [pc_ - 1 if k6_ else pc_ + 1 for k6_, pc_ in zip(df["k6"], pc)]
        # k7: true or false
        df["k7"] = df["k7"].apply(lambda x: "true" if x else "false")
        # k8: set among 0.50-0.75, default 0.6, precision:2
        df["k8"] = df["k8"].round(2).astype(str)
        # s4: set spark.sql.shuffle.partitions = spark.default.parallelism when true, > 2000 when False.
        df["s4"] = [pc_ if s4_ else 2048 for s4_, pc_ in zip(df["s4"], pc)]

        for k in self.knobs:
            if k.id in ["k1", "k2", "k3", "k4", "k5", "s1", "s2", "s3"]:
                df[k.id] = df[k.id].astype(str) + (k.unit if k.unit is not None else "")

        conf_df = df.rename(columns={k.id: k.name for k in self.knobs})
        return conf_df

    def df_conf2knob(self, conf_df):
        """
        convert the parameter values to the knob values, reverse engineering for `knob2conf`
        :param conf_df: a dataframe of all parameter values in the knob
        :return: a dataframe of all knob values
        """
        pc = conf_df["spark.default.parallelism"].astype(int)
        df = conf_df.rename(columns={k.name: k.id for k in self.knobs})
        for k in self.knobs:
            if k.id in ["k1", "k2", "k3", "k4", "k5", "s1", "s2", "s3"]:
                df[k.id] = df[k.id].str.replace(k.unit, "").astype(float) if k.unit is not None \
                    else df[k.id].astype(float)

        # k4: set spark.default.parallelism = k2 * k3 * k4
        df["k4"] = (pc / df["k2"] / df["k3"]).astype(int)
        # k6: set spark.shuffle.sort.bypassMergeThreshold < (k2 * k3 * k4) when True, > (k2 * k3 * k4) when False
        bmt = conf_df["spark.shuffle.sort.bypassMergeThreshold"].astype(int)
        df["k6"] = [bmt_ < pc_ for bmt_, pc_ in zip(bmt, pc)]
        # k7: true or false
        df["k7"] = [k7_ == "true" for k7_ in df["k7"]]
        # k8: set among 0.50-0.75, default 0.6, precision:2
        df["k8"] = df["k8"].astype(float)
        # s4: set spark.sql.shuffle.partitions = spark.default.parallelism when true, > 2000 when False.
        df["s4"] = [int(s4_) <= 2000 for s4_ in df["s4"]]

        # convert Integer parameter to the knob values
        for k in self.knobs:
            if k.type == VarTypes.INTEGER:
                if k.scale == "linear":
                    df[k.id] = (df[k.id] / k.factor).round().astype(int)
                elif k.scale == "log":
                    df[k.id] = (np.log2(df[k.id] / k.factor) / np.log2(k.base)).round().astype(int)
                else:
                    raise Exception(f"unsupported scale attribute {k.scale}")
        return df

    def knobs2conf(self, knob_dict: dict) -> dict:
        """
        convert a dict of knob values to a dict of configuration parameter values
        :param knob_dict: a dict of knob values
        :return: a dict of configuration parameter values
        """
        df = pd.DataFrame.from_records([knob_dict])
        assert df.columns.to_list() == [k.id for k in self.knobs]
        conf_df = self.df_knob2conf(df)
        return conf_df.to_dict(orient="records")[0]

    def conf2knobs(self, conf_dict: dict) -> dict:
        """
        convert a dict of configuration parameter values to a dict of knob values
        :param conf_dict: a dict of configuration parameter values
        :return: a dict of knob values
        """
        df = pd.DataFrame.from_records([conf_dict])
        assert df.columns.to_list() == [k.name for k in self.knobs]
        knob_df = self.df_conf2knob(df)
        return knob_df.to_dict(orient="records")[0]


class PostgresKnobs:
    def __init__(self):
        # todo: add for PSQL
        raise NotImplementedError
