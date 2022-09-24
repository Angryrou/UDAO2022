# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: the framework for trace colleciton
#
# Created at 9/19/22
import os
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from trace.collect.sampler import LHSSampler, BOSampler
from utils.common import PickleUtils
from utils.data.configurations import KnobUtils, SparkKnobs


class QueryQueue(object):
    def __init__(self, n_templates, qpt: int, seed: int):
        np.random.seed(seed)
        queries = np.tile(np.arange(1, n_templates + 1), [qpt, 1])
        self.queries = np.apply_along_axis(np.random.permutation, axis=1, arr=queries).flatten()
        self.total = n_templates * qpt
        self.n_templates = n_templates
        self.qpt = qpt

    def index_to_tid_and_qid(self, i):
        if i >= self.total:
            print(f"no more queries")
            return -1, -1
        else:
            tid = self.queries[i]
            qid = (i // self.n_templates) + 1
            return tid, qid


class Collection(object, metaclass=ABCMeta):
    def __init__(self, benchmark: str, scale_factor: int, knobs: list, seed=42):
        self.benchmark = benchmark
        self.scale_factor = scale_factor
        self.knobs = knobs
        self.lhs_sampler = LHSSampler(knobs, seed=seed)
        self.bo_sampler = BOSampler(knobs, seed=seed)
        self.seed = seed

    @abstractmethod
    def get_queries(self, n_templates: int, qpt: int):
        pass

    @abstractmethod
    def get_configurations_lhs(self, n_templates: int, qpt: int, cache_header: str):
        pass

    @abstractmethod
    def get_configurations_bo(self):
        pass

    @abstractmethod
    def run_unit(self, q_sign: str, knob_sign: str):
        pass

    @abstractmethod
    def save_trace(self):
        pass


class SparkCollect(Collection):
    def __init__(self, benchmark, scale_factor: int, spark_knobs: SparkKnobs, query_header: str, seed=42):
        super(SparkCollect, self).__init__(benchmark, scale_factor, spark_knobs.knobs, seed)
        self.spark_knobs = spark_knobs
        self.query_header = query_header

    def get_queries(self, n_templates: int, qpt: int):
        """
        get 2D query list, each row is a permutation of full template queries.
        :param n_templates: number of templates
        :param qpt: number of queries per templates
        :return:
        """
        return QueryQueue(n_templates=n_templates, qpt=qpt, seed=self.seed)

    def get_configurations_lhs(self, n_templates, qpt: int, cache_header: str):
        file_name = f"lhs_{n_templates}x{qpt}.pkl"
        try:
            conf_df_dict = PickleUtils.load(cache_header, file_name)
        except:
            conf_df_dict = {}
            for tid in range(1, n_templates + 1):
                start = time.time()
                conf_df_dict[tid] = self.spark_knobs.df_knob2conf(self.lhs_sampler.get_samples(qpt, random_state=tid))
                print(f"generated {qpt} configurations for {tid}, cost {time.time() - start} s")
            PickleUtils.save(conf_df_dict, cache_header, file_name)
        return conf_df_dict

    def get_configurations_bo(self):
        pass

    def run_unit(self, q_sign: str, knob_sign: str):
        pass

    def save_trace(self):
        pass

    def make_script(self, tid: str, qid: str, knob_sign: str, conf_dict: dict,
                    spath="/opt/hex_users/$USER/chenghao/spark-sql-perf",
                    jpath="/opt/hex_users/$USER/spark-3.2.1-hadoop3.3.0/jdk1.8") -> str:
        conf_str = "\n".join(f"--conf {k}={v} \\" for k, v in conf_dict.items())
        bm, sf = self.benchmark, self.scale_factor
        name = f"{self.benchmark}{self.scale_factor}_q{tid}-{qid}_{knob_sign}"

        return f"""\
# {tid}-{qid}
# {knob_sign}

spath={spath}
jpath={jpath}
lpath={spath}/src/main/resources/log4j.properties
name={name}

~/spark/bin/spark-submit \\
--class com.databricks.spark.sql.perf.MyRunTemplateQuery \\
--name {name} \\
--master yarn \\
--deploy-mode client \\
--conf spark.executorEnv.JAVA_HOME=${{jpath}} \\
--conf spark.yarn.appMasterEnv.JAVA_HOME=${{jpath}} \\
{conf_str}
--conf spark.yarn.am.cores={conf_dict["spark.executor.cores"]} \\
--conf spark.yarn.am.memory={conf_dict["spark.executor.memory"]} \\
--conf spark.sql.adaptive.enabled=false \\
--conf spark.sql.parquet.compression.codec=snappy \\
--conf spark.sql.broadcastTimeout=10000 \\
--conf spark.rpc.askTimeout=12000 \\
--conf spark.shuffle.io.retryWait=60 \\
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \\
--conf spark.kryoserializer.buffer.max=512m \\
--driver-java-options "-Dlog4j.configuration=file:$lpath" \\
--conf "spark.driver.extraJavaOptions=-Xms20g" \\
--conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \\
--files "$lpath" \\
--jars ~/spark/examples/jars/scopt_2.12-3.7.1.jar \\
$spath/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar \\
-b {bm} -t {tid} -q {qid} -s {sf} -l {self.query_header} 

"""

    def save_one_script(self, tid: str, qid: str, conf_dict: dict, out_header: str):
        knob_dict = self.spark_knobs.conf2knobs(conf_dict)
        knob_sign = KnobUtils.knobs2sign([knob_dict[k.id] for k in self.knobs], self.knobs)
        # dropped > {log_header}/{name}.log 2>&1
        spark_script = self.make_script(
            tid=str(tid),
            qid=str(qid),
            knob_sign=knob_sign,
            conf_dict=conf_dict
        )
        file_name = f"q{tid}-{qid}_{knob_sign}.sh"
        os.makedirs(f"{out_header}", exist_ok=True)
        with open(f"{out_header}/{file_name}", "w") as f:
            f.write(spark_script)
        print(f"script {tid}-{qid} prepared for running")
        return file_name


class MultiQueryEnvironment(object):

    def __init__(self, cpu_guide_str="resources/system-guide/A_system_prior.csv"):
        df = pd.read_csv(cpu_guide_str, header=None, names=[
            "timestep",
            "cpu_utils_avg",
            "cpu_utils_std",
            "cpu_utils_avg_daily",
            "cpu_utils_std_daily",
            "mem_utils_avg",
            "mem_utils_std",
            "mem_utils_avg_daily",
            "mem_utils_std_daily",
            "io_bytes_avg",
            "io_bytes_std",
            "io_bytes_avg_daily",
            "io_bytes_std_daily",
            "io_rqs_avg",
            "io_rqs_std",
            "io_rqs_avg_daily",
            "io_rqs_std_daily",
        ]).set_index("timestep").sort_index()
