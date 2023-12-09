import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool, Manager

from udao_trace.configuration import SparkConf
from udao_trace.environment.cluster import Cluster
from udao_trace.utils import BenchmarkType, ClusterName, PickleHandler
from udao_trace.utils.handler import error_handler
from udao_trace.utils.logging import logger
from udao_trace.workload import QueryMatrix, Benchmark


class SparkCollector:

    def __init__(
        self,
        knob_meta_file: str,
        benchmark_type: BenchmarkType,
        scale_factor: int,
        cluster_name: ClusterName,
        parametric_bash_file: str = "assets/run_spark_tpc.sh",
        header: str = "spark_collector",
        debug: bool = False
    ):
        spark_conf = SparkConf(knob_meta_file)
        benchmark = Benchmark(benchmark_type=benchmark_type, scale_factor=scale_factor)
        cluster = Cluster(cluster_name)

        self.spark_conf = spark_conf
        self.benchmark = benchmark
        self.cluster = cluster
        self.header = f"{header}/{benchmark.get_prefix()}"
        os.makedirs(self.header, exist_ok=True)

        assert (os.path.exists(parametric_bash_file)), FileNotFoundError(parametric_bash_file)
        self.parametric_bash_file = parametric_bash_file

        self.debug = debug
        self.current_cores = None
        self.lock = None
        self.query_matrix = None

    def _get_lhs_conf_dict(self, n_data_per_template: int) -> None:
        cache_header = f"{self.header}/cache/template_to_conf_dict"
        spark_conf = self.spark_conf
        benchmark = self.benchmark
        templates = benchmark.templates
        try:
            template_to_conf_dict = PickleHandler.load(cache_header, f"{len(templates)}x{n_data_per_template}.pkl")
            logger.debug("template_to_conf_dict loaded")
        except:
            logger.debug("template_to_conf_dict not found, generating...")
            template_to_conf_dict = {
                template: spark_conf.get_lhs_configurations(n_data_per_template, benchmark.template2id[template])
                for template in templates
            }
            PickleHandler.save(template_to_conf_dict, cache_header, f"{len(templates)}x{n_data_per_template}.pkl")
            logger.debug("template_to_conf_dict saved")
        self.template_to_conf_dict = template_to_conf_dict

    def _prepare_lhs_i(self, index: int) -> (str, int, str, int):
        template, qid = self.query_matrix.get_query_as_template_variant(index)
        conf_df = self.template_to_conf_dict[template].iloc[qid - 1]
        knob_sign = conf_df.name
        cores = int(conf_df["spark.executor.cores"]) * (int(conf_df["spark.executor.instances"]) + 1)
        return template, qid, knob_sign, cores

    @staticmethod
    def _exec_cmd(template, qid, cmd: str) -> bool:
        logger.debug(f"[{template}-{qid}]: {cmd}")
        try:
            # Run the command and capture both stdout and stderr
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check the return code
            if result.returncode == 0:
                logger.debug(f"[{template}-{qid}]: success")
                return True
            else:
                print(f"Error executing command. Return code: {result.returncode}")
                print("Error message:")
                print(result.stderr)
                logger.error(
                    f"[{template}-{qid}]: failed, return code: {result.returncode}, error message: {result.stderr}")
                return False
        except Exception as e:
            print(f"Exception: {e}")
            sys.exit(1)

    def _submit(self, template, qid, knob_sign, cores, header):
        trace_header = f"{header}/trace"
        log_header = f"{header}/log"
        os.makedirs(trace_header, exist_ok=True)
        os.makedirs(log_header, exist_ok=True)

        app_name = self.benchmark.get_prefix() + f"_{template}-{qid}_{knob_sign}"
        logger.info(f"-[{template}-{qid}]: start running")
        start = time.time()
        conf_str = knob_sign.replace(",", " ")
        exec_str = f"bash {self.parametric_bash_file} -q \"{template} {qid}\" -c \"{conf_str}\" " \
                   f"-b {self.benchmark.get_name()} -n {app_name} -x {trace_header} 2>&1 " \
                   f"| tee {log_header}/{template}-{qid}.log"

        logger.debug(f"[{template}-{qid}]: {exec_str}")
        if self.debug:
            time.sleep(random.randint(1, 5))
            success = True
        else:
            success = self._exec_cmd(template, qid, exec_str)
        dt = time.time() - start
        logger.debug(f"[{template}-{qid}]: finished, took {dt:0f}s")

        with self.lock:
            self.current_cores.value -= cores
            if not success:
                self.shared_failure_list.append((template, qid, knob_sign))
            logger.info(f"-[{template}-{qid}]: finished, took {dt:0f}s, current_cores={self.current_cores.value}")

    def start_lhs(
        self,
        n_data_per_template: int,
        cluster_cores: int = 150,
        seed: int = 42,
        n_processes: int = 6
    ):

        m = Manager()
        self.lock = m.RLock()
        self.current_cores = m.Value("i", 0)
        self.shared_failure_list = m.list()

        templates = self.benchmark.templates
        self._get_lhs_conf_dict(n_data_per_template)
        self.query_matrix = QueryMatrix(templates=templates, n_data_per_template=n_data_per_template, seed=seed)
        total = self.query_matrix.total

        lhs_header = f"{self.header}/lhs_{n_data_per_template}x{len(templates)}"
        submit_index = 0

        pool = Pool(processes=n_processes)
        while submit_index < total:
            template, qid, knob_sign, cores = self._prepare_lhs_i(submit_index)
            with self.lock:
                if cores + self.current_cores.value < cluster_cores:
                    self.current_cores.value += cores
                    if_submit = True
                    logger.debug(f"Main Process: submit {template}-{qid} for {knob_sign}, "
                                 f"current_cores = {self.current_cores.value}")
                else:
                    if_submit = False
            if if_submit:
                pool.apply_async(func=self._submit,
                                 args=(template, qid, knob_sign, cores, lhs_header),
                                 error_callback=error_handler)
                submit_index += 1
            time.sleep(1)
        pool.close()
        pool.join()

        print(f"Total {total} queries submitted, {len(self.shared_failure_list)} failed:")
        print(self.shared_failure_list)
