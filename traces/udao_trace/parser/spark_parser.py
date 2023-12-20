import glob
import os
from typing import Dict
from logging import Logger

from ..utils.logging import _get_logger
from ..utils import JsonHandler, BenchmarkType
from ..workload import Benchmark

import pandas as pd


class SparkParser:

    def __init__(
        self,
        benchmark_type: BenchmarkType,
        scale_factor: int,
        logger: Logger = None
    ):
        benchmark = Benchmark(benchmark_type=benchmark_type, scale_factor=scale_factor)
        self.benchmark = benchmark
        self.benchmark_prefix = benchmark.get_prefix()
        self.logger = logger

    @staticmethod
    def _parse_conf(conf: Dict) -> Dict:
        return {f"{k_type}-{k}": v for k_type, k_list in conf.items() for kv in k_list for k, v in kv.items()}

    def _parse_base(self, d: Dict) -> Dict:
        im = {f"IM-{k}": v for k, v in d["IM"].items()}
        if "Configuration" in d:
            conf_key = "Configuration"
        else:
            assert ("RuntimeConfiguration" in d)
            conf_key = "RuntimeConfiguration"
        conf = self._parse_conf(d[conf_key])
        pd = d["PD"]
        ss_dict = {}
        if "RunningQueryStageSnapshot" in d:
            for k, v in d["RunningQueryStageSnapshot"].items():
                if isinstance(v, list):
                    for tile, v_ in zip([0, 25, 50, 75, 100], v):
                        ss_dict[f"SS-{k}-{tile}tile"] = v_
                else:
                    ss_dict[f"SS-{k}"] = v
        else:
            ss_dict = {
                "SS-RunningTasksNum": 0,
                "SS-FinishedTasksNum": 0,
                "SS-FinishedTasksTotalTimeInMs": 0.0,
                "SS-FinishedTasksDistributionInMs-0tile": 0.0,
                "SS-FinishedTasksDistributionInMs-25tile": 0.0,
                "SS-FinishedTasksDistributionInMs-50tile": 0.0,
                "SS-FinishedTasksDistributionInMs-75tile": 0.0,
                "SS-FinishedTasksDistributionInMs-100tile": 0.0,
            }
        return {**{"PD": pd}, **im, **conf, **ss_dict}

    def _parse_lqp_features(self, d: Dict) -> Dict:
        lqp_str = JsonHandler.dump_to_string(d["LQP"], indent=None)
        base = self._parse_base(d)
        return {**{"lqp": lqp_str}, **base}

    @staticmethod
    def _parse_lqp_objectives(d: Dict) -> Dict:
        return {
            "latency_s": d["DurationInMs"] / 1000,
            "io_mb": d["IOBytes"]["Total"] / 1024 / 1024
        }

    @staticmethod
    def _parse_qs_objectives(d: Dict) -> Dict:
        return {
            "latency_s": d["DurationInMs"] / 1000,
            "io_mb": d["IOBytes"]["Total"] / 1024 / 1024,
            "ana_latency_s": d["TotalTasksDurationInMs"] / 1000
        }

    def _parse_lqp(self, feat: Dict, obj: Dict, meta: Dict) -> Dict:
        feat_dict = self._parse_lqp_features(feat)
        obj_dict = self._parse_lqp_objectives(obj)
        return {**meta, **feat_dict, **obj_dict}

    def _parse_qs(self, d: Dict, meta: Dict) -> Dict:
        qs_lqp_str = JsonHandler.dump_to_string(d["QSLogical"], indent=None)
        qs_pqp_str = JsonHandler.dump_to_string(d["QSPhysical"], indent=None)
        local = {"qs_lqp": qs_lqp_str, "qs_pqp": qs_pqp_str, "InitialPartitionNum": d["InitialPartitionNum"]}
        base = self._parse_base(d)
        obj_dict = self._parse_qs_objectives(d["Objectives"])
        return {**meta, **local, **base, **obj_dict}

    def parse_one_file(self, file: str) -> (pd.DataFrame, pd.DataFrame):
        appid = "application_" + file.split("_application_")[-1][:-5]
        tq = file.split("/")[-1].split("_")[1]
        tid, qid = tq.split("-")
        meta = {"appid": appid, "tid": tid, "qid": qid}

        d = JsonHandler.load_json(file)

        # work on compile_time LQP
        q_dict_list = []
        compile_time_lqp_dict = self._parse_lqp(d["CompileTimeLQP"], d["Objectives"], meta)
        compile_time_lqp_dict["lqp_id"] = 0
        theta_c = {k: v for k, v in compile_time_lqp_dict.items() if k.startswith("theta_c")}
        q_dict_list.append(compile_time_lqp_dict)
        # work on runtime LQP
        for lqp_id, runtime_lqp in d["RuntimeLQPs"].items():
            runtime_lqp_dict = self._parse_lqp(runtime_lqp, runtime_lqp["Objectives"], meta)
            runtime_lqp_dict["lqp_id"] = lqp_id
            runtime_lqp_dict = {**runtime_lqp_dict, **theta_c}
            q_dict_list.append(runtime_lqp_dict)
        # q_lens = [len(d) for d in q_dict_list]
        # assert (np.std(q_lens) == 0, f"inconsistent number q_dict_list: {q_lens}")

        # work on QSs
        qs_dict_list = []
        for qs_id, qs in d["RuntimeQSs"].items():
            qs_dict = self._parse_qs(qs, meta)
            qs_dict["qs_id"] = qs_id
            qs_dict = {**qs_dict, **theta_c}
            qs_dict_list.append(qs_dict)
        # qs_lens = [len(d) for d in qs_dict_list]
        # assert (np.std(qs_lens) == 0, f"inconsistent length of qs_dict_list: {qs_lens}")

        return q_dict_list, qs_dict_list


    def prepare_headers(self, header):
        trace_header = f"{header}/trace"
        assert os.path.exists(trace_header), FileNotFoundError(trace_header)
        csv_header = f"{header}/csv"
        os.makedirs(csv_header, exist_ok=True)
        return trace_header, csv_header

    def parse(self, header, templates, upto=10):
        trace_header, csv_header = self.prepare_headers(header)
        logger = self.logger
        tq2path = {f"{template}-{qid}": None for template in templates for qid in range(1, upto + 1)}
        for i, tq in enumerate(tq2path):
            trace_prefix = f"{trace_header}/{self.benchmark_prefix}_{tq}_"
            files = glob.glob(f"{trace_prefix}*.json")
            if len(files) == 0:
                logger.warning(f"{trace_prefix}* does not exist")
                continue
            elif len(files) > 1:
                raise Exception(f"multiple files found for {trace_prefix}*")
            else:
                tq2path[tq] = files[0]

        tq_parsed = len({tq: path for tq, path in tq2path.items() if path is not None})
        tq_not_found = len(tq2path) - tq_parsed
        tp_total = len(tq2path)
        logger.info(f"queries parsed|not_found|total: {tq_parsed}|{tq_not_found}|{tp_total}")


if __name__ == "__main__":
    sp = SparkParser(
        benchmark_type=BenchmarkType.TPCH,
        scale_factor=100,
        logger=_get_logger(__name__)
    )
    templates = sp.benchmark.templates
    sp.parse(
        header="spark_collector/tpch100/lhs_22x2273",
        templates=templates,
        upto=10
    )
