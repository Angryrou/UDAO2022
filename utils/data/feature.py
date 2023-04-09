# Author(s): chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: utilization functions for
#              - feature capture (NmonUtils)
#              - featurization
#              - normalization
#
# Created at 9/14/22


CH1_FEATS = ["sql_struct_id", "sql_struct_svid", "qid"]
CH2_FEATS = ["input_mb", "input_records", "input_mb_log", "input_records_log"]
CH3_FEATS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
CH4_FEATS = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2", "s3", "s4"]
OBJS = ["latency"]

CH1_FEATS_STAGE = ["sql_struct_id", "sql_struct_svid", "mapping_sign_id", "qid", "qs_id"]
CH2_FEATS_STAGE = ["task_num",
                   "input_mb", "sr_mb", "input_records", "sr_records",
                   "input_mb_log", "sr_mb_log", "input_records_log", "sr_records_log"]
CH3_FEATS_STAGE = CH3_FEATS
CH4_FEATS_STAGE = CH4_FEATS
OBJS_STAGE = ["stage_latency", "stage_dt", "output_mb_log", "sw_mb_log", "output_records_log", "sw_records_log"]

L2P_MAP = {
    "tpch": {
        0: [0, 1, 1, 1, 1, 2, 2, 2],
        1: [0, 3, 4, 5, 6, 7, 7, 8, 9, 9, 9, 9, 9, 12, 13, 13, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 20, 23, 23,
            23, 23, 23, 26, 26, 26, 29, 29, 29, 32, 32, 32, 35, 35, 36, 37, 37, 38, 39, 39, 39, 45, 45, 45],
        2: [0, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 9, 9, 9, 12, 12, 13, 13, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20,
            20, 20, 23, 23, 23, 23, 23, 26, 26, 26, 29, 29, 20, 32, 32, 32, 32, 32, 35, 35, 35, 36, 37, 37, 38, 39, 39,
            39, 45, 45, 45],
        3: [0, 3, 4, 5, 5, 5, 6, 7, 7, 8, 9, 9, 9, 9, 9, 12, 13, 13, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 20, 23,
            23, 23, 23, 23, 26, 26, 26, 29, 29, 29, 32, 32, 32, 35, 35, 35, 36, 37, 37, 38, 39, 39, 39, 45, 45, 45],
        4: [0, 3, 3, 4, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11, 14, 14, 14, 14, 14, 14],
        5: [0, 3, 3, 4, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 11, 11, 11, 14, 14, 14, 14, 14, 14],
        6: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7],
        7: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8, 9, 13, 13, 13, 13, 13, 13, 10, 10, 10, 10, 10, 10, 16, 16,
            16, 16, 16, 19, 19, 20, 21, 21, 21, 24, 24, 24, 24, 24, 27, 27, 27, 27, 27, 27],
        8: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 13, 13, 13, 13, 13, 13, 10, 10, 10, 10, 10, 10, 16, 16, 16, 16,
            19, 19, 20, 21, 21, 21, 24, 24, 24, 24, 24, 27, 27, 27, 27, 27, 27],
        9: [0, 0, 0, 1, 1, 1, 1],
        10: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 14, 14, 14, 14, 14, 17,
             17, 17, 18, 19, 19, 20, 20, 20, 23, 23, 23, 23, 26, 26, 26, 26],
        11: [0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 14, 14, 14, 14, 17, 17, 18, 19, 19,
             20, 20, 20, 23, 23, 23, 23, 26, 26, 26, 26],
        12: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11, 11, 14, 14, 14, 14, 17, 17, 17,
             18, 19, 19, 20, 20, 20, 23, 23, 23, 23, 26, 26, 26, 26],
        13: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 12, 12, 15, 15, 15, 15, 18,
             18, 18, 18, 18, 21, 21, 21, 21, 21, 24, 24, 24, 24, 27, 27, 27, 28, 29, 29, 30, 31, 31, 31, 34, 34, 34, 34,
             34, 37, 37, 37, 37],
        14: [0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 12, 12, 12, 12, 12, 15, 15, 15, 15, 18, 18, 18, 18,
             18, 21, 21, 21, 21, 24, 24, 24, 24, 27, 27, 28, 29, 29, 30, 31, 31, 31, 34, 34, 34, 34, 34, 37, 37, 37,
             37],
        15: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 12, 12, 12, 12, 12, 15, 15, 15, 15, 18, 18,
             18, 18, 18, 21, 21, 21, 21, 24, 24, 24, 24, 27, 27, 27, 28, 29, 29, 30, 31, 31, 31, 34, 34, 34, 34, 34, 37,
             37, 37, 37],
        16: [0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12, 12, 12, 12, 15, 15, 15, 15, 18, 18, 18, 21, 21,
             21, 21, 24, 24, 24, 24, 27, 27, 28, 29, 29, 30, 31, 31, 31, 34, 34, 34, 34, 34, 37, 37, 37, 37],
        17: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15,
             18, 18, 18, 18, 18, 18, 21, 21, 21, 21, 24, 24, 24, 24, 27, 27, 27, 27, 27],
        18: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15,
             15, 18, 18, 18, 18, 18, 18, 21, 21, 21, 21, 21, 24, 24, 24, 24, 27, 27, 27, 27, 27],
        19: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 18, 18,
             18, 18, 18, 21, 21, 21, 21, 24, 24, 24, 24, 27, 27, 27, 27, 27],
        20: [0, 3, 3, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 10, 10, 10, 13, 13, 13, 13, 13, 16, 16, 16, 16, 19, 19, 19, 19, 19,
             19],
        21: [0, 3, 3, 3, 4, 5, 6, 6, 6, 7, 8, 9, 13, 13, 13, 13, 13, 13, 10, 10, 10, 10, 10, 10, 16, 16, 16, 16, 19, 19,
             19, 19, 19, 19],
        22: [0, 1, 1, 16, 16, 16, 17, 18, 19, 19, 20, 21, 21, 21, 24, 24, 24, 24, 24, 27, 27, 27, 2, 2, 2, 2, 3, 4, 13,
             13, 13],
        23: [0, 1, 1, 1, 1, 2, 3, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 4],
        24: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 7, 7, 7, 7, 7],
        25: [0, 1, 1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7],
        26: [0, 0, 0, 1, 2, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3],
        27: [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 6, 6, 6, 6],
        28: [0, 1, 1, 2, 3, 3, 3, 6, 6, 12, 12, 12, 13, 13, 13, 13, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9],
        29: [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 8, 8, 8, 8, 8, 11, 11, 11, 11],
        30: [0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 8, 8, 8, 8, 8, 11, 11, 11, 11, 11],
        31: [0, 0, 0, 1, 2, 3, 3, 4, 5, 5, 5, 5, 5, 8, 9, 9, 9, 10, 10, 10, 10, 13, 13, 13, 13],
        32: [0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 8, 8, 9, 9, 9, 10, 10, 10, 10, 13, 13, 13, 13, 13, 13],
        33: [0, 3, 3, 4, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 11, 11, 11, 12, 12, 12, 12, 12, 15, 15, 16, 17, 17, 17, 18,
             18, 18, 20, 21, 21, 21, 21, 21, 21, 24, 24, 25, 26],
        34: [0, 0, 0, 1, 2, 3, 3, 3, 3, 6, 6, 6, 6],
        35: [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6],
        36: [0, 0, 0, 1, 2, 3, 3, 3, 3, 6, 6, 6, 6, 6],
        37: [0, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 11, 14, 14, 14, 14, 14, 14, 17,
             17, 17, 18, 18, 19, 20, 20, 20, 20, 20, 20, 23, 26, 26, 26, 26, 26],
        38: [0, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 10, 10, 10, 11, 11, 11, 14, 14, 14, 14, 14, 17, 17, 18, 18,
             18, 19, 20, 20, 20, 20, 26, 26, 26, 26, 26],
        39: [0, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17,
             17, 20, 20, 20, 20, 20, 20, 23, 23, 23, 23, 23, 26, 26, 26, 26, 26],
        40: [0, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17, 17,
             20, 20, 20, 20, 20, 20, 23, 23, 23, 23, 26, 26, 26, 26, 26],
        41: [0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 11, 11, 11, 11, 11]
    }
}


class NmonUtils(object):

    @staticmethod
    def nmon_remote_reset(workers, remote_header):
        NmonUtils.nmon_remote_stop(workers)
        return "\n".join(f"""ssh {worker} rm -f {remote_header}/*.nmon""" for worker in workers)

    @staticmethod
    def nmon_remote_start(workers, remote_header, name_suffix, counts, freq):
        return "\n".join(
            f"""ssh {worker} "nmon -s{freq} -c{counts} -F {worker}{name_suffix}.nmon -m {remote_header}" """
            for worker in workers)

    @staticmethod
    def nmon_remote_stop(workers):
        return "\n".join(f"""ssh {worker} kill $(ssh {worker} ps -ef | grep nmon | tr -s ' '| cut -d ' ' -f2)"""
                         for worker in workers)

    @staticmethod
    def nmon_remote_agg(workers, remote_header, local_header, name_suffix):
        cmd = f"mkdir -p {local_header}/nmon" + "\n"
        cmd += "\n".join(f"""scp {worker}:{remote_header}/{worker}{name_suffix}.nmon {local_header}/nmon"""
                         for worker in workers)
        return cmd
