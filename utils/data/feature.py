# Author(s): chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: utilization functions for
#              - feature capture (NmonUtils)
#              - featurization
#              - normalization
#
# Created at 9/14/22


CH1_FEATS = ["sql_struct_id", "sql_struct_svid"]
CH2_FEATS = ["input_mb", "input_records", "input_mb_log", "input_records_log"]
CH3_FEATS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
CH4_FEATS = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2", "s3", "s4"]
OBJS = ["latency"]


class NmonUtils(object):

    @staticmethod
    def nmon_remote_reset(workers, remote_header):
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