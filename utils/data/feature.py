# Author(s): chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: utilization functions for
#              - feature capture (NmonUtils)
#              - featurization
#              - normalization
#
# Created at 9/14/22


class NmonUtils(object):

    @staticmethod
    def nmon_remote_reset(workers, remote_header):
        return "\n".join(f"""ssh {worker} rm -f {remote_header}/*.nmon""" for worker in workers)

    @staticmethod
    def nmon_remote_start(workers, remote_header, name_suffix, duration, freq):
        return "\n".join(
            f"""ssh {worker} "nmon -s{freq} -c{duration} -F {worker}{name_suffix}.nmon -m {remote_header}" """
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