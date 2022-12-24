# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: download the traces from JSON of REST APIs to CSV files
#
# Created at 12/12/22

import argparse, os, time
import numpy as np
class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--sampling", type=str, default="lhs")
        self.parser.add_argument("--dst-path-header", type=str, default="examples/trace/spark-parser/outs")
        self.parser.add_argument("--url-header", type=str,
                                 default="http://10.0.0.1:18088/api/v1/applications/application_1663600377480")
        self.parser.add_argument("--url-suffix-start", type=int, default=3827, help="the number is inclusive")
        self.parser.add_argument("--url-suffix-end", type=int, default=83840, help="the number is inclusive")
        self.parser.add_argument("--num-processes", type=int, default=6)
        self.parser.add_argument("--lamda", type=int, default=100)
        self.parser.add_argument("--num-steps", type=int, default=20)

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':
    args = Args().parse()
    bm = args.benchmark.lower()
    sf = args.scale_factor
    sampling = args.sampling
    dst_path_header = args.dst_path_header
    url_header = args.url_header
    url_suffix_start = args.url_suffix_start
    url_suffix_end = args.url_suffix_end
    n_processes = args.num_processes
    lamda = args.lamda
    n_steps = args.num_steps
    step = int(np.ceil((url_suffix_end - url_suffix_start + 1) / n_steps))
    print(f"n_steps = {n_steps}, step = {step}")
    for i in range(n_steps):
        start = url_suffix_start + i * step
        end = start + step - 1
        if end > url_suffix_end:
            end = url_suffix_end
        cmd = f"""\
python -u examples/trace/spark-parser/2.tabular_downloader.py -b {bm} --scale-factor {sf} --sampling {sampling} \
--dst-path-header {dst_path_header} --url-header {url_header} --url-suffix-start {start} --url-suffix-end {end} \
--num-processes {n_processes} --lamda {lamda}"""
        print(cmd)

        os.system(cmd)
        os.system("bash ~/chenghao/kill_jps.sh")
        os.system("bash ~/chenghao/start-hex.sh")
        time.sleep(10 * 60)