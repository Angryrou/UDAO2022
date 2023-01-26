# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 26/01/2023

import argparse
import glob, os
import re

from utils.common import BenchmarkUtils


class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--out-header", type=str, default="examples/trace/spark/internal/2.knob_hp_tuning")
        self.parser.add_argument("--q-sign", type=str, default="1")
        self.parser.add_argument("--if-aqe", type=int, default=1)

    def parse(self):
        return self.parser.parse_args()


def get_s3_adjustable(bm: str):
    if bm.lower() == "tpch":
        return {
            'q1': [0, 1, 2, 3, 4, 5],
            'q2': [4, 5],
            'q3': [4, 5],
            'q4': [0, 1, 2, 3, 4, 5],
            'q5': [3, 4, 5],
            'q6': [0, 1, 2, 3, 4, 5],
            'q7': [5],
            'q8': [],
            'q9': [],
            'q10': [5],
            'q11': [0, 1, 2, 3, 4, 5],
            'q12': [5],
            'q13': [0, 1, 2, 3, 4, 5],
            'q14': [],
            'q15': [0, 1, 2, 3, 4, 5],
            'q16': [],
            'q17': [1, 2, 3, 4, 5],
            'q18': [0, 1, 2, 3, 4, 5],
            'q19': [3, 4, 5],
            'q20': [],
            'q21': [4, 5],
            'q22': [0, 1, 2, 3, 4, 5]}
    else:
        raise ValueError(bm)


args = Args().parse()
benchmark = args.benchmark

s3_adjustable = get_s3_adjustable(benchmark)
if bool(re.match(r"^q[0-9]+-[0-9]+$", args.q_sign)):
    q_sign = args.q_sign
else:
    try:
        q_sign = BenchmarkUtils.get_sampled_q_signs(benchmark)[int(args.q_sign) - 1]
    except:
        raise ValueError(args.q_sign)
if_aqe = False if args.if_aqe == 0 else True

out_header = f"{args.out_header}/{benchmark.lower()}_aqe_{'on' if if_aqe else 'off'}/{q_sign}"
for i in range(1, 23):
    names = [os.path.basename(x) for x in glob.glob(f"{out_header}/q{i}-1*") if x[-3:] != ".sh"]
    for name in names:
        ns = name.split(".")
        nss = ns[0].split(",")
        if int(nss[-2]) in s3_adjustable[f"q{i}"]:
            nss_new = ",".join(nss[:-2] + ["6", nss[-1]])
            name_new = ".".join([nss_new] + ns[1:])
            if not os.path.exists(f"q{i}-1/{name_new}"):
                cmd = f"ln -s {name} {out_header}/{name_new}"
                os.system(cmd)
