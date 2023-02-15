# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: provide args for different training purpose
#
# Created at 03/01/2023

import argparse

class ArgsBase():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="tpch")
        self.parser.add_argument("-s", "--scale-factor", type=int, default=100)
        self.parser.add_argument("-g", "--granularity", type=str, default="Q",
                                 help="Q|QS, Q for Query-level, QS for QueryStage-level")
        self.parser.add_argument("-d", "--data-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--debug", type=int, default=0)

    def parse(self):
        return self.parser.parse_args()


class ArgsRecoQ(ArgsBase):
    def __init__(self):
        super(ArgsRecoQ, self).__init__()
        self.parser.add_argument("--ch1-type", type=str, default="on", help="off|on")
        self.parser.add_argument("--ch1-cbo", type=str, default="off", help="off|on")
        self.parser.add_argument("--ch1-enc", type=str, default="off", help="off|d2v|w2v")
        self.parser.add_argument("--ch2", type=str, default="on", help="off|on, input data meta")
        self.parser.add_argument("--ch3", type=str, default="on", help="off|on, system states")
        self.parser.add_argument("--ch4", type=str, default="on", help="off|on, configuration")
        self.parser.add_argument("--obj", type=str, default="latency")
        self.parser.add_argument("--model-name", type=str, default="GTN")
        self.parser.add_argument("--q-signs", type=str, default=None)
        self.parser.add_argument("--ckp-sign", type=str, default="b7698e80492e5d72")
        self.parser.add_argument("--n-samples", type=int, default=5000)
        self.parser.add_argument("--n-weights", type=int, default=1000)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--query-header", type=str, default="resources/tpch-kit/spark-sqls")
        self.parser.add_argument("--gpu", type=str, default="-1")


class ArgsRecoQRun(ArgsRecoQ):
    def __init__(self):
        super(ArgsRecoQRun, self).__init__()
        self.parser.add_argument("--algo", type=str, required=True, help="robust|vc")
        self.parser.add_argument("--moo", type=str, default="ws", help="ws|bf")
        self.parser.add_argument("--alpha", type=int, default=0, help="for robust optimization")
        self.parser.add_argument("--if-aqe", type=int, default=0)
        self.parser.add_argument("--worker", type=str, default="debug")

class ArgsGTN(ArgsBase):
    def __init__(self):
        super(ArgsGTN, self).__init__()
        # data_params
        self.parser.add_argument("--ch1-type", type=str, default="on", help="off|on")
        self.parser.add_argument("--ch1-cbo", type=str, default="off", help="off|on")
        self.parser.add_argument("--ch1-enc", type=str, default="off", help="off|d2v|w2v")
        self.parser.add_argument("--ch2", type=str, default="on", help="off|on, input data meta")
        self.parser.add_argument("--ch3", type=str, default="on", help="off|on, system states")
        self.parser.add_argument("--ch4", type=str, default="on", help="off|on, configuration")
        self.parser.add_argument("--obj", type=str, default="latency")
        self.parser.add_argument("--model-name", type=str, default="GTN")

        # learning_params
        self.parser.add_argument("--gpu", type=str, default="-1")
        self.parser.add_argument("--nworkers", type=int, default=0)
        self.parser.add_argument("--bs", type=int, default=64)
        self.parser.add_argument("--epochs", type=int, default=2)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument('--init-lr', type=float)
        self.parser.add_argument('--min-lr', type=float)
        self.parser.add_argument('--weight-decay', type=float)
        self.parser.add_argument('--ckp-interval', type=int)
        self.parser.add_argument('--loss-type', type=str)

        # net_params
        self.parser.add_argument("--ped", type=int, help="positional encoding dimension", default=8)
        self.parser.add_argument("--L-gtn", type=int, help="nlayers of gtn", default=4)
        self.parser.add_argument("--L-mlp", type=int, help="nlayers of mlp", default=4)
        self.parser.add_argument("--n-heads", type=int)
        self.parser.add_argument('--hidden-dim', type=int)
        self.parser.add_argument('--out-dim', type=int)
        self.parser.add_argument('--mlp-dim', type=int)
        self.parser.add_argument('--residual', type=int)
        self.parser.add_argument('--readout', type=str)
        self.parser.add_argument('--dropout', type=float)
        self.parser.add_argument('--dropout2', type=float)
        self.parser.add_argument('--batch-norm', type=int)
        self.parser.add_argument('--layer-norm', type=int)
        self.parser.add_argument("--ch1-type-dim", type=int)
        self.parser.add_argument("--ch1-cbo-dim", type=int)
        self.parser.add_argument("--ch1-enc-dim", type=int)
        self.parser.add_argument("--out-norm", type=str)