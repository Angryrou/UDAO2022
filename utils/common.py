# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Created at 9/16/22

import os, json, pickle


class JsonUtils(object):

    @staticmethod
    def load_json(file):
        assert os.path.exists(file), FileNotFoundError(file)
        with open(file) as f:
            try:
                return json.load(f)
            except:
                raise Exception(f"{f} cannot be parsed as a JSON file")

    @staticmethod
    def print_dict(d: dict):
        print(json.dumps(d, indent=2))


class PickleUtils(object):

    @staticmethod
    def save(obj, header, file_name, overwrite=False):
        path = f"{header}/{file_name}"
        if os.path.exists(path) and not overwrite:
            return f"{path} already exists"
        os.makedirs(header, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(header, file_name):
        path = f"{header}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return pickle.load(f)

class BenchmarkUtils(object):

    @staticmethod
    def get(benchmark: str):
        if benchmark.lower() == "tpch":
            return [str(i) for i in range(1, 23)]
        elif benchmark.lower() == "tpcds":
            return "1 2 3 4 5 6 7 8 9 10 11 12 13 14a 14b 15 16 17 18 19 20 21 22 23a 23b 24a 24b 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39a 39b 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99".split(" ")
        else:
            raise ValueError(f"{benchmark} is not supported")

    @staticmethod
    def get_workers(benchmark: str):
        if benchmark.lower() == "tpch":
            return ["node2", "node3", "node4", "node5", "node6"]
        elif benchmark.lower() == "tpcds":
            return ["node8", "node9", "node10", "node11", "node12"]
        else:
            raise ValueError(f"{benchmark} is not supported")