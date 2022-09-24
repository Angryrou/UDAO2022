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
        with open(path) as f:
            return pickle.load(f)

