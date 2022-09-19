# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Created at 9/16/22

import os, json


def load_json(file):
    assert os.path.exists(file), FileNotFoundError(file)
    with open(file) as f:
        try:
            return json.load(f)
        except:
            raise Exception(f"{f} cannot be parsed as a JSON file")
