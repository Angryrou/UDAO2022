import json
import os


class JsonHandler:

    @staticmethod
    def load_json(file):
        assert os.path.exists(file), FileNotFoundError(file)
        with open(file) as f:
            try:
                return json.load(f)
            except:
                raise Exception(f"{f} cannot be parsed as a JSON file")


