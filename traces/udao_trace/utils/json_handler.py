import json
import os
import urllib.request


class JsonHandler:

    @staticmethod
    def load_json(file):
        assert os.path.exists(file), FileNotFoundError(file)
        with open(file) as f:
            try:
                return json.load(f)
            except:
                raise Exception(f"{f} cannot be parsed as a JSON file")

    @staticmethod
    def load_json_from_str(s: str):
        return json.loads(s)

    @staticmethod
    def print_dict(d: dict):
        print(json.dumps(d, indent=2))

    @staticmethod
    def load_json_from_url(url_str, timeout=10):
        try:
            with urllib.request.urlopen(url_str, timeout=timeout) as url:
                data = json.load(url)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            raise Exception(f"failed to load from {url_str}")
        return data

    @staticmethod
    def extract_json_list(l, keys: list):
        for key in keys:
            for l_ in l:
                assert key in l_, f"{key} is not in {l_}"
        return [[l_[key] for l_ in l] for key in keys]

    @staticmethod
    def save_json(d, des_file):
        json_data = json.dumps(d, indent=2)
        with open(des_file, "w") as f:
            f.write(json_data)

    @staticmethod
    def dump2str(d, indent: int = None):
        return json.dumps(d, indent=indent)
