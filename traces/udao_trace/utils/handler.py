import json
import os
import pickle
import traceback

from typing import List

from . import ClusterName


class JsonHandler:

    @staticmethod
    def load_json(file):
        assert os.path.exists(file), FileNotFoundError(file)
        with open(file) as f:
            try:
                return json.load(f)
            except:
                raise Exception(f"{f} cannot be parsed as a JSON file")


class PickleHandler(object):

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


class FileHandler:

    @staticmethod
    def create_script(header, file, content):
        os.makedirs(header, exist_ok=True)
        with open(f"{header}/{file}", "w") as f:
            f.write(content)


def error_handler(e):
    def error_handler(e):
        print('An error occurred:')

        # Print the exception type
        print(f"Exception Type: {type(e).__name__}")

        # Print the exception message
        print(f"Exception Message: {str(e)}")

        # Print the traceback information
        print("Traceback:")
        traceback.print_exc()
