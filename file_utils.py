import json
import os


def read_data_json(data_path, filename):
    filepath = os.path.join(os.path.dirname(__file__), "data", data_path, filename)
    with open(filepath, "r") as file:
        return json.load(file)


def read_user_json(filename):
    return read_data_json("user", filename)


def read_workerman_json(filename):
    return read_data_json("workerman", filename)


def read_sample_json(filename):
    return read_data_json("sample", filename)


def write_sample_json(filename, data):
    filepath = os.path.join(os.path.dirname(__file__), "data", "sample", filename)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)
