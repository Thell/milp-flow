import json
import os


def read_data_json(data_sub_path, filename):
    filepath = os.path.join(os.path.dirname(__file__), "data", data_sub_path, filename)
    with open(filepath, "r") as file:
        return json.load(file)


def write_data_json(data_sub_path, filename, data):
    filepath = os.path.join(os.path.dirname(__file__), "data", data_sub_path, filename)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


def read_sample_json(filename):
    return read_data_json("sample", filename)


def write_sample_json(filename, data):
    return write_data_json("sample", filename, data)


def read_user_json(filename):
    return read_data_json("user", filename)


def write_user_json(filename, data):
    return write_data_json("user", filename, data)


def read_workerman_json(filename):
    return read_data_json("workerman", filename)


def read_empire_json(filename):
    if filename.startswith("sample_"):
        return read_sample_json(filename)
    return read_user_json(filename)


def write_empire_json(filename, data):
    if filename.startswith("sample_"):
        return write_sample_json(filename, data)
    return write_user_json(filename, data)
