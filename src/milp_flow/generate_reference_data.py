# generate_reference_data.py

import hashlib
import json

import milp_flow.data_store as ds
from milp_flow.generate_value_data import generate_value_data


def get_data_files(data: dict) -> None:
    print("Reading data files...")

    # Wont need this since is_workerman_plantzone will be in exploration.json
    data["all_plantzones"] = ds.read_json("plantzone.json")

    # Still need this.
    data["plantzone_drops"] = ds.read_json("plantzone_drops.json")

    # Will use the all_lodging_storage.json from housecraft
    data["lodging_data"] = ds.read_json("all_lodging_storage.json")

    # Group is actually the affiliated region. Can use exploration.json for these.
    data["town_to_region"] = ds.read_json("town_node_translate.json")["tnk2tk"]
    data["region_to_town"] = ds.read_json("town_node_translate.json")["tk2tnk"]

    # Can use either Region strings or Exploration strings for this.
    data["region_to_townname"] = ds.read_json("warehouse_to_townname.json")

    # Will rename waypoint to exploration.
    data["waypoint_data"] = ds.read_json("exploration.json")

    # Won't need this since links are a part of exploration.json.
    data["waypoint_links"] = ds.read_json("deck_links.json")


def get_value_data(prices: dict, modifiers: dict, data: dict) -> None:
    print("Generating node values...")
    sha_filename = "values_hash.txt"
    current_sha = ds.read_text(sha_filename) if ds.is_file(sha_filename) else None

    encoded = json.dumps({"p": prices, "m": modifiers}).encode()
    latest_sha = hashlib.sha256(encoded).hexdigest()

    if latest_sha == current_sha:
        print("  ...re-using existing node values data.")
    else:
        generate_value_data(prices, modifiers)
        ds.path().joinpath(sha_filename).write_text(latest_sha)

    data["plant_values"] = ds.read_json("node_values_per_town.json")
    data["plants"] = data["plant_values"].keys()
    data["regions"] = data["plant_values"][list(data["plants"])[0]].keys()
    data["towns"] = [data["region_to_town"][w] for w in data["regions"]]
    data["max_ub"] = len(data["plants"])


def get_lodging_data(lodging: dict, data: dict) -> None:
    print("Generating lodging data...")
    for region, lodgings in data["lodging_data"].items():
        if region not in data["regions"]:
            continue
        townname = data["region_to_townname"][region]
        max_lodging = 1 + lodging[townname] + max([int(k) for k in lodgings.keys()])
        data["lodging_data"][region]["max_ub"] = max_lodging
        data["lodging_data"][region]["lodging_bonus"] = lodging[townname]


def generate_reference_data(
    config: dict, prices: dict, modifiers: dict, lodging: dict, force_active_node_ids: list[int]
) -> dict:
    data = {}
    data["config"] = config
    data["force_active_node_ids"] = force_active_node_ids
    get_data_files(data)
    get_value_data(prices, modifiers, data)
    get_lodging_data(lodging, data)
    return data
