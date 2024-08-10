"""Parse optimizer output for workerman import."""

from collections import Counter
import json
import locale
import logging
import os
from typing import Dict, Any

import natsort
import networkx as nx
from tabulate import tabulate

from generate_empire_data import generate_empire_data, GraphData, get_reference_data


def read_solution_vars_json(filepath):
    """Read and return json as dict."""
    with open(os.path.join(os.path.dirname(__file__), filepath), "r") as file:
        data = json.load(file)
    return data


def write_workerman_json(filename, data):
    filepath = os.path.join(os.path.dirname(__file__), "workerman_output", filename)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)
        print(f"Workerman json written to: {filepath}")


def get_workerman_json(lodging_bonuses, workers):
    """Populate and return a standard 'dummy' instance of the workerman dict."""
    workerman_json = {
        "activateAncado": False,
        "lodgingP2W": lodging_bonuses,
        "userWorkers": workers,
        "farmingEnable": False,
        "farmingProfit": 0,
        "farmingBareProfit": 0,
        "grindTakenList": [],
    }
    return workerman_json


def get_workerman_lodging_bonuses(lodging_bonus):
    """Populate and return a 'dummy' instance of the workerman lodging_bonuses dict."""
    # TODO: read from a user controlled lodging config json file with this as fallback.
    lodging_bonuses = {
        "5": 0,
        "32": 0,
        "52": 0,
        "77": 0,
        "88": 0,
        "107": 0,
        "120": 0,
        "126": 0,
        "182": 0,
        "202": 0,
        "218": 0,
        "221": 0,
        "229": 0,
        "601": 0,
        "605": 0,
        "619": 0,
        "693": 0,
        "706": 0,
        "735": 0,
        "873": 0,
        "955": 0,
        "1000": 0,
        "1124": 0,
        "1210": 0,
        "1219": 0,
        "1246": 0,
        "1375": 0,
        "1382": 0,
    }
    return {k: lodging_bonus for k, v in lodging_bonuses.items()}


def get_workerman_storage_stashes():
    """Populate and return a 'dummy' instance of origin/warehouse stash locations json."""
    # TODO: read from a user controlled stashes config json file with this as fallback.
    pass


def make_workerman_worker(town_id: int, origin_id: int, worker_data: Dict[str, Any], stash_id: int):
    """Populate and return a 'dummy' instance of a workerman user worker dict."""
    worker = {
        "tnk": town_id,
        "charkey": str(worker_data["charkey"]),
        "label": "default",
        "level": 40,
        "wspdSheet": worker_data["wspd"],
        "mspdSheet": worker_data["mspd"],
        "luckSheet": worker_data["luck"],
        "skills": [int(s) for s in worker_data["skills"]],
        "job": {"kind": "plantzone", "pzk": int(origin_id), "storage": stash_id},
    }
    return worker


def order_workerman_workers(graph, user_workers: list[Dict[str, Any]], solution_distances):
    """Order user workers into import order for correct workerman paths construction."""

    # Order by shortest origin -> town paths to break ties by nearest nodes.
    distance_indices = zip(list(range(len(solution_distances))), solution_distances)
    distance_indices = sorted(distance_indices, key=lambda x: x[1])
    workerman_user_workers = [user_workers[i] for i, _ in distance_indices]

    # Iterative ordering of user workers by shortest paths with weight removal on used arcs.
    ordered_workers = []
    while workerman_user_workers:
        distances = []
        all_pairs = dict(nx.all_pairs_bellman_ford_path_length(graph, weight="weight"))
        for worker in workerman_user_workers:
            distance = all_pairs[str(worker["tnk"])][str(worker["job"]["pzk"])]
            distances.append(distance)
        min_value = min(distances)
        min_indice = distances.index(min_value)
        worker = workerman_user_workers[min_indice]
        ordered_workers.append(worker)
        workerman_user_workers.pop(min_indice)

        short_path = nx.shortest_path(graph, str(worker["tnk"]), str(worker["job"]["pzk"]), "weight")
        for s, d in zip(short_path, short_path[1:]):
            if graph.edges[(s, d)]["weight"] >= 1:
                for edge in graph.in_edges(d):
                    graph.edges[edge]["weight"] = 0
                break

    return ordered_workers


def print_summary(outputs, counts: Dict[str, int], costs: Dict[str, int], total_value: float):
    """Print town, origin, worker summary report."""
    outputs = natsort.natsorted(outputs, key=lambda x: (x["warehouse"], x["node"]))
    colalign = ("right", "right", "left", "right", "right")
    print(tabulate(outputs, headers="keys", colalign=colalign))
    print("Worker Nodes:", counts["origins"], "cost:", costs["origins"])
    print("Waypoints:", counts["waypoints"], "cost:", costs["waypoints"])
    print("Lodgings cost:", costs["lodgings"])
    print("Total Cost:", sum(c for c in costs.values()))
    print("Value:", locale.currency(round(total_value), grouping=True, symbol=True)[:-3])


def generate_graph(graph_data: GraphData, solution_data: Dict[str, Any]):
    graph = nx.DiGraph()
    exclude_keywords = ["lodging", "supply", "source", "sink", "warehouse"]

    for var_key in solution_data["solution_vars"].keys():
        is_edge = "_to_" in var_key
        exclude = any(keyword in var_key for keyword in exclude_keywords)
        if not var_key.startswith("HasLoad_") or not is_edge or exclude:
            continue

        source, destination = var_key.split("_to_")
        source = source.replace("HasLoad_", "")
        weight = graph_data["nodes"][source].cost
        graph.add_edge(destination.split("_")[1], source.split("_")[1], weight=weight)

    return graph


def process_solution_vars_file(solution_data):
    """Parse solution vars file."""

    ref_data = get_reference_data(solution_data["config"])
    graph_data = generate_empire_data(solution_data["config"])

    graph = generate_graph(graph_data, solution_data)
    all_pairs = dict(nx.all_pairs_bellman_ford_path_length(graph, weight="weight"))

    lodging_vars = {}
    origin_vars = {}
    waypoint_vars = {}
    for k, v in solution_data["solution_vars"].items():
        if k.startswith("Load_lodging") and "_to_" not in k:
            lodging_vars[k.replace("Load_", "")] = v
        elif k.startswith("LoadForWarehouse") and "_to_" not in k and "_at_origin_" in k:
            origin_vars[k.split("_")[4]] = k.split("_")[1]
        elif k.startswith("Load_waypoint") and "_to_" not in k:
            waypoint_vars[k.replace("Load_", "")] = v

    calculated_value = 0
    distances = []
    origin_cost = 0
    outputs = []
    town_ids = set()
    workerman_user_workers = []
    warehouse_ranks = []
    for k, v in origin_vars.items():
        town_id = ref_data["warehouse_to_town"][v]

        town_ids.add(town_id)
        distances.append(all_pairs[town_id][k])

        origin = graph_data["nodes"][f"origin_{k}"]
        worker_data = origin.warehouse_values[v]["worker_data"]
        user_worker = make_workerman_worker(int(town_id), int(origin.id), worker_data, int(town_id))
        workerman_user_workers.append(user_worker)

        value = origin.warehouse_values[v]["value"]
        worker = origin.warehouse_values[v]["worker"]
        warehouse_rank = list(origin.warehouse_values.keys()).index(v) + 1
        warehouse_ranks.append(warehouse_rank)

        origin_cost += origin.cost
        calculated_value += value

        output = {
            "warehouse": v,
            "node": origin.id,
            "worker": worker,
            "value": locale.currency(round(value), grouping=True, symbol=True)[:-3],
            "value_rank": warehouse_rank,
        }
        outputs.append(output)

    counts = {"origins": len(origin_vars), "waypoints": len(waypoint_vars)}
    costs = {
        "lodgings": sum(graph_data["nodes"][k].cost for k in lodging_vars.keys()),
        "origins": origin_cost,
        "waypoints": sum(graph_data["nodes"][k].cost for k in waypoint_vars.keys()),
    }
    print_summary(outputs, counts, costs, calculated_value)
    print("Warehouse rank counts:", {k: v for k, v in Counter(warehouse_ranks).items()})

    lodging_bonuses = get_workerman_lodging_bonuses(solution_data["config"]["lodging_bonus"])
    workerman_ordered_workers = order_workerman_workers(graph, workerman_user_workers, distances)
    workerman_json = get_workerman_json(lodging_bonuses, workerman_ordered_workers)

    return workerman_json


def main():
    filepath = "highs_output/solution-vars"
    input_files = [os.path.join(filepath, f) for f in os.listdir(filepath)]

    for filepath in input_files:
        if "NoCostObj_mc501_lb3_tn4_nn5_wc25" not in filepath:
            continue
        logging.info("Processing:", filepath.split("/")[-1])

        solution_data = read_solution_vars_json(filepath)
        workerman_json = process_solution_vars_file(solution_data)

        outfile = f"workerman_{filepath.split("/")[-1]}"
        write_workerman_json(outfile, workerman_json)


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")
    main()
