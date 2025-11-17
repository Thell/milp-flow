# experiment_fixed_prize.py
"""
- A fixed prize of '1' will yield the maximum terminal count of a budget.
- TODO: Figure out a good min terminal experiment
"""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from random import seed, randint

from loguru import logger

import data_store as ds

from api_common import set_logger
from generate_reference_data import generate_reference_data
from generate_graph_data import generate_graph_data
from optimize import optimize
from reduce_prize_data import reduce_prize_data
from reduce_transit_data import reduce_transit_data


def main():
    import json

    config: dict = {
        "name": "Empire",
        "budget": 25,
        "top_n": 6,
        "nearest_n": 7,
        "max_waypoint_ub": 17,
        "prune_prizes": False,
        "capacity_mode": "min",
        "transit_prune": False,
        "transit_reduce": False,
        "budget_equality": "eq",
    }
    solver_config = {
        "log_to_console": True,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_min_logging_interval": 60,
        "mip_share_incumbent_solution": False,
        "num_processes": 1,
        "random_seed": 0,  # set to 0 for internal HiGHS seed and seed + i for concurrent HiGHS instances
        "threads": 8,  # HiGHS internal threads
    }
    config["solver"] = solver_config

    modifiers = {}
    grindTakenList = []

    # Reproducability while using this specific price file
    # prices_file = r"C:\Users\thell\Downloads\custom_prices (22).json"
    prices_file = r"src\milp_flow\data\en_lta_prices.json"

    # NOTE: Since this is a fixed prize value experiment, the price list calculated values will all be the same
    prices = json.loads(Path(prices_file).read_text(encoding="utf-8"))["effectivePrices"]

    seed(prices_file)

    experiments = {}

    for capacity_mode in ["min", "max"]:
        data = {}
        config["capacity_mode"] = capacity_mode

        step = 25
        prev_terminal_count = 0

        for budget in range(50, 626, step):
            highs_seed = randint(0, 2**31 - 1)  # HiGHS ranges from 0 to 2^31 - 1
            config["solver"]["random_seed"] = highs_seed
            config["budget"] = budget
            # Do half of the step when budget > 200 for max capacity
            config["terminal_count_min_limit"] = 0  # use prev_terminal_count for max
            config["terminal_count_max_limit"] = prev_terminal_count + step

            lodging_specification = ds.read_json("lodging_specifications.json")[capacity_mode]
            print(f"== Budget: {budget} (highs_seed: {highs_seed}) ==")

            if not data:
                data = generate_reference_data(
                    config, prices, modifiers, lodging_specification, grindTakenList
                )

                # Modify all prizes values to be 1 for max terminals
                for t, entries in data["plant_values"].items():
                    for r, entry in entries.items():
                        entry["value"] = 1

                generate_graph_data(data)

            data["config"] = config

            solver_graph = deepcopy(data["G"].copy())
            solver_graph.attrs = deepcopy(data["G"].attrs)
            data["solver_graph"] = solver_graph

            if config["prune_prizes"]:
                reduce_prize_data(data)
            if config["transit_reduce"]:
                reduce_transit_data(data)

            model, terminal_sets = optimize(data, cr_pairs=True, prize_scale=False, prev_terminals_sets=None)

            highs_runtime = model.getRunTime()
            objective = model.getObjectiveValue()
            root_counts = Counter(terminal_sets.values())
            terminal_counts = len(terminal_sets)
            prev_terminal_count = terminal_counts

            logger.success(
                f"{budget=} {capacity_mode=} {objective=:.2f} {terminal_counts=} {root_counts=} {highs_runtime=:.2f}\n"
                f"{terminal_sets=}"
            )
            logger.info("-" * 80)

            # Store context information for experiements
            experiment_data = {
                "budget": budget,
                "highs_seed": highs_seed,
                "highs_runtime": highs_runtime,
                "objective": model.getObjectiveValue(),
                "terminal_sets": terminal_sets,
            }
            experiment_key = f"{capacity_mode}-{budget}"
            experiments[experiment_key] = experiment_data

    out_file = "fixed_prize_experiment.json"
    with open(out_file, "w") as f:
        json.dump(experiments, f, indent=4)

    print(f"Output written to {out_file}")

    return


if __name__ == "__main__":
    config = {}
    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    set_logger(config)
    main()
