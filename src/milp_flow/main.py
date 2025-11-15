# main.py

from copy import deepcopy
from math import floor, inf
from pathlib import Path
from random import seed, randint

from loguru import logger

import data_store as ds
from generate_reference_data import generate_reference_data
from generate_graph_data import generate_graph_data
from generate_workerman_data import generate_workerman_data
from optimize import optimize

from api_common import set_logger
from reduce_prize_data import reduce_prize_data
from reduce_transit_data import reduce_transit_data

min_lodging_specifications = {
    "Velia": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Heidel": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Glish": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Calpheon City": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Olvia": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Keplan": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Port Epheria": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Trent": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Iliya Island": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 0},
    "Altinova": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 8},
    "Tarif": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Valencia City": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Shakatu": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Sand Grain Bazaar": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Ancado Inner Harbor": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 0},
    "Arehaza": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Old Wisdom Tree": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Grána": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Duvencrune": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "O'draxxia": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 9},
    "Eilton": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Dalbeol Village": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Nampo's Moodle Village": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Nopsae's Byeot County": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Asparkan": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Muzgar": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Yukjo Street": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Godu Village": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Bukpo": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Hakinza Sanctuary": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
}
max_lodging_specifications = {
    "Velia": {"bonus": 7, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Heidel": {"bonus": 7, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Glish": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Calpheon City": {"bonus": 7, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Olvia": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Keplan": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Port Epheria": {"bonus": 5, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Trent": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Iliya Island": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 0},
    "Altinova": {"bonus": 8, "reserved": 0, "prepaid": 0, "bonus_ub": 8},
    "Tarif": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Valencia City": {"bonus": 7, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Shakatu": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Sand Grain Bazaar": {"bonus": 5, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Ancado Inner Harbor": {"bonus": 0, "reserved": 0, "prepaid": 0, "bonus_ub": 0},
    "Arehaza": {"bonus": 5, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Old Wisdom Tree": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Grána": {"bonus": 7, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "Duvencrune": {"bonus": 7, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
    "O'draxxia": {"bonus": 9, "reserved": 0, "prepaid": 0, "bonus_ub": 9},
    "Eilton": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Dalbeol Village": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Nampo's Moodle Village": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Nopsae's Byeot County": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Asparkan": {"bonus": 5, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Muzgar": {"bonus": 5, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Yukjo Street": {"bonus": 5, "reserved": 0, "prepaid": 0, "bonus_ub": 5},
    "Godu Village": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Bukpo": {"bonus": 6, "reserved": 0, "prepaid": 0, "bonus_ub": 6},
    "Hakinza Sanctuary": {"bonus": 7, "reserved": 0, "prepaid": 0, "bonus_ub": 7},
}


def main():
    # import sys
    # sys.activate_stack_trampoline("perf")

    import datetime
    import json
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    prices = json.loads(Path(file_path).read_text(encoding="utf-8"))["effectivePrices"]

    modifiers = {}
    grindTakenList = []
    lodging = min_lodging_specifications

    today = datetime.datetime.now().strftime("%y%m%d_%H%M")
    logfile = Path(ds.path()).parent.parent.parent.joinpath("zzz_out_new", "logs")
    workerman_file = Path(ds.path()).parent.parent.parent.joinpath("zzz_out_new", "workerman")

    # all combos of arg_set options
    arg_sets = []
    for cr_pairs in [False, False]:
        for prize_scale in [True, False]:
            for root_cost in [False, False]:
                arg_sets.append({
                    "do_prune": True,
                    "do_reduce": True,
                    "cr_pairs": cr_pairs,
                    "prize_scale": prize_scale,
                    "root_cost": root_cost,
                    "basin_type": 2,
                })

    # Reproducability while using this specific price file
    # seed(r"C:\Users\thell\Downloads\custom_prices (22).json")
    seed(r"src\milp_flow\data\en_lta_prices.json")

    budgets = range(50, 625, 25)
    SKIP_TO = (550, 1, 8, False, False, True, False)
    FOUND = False
    break_out = False
    data = {}
    for budget in budgets:
        # Each budget gets its own seed so we can compare performance between arg combinations
        highs_seed = randint(0, 2**31 - 1)  # HiGHS ranges from 0 to 2^31 - 1

        print(f"== Budget: {budget} (highs_seed: {highs_seed}) ==")
        # for num_processes in [1, 7]:
        for num_processes in [1]:
            # for num_threads in [1, 8]:
            for num_threads in [8]:
                if num_processes == 1 and num_threads == 1:  # regularly worse than (1, 8) by milliseconds
                    continue
                for do_solution_share in [True, False]:
                    if num_processes == 1 and do_solution_share:  # invalid combination
                        continue

                    for args in [arg_sets[0]]:
                        if break_out:
                            break

                        args["processor"] = (num_processes, num_threads)
                        test_args = (
                            budget,
                            num_processes,
                            num_threads,
                            do_solution_share,
                            args["cr_pairs"],
                            args["prize_scale"],
                            args["root_cost"],
                        )
                        FOUND = True if test_args == SKIP_TO else FOUND
                        if not FOUND:
                            # advance rng by the count of clones missed
                            if num_processes > 1:
                                seed(highs_seed)
                                _ = [randint(0, 2**31 - 1) for _ in range(num_processes - 1)]
                                # logger.info(
                                #     f"  skipping seeds: {[randint(0, 2**31 - 1) for _ in range(num_processes - 1)]}"
                                # )
                            continue

                        terminal_count_max_limit = round(
                            3.98980981766944
                            + 0.552618716162738 * budget
                            + 2.98921916296481e-7 * budget**3
                            - 0.000503850929485882 * budget**2
                        )
                        config: dict = {
                            "name": "Empire",
                            "budget": budget,
                            "top_n": 10,
                            "nearest_n": 10,
                            "max_waypoint_ub": 25,
                            "prune_prizes": True,  # This currently discards optimal solution on budgets >= 550 even with .5 threshold
                            "capacity_mode": "min",
                            "transit_prune": True,
                            "transit_reduce": True,
                            "transit_basin_type": 2,
                            "transit_prune_low_asp": False,
                            "terminal_count_min_limit": floor(terminal_count_max_limit * 0.60),
                            "terminal_count_max_limit": terminal_count_max_limit,
                            "prize_pruning_threshold_factors": {
                                "min": {"only_child": 0.5, "dominant": 0.5, "protected": 0.5},
                                "max": {"only_child": 0.5, "dominant": 0.5, "protected": 0.5},
                            },
                        }
                        solver_config = {
                            # "highs_analysis_level": 128,
                            "log_to_console": False if budget <= 300 else True,
                            "mip_feasibility_tolerance": 1e-4,
                            "mip_heuristic_run_root_reduced_cost": False,
                            "mip_min_logging_interval": 60,
                            "mip_rel_gap": 1e-4,
                            "mip_share_incumbent_solution": do_solution_share,
                            "num_processes": num_processes,  # concurrent processes
                            # "primal_feasibility_tolerance": 1e-4,
                            "random_seed": highs_seed,  # set to 0 for internal HiGHS seed and seed + i for concurrent HiGHS instances
                            "threads": num_threads,  # HiGHS internal threads
                            "time_limit": inf,
                        }
                        config["solver"] = solver_config

                        logger.info("-" * 80)
                        logger.info(f"  Budget: {budget} shared: {do_solution_share} {args=}")
                        logger.info(f"  {config=}")

                        if not data:
                            data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
                            generate_graph_data(data)

                        data["config"] = config
                        solver_graph = deepcopy(data["G"].copy())
                        solver_graph.attrs = deepcopy(data["G"].attrs)
                        data["solver_graph"] = solver_graph

                        reduce_prize_data(data)
                        reduce_transit_data(data)

                        model, terminal_sets = optimize(data, args["cr_pairs"], args["prize_scale"])

                        highs_runtime = model.getRunTime()
                        logger.success(
                            f"Elapsed Time: {highs_runtime:.2f}  Budget: {budget} Objective: {model.getObjectiveValue()} Shared: {do_solution_share}\n{args=}"
                        )
                        logger.info("-" * 80)
                        # break_out = True


if __name__ == "__main__":
    config = {}
    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    set_logger(config)
    main()
