# main.py

from math import inf
from pathlib import Path

from loguru import logger

import data_store as ds
from generate_reference_data import generate_reference_data
from generate_graph_data import generate_graph_data
from generate_workerman_data import generate_workerman_data
from optimize import optimize

from api_common import set_logger


min_lodging = {
    "Velia": 0,
    "Heidel": 0,
    "Glish": 0,
    "Calpheon City": 0,
    "Olvia": 0,
    "Keplan": 0,
    "Port Epheria": 0,
    "Trent": 0,
    "Iliya Island": 0,
    "Altinova": 0,
    "Tarif": 0,
    "Valencia City": 0,
    "Shakatu": 0,
    "Sand Grain Bazaar": 0,
    "Ancado Inner Harbor": 0,
    "Arehaza": 0,
    "Old Wisdom Tree": 0,
    "Grána": 0,
    "Duvencrune": 0,
    "O'draxxia": 0,
    "Eilton": 0,
    "Dalbeol Village": 0,
    "Nampo's Moodle Village": 0,
    "Nopsae's Byeot County": 0,
    "Asparkan": 0,
    "Muzgar": 0,
    "Yukjo Street": 0,
    "Godu Village": 0,
    "Bukpo": 0,
    "Hakinza Sanctuary": 0,
}
max_lodging = {
    "Velia": 7,
    "Heidel": 7,
    "Glish": 6,
    "Calpheon City": 7,
    "Olvia": 6,
    "Keplan": 6,
    "Port Epheria": 5,
    "Trent": 6,
    "Iliya Island": 0,
    "Altinova": 8,
    "Tarif": 6,
    "Valencia City": 7,
    "Shakatu": 6,
    "Sand Grain Bazaar": 5,
    "Ancado Inner Harbor": 0,
    "Arehaza": 5,
    "Old Wisdom Tree": 6,
    "Grána": 7,
    "Duvencrune": 7,
    "O'draxxia": 9,
    "Eilton": 6,
    "Dalbeol Village": 6,
    "Nampo's Moodle Village": 6,
    "Nopsae's Byeot County": 6,
    "Asparkan": 5,
    "Muzgar": 5,
    "Yukjo Street": 5,
    "Godu Village": 6,
    "Bukpo": 6,
    "Hakinza Sanctuary": 7,
}
lodging_specifications = {
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


def main():
    # import sys
    # sys.activate_stack_trampoline("perf")

    import datetime
    import json
    import tkinter as tk
    from tkinter import filedialog

    global min_lodging, max_lodging

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    prices = json.loads(Path(file_path).read_text(encoding="utf-8"))["effectivePrices"]

    modifiers = {}
    grindTakenList = []
    lodging = min_lodging

    today = datetime.datetime.now().strftime("%y%m%d_%H%M")
    logfile = Path(ds.path()).parent.parent.parent.joinpath("zzz_out_new", "logs")
    workerman_file = Path(ds.path()).parent.parent.parent.joinpath("zzz_out_new", "workerman")

    # Empirical testing (prior to adding the clustering logic) shows that the best parameters are:
    # For < 550 budget:
    #   "do_prune" = True
    #   "do_reduce" = True
    #   "cr_pairs" = True
    #   "prize_scale" = True
    #   "root_cost" = False
    # and for > 550 budget:
    #   "do_prune" = True
    #   "do_reduce" = True
    #   "cr_pairs" = False
    #   "prize_scale" = True
    #   "root_cost" = True
    arg_sets = [
        {
            "do_prune": True,
            "do_reduce": True,
            "cr_pairs": True,
            "prize_scale:": True,
            "root_cost": False,
        },
        {
            "do_prune": True,
            "do_reduce": True,
            "cr_pairs": False,
            "prize_scale:": True,
            "root_cost": True,
        },
        {
            "do_prune": True,
            "do_reduce": True,
            "cr_pairs": True,
            "prize_scale:": True,
            "root_cost": True,
        },
        {
            "do_prune": True,
            "do_reduce": True,
            "cr_pairs": False,
            "prize_scale:": True,
            "root_cost": False,
        },
        {
            "do_prune": True,
            "do_reduce": True,
            "cr_pairs": False,
            "prize_scale:": False,
            "root_cost": True,
        },
        {
            "do_prune": True,
            "do_reduce": True,
            "cr_pairs": False,
            "prize_scale:": False,
            "root_cost": False,
        },
    ]

    # all combos of arg_set options
    arg_sets = []
    for cr_pairs in [True, False]:
        for prize_scale in [True, False]:
            for root_cost in [True, False]:
                arg_sets.append({
                    "do_prune": True,
                    "do_reduce": True,
                    "cr_pairs": cr_pairs,
                    "prize_scale": prize_scale,
                    "root_cost": root_cost,
                    "basin_type": 2,
                })
    # Remove leading arg_sets until True, False, False is found
    # arg_sets = arg_sets[7:]
    # arg_sets = [arg_sets[0]]  # test using just the first arg_set

    budgets = range(625, 650, 25)
    for budget in budgets:
        args = {
            "do_prune": True,
            "do_reduce": True,
            "cr_pairs": False,
            "prize_scale": False,
            "root_cost": False,
            "basin_type": 2,
        }
        if budget >= 300:
            args["root_cost"] = True

        # Initial base test
        print(f"== Budget: {budget} ==")
        config: dict = {
            "name": "Empire",
            "budget": budget,
            "top_n": 6,
            "nearest_n": 7,
            "waypoint_ub": 17,
        }
        solver_config = {
            "num_threads": 7,
            "mip_rel_gap": 1e-4,
            "mip_feasibility_tolerance": 1e-4,
            "mip_heuristic_run_root_reduced_cost": args["root_cost"],
            "primal_feasibility_tolerance": 1e-4,
            "time_limit": inf,
            "random_seed": 0,
            "log_to_console": False if budget < 20 else True,
            "log_file": "",
            # "highs_analysis_level": 128,
        }
        config["solver"] = solver_config

        data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
        graph_data = generate_graph_data(data, args["do_prune"], args["do_reduce"], args["basin_type"])
        logger.info(f"  Budget: {budget} {args=}")
        model, terminal_sets = optimize(graph_data, args["cr_pairs"], args["prize_scale"])
        logger.info(f"  Budget: {budget} {args=}")


if __name__ == "__main__":
    config = {}
    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    set_logger(config)
    main()
