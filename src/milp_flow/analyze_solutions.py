from copy import deepcopy
import json
from math import inf
from typing import Any
import re

import pandas as pd
from loguru import logger
from rustworkx import PyDiGraph

import data_store as ds
from api_common import set_logger
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
from terminal_prize_utils import Category


def main(config: dict[str, Any], kwargs: dict[str, Any]):
    solutions_json = kwargs["solutions_json"]
    prices = kwargs["prices"]
    modifiers = kwargs["modifiers"]
    grind_taken_list = kwargs["grind_taken_list"]
    min_lodging = kwargs["min_lodging"]
    max_lodging = kwargs["max_lodging"]

    budgets = set()
    for key in solutions_json:
        match = re.match(r".*(min|max)-(\d+)$", key)
        if match:
            _, budget_str = match.groups()
            budgets.add(int(budget_str))

    all_analyses = []

    # NOTE: Reference graph only needs to be generated once for each capacity mode!
    for cap in ["min", "max"]:
        lodging = min_lodging if cap == "min" else max_lodging
        config["capacity_mode"] = cap

        data = generate_reference_data(config, prices, modifiers, lodging, grind_taken_list)
        generate_graph_data(data)

        data["config"] = config

        solver_graph = deepcopy(data["G"].copy())
        solver_graph.attrs = deepcopy(data["G"].attrs)
        data["solver_graph"] = solver_graph

        logger.info(f"=== Running analysis for cap={cap} (all budgets) ===")
        df = run_analysis(data, cap)
        if df is not None and not df.empty:
            all_analyses.append(df)

    if all_analyses:
        all_df = pd.concat(all_analyses, ignore_index=True)
        print(all_df.to_csv(index=False))
        print(all_df)
    else:
        raise ValueError("No valid budget/cap keys found in JSON.")


def run_analysis(data: dict[str, Any], cap_mode: str) -> pd.DataFrame | None:
    """Run analysis for all budgets for cap_mode in JSON using the provided solver_graph."""
    solver_graph = data["solver_graph"]
    solutions_json = ds.read_json("highs_solutions.json")
    if not solutions_json:
        logger.warning("No solutions JSON loaded.")
        return None

    prize_ranks = data["prize_ranks"]

    analyses = []
    for budget in range(50, 601, 25):
        key = f"highs-{cap_mode}-{budget}"
        solution = solutions_json[key][0]
        if solution:
            df = analyze_used_prizes(
                solution,
                solver_graph,
                budget,
                cap_mode,
                prize_ranks,
            )
            if not df.empty:
                analyses.append(df)

    if analyses:
        all_df = pd.concat(analyses, ignore_index=True)
        return all_df
    else:
        raise ValueError("No valid budget/cap keys found in JSON.")


def analyze_used_prizes(
    solution: dict[str, Any],
    solver_graph: PyDiGraph,
    budget: int,
    capacity: str,
    prize_ranks: dict[int, dict[int, dict[str, Any]]],
) -> pd.DataFrame:
    """Identify min rank of used prizes for each terminal category."""
    terminal_sets_str = solution.get("terminal_sets", "{}")
    terminal_sets = json.loads(terminal_sets_str) if isinstance(terminal_sets_str, str) else terminal_sets_str
    if not terminal_sets:
        raise ValueError("No terminal_sets in JSON.")

    node_key_by_index = solver_graph.attrs["node_key_by_index"]
    terminal_sets = {
        node_key_by_index.inv[int(k)]: node_key_by_index.inv[int(v)] for k, v in terminal_sets.items()
    }

    min_rankings = {
        "budget": budget,
        "capacity": capacity,
        "only_child": inf,
        "dominant": inf,
        "protected": inf,
        "prize_rank_by_root": -inf,
        "prize_rank_by_terminal": -inf,
    }
    for t, r in terminal_sets.items():
        t_node = solver_graph[t]
        rank_info = prize_ranks.get(t, {}).get(r, {})
        global_pct_rank = rank_info["terminal_prize_net_path_cost_global_pct"]

        prize_rank_by_terminal = rank_info["root_prize_value_rank_by_terminal_view"]
        min_rankings["prize_rank_by_terminal"] = max(
            min_rankings["prize_rank_by_terminal"], prize_rank_by_terminal
        )
        prize_rank_by_root = rank_info["terminal_prize_value_rank_by_root_view"]
        min_rankings["prize_rank_by_root"] = max(min_rankings["prize_rank_by_root"], prize_rank_by_root)

        category = t_node["category"]
        if category == Category.ONLY_CHILD.value:
            min_rankings["only_child"] = min(min_rankings["only_child"], global_pct_rank)
        elif category == Category.DOMINANT.value:
            min_rankings["dominant"] = min(min_rankings["dominant"], global_pct_rank)
        elif category == Category.PROTECTED.value:
            min_rankings["protected"] = min(min_rankings["protected"], global_pct_rank)
            # Print warning if protected terminal has no protector present in solution
            protector = t_node["protector"]
            if protector not in terminal_sets:
                logger.warning(
                    f"{budget=}, {capacity=}, Protected terminal {node_key_by_index[t]} is missing protector {node_key_by_index[protector]} in solution."
                )

    df = pd.DataFrame([min_rankings])
    return df


if __name__ == "__main__":
    # Common config
    config = {}
    config["name"] = "Empire"
    config["budget"] = 50
    config["top_n"] = 0
    config["nearest_n"] = 30
    config["max_waypoint_ub"] = 30
    config["prune_prizes"] = False
    config["solver"] = {}  # Solver not used during analysis

    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    set_logger(config)

    # Common data
    kwargs = {
        "prices": ds.read_json("en_lta_prices.json")["effectivePrices"],
        "modifiers": {},
        "grind_taken_list": [],
        "solutions_json": ds.read_json("highs_solutions.json"),
        "min_lodging": ds.read_json("lodging_specifications.json")["min"],
        "max_lodging": ds.read_json("lodging_specifications.json")["max"],
    }

    main(config, kwargs)
