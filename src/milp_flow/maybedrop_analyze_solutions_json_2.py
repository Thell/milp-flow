from copy import deepcopy
from typing import Any
from collections import defaultdict
import re

from loguru import logger
from natsort import natsorted
import pandas as pd
from rustworkx import PyDiGraph
from scipy.stats import percentileofscore

from api_common import set_logger
import data_store as ds
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data  # For parsing budget/cap from keys


def recompute_ranks(
    solver_graph: PyDiGraph, terminal_indices, all_pairs_path_lengths, affiliated_town_region
):
    """Recompute ranks from prizes; return {t: {r: {'RootRank_Value': int, 'ValueRank_Root': int, 'RootRank_Value_Pct': float, 'ValueRank_Root_Pct': float, 'GlobalPct': float, 'GlobalVPC': float}}}."""
    # Build tr_data (from prize_pruning logic)
    tr_data = []
    for i in terminal_indices:
        node = solver_graph[i]
        t_cost = node["need_exploration_point"]
        for r_idx, value in node["prizes"].items():
            if value == 0:
                continue
            path_cost = all_pairs_path_lengths.get((r_idx, i), float("inf"))
            if path_cost == float("inf"):
                raise ValueError(f"No path from {r_idx} to {i}")
            tr_data.append({
                "t_idx": i,
                "r_idx": r_idx,
                "value": value,
                "cost": t_cost,
                "path_cost": path_cost,
            })

    # Compute absolute ranks (from prize_pruning)
    root_ts = defaultdict(list)
    for entry in tr_data:
        root_ts[entry["r_idx"]].append(entry)
    for r, entries in root_ts.items():
        sorted_entries = sorted(entries, key=lambda e: e["value"], reverse=True)
        for rank, entry in enumerate(sorted_entries, 1):
            entry["ValueRank_Root"] = rank

    t_rs = defaultdict(list)
    for entry in tr_data:
        t_rs[entry["t_idx"]].append(entry)
    for t, entries in t_rs.items():
        sorted_entries = sorted(entries, key=lambda e: e["value"] / e["path_cost"], reverse=True)
        for rank, entry in enumerate(sorted_entries, 1):
            entry["RootRank_Value"] = rank

    # Compute percentiles
    all_values = [e["value"] for e in tr_data]  # Global raw values
    all_net = [e["value"] / e["path_cost"] for e in tr_data]  # Global net values

    for entry in tr_data:
        # Per-root percentile (local desirability from root view)
        root_vals = [e["value"] for e in root_ts[entry["r_idx"]]]
        entry["ValueRank_Root_Pct"] = percentileofscore(root_vals, entry["value"], kind="rank")

        # Per-terminal percentile (local desirability from terminal view)
        term_vals = [e["value"] for e in t_rs[entry["t_idx"]]]
        entry["RootRank_Value_Pct"] = percentileofscore(term_vals, entry["value"], kind="rank")

        # Global raw percentile (cross all pairs)
        entry["GlobalPct"] = percentileofscore(all_values, entry["value"], kind="rank")

        # Global VPC percentile (cost-adjusted desirability)
        net = entry["value"] / entry["path_cost"]
        entry["GlobalVPC"] = percentileofscore(all_net, net, kind="rank")

    # Aggregate to prizes_ranks
    prizes_ranks = defaultdict(dict)
    for entry in tr_data:
        t, r = entry["t_idx"], entry["r_idx"]
        prizes_ranks[t][r] = {
            "RootRank_Value": entry["RootRank_Value"],  # Absolute
            "ValueRank_Root": entry["ValueRank_Root"],  # Absolute
            "RootRank_Value_Pct": entry["RootRank_Value_Pct"],  # Pct
            "ValueRank_Root_Pct": entry["ValueRank_Root_Pct"],  # Pct
            "GlobalPct": entry["GlobalPct"],
            "GlobalVPC": entry["GlobalVPC"],
        }
    return prizes_ranks


def compute_families(solver_graph: PyDiGraph, terminal_indices):
    """Compute families {parent: [ts]}."""
    families = defaultdict(list)
    for i in terminal_indices:
        parents = list(solver_graph.predecessor_indices(i))
        if parents:
            parent = parents[0]
            families[parent].append(i)

    # Output family size and counts breakdown using a Counter
    family_counts = {}
    for parent, ts in families.items():
        family_counts[len(ts)] = family_counts.get(len(ts), 0) + 1
    # print(f"Family counts: {family_counts}")

    return families


def analyze_used_prizes(
    solutions_json,
    solver_graph: PyDiGraph,
    budget,
    capacity="min",
    affiliated_town_region=None,
    all_pairs_path_lengths=None,
):
    """Analyze used prizes by rank/family from solved JSON + graph."""
    # Adjust key format based on your JSON structure (e.g., 'min_50' or similar)
    key = f"highs-{capacity}-{budget}"
    if key not in solutions_json:
        raise ValueError(f"No data for {key}")

    sol = solutions_json[key][0]
    terminal_sets = sol.get("terminal_sets", "{}")  # {t: r} as graph indices
    if not terminal_sets:
        raise ValueError("No terminal_sets in JSON.")

    node_map = solver_graph.attrs["node_key_by_index"]
    terminal_sets = {node_map.inv[int(k)]: node_map.inv[int(v)] for k, v in terminal_sets.items()}
    solver_graph.attrs["terminal_sets"] = terminal_sets

    terminal_indices = solver_graph.attrs["terminal_indices"]
    if not terminal_indices:
        raise ValueError("No terminals in solver_graph attributes.")

    prizes_ranks = recompute_ranks(
        solver_graph, terminal_indices, all_pairs_path_lengths, affiliated_town_region
    )
    families = compute_families(solver_graph, terminal_indices)

    used_data = []
    for t, r in terminal_sets.items():
        if not solver_graph.has_node(t):
            logger.warning(f"Terminal {t} not in solver_graph (pruned but used in solution). Skipping.")
            continue
        t_node = solver_graph[t]

        rank_info = prizes_ranks.get(t, {}).get(
            r,
            {
                "RootRank_Value_Pct": float("inf"),
                "ValueRank_Root_Pct": float("inf"),
                "GlobalPct": float("inf"),
                "GlobalVPC": float("inf"),
            },
        )
        if rank_info["RootRank_Value_Pct"] == float("inf"):
            logger.warning(f"No rank info for terminal {t} and root {r}.")

        parents = list(solver_graph.predecessor_indices(t))
        parent = parents[0] if parents else None
        if parent and parent not in families:
            raise ValueError(f"Parent {parent} not in families.")
        family_size = len(families.get(parent, [t])) if parent else 1

        used_data.append({
            "budget": budget,
            "capacity": capacity,
            "terminal": t_node["waypoint_key"],
            "root": solver_graph[r]["waypoint_key"],
            "term_rank": rank_info["RootRank_Value_Pct"],  # Pct
            "root_rank": rank_info["ValueRank_Root_Pct"],  # Pct
            "global_pct": rank_info["GlobalPct"],
            "global_vpc": rank_info["GlobalVPC"],
            "family_size": family_size,
            "is_family": family_size > 1,
            "value": t_node["prizes"].get(r, 0),
        })

    df = pd.DataFrame(used_data)
    if df.empty:
        return df

    return df


def run_analysis(data: dict[str, Any]):
    """Entry point: Run analysis for all budgets/capacities in JSON using the provided solver_graph."""
    solver_graph = data["solver_graph"]
    solutions_json = ds.read_json("highs_solutions.json")
    if not solutions_json:
        print("No solutions JSON loaded.")
        return

    all_pairs_path_lengths = data["all_pairs_path_lengths"]
    affiliated_town_region = data["affiliated_town_region"]

    # Parse budgets/caps from keys (assume format 'min_50', 'max_75', etc.)
    analyses = []
    for key, runs in solutions_json.items():
        match = re.match(r".*(min|max)-(\d+)$", key)
        if match:
            cap, budget_str = match.groups()
            budget = int(budget_str)
            df = analyze_used_prizes(
                solutions_json, solver_graph, budget, cap, affiliated_town_region, all_pairs_path_lengths
            )
            if not df.empty:
                analyses.append(df)

    if analyses:
        all_df = pd.concat(analyses, ignore_index=True)
        return all_df
    else:
        raise ValueError("No valid budget/cap keys found in JSON.")


if __name__ == "__main__":
    prices = ds.read_json("en_lta_prices.json")
    prices = prices["effectivePrices"]

    modifiers = {}
    grindTakenList = []
    min_lodging = ds.read_json("lodging_specifications.json")["min"]
    max_lodging = ds.read_json("lodging_specifications.json")["max"]

    config: dict = {
        "name": "Empire",
        "budget": 50,
        "top_n": 30,
        "nearest_n": 30,
        "max_waypoint_ub": 30,
        "prune_prizes": True,
        "capacity_mode": "min",
        "analyze_solutions": True,
    }
    config["logger"] = {"level": "DEBUG", "format": "<level>{message}</level>"}
    config["solver"] = {}

    set_logger(config)

    # Load budgets/caps from completed solutions
    solutions_json = ds.read_json("highs_solutions.json")
    budgets = set()
    for key in solutions_json:
        match = re.match(r".*(min|max)-(\d+)$", key)
        if match:
            _, budget_str = match.groups()
            budgets.add(int(budget_str))

    all_analyses = []
    data = {}
    for cap in ["min", "max"]:
        lodging = min_lodging if cap == "min" else max_lodging
        refresh_data = True

        for budget in natsorted(budgets):
            config["budget"] = budget

            if not data or refresh_data:
                data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
                generate_graph_data(data)
                refresh_data = False

            data["config"] = config
            solver_graph = deepcopy(data["G"].copy())
            solver_graph.attrs = deepcopy(data["G"].attrs)
            data["solver_graph"] = solver_graph

            logger.info(f"=== Running analysis for budget={budget}, cap={cap}")
            df = analyze_used_prizes(
                solutions_json,
                solver_graph,
                budget,
                cap,
                data["affiliated_town_region"],
                data["all_pairs_path_lengths"],
            )
            if not df.empty:
                all_analyses.append(df)

    if all_analyses:
        all_df = pd.concat(all_analyses, ignore_index=True)

        used_summary = (
            all_df.groupby(["budget", "capacity", "is_family", "vpc_bin"])
            .size()
            .unstack(fill_value=0, level="vpc_bin")
        )
        used_summary["Total"] = used_summary.sum(axis=1)

        print("\n=== Used Prizes/Terminals by VPC Bin (by Budget/Cap/Family) ===\n", used_summary)
        used_summary.to_csv("used_vpc_summary_2.csv")

    else:
        raise ValueError("No valid budget/cap keys found in JSON.")
