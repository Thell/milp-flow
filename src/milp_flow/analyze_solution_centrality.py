# analyze_solution_centrality.py

from copy import deepcopy
from typing import Any
import json
import re

from loguru import logger
import pandas as pd
from rustworkx import PyDiGraph
import rustworkx as rx
from collections import defaultdict

from api_common import set_logger
import data_store as ds
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
from api_rx_pydigraph import subgraph_stable


def analyze_used_centrality(
    solutions_json: dict[str, Any],
    solver_graph: PyDiGraph,
    budget: int,
    capacity: str,
    bc_tr_asp: dict[int, float],
    families: dict[int, list[int]],
) -> pd.DataFrame:
    """Analyze used centrality (bc_tr_asp) by rank from solved JSON + graph, using solution subgraph paths."""
    key = f"highs-{capacity}-{budget}"
    if key not in solutions_json:
        raise ValueError(f"No data for {key}")

    node_key_by_index = solver_graph.attrs["node_key_by_index"]
    sol = solutions_json[key][0]

    terminal_sets_str = sol.get("terminal_sets", "{}")
    terminal_sets = json.loads(terminal_sets_str) if isinstance(terminal_sets_str, str) else terminal_sets_str
    if not terminal_sets:
        raise ValueError("No terminal_sets in JSON.")
    terminal_sets = {
        node_key_by_index.inv[int(k)]: node_key_by_index.inv[int(v)] for k, v in terminal_sets.items()
    }

    solution_waypoints = set(sol.get("nodes", []))
    if not solution_waypoints:
        raise ValueError(f"No nodes in solution for {key}")
    solution_indices = set([
        node_key_by_index.inv[node] for node in solution_waypoints if node in node_key_by_index.inv
    ])

    solution_subG = subgraph_stable(list(solution_indices), solver_graph)

    used_data = []
    skipped_zero = 0
    total_incidents = 0
    for t, r in terminal_sets.items():
        if not solution_subG.has_node(t) or not solution_subG.has_node(r):
            continue
        t_node = solver_graph[t]  # Use original for prizes
        value = t_node["prizes"].get(r, 0)
        if value == 0:
            logger.warning(f"{key}: zero-prize skip for t={t} r={r} value={value}")
            skipped_zero += 1
            continue

        paths = rx.dijkstra_shortest_paths(solution_subG, t, weight_fn=None)
        path = paths[r]
        interm_nodes = path[1:-1]

        # Full-graph category and family_size (per terminal)
        category = t_node.get("category", "only_child")
        parents = list(solver_graph.predecessor_indices(t))
        parent = parents[0] if parents else None
        family_size = len(families.get(parent, [t]))

        # Per interm node row (only for used pairs)
        for n in interm_nodes:
            total_incidents += 1
            used_data.append({
                "budget": budget,
                "capacity": capacity,
                "terminal": t_node["waypoint_key"],
                "root": solver_graph[r]["waypoint_key"],
                "node": solver_graph[n]["waypoint_key"],
                "bc_tr_asp": bc_tr_asp[n],
                "num_interm_nodes": len(interm_nodes),
                "family_size": family_size,
                "category": category,
                "value": value,
            })

    logger.info(f"{key}: skipped {skipped_zero} zero-prize entries; total incidents = {total_incidents}")

    df = pd.DataFrame(used_data)
    if df.empty:
        logger.warning(f"{key}: No data after processing")
    return df


def run_analysis(data: dict[str, Any], cap_mode: str) -> pd.DataFrame | None:
    """Entry point: Run analysis for all budgets/capacities in JSON using the provided solver_graph."""
    solver_graph = data["solver_graph"]
    solutions_json = ds.read_json("highs_solutions.json")
    if not solutions_json:
        logger.warning("No solutions JSON loaded.")
        return None

    bc_tr_asp = solver_graph.attrs["exploration_metrics"]["bc_tr_asp"]
    families = data["families"]

    analyses = []
    for key, _ in solutions_json.items():
        match = re.match(r".*(min|max)-(\d+)$", key)
        if match:
            cap, budget_str = match.groups()
            if cap != cap_mode:
                continue
            budget = int(budget_str)
            df = analyze_used_centrality(
                solutions_json,
                solver_graph,
                budget,
                cap,
                bc_tr_asp,
                families,
            )
            if not df.empty:
                analyses.append(df)

    if analyses:
        all_df = pd.concat(analyses, ignore_index=True)
        return all_df
    else:
        raise ValueError("No valid budget/cap keys found in JSON.")


if __name__ == "__main__":
    prices = ds.read_json("en_lta_prices.json")["effectivePrices"]
    modifiers = {}
    grind_taken_list = []
    min_lodging = ds.read_json("lodging_specifications.json")["min"]
    max_lodging = ds.read_json("lodging_specifications.json")["max"]

    config: dict[str, Any] = {
        "name": "Empire",
        "budget": 50,
        "top_n": 0,
        "nearest_n": 30,
        "max_waypoint_ub": 30,
        "prune_prizes": False,
        "capacity_mode": "min",
        "use_asp_bc": True,  # Enable for metrics
    }
    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    config["solver"] = {}

    set_logger(config)

    solutions_json = ds.read_json("highs_solutions.json")
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
        generate_graph_data(data)  # Populates data["prize_ranks"], data["families"], etc. + metrics

        # NOTE: Don't do any top_n filtering or prize pruning for reference graph - all rankings are included!

        data["config"] = config
        solver_graph = deepcopy(data["G"].copy())
        solver_graph.attrs = deepcopy(data["G"].attrs)
        data["solver_graph"] = solver_graph

        logger.info(f"=== Running analysis for cap={cap} (all budgets) ===")
        df = run_analysis(data, cap)
        if df is not None and not df.empty:
            all_analyses.append(df)

    import numpy as np

    if all_analyses:
        all_df = pd.concat(all_analyses, ignore_index=True)

        # Save raw data for stats
        all_df.to_csv("raw_used_centrality.csv", index=False)

        # Compute min used bc_tr_asp per budget/capacity
        min_stats = all_df.groupby(["budget", "capacity"])["bc_tr_asp"].min().reset_index()
        min_stats.to_csv("min_used_bc_per_budget_cap.csv", index=False)
        logger.info("Saved raw_used_centrality.csv and min_used_bc_per_budget_cap.csv")

        # Define bins for bc_tr_asp (0-1 normalized)
        category_order = ["only_child", "dominant", "protected"]

        # Convert category to Categorical to enforce order
        all_df["category"] = pd.Categorical(all_df["category"], categories=category_order, ordered=True)

        # Add percentile column
        all_df["bc_tr_asp_pct"] = all_df["bc_tr_asp"].rank(pct=True) * 100

        # Metrics to produce reports for (focus on bc_tr_asp per used node)
        base_out = "solution_used_centrality_by_root_view_summary"

        # Option 1: Log-scale bins for low-end detail
        log_bins = np.logspace(-5, 0, 21)  # 20 bins: 1e-5 to 1
        all_df["bc_tr_asp_log_bin"] = pd.cut(all_df["bc_tr_asp"], bins=log_bins, duplicates="drop")  # type: ignore[reportCallIssue]

        df_clean_log = all_df.dropna(subset=["bc_tr_asp_log_bin"])

        summary_log = (
            df_clean_log.groupby(
                ["budget", "capacity", "root", "terminal", "category", "bc_tr_asp_log_bin"], observed=True
            )
            .size()
            .unstack(fill_value=0, level="bc_tr_asp_log_bin")
        )

        # Trim leading zero bins
        non_zero_cols_log = summary_log.columns[summary_log.sum() > 0]
        if len(non_zero_cols_log) > 0:
            first_non_zero_log = non_zero_cols_log.min()
            summary_log = summary_log.loc[:, summary_log.columns >= first_non_zero_log]

        summary_log["Total"] = summary_log.sum(axis=1)

        # Sort by budget, capacity, root, terminal, category
        summary_log = summary_log.sort_index(level=["budget", "capacity", "root", "terminal", "category"])

        print(f"\n=== Used Centrality Log Bins (by Budget/Cap/Root/Terminal/Category) ===\n", summary_log)
        summary_log.to_csv(f"{base_out}_log_scale.csv")

        # Option 2: Percentile bins (uniform on [0,100])
        pct_bins = np.linspace(0, 100, 21)  # 20 equal-width bins on [0,100]
        all_df["bc_tr_asp_pct_bin"] = pd.cut(all_df["bc_tr_asp_pct"], bins=pct_bins, duplicates="drop")  # type: ignore[reportCallIssue]

        df_clean_pct = all_df.dropna(subset=["bc_tr_asp_pct_bin"])

        summary_pct = (
            df_clean_pct.groupby(
                ["budget", "capacity", "root", "terminal", "category", "bc_tr_asp_pct_bin"], observed=True
            )
            .size()
            .unstack(fill_value=0, level="bc_tr_asp_pct_bin")
        )

        # Trim leading zero bins
        non_zero_cols_pct = summary_pct.columns[summary_pct.sum() > 0]
        if len(non_zero_cols_pct) > 0:
            first_non_zero_pct = non_zero_cols_pct.min()
            summary_pct = summary_pct.loc[:, summary_pct.columns >= first_non_zero_pct]

        summary_pct["Total"] = summary_pct.sum(axis=1)

        # Sort by budget, capacity, root, terminal, category
        summary_pct = summary_pct.sort_index(level=["budget", "capacity", "root", "terminal", "category"])

        print(f"\n=== Used Centrality Pct Bins (by Budget/Cap/Root/Terminal/Category) ===\n", summary_pct)
        summary_pct.to_csv(f"{base_out}_pct.csv")
    else:
        raise ValueError("No valid budget/cap keys found in JSON.")
