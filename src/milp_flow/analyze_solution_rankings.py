# analyze_solution_rankings.py

from copy import deepcopy
from typing import Any
import json
import re

from loguru import logger
import pandas as pd
from rustworkx import PyDiGraph
from collections import defaultdict

from api_common import set_logger
import data_store as ds
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data


def analyze_used_prizes(
    solutions_json: dict[str, Any],
    solver_graph: PyDiGraph,
    budget: int,
    capacity: str,
    prize_ranks: dict[int, dict[int, dict[str, Any]]],
    all_pairs_path_lengths: dict[tuple[int, int], float],
) -> pd.DataFrame:
    """Analyze used prizes by rank from solved JSON + graph, using full-graph categories."""
    key = f"highs-{capacity}-{budget}"
    if key not in solutions_json:
        raise ValueError(f"No data for {key}")

    sol = solutions_json[key][0]
    terminal_sets_str = sol.get("terminal_sets", "{}")
    terminal_sets = json.loads(terminal_sets_str) if isinstance(terminal_sets_str, str) else terminal_sets_str
    if not terminal_sets:
        raise ValueError("No terminal_sets in JSON.")

    node_key_by_index = solver_graph.attrs["node_key_by_index"]
    terminal_sets = {
        node_key_by_index.inv[int(k)]: node_key_by_index.inv[int(v)] for k, v in terminal_sets.items()
    }
    logger.info(f"{key}: mapped len(terminal_sets) = {len(terminal_sets)}")
    solver_graph.attrs["terminal_sets"] = terminal_sets  # {t_idx: r_idx}

    terminal_indices = solver_graph.attrs["terminal_indices"]
    if not terminal_indices:
        raise ValueError("No terminals in solver_graph attributes.")

    # Build selected_tr for value/path metrics (skip zero-value)
    inf = float("inf")
    selected_tr = {}
    skipped_zero = 0
    for t, r in terminal_sets.items():
        if not solver_graph.has_node(t):
            continue
        t_node = solver_graph[t]
        value = t_node["prizes"].get(r, 0)
        if value == 0:
            logger.warning(f"{key}: zero-prize skip for t={t} r={r} value={value}")
            skipped_zero += 1
            continue
        path_cost = all_pairs_path_lengths.get((t, r), inf)
        if path_cost == inf:
            raise ValueError(f"No path from {t} to {r}")
        selected_tr[(t, r)] = {
            "value": value,
            "path_cost": path_cost,
            "vpc": value / path_cost,
        }
    logger.info(f"{key}: skipped {skipped_zero} zero-prize entries; len(selected_tr) = {len(selected_tr)}")

    used_data = []
    default_ranks = {
        "root_prize_value_pct_by_terminal_view": inf,
        "terminal_prize_value_pct_by_root_view": inf,
        "terminal_prize_value_global_pct": inf,
        "terminal_prize_net_t_cost_global_pct": inf,
        "terminal_prize_net_path_cost_global_pct": inf,
    }
    category_counts = defaultdict(int)

    families = data["families"]
    for t, r in terminal_sets.items():
        if not solver_graph.has_node(t) or (t, r) not in selected_tr:
            logger.warning(f"Terminal {t} not in solver_graph or zero-prize (skipping).")
            continue
        t_node = solver_graph[t]

        rank_info = prize_ranks.get(t, {}).get(r, default_ranks)
        if rank_info["terminal_prize_value_pct_by_root_view"] == inf:
            logger.warning(f"No rank info for terminal {t} and root {r}.")

        # Full-graph category and family_size
        category = t_node.get("category", "only_child")
        parents = list(solver_graph.predecessor_indices(t))
        parent = parents[0] if parents else None
        family_size = len(families.get(parent, [t]))

        category_counts[category] += 1
        logger.debug(f"{key}: t={t} r={r} parent={parent} family_size={family_size} category={category}")

        used_data.append({
            "budget": budget,
            "capacity": capacity,
            "terminal": t_node["waypoint_key"],
            "root": solver_graph[r]["waypoint_key"],
            "root_prize_value_pct_by_terminal_view": rank_info["root_prize_value_pct_by_terminal_view"],
            "terminal_prize_value_pct_by_root_view": rank_info["terminal_prize_value_pct_by_root_view"],
            "terminal_prize_value_global_pct": rank_info["terminal_prize_value_global_pct"],
            "terminal_prize_net_t_cost_global_pct": rank_info["terminal_prize_net_t_cost_global_pct"],
            "terminal_prize_net_path_cost_global_pct": rank_info["terminal_prize_net_path_cost_global_pct"],
            "family_size": family_size,
            "category": category,
            "value": selected_tr[(t, r)]["value"],
            "vpc": selected_tr[(t, r)]["vpc"],
        })

    df = pd.DataFrame(used_data)
    logger.info(f"{key}: len(df) = {len(df)}; category counts: {dict(category_counts)}")
    return df


def run_analysis(data: dict[str, Any], cap_mode: str) -> pd.DataFrame | None:
    """Entry point: Run analysis for all budgets/capacities in JSON using the provided solver_graph."""
    solver_graph = data["solver_graph"]
    solutions_json = ds.read_json("highs_solutions.json")
    if not solutions_json:
        logger.warning("No solutions JSON loaded.")
        return None

    all_pairs_path_lengths = data["all_pairs_path_lengths"]
    prize_ranks = data["prize_ranks"]

    analyses = []
    for key, _ in solutions_json.items():
        match = re.match(r".*(min|max)-(\d+)$", key)
        if match:
            cap, budget_str = match.groups()
            if cap != cap_mode:
                continue
            budget = int(budget_str)
            df = analyze_used_prizes(
                solutions_json,
                solver_graph,
                budget,
                cap,
                prize_ranks,
                all_pairs_path_lengths,
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
        generate_graph_data(data)  # Populates data["prize_ranks"], data["families"], etc.

        # NOTE: Don't do any top_n filtering or prize pruning for reference graph - all rankings are included!

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

        # Define bins and category order
        bins = [f"{i}-{i + 0.99}%" for i in range(0, 101)]
        category_order = ["only_child", "dominant", "protected"]

        # Convert category to Categorical to enforce order
        all_df["category"] = pd.Categorical(all_df["category"], categories=category_order, ordered=True)

        # Metrics to produce reports for
        metrics = [
            ("terminal_prize_net_path_cost_global_pct", "solution_used_net_path_cost_summary.csv"),
            ("terminal_prize_net_t_cost_global_pct", "solution_used_net_t_cost_summary.csv"),
            ("terminal_prize_value_global_pct", "solution_used_value_summary.csv"),
        ]

        for metric_col, out_csv in metrics:
            all_df[f"{metric_col}_bin"] = pd.cut(
                all_df[metric_col],
                bins=range(0, 102, 1),
                right=False,
                labels=bins,
            )

            dropped_rows = all_df[all_df[f"{metric_col}_bin"].isna()]
            if not dropped_rows.empty:
                logger.info(
                    f"Dropped {len(dropped_rows)} rows with invalid {metric_col}_bin (inf ranks): "
                    f"{dropped_rows[['budget', 'capacity', 'terminal', 'root', metric_col]].to_dict('records')}"
                )

            df_clean = all_df.dropna(subset=[f"{metric_col}_bin"])

            summary = (
                df_clean.groupby(["budget", "capacity", "category", f"{metric_col}_bin"])
                .size()
                .unstack(fill_value=0, level=f"{metric_col}_bin")
            )

            # Trim leading zero bins
            non_zero_cols = summary.columns[summary.sum() > 0]
            if len(non_zero_cols) > 0:
                first_non_zero = non_zero_cols.min()
                summary = summary.loc[:, summary.columns >= first_non_zero]

            summary["Total"] = summary.sum(axis=1)

            # Sort by category
            summary = summary.sort_index(level=["budget", "capacity", "category"])

            print(f"\n=== Used Prizes/Terminals by {metric_col} Bin (by Budget/Cap/Category) ===\n", summary)
            summary.to_csv(out_csv)
    else:
        raise ValueError("No valid budget/cap keys found in JSON.")
