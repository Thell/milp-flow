# analyze_top_n.py
from copy import deepcopy
from typing import Any

from loguru import logger
import pandas as pd
from collections import Counter, defaultdict

from api_common import set_logger
import data_store as ds
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
from reduce_prize_data import reduce_prize_data


def analyze_top_n_remaining(
    data: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze rankings of prizes remaining after top_n filtering."""
    solver_graph = data["solver_graph"]
    prize_ranks = data["prize_ranks"]  # Reference global ranks
    families = data["families"]  # Full graph families
    all_pairs_path_lengths = data["all_pairs_path_lengths"]

    terminal_indices = solver_graph.attrs["terminal_indices"]
    if not terminal_indices:
        raise ValueError("No terminals in solver_graph attributes.")

    # Compute global per-terminal categories (fixed by graph structure)
    terminal_categories = {}
    for parent, family_ts in families.items():
        family_size = len(family_ts)
        if family_size == 1:
            terminal_categories[family_ts[0]] = "only_child"
            continue
        # Dominant: max total prize value across all roots (sum prizes for this t)
        dominant_t = max(family_ts, key=lambda t: sum(solver_graph[t]["prizes"].values()))
        for t in family_ts:
            terminal_categories[t] = "dominant" if t == dominant_t else "protected"
    logger.info(f"Terminal categories: {dict(Counter(terminal_categories.values()))}")

    # Build remaining_tr for prizes >0 in solver_graph
    inf = float("inf")
    remaining_tr = {}
    for t_idx in terminal_indices:
        node = solver_graph[t_idx]
        t_cost = node["need_exploration_point"]
        for r_idx, value in node["prizes"].items():
            if value == 0:
                continue
            path_cost = all_pairs_path_lengths.get((t_idx, r_idx), inf)
            if path_cost == inf:
                raise ValueError(f"No path from {t_idx} to {r_idx}")
            remaining_tr[(t_idx, r_idx)] = {
                "t_idx": t_idx,
                "r_idx": r_idx,
                "value": value,
                "t_cost": t_cost,
                "path_cost": path_cost,
                "vpc": value / path_cost,
            }
    logger.info(f"Remaining prizes after top_n: {len(remaining_tr)}")

    remaining_data = []
    default_ranks = {
        "root_prize_value_pct_by_terminal_view": inf,
        "terminal_prize_value_pct_by_root_view": inf,
        "terminal_prize_value_global_pct": inf,
        "terminal_prize_net_t_cost_global_pct": inf,
        "terminal_prize_net_path_cost_global_pct": inf,
    }
    category_counts = defaultdict(int)
    for (t, r), entry in remaining_tr.items():
        rank_info = prize_ranks.get(t, {}).get(r, default_ranks)
        if rank_info["terminal_prize_value_pct_by_root_view"] == inf:
            logger.warning(f"No rank info for terminal {t} and root {r}.")

        # Category per-terminal (global)
        category = terminal_categories.get(t, "only_child")  # Fallback
        category_counts[category] += 1

        remaining_data.append({
            "terminal": solver_graph[t]["waypoint_key"],
            "root": solver_graph[r]["waypoint_key"],
            "terminal_prize_value_global_pct": rank_info["terminal_prize_value_global_pct"],
            "terminal_prize_net_t_cost_global_pct": rank_info.get(
                "terminal_prize_net_t_cost_global_pct", inf
            ),
            "terminal_prize_net_path_cost_global_pct": rank_info["terminal_prize_net_path_cost_global_pct"],
            "family_size": len(
                families.get(
                    list(solver_graph.predecessor_indices(t))[0]
                    if list(solver_graph.predecessor_indices(t))
                    else None,
                    [t],
                )
            ),
            "category": category,
            "value": entry["value"],
            "vpc": entry["vpc"],
        })

    df = pd.DataFrame(remaining_data)
    logger.info(f"len(df) = {len(df)}; category counts: {dict(category_counts)}")

    # Compute unique terminal max per metric, with terminal category
    term_max_data = []
    for metric in ["value", "net_t_cost", "net_path_cost"]:
        pct_col = f"terminal_prize_{metric}_global_pct"
        for term_key in df["terminal"].unique():
            term_df = df[df["terminal"] == term_key]
            if term_df.empty:
                continue
            max_idx = term_df[pct_col].idxmax()
            max_pct = term_df.loc[max_idx, pct_col]
            term_category = term_df.loc[max_idx, "category"]  # Terminal-global category
            term_max_data.append({
                "terminal": term_key,
                "metric": metric,
                "max_pct": max_pct,
                "category": term_category,
            })
    term_max = pd.DataFrame(term_max_data)

    if df.empty:
        return df, term_max
    return df, term_max


if __name__ == "__main__":
    prices = ds.read_json("en_lta_prices.json")["effectivePrices"]
    modifiers = {}
    grind_taken_list = []
    min_lodging = ds.read_json("lodging_specifications.json")["min"]

    config: dict[str, Any] = {
        "name": "Empire",
        "budget": 50,
        "top_n": 6,
        "nearest_n": 30,
        "max_waypoint_ub": 30,
        "prune_prizes": True,
        "capacity_mode": "min",
    }
    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    config["solver"] = {}

    set_logger(config)

    data = generate_reference_data(config, prices, modifiers, min_lodging, grind_taken_list)
    generate_graph_data(data)

    data["config"] = config
    solver_graph = deepcopy(data["G"].copy())
    solver_graph.attrs = deepcopy(data["G"].attrs)
    data["solver_graph"] = solver_graph
    reduce_prize_data(data)

    df_prizes, df_term_max = analyze_top_n_remaining(data)

    if df_prizes.empty:
        raise ValueError("No valid top_n data found.")

    metrics = ["value", "net_t_cost", "net_path_cost"]
    bins = [f"{i}-{i + 0.99}%" for i in range(0, 101)]
    combined_rows = []

    # Define the category order
    category_order = ["only_child", "dominant", "protected"]

    # --- Prize-level summaries ---
    for metric in metrics:
        pct_col = f"terminal_prize_{metric}_global_pct"
        df_prizes[f"{metric}_bin"] = pd.cut(
            df_prizes[pct_col], bins=range(0, 102, 1), right=False, labels=bins
        )
        grouped = df_prizes.groupby(["category", f"{metric}_bin"]).size().unstack(fill_value=0)
        grouped = grouped[bins]  # Ensure column order
        grouped["Total"] = grouped.sum(axis=1)
        grouped = grouped.reset_index()

        # Reorder categories
        grouped["category"] = pd.Categorical(grouped["category"], categories=category_order, ordered=True)
        grouped = grouped.sort_values("category").reset_index(drop=True)

        grouped.insert(0, "Scope", f"top_n_{metric}")
        combined_rows.append(grouped)

    # --- Terminal-level max summaries ---
    for metric in metrics:
        term_df = df_term_max[df_term_max["metric"] == metric].copy()
        term_df[f"{metric}_bin"] = pd.cut(term_df["max_pct"], bins=range(0, 102, 1), right=False, labels=bins)
        grouped = term_df.groupby(["category", f"{metric}_bin"]).size().unstack(fill_value=0)
        grouped = grouped[bins]  # Ensure column order
        grouped["Total"] = grouped.sum(axis=1)
        grouped = grouped.reset_index()

        # Reorder categories
        grouped["category"] = pd.Categorical(grouped["category"], categories=category_order, ordered=True)
        grouped = grouped.sort_values("category").reset_index(drop=True)

        grouped.insert(0, "Scope", f"top_n_term_best_{metric}")
        combined_rows.append(grouped)

    # Concatenate everything into a single hierarchical report
    hierarchical_report = pd.concat(combined_rows, axis=0, ignore_index=True)

    print("\n=== Hierarchical Top-N Prize Report ===\n")
    print(hierarchical_report)

    # Save single CSV
    hierarchical_report.to_csv("top_n_analysis_report.csv", index=False)
