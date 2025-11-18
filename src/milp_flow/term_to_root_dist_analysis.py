# term_to_root_dist_analysis.py

"""
- For each terminal, compute dist to nearest root and dist to root with highest prize.
- Output CSV: terminal, nearest_dist, best_dist, best_prize.
- Console: 5-num summary stats for nearest/best dists across terminals.
"""

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from api_common import set_logger
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
import data_store as ds


def run_analysis(config: dict[str, Any], data: dict[str, Any]):
    G = data["G"]
    terminal_indices = G.attrs["terminal_indices"]
    root_indices = G.attrs["root_indices"]
    all_pairs_path_lengths = data["all_pairs_path_lengths"]

    terminal_data = []
    for t_idx in terminal_indices:
        node = G[t_idx]
        prizes = node["prizes"]
        if not prizes:
            continue

        # Nearest dist: min over all roots
        dists = [all_pairs_path_lengths.get((t_idx, r), np.inf) for r in root_indices]
        nearest_dist = min(dists)

        # Best dist: dist to root with max prize
        best_r = max(prizes, key=prizes.get)
        best_dist = all_pairs_path_lengths.get((t_idx, best_r), np.inf)
        best_prize = prizes[best_r]

        terminal_data.append({
            "terminal": node["waypoint_key"],
            "nearest_dist": nearest_dist,
            "best_dist": best_dist,
            "best_prize": best_prize,
        })

    df = pd.DataFrame(terminal_data)
    logger.info(f"Analyzed {len(df)} terminals.")

    # 5-num summaries
    nearest_stats = df["nearest_dist"].describe()
    best_stats = df["best_dist"].describe()

    return df, nearest_stats, best_stats


def generate_report(df: pd.DataFrame, nearest_stats, best_stats):
    df.to_csv("term_to_root_dist_analysis.csv", index=False)
    logger.info("Generated term_to_root_dist_analysis.csv")

    print("\n=== Nearest Dist 5-Num Summary ===")
    print(nearest_stats)
    print("\n=== Best Dist 5-Num Summary ===")
    print(best_stats)

    # Quantile breakdown (optional: 25% intervals)
    print("\n=== Nearest Dist Quantiles (cum % terminals) ===")
    nearest_quants = df["nearest_dist"].quantile([0, 0.25, 0.5, 0.75, 1.0])
    print(nearest_quants)

    print("\n=== Best Dist Quantiles (cum % terminals) ===")
    best_quants = df["best_dist"].quantile([0, 0.25, 0.5, 0.75, 1.0])
    print(best_quants)


def main(config: dict[str, Any], kwargs: dict[str, Any]):
    lodging = kwargs["min_lodging"]
    prices = kwargs["prices"]
    modifiers = kwargs["modifiers"]
    grindTakenList = kwargs["grind_taken_list"]

    data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
    generate_graph_data(data)

    df, nearest_stats, best_stats = run_analysis(config, data)
    generate_report(df, nearest_stats, best_stats)


if __name__ == "__main__":
    # Common config
    config = {}
    config["name"] = "Empire"
    config["budget"] = 50  # N/A
    config["top_n"] = 324  # all prizes
    config["nearest_n"] = 30  # all roots
    config["max_waypoint_ub"] = 30  # N/A
    config["prune_prizes"] = False
    config["solver"] = {}  # N/A

    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    set_logger(config)

    # Common data
    kwargs = {
        "prices": ds.read_json("en_lta_prices.json")["effectivePrices"],
        "modifiers": {},
        "grind_taken_list": [],
        "min_lodging": ds.read_json("lodging_specifications.json")["min"],
        "max_lodging": ds.read_json("lodging_specifications.json")["max"],
    }

    main(config, kwargs)
