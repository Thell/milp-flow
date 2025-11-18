# analyze_tr_potential_reach.py

"""
- For each root, analyze the distances from the root to each terminal with a prize for that root.
- Report the 5 num summary stats for each root and the average prize values associated with terminals
  in the range of each of those stats.
- Report a quantile breakdown for the distances from the root to each terminal with a prize for that root
  along with the associated aggregate prize values.

  Note: this analysis can be done using the data within 'data' container.
"""

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from api_common import set_logger
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
import data_store as ds


def run_analysis(config: dict[str, Any], data: dict[str, Any], bin_count: int = 5):
    G = data["G"]
    terminal_indices = G.attrs["terminal_indices"]
    root_indices = G.attrs["root_indices"]
    all_pairs_path_lengths = data["all_pairs_path_lengths"]

    all_analysis = {}
    for r_idx in root_indices:
        terminal_entries = [
            {
                "t_idx": t,
                "r_idx": r_idx,
                "value": round(G[t]["prizes"][r_idx] / 1e6, 3),
                "dist": all_pairs_path_lengths[(t, r_idx)],
            }
            for t in terminal_indices
            if G[t]["prizes"].get(r_idx) is not None
        ]
        if not terminal_entries:
            continue

        logger.info(
            f"    analyzing root {G[r_idx]['waypoint_key']} with {len(terminal_entries)} terminals..."
        )

        df = pd.DataFrame(terminal_entries)
        df["root_key"] = G[r_idx]["waypoint_key"]

        # 5-num summary stats on distances
        dist_stats = df["dist"].describe()

        # Equal-width dist bins for breakdown (bin_count bins from min to max + Full)
        min_dist = df["dist"].min()
        max_dist = df["dist"].max()
        bin_edges = np.linspace(min_dist, max_dist, bin_count + 1)  # bin_count bins
        dist_cuts = pd.cut(df["dist"], bins=bin_edges, duplicates="drop")
        grouped = df.groupby(dist_cuts, observed=True)
        bin_stats = grouped.agg({
            "dist": "count",  # num_terms
            "value": "sum",  # sum_prize per bin
        })
        bin_stats.columns = ["num_terms", "sum_prize"]
        bin_stats["avg_prize"] = bin_stats["sum_prize"] / bin_stats["num_terms"]
        # Cumsum across bins
        cum_terms = bin_stats["num_terms"].cumsum()
        cum_prize = bin_stats["sum_prize"].cumsum()
        bin_stats["cum_terms"] = cum_terms
        bin_stats["cum_prize"] = cum_prize
        total_prize = cum_prize.iloc[-1]
        bin_stats["per_%_total_prize"] = (bin_stats["sum_prize"] / total_prize) * 100
        bin_stats["cum_%_total_prize"] = (bin_stats["cum_prize"] / total_prize) * 100
        bin_stats["dist_range"] = bin_stats.index.astype(str)
        bin_stats = bin_stats.reset_index(drop=True)
        # Count other roots in each bin
        root_dists = {s: all_pairs_path_lengths.get((r_idx, s), np.inf) for s in root_indices if s != r_idx}
        for i, (_, row) in enumerate(bin_stats.iterrows()):
            if pd.isna(row["dist_range"]):
                continue
            # Parse Interval safely
            interval_str = row["dist_range"]
            if "nan" in interval_str:
                low, high = 0, np.inf
            else:
                # Extract low, high from '(low, high]'
                low = float(interval_str.split(",")[0].replace("(", ""))
                high = float(interval_str.split(",")[1].replace("]", ""))
            roots_in_bin = sum(1 for d in root_dists.values() if low < d <= high)
            bin_stats.at[i, "roots_in_bin"] = roots_in_bin
        bin_stats["roots_in_bin"] = bin_stats["roots_in_bin"].fillna(0).astype(int)
        # Add Full row
        full_row = pd.DataFrame(
            {
                "num_terms": [len(df)],
                "sum_prize": [total_prize],
                "avg_prize": [total_prize / len(df)],
                "cum_terms": [len(df)],
                "cum_prize": [total_prize],
                "per_%_total_prize": [100.0],
                "cum_%_total_prize": [100.0],
                "dist_range": [f"{min_dist:.1f}-{max_dist:.1f}"],
                "roots_in_bin": [len(root_indices) - 1],
            },
            index=[0],
        )
        bin_stats = pd.concat([bin_stats, full_row], ignore_index=True)

        all_analysis[r_idx] = {
            "stats": dist_stats,
            "dist_bins": bin_stats,
            "df": df,  # For further if needed
        }

    return all_analysis


def generate_report(all_analysis, bin_count: int = 5):
    report_data = []
    for r_idx, analysis in all_analysis.items():
        stats = analysis["stats"]
        dist_bins = analysis["dist_bins"]

        # Flatten 5-num (one row for summary)
        report_data.append({
            "root": analysis["df"]["root_key"].iloc[0],
            "stat": "5-num Summary",
            "count": stats["count"],
            "mean_dist": stats["mean"],
            "avg_prize_in_bin": np.nan,
            "sum_prize_in_bin": np.nan,
            "num_terms": stats["count"],
            "cum_prize": np.nan,
            "per_%_total_prize": np.nan,
            "cum_%_total_prize": np.nan,
            "roots_in_bin": np.nan,
        })

        # Dist bins rows
        for _, row in dist_bins.iterrows():
            report_data.append({
                "root": analysis["df"]["root_key"].iloc[0],
                "stat": row["dist_range"],
                "count": row["num_terms"],
                "mean_dist": stats["mean"],
                "avg_prize_in_bin": row["avg_prize"],
                "sum_prize_in_bin": row["sum_prize"],
                "num_terms": row["num_terms"],
                "cum_prize": row["cum_prize"],
                "per_%_total_prize": row["per_%_total_prize"],
                "cum_%_total_prize": row["cum_%_total_prize"],
                "roots_in_bin": row["roots_in_bin"],
            })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv("tr_potential_reach_report.csv", index=False)
    logger.info("Generated tr_potential_reach_report.csv")

    # Per-root summary print
    for r_idx in sorted(all_analysis.keys()):
        analysis = all_analysis[r_idx]
        print(f"\nRoot {analysis['df']['root_key'].iloc[0]} (bin_count={bin_count}):")
        print("Dist 5-num:", analysis["stats"])
        print("Dist bins breakdown:\n", analysis["dist_bins"])


def main(config: dict[str, Any], kwargs: dict[str, Any]):
    # I'm not sure if we need to analyze both lodging capacity modes so we'll just use min
    lodging = kwargs["min_lodging"]
    prices = kwargs["prices"]
    modifiers = kwargs["modifiers"]
    grindTakenList = kwargs["grind_taken_list"]
    bin_count = kwargs["bin_count"]

    data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
    generate_graph_data(data)

    all_analysis = run_analysis(config, data, bin_count)
    generate_report(all_analysis, bin_count)


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
        "bin_count": 20,
    }

    main(config, kwargs)
