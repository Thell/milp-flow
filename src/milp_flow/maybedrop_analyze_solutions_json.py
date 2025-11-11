from copy import deepcopy
from typing import Any
from collections import defaultdict
import re

from loguru import logger
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
            # raise ValueError(f"Terminal {t} not in solver_graph.")
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

    num_terminals = df["terminal"].nunique()
    print("-" * 70)
    print(f"=== Budget {budget} ({capacity}): Num Terminals = {num_terminals} ===")

    # Rank usage at terminals (pct)
    rank_counts = df["term_rank"].value_counts().sort_index()
    for rank, count in rank_counts.items():
        print(f"Num Rank {rank} Prizes Used at Terminals: {count}")

    # Global Pct Usage
    global_counts = df["global_pct"].value_counts().sort_index()
    print("\nGlobal Pct Usage:\n", global_counts)

    # Family vs only-child per rank
    family_df = df.groupby(["term_rank", "is_family"]).size().unstack(fill_value=0)
    family_df.columns = ["Only Child", "Family Member"]
    print("\nFamily Breakdown by Term Rank:\n", family_df)

    # Root rank dist per term rank (%)
    crosstab = pd.crosstab(df["term_rank"], df["root_rank"], normalize="index") * 100
    print("\nRoot Rank Dist by Term Rank (%):\n", crosstab.round(1))

    # Inter-prize: Corr on ranks/values (incl global)
    numeric_cols = ["term_rank", "root_rank", "global_pct", "value", "family_size"]
    corr_matrix = df[numeric_cols].corr()
    print("\nInter-Prize Correlations (incl Global):\n", corr_matrix.round(2))

    # Pairwise % co-use (e.g., % terminals using multiple ranks)
    multi_rank = (
        df.groupby("terminal")["term_rank"].apply(lambda x: x.nunique() > 1).sum() / num_terminals * 100
    )
    print(f"\n% Terminals Using Multiple Ranks: {multi_rank:.1f}%")

    # Family split on global pct
    for group, sub_df in df.groupby("is_family"):
        label = "Family" if group else "Only Child"
        print(f"\n{label} Global Pct:\n", sub_df["global_pct"].value_counts().sort_index())

    # Global VPC Usage
    vpc_counts = df["global_vpc"].value_counts().sort_index()
    print("\nGlobal VPC Usage:\n", vpc_counts)

    # Corrs incl VPC
    numeric_cols = ["term_rank", "root_rank", "global_pct", "global_vpc", "value", "family_size"]
    corr_matrix = df[numeric_cols].corr()
    print("\nCorrs incl Global & VPC:\n", corr_matrix.round(2))

    # Family split on VPC
    for group, sub_df in df.groupby("is_family"):
        label = "Family" if group else "Only Child"
        print(f"\n{label} Global VPC:\n", sub_df["global_vpc"].value_counts().sort_index())

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
    lodging = min_lodging_specifications

    config: dict = {
        "name": "Empire",
        "budget": 50,
        "top_n": 30,
        "nearest_n": 10,
        "max_waypoint_ub": 25,
        "prune_prizes": True,
        "capacity_mode": "min",
        "analyze_solutions": True,
    }
    config["logger"] = {"level": "DEBUG", "format": "<level>{message}</level>"}
    config["solver"] = {}

    set_logger(config)

    # data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
    # # Since this graph is for analysis, we don't want to prune or reduce
    # graph_data = generate_graph_data(data, do_prune=False, do_reduce=False, basin_type=0)

    # run_analysis(graph_data)

    # Load budgets/caps from JSON
    solutions_json = ds.read_json("highs_solutions.json")
    budgets = set()
    for key in solutions_json:
        match = re.match(r".*(min|max)-(\d+)$", key)
        if match:
            _, budget_str = match.groups()
            budgets.add(int(budget_str))

    # Loop per budget
    all_analyses = []
    data = {}
    for budget in sorted(budgets):
        for cap in ["min", "max"]:
            config: dict = {
                "name": "Empire",
                "budget": budget,
                "top_n": 30,  # default: 6
                "nearest_n": 30,  # default: 7
                "max_waypoint_ub": 30,  # default: 17
                "prune_prizes": False,  # default: True
                "capacity_mode": cap,
            }
            config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
            config["solver"] = {}

            lodging = min_lodging_specifications if cap == "min" else max_lodging_specifications

            set_logger(config)
            if not data:
                data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
                generate_graph_data(data)

            data["config"] = config
            solver_graph = deepcopy(data["G"].copy())
            solver_graph.attrs = deepcopy(data["G"].attrs)
            data["solver_graph"] = solver_graph

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

        # Pivot: % Usage by Budget/Cap/Rank
        pivot_df = all_df.groupby(["budget", "capacity", "term_rank"]).size().reset_index(name="count")
        pivot_df["pct"] = pivot_df.groupby(["budget", "capacity"])["count"].transform(
            lambda x: x / x.sum() * 100
        )
        pivot_df = pivot_df.pivot_table(
            index=["budget", "capacity"], columns="term_rank", values="pct", fill_value=0
        ).round(1)
        print("\n=== Term Rank Decay Pivot (% Usage) ===\n", pivot_df)
        pivot_df.to_csv("term_rank_decay_pivot.csv")  # Export for review

        # Root Rank Pivot: % Usage by Budget/Cap/Root Rank
        root_pivot_df = all_df.groupby(["budget", "capacity", "root_rank"]).size().reset_index(name="count")
        root_pivot_df["pct"] = root_pivot_df.groupby(["budget", "capacity"])["count"].transform(
            lambda x: x / x.sum() * 100
        )
        root_pivot_df = root_pivot_df.pivot_table(
            index=["budget", "capacity"], columns="root_rank", values="pct", fill_value=0
        ).round(1)
        print("\n=== Root Rank Decay Pivot (% Usage) ===\n", root_pivot_df)
        root_pivot_df.to_csv("root_rank_decay_pivot.csv")
        root_cum = root_pivot_df.cumsum(axis=1).round(1)
        print("\nCum % Coverage:\n", root_cum.iloc[:, :20])
        root_cum.to_csv("cum_root_coverage.csv")

        print("\n=== Aggregate Across All Budgets ===")
        print(f"Total Analyses: {len(all_analyses)}")
        print(f"Overall Num Terminals: {all_df['terminal'].nunique()}")
        # Quick aggregate rank usage
        agg_term_ranks = all_df["term_rank"].value_counts().sort_index()
        print("Aggregate Rank Usage:\n", agg_term_ranks)
        agg_root_ranks = all_df["root_rank"].value_counts().sort_index()
        print("Aggregate Root Rank Usage:\n", agg_root_ranks)

        # Split aggregates
        for group, sub_df in all_df.groupby("is_family"):
            label = "Family" if group else "Only Child"
            print(f"\n==={label} Aggregate Rank Usage ===\n", sub_df["term_rank"].value_counts().sort_index())
            print(f"{label} Root Rank Usage:\n", sub_df["root_rank"].value_counts().sort_index())

        # Pivot split (add to existing)
        split_pivot = (
            all_df.groupby(["budget", "capacity", "term_rank", "is_family"])
            .size()
            .unstack(fill_value=0, level="is_family")
        )
        split_pivot.columns = ["Only Child", "Family"]
        split_pivot.to_csv("split_rank_pivot.csv")

        # Global Pct Usage (aggregate)
        global_counts = all_df["global_pct"].value_counts().sort_index()
        print("\n=== Aggregate Global Pct Usage ===\n", global_counts)

        # Corrs incl Global (aggregate)
        numeric_cols = ["term_rank", "root_rank", "global_pct", "value", "family_size"]
        corr_matrix = all_df[numeric_cols].corr()
        print("\n=== Aggregate Corrs incl Global ===\n", corr_matrix.round(2))

        # Family split on global pct (aggregate)
        for group, sub_df in all_df.groupby("is_family"):
            label = "Family" if group else "Only Child"
            print(
                f"\n=== {label} Aggregate Global Pct ===\n", sub_df["global_pct"].value_counts().sort_index()
            )

        # Only-child global pct by 10% bins
        only_df = all_df[all_df["is_family"] == False]
        only_df["global_bin"] = pd.cut(
            only_df["global_pct"],
            bins=range(0, 101, 10),
            right=False,
            labels=[f"{i}-{i + 9.99}%" for i in range(0, 100, 10)],
        )
        only_bins = only_df["global_bin"].value_counts().sort_index().reset_index(name="Count")
        only_bins["% of Only-Child"] = (only_bins["Count"] / 113 * 100).round(1)
        print("\nOnly-Child Global Pct Breakdown (10% Bins):\n", only_bins)
        only_bins.to_csv("only_child_global_bins.csv", index=False)

        # Aggregate Global VPC Usage
        vpc_counts = all_df["global_vpc"].value_counts().sort_index()
        print("\n=== Aggregate Global VPC Usage ===\n", vpc_counts)

        # Corrs incl VPC (aggregate)
        numeric_cols = ["term_rank", "root_rank", "global_pct", "global_vpc", "value", "family_size"]
        corr_matrix = all_df[numeric_cols].corr()
        print("\n=== Aggregate Corrs incl Global & VPC ===\n", corr_matrix.round(2))

        # Family split on VPC (aggregate)
        for group, sub_df in all_df.groupby("is_family"):
            label = "Family" if group else "Only Child"
            print(
                f"\n=== {label} Aggregate Global VPC ===\n", sub_df["global_vpc"].value_counts().sort_index()
            )

        # Summary: Only-Child/Family Aggregate by Budget/Cap (VPC Bins)
        all_df["vpc_bin"] = pd.cut(
            all_df["global_vpc"],
            bins=range(0, 101, 1),
            right=False,
            labels=[f"{i}-{i + 0.99}%" for i in range(0, 100, 1)],
        )
        # summary = (
        #     all_df.groupby(["budget", "capacity", "is_family", "vpc_bin"])
        #     .size()
        #     .unstack(fill_value=0, level="vpc_bin")
        # )
        # summary.columns = [col if col != "nan" else "100%" for col in summary.columns]  # Fix 100%
        # summary["Total"] = summary.sum(axis=1)
        # non_zero_cols = summary.columns[summary.sum() > 0]  # Keep only columns with any non-zero
        # summary_filtered = summary[non_zero_cols]
        # with pd.option_context("display.max_rows", None):
        #     print("\n=== Only-Child/Family VPC Bin Summary by Budget/Cap ===\n", summary_filtered.round(0))
        # summary.to_csv("vpc_summary_by_budget.csv")

        # TODO: We cant do these unless we store the solver_graph data for each of the analyses
        # # Potentials from tr_data (fixed across budgets; compute once)
        # potential_data = []
        # solver_graph = graph_data["solver_graph"]
        # all_pairs_path_lengths = data["all_pairs_path_lengths"]
        # affiliated_town_region = data["affiliated_town_region"]

        # for df in all_analyses:  # Or single if graph fixed
        #     terminal_indices = solver_graph.attrs["terminal_indices"]
        #     prizes_ranks = recompute_ranks(
        #         solver_graph, terminal_indices, all_pairs_path_lengths, affiliated_town_region
        #     )
        #     families = compute_families(solver_graph, terminal_indices)
        #     for t in terminal_indices:
        #         parents = list(solver_graph.predecessor_indices(t))
        #         parent = parents[0] if parents else None
        #         is_family = len(families.get(parent, [t])) > 1
        #         for r, info in prizes_ranks[t].items():
        #             potential_data.append({
        #                 "budget": df["budget"].iloc[0],
        #                 "capacity": df["capacity"].iloc[0],
        #                 "term_rank": info["RootRank_Value_Pct"],
        #                 "root_rank": info["ValueRank_Root_Pct"],
        #                 "global_pct": info["GlobalPct"],
        #                 "global_vpc": info["GlobalVPC"],
        #                 "is_family": is_family,
        #             })

        # pot_df = pd.DataFrame(potential_data)
        # pot_df["vpc_bin"] = pd.cut(
        #     pot_df["global_vpc"],
        #     bins=range(0, 101, 1),
        #     right=False,
        #     labels=[f"{i}-{i + 4.99}%" for i in range(0, 100, 5)],
        # )
        # pot_summary = (
        #     pot_df.groupby(["budget", "capacity", "is_family", "vpc_bin"])
        #     .size()
        #     .unstack(fill_value=0, level="vpc_bin")
        # )
        # pot_summary["Total"] = pot_summary.sum(axis=1)
        # print("\n=== Potential Prizes/Terminals by VPC Bin (by Budget/Cap/Family) ===\n", pot_summary)
        # pot_summary.to_csv("potential_vpc_summary.csv")

        # Compare to used (all_df)
        used_summary = (
            all_df.groupby(["budget", "capacity", "is_family", "vpc_bin"])
            .size()
            .unstack(fill_value=0, level="vpc_bin")
        )
        used_summary["Total"] = used_summary.sum(axis=1)
        print("\n=== Used Prizes/Terminals by VPC Bin (by Budget/Cap/Family) ===\n", used_summary)
        used_summary.to_csv("used_vpc_summary.csv")

    else:
        raise ValueError("No valid budget/cap keys found in JSON.")
