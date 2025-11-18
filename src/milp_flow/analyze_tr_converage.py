# analyze_tr_converage.py

"""
- For each root, repeatedly add frontier terminals until all terminals are covered.
- Report the terminal and prize value coverage details for each root by ring.

  Note: this analysis can be done using the data within 'data' container.
"""

from typing import Any

import alphashape
from loguru import logger
import numpy as np
import pandas as pd
from rustworkx import PyDiGraph
import rustworkx as rx
from shapely import Point

from api_common import set_logger
from api_exploration_graph import has_edge_weights, populate_edge_weights
from api_rx_pydigraph import subgraph_stable
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
import data_store as ds

from api_common import OQUILLAS_EYE_KEY


def get_cartesian(G: PyDiGraph, idx: int) -> tuple[float, float]:
    p = G[idx]["position"]
    return (float(p["x"]), float(p["z"]))


def get_hull(G: PyDiGraph, settlement: set[int], alpha=0.25):
    if len(settlement) < 3:
        return None
    coords = [get_cartesian(G, idx) for idx in settlement]
    try:
        hull = alphashape.alphashape(coords, alpha)  # type: ignore
        return hull.buffer(0.5)  # type: ignore
    except Exception as e:
        logger.warning(f"Hull failed: {e}")
        return None


def get_hull_covered_settlement(G: PyDiGraph, settlement: set[int], hull) -> bool:
    """Updates settlement in place with nodes covered by hull.
    Returns true if any new nodes covered by hull have been settled.
    """
    if hull is None:
        return False
    new_settlement = settlement.copy()
    new_settlement.update({
        idx
        for idx in G.node_indices()
        if idx not in settlement and hull.covers(Point(*get_cartesian(G, idx)))
    })
    if len(new_settlement) == len(settlement):
        return False
    settlement.update(new_settlement)
    return True


def find_frontier_nodes(G: PyDiGraph, settlement: set[int]) -> set[int]:
    """Finds and returns nodes not in settlement with neighbors in settlement."""
    frontier = set()
    for v in settlement:
        frontier.update(G.neighbors(v))
    frontier.difference_update(settlement)
    return frontier


def expand_frontier_to_new_terminals(G: PyDiGraph, settlement: set[int], terminals: set[int]) -> set[int]:
    """Expands frontier until new terminals are found.
    Returns previously frontier nodes that are now settled.
    """
    wild_frontier_terminals = terminals - settlement
    new_settlement = settlement.copy()
    new_frontier = find_frontier_nodes(G, new_settlement)
    while not new_frontier.intersection(wild_frontier_terminals):
        new_settlement.update(new_frontier)
        new_frontier = find_frontier_nodes(G, new_settlement)
    new_settlement.update(new_frontier)
    new_settlement.difference_update(settlement)

    return new_settlement


def ensure_all_inter_terminal_paths_are_covered(
    G: PyDiGraph, settlement: set[int], settled_terminals: set[int], newly_settled_terminals: set[int]
):
    """Adds any shortest path nodes between root, settled terminals and newly settled terminals to settlement.
    Returns true if any new nodes were added and hull needs to be updated.
    """
    new_settlement = settlement.copy()
    for terminal in newly_settled_terminals:
        for s_terminal in settled_terminals | newly_settled_terminals:
            if s_terminal == terminal:
                continue
            path = rx.dijkstra_shortest_paths(
                G, s_terminal, terminal, weight_fn=lambda e: e["need_exploration_point"]
            )
            new_settlement.update(path[terminal])
    if new_settlement != settlement:
        settlement.update(new_settlement)
        return True
    return False


def prune_NTD1(graph: PyDiGraph, non_removables: set[int] | None = None):
    """Prunes non terminal nodes of degree 1."""
    if non_removables is None:
        non_removables = set(graph.attrs.get("root_indices", []))
        non_removables.update(set(graph.attrs.get("terminal_indices", [])))
        if graph.attrs.get("super_root_index", None) is not None:
            non_removables.add(graph.attrs["super_root_index"])
            non_removables.update(set(graph.attrs.get("super_terminal_indices", [])))

    num_removed = 0
    while removal_nodes := [
        v for v in graph.node_indices() if graph.out_degree(v) == 1 and v not in non_removables
    ]:
        graph.remove_nodes_from(removal_nodes)
        num_removed += len(removal_nodes)
        removal_nodes = []

    return num_removed


def get_root_connected_subset(graph: PyDiGraph, root: int, settlement: set[int]):
    """Prunes disconnected nodes from settlement returning the set of remaining nodes."""
    new_settlement = settlement.copy()
    subG = subgraph_stable(graph, new_settlement)
    for v in settlement:
        if not rx.has_path(subG, root, v):
            new_settlement.remove(v)
    return new_settlement


def generate_root_coverage_ring_subgraphs(G: PyDiGraph, root: int, terminals: set[int], alpha=0.25):
    """Generates root coverage rings for a given graph, root and set of terminals."""
    # NOTE: Starting at root build eccentric rings by expanding frontier until no new terminals are found
    # each ring consists of the next layer of nodes to be settled for each row of the dataframe.
    rings = []
    remaining_terminals = set(terminals)
    settlement = {root}
    settled_terminals = set()

    while remaining_terminals:
        newly_settled_nodes = expand_frontier_to_new_terminals(G, settlement, remaining_terminals)
        if not newly_settled_nodes:
            break

        settlement.update(newly_settled_nodes)
        newly_from_frontier = remaining_terminals.intersection(settlement)
        remaining_terminals.difference_update(newly_from_frontier)

        # settle until hull + inter-terminal connectivity stops changing
        while True:
            changed = False

            hull = get_hull(G, settlement, alpha=alpha)
            if hull is not None:
                if get_hull_covered_settlement(G, settlement, hull):
                    changed = True

            newly_from_hull = remaining_terminals.intersection(settlement)
            if newly_from_hull:
                remaining_terminals.difference_update(newly_from_hull)

            if ensure_all_inter_terminal_paths_are_covered(
                G, settlement, settled_terminals, newly_from_frontier | newly_from_hull
            ):
                changed = True

            if newly_from_frontier or newly_from_hull:
                settled_terminals.update(newly_from_frontier)
                settled_terminals.update(newly_from_hull)
                newly_from_frontier = set()

            if not changed:
                break

        rings.append(settlement.copy())
        settlement = get_root_connected_subset(G, root, settlement)

    ring_subgraphs = [subgraph_stable(G, r) for r in rings]
    for subgraph in ring_subgraphs:
        assert isinstance(subgraph, PyDiGraph)
        prune_NTD1(subgraph)

    return ring_subgraphs


def analyze_ring(
    G: PyDiGraph,
    prev_ring: PyDiGraph | None,
    ring: PyDiGraph,
    r_idx: int,
    terminal_entries: list[dict[str, Any]],
):
    nodes_in_ring = (
        set(ring.node_indices()) - set(prev_ring.node_indices())
        if prev_ring is not None
        else set(ring.node_indices())
    )

    terminals_in_ring = [i for i in nodes_in_ring if ring[i]["is_workerman_plantzone"]]
    terminal_value_in_ring = sum(round(G[i]["prizes"][r_idx] / 1e6, 3) for i in terminals_in_ring)

    terminals_in_hull = [i for i in ring.node_indices() if G[i]["is_workerman_plantzone"]]
    terminal_value_in_hull = sum(round(G[i]["prizes"][r_idx] / 1e6, 3) for i in terminals_in_hull)

    ttl_terminal_count = len(terminal_entries)
    ttl_terminal_value = sum(e["value"] for e in terminal_entries)

    return {
        "terminal_count": len(terminals_in_ring),
        "cum_terminal_count": len(terminals_in_hull),
        "terminal_count_pct": (len(terminals_in_ring) / ttl_terminal_count) if ttl_terminal_count else 0.0,
        "cum_terminal_count_pct": (len(terminals_in_hull) / ttl_terminal_count)
        if ttl_terminal_count
        else 0.0,
        "terminal_value": terminal_value_in_ring,
        "cum_terminal_value": terminal_value_in_hull,
        "terminal_value_pct": (terminal_value_in_ring / ttl_terminal_value) if ttl_terminal_value else 0.0,
        "cum_terminal_value_pct": (terminal_value_in_hull / ttl_terminal_value)
        if ttl_terminal_value
        else 0.0,
        "ring_nodes": sorted(nodes_in_ring),
        "hull_nodes": sorted(ring.node_indices()),
    }


def generate_root_df_analysis(ring_analysis: list[dict], bin_count: int = 5):
    """Generate root-level statistics and value-bin aggregation based on per-ring analysis."""

    total_rings = len(ring_analysis)
    if total_rings == 0:
        return pd.DataFrame()

    # Ensure bin_count does not exceed total_rings
    bin_count = min(bin_count, total_rings)
    rings_per_bin = total_rings // bin_count
    extra = total_rings % bin_count  # distribute remainder across first few bins

    bins = []
    start = 0
    for b in range(bin_count):
        end = start + rings_per_bin + (1 if b < extra else 0)
        bin_rings = ring_analysis[start:end]

        if not bin_rings:
            continue  # safety, should not happen

        bin_row = {
            "num_terms": sum(r["terminal_count"] for r in bin_rings),
            "cum_terms": bin_rings[-1]["cum_terminal_count"],
            "sum_prize": sum(r["terminal_value"] for r in bin_rings),
            "cum_prize": bin_rings[-1]["cum_terminal_value"],
        }
        total_prize = sum(r["terminal_value"] for r in ring_analysis)
        bin_row["per_%_total_prize"] = (bin_row["sum_prize"] / total_prize * 100) if total_prize else 0.0
        bin_row["cum_%_total_prize"] = (bin_row["cum_prize"] / total_prize * 100) if total_prize else 0.0
        bin_row["value_range"] = (
            f"{bin_row['cum_%_total_prize'] - bin_row['per_%_total_prize']:.1f}-{bin_row['cum_%_total_prize']:.1f}"
        )
        bins.append(bin_row)
        start = end

    value_bins = pd.DataFrame(bins)
    full_row = {
        "num_terms": sum(r["terminal_count"] for r in ring_analysis),
        "cum_terms": ring_analysis[-1]["cum_terminal_count"],
        "sum_prize": sum(r["terminal_value"] for r in ring_analysis),
        "cum_prize": ring_analysis[-1]["cum_terminal_value"],
        "per_%_total_prize": 100.0,
        "cum_%_total_prize": 100.0,
        "value_range": f"0.0-100.0",
    }
    value_bins = pd.concat([value_bins, pd.DataFrame([full_row])], ignore_index=True)

    return value_bins


def run_analysis(config: dict[str, Any], data: dict[str, Any], bin_count: int = 5):
    G = data["G"]

    G.remove_node(G.attrs["node_key_by_index"].inv[OQUILLAS_EYE_KEY])
    prune_NTD1(G)

    terminal_indices = G.attrs["terminal_indices"]
    root_indices = G.attrs["root_indices"]
    all_pairs_path_lengths = data["all_pairs_path_lengths"]

    if not has_edge_weights(G):
        populate_edge_weights(G)

    all_analysis = {}
    for r_idx in root_indices[:2]:
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

        ring_subgraphs = generate_root_coverage_ring_subgraphs(
            G, r_idx, {t["t_idx"] for t in terminal_entries}
        )

        ring_analysis = []
        for prev_ring, ring in zip([None] + ring_subgraphs, ring_subgraphs):
            assert isinstance(ring, PyDiGraph)
            assert isinstance(prev_ring, PyDiGraph | None)
            ring_analysis.append(analyze_ring(G, prev_ring, ring, r_idx, terminal_entries))

        # terminal dataframe
        df = pd.DataFrame(terminal_entries)
        df["root_key"] = G[r_idx]["waypoint_key"]

        value_bins = generate_root_df_analysis(ring_analysis, bin_count)

        all_analysis[r_idx] = {
            "value_bins": value_bins,
            "df": df,
            "ring_analysis": ring_analysis,
        }

    return all_analysis


def generate_report(all_analysis, bin_count: int = 5):
    report_data = []

    for r_idx, analysis in all_analysis.items():
        df = analysis["df"]
        ring_analysis = pd.DataFrame(analysis["ring_analysis"])
        value_bins = analysis["value_bins"]
        root_key = df["root_key"].iloc[0]

        # Compute totals
        total_terminals = ring_analysis["terminal_count"].sum()
        total_prize = ring_analysis["terminal_value"].sum()

        print(f"\nRoot {root_key}:")
        print(f"  Total rings: {len(ring_analysis)}")
        print(f"  Total terminals covered: {total_terminals}")
        print(f"  Total terminal prize: {total_prize:.3f}")

        # Per-ring breakdown using vectorized calculations
        ring_analysis = ring_analysis.assign(
            CumTerminal=ring_analysis["terminal_count"].cumsum(),
            CumPrize=ring_analysis["terminal_value"].cumsum(),
        )
        ring_analysis["%Term"] = ring_analysis["terminal_count"] / total_terminals * 100
        ring_analysis["Cum%Term"] = ring_analysis["CumTerminal"] / total_terminals * 100
        ring_analysis["%Prize"] = ring_analysis["terminal_value"] / total_prize * 100
        ring_analysis["Cum%Prize"] = ring_analysis["CumPrize"] / total_prize * 100
        ring_analysis["Ring"] = ring_analysis.index

        print("  Per-ring breakdown:")
        print(
            ring_analysis[
                [
                    "Ring",
                    "terminal_count",
                    "CumTerminal",
                    "%Term",
                    "Cum%Term",
                    "terminal_value",
                    "CumPrize",
                    "%Prize",
                    "Cum%Prize",
                ]
            ]
            .rename(columns={"terminal_count": "Terminal", "terminal_value": "Prize"})
            .to_string(index=False, float_format="{:7.3f}".format)
        )

        for _, row in ring_analysis.iterrows():
            report_data.append({
                "root": root_key,
                "ring_index": row["Ring"],
                "Terminal": row["Terminal"] if "Terminal" in row else row["terminal_count"],
                "CumTerminal": row["CumTerminal"],
                "%Term": row["%Term"],
                "Cum%Term": row["Cum%Term"],
                "Prize": row["Prize"] if "Prize" in row else row["terminal_value"],
                "CumPrize": row["CumPrize"],
                "%Prize": row["%Prize"],
                "Cum%Prize": row["Cum%Prize"],
            })

        bin_df = value_bins.copy()
        bin_df["%Term"] = bin_df["num_terms"] / total_terminals * 100
        bin_df["Cum%Term"] = bin_df["cum_terms"] / total_terminals * 100
        bin_df["%Prize"] = bin_df["sum_prize"] / total_prize * 100
        bin_df["Cum%Prize"] = bin_df["cum_prize"] / total_prize * 100

        bin_df = bin_df.rename(
            columns={
                "value_range": "BinRange",
                "num_terms": "Terminal",
                "sum_prize": "Prize",
                "cum_terms": "CumTerminal",
                "cum_prize": "CumPrize",
            }
        )

        print("  Value-bin summary:")
        print(
            bin_df[
                [
                    "BinRange",
                    "Terminal",
                    "CumTerminal",
                    "%Term",
                    "Cum%Term",
                    "Prize",
                    "CumPrize",
                    "%Prize",
                    "Cum%Prize",
                ]
            ].to_string(index=False, float_format="{:7.3f}".format)
        )

        for _, row in bin_df.iterrows():
            report_data.append({
                "root": root_key,
                "ring_index": f"bin_{row['BinRange']}",
                "Terminal": row["Terminal"],
                "CumTerminal": row["CumTerminal"],
                "%Term": row["%Term"],
                "Cum%Term": row["Cum%Term"],
                "Prize": row["Prize"],
                "CumPrize": row["CumPrize"],
                "%Prize": row["%Prize"],
                "Cum%Prize": row["Cum%Prize"],
            })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv("tr_coverage_report.csv", index=False)
    logger.info("Generated tr_coverage_report.csv")


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
        "bin_count": 10,
    }

    main(config, kwargs)
