# analyze_root_converage.py

"""
- For each root, repeatedly add frontier terminals until all terminals are covered.
- Report the terminal and prize value coverage details for each root by ring.

  Note: this analysis can be done using the data within 'data' container.
"""

from typing import Any

import alphashape
from loguru import logger
import pandas as pd
import rustworkx as rx
from rustworkx import PyDiGraph, weakly_connected_components

from api_common import set_logger
from api_exploration_graph import (
    get_all_pairs_shortest_paths,
    has_edge_weights,
    populate_edge_weights,
    add_scaled_coords_to_graph,
)
from api_rx_pydigraph import subgraph_stable
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
import data_store as ds

from api_common import OQUILLAS_EYE_KEY


def find_hull_based_frontier(G, settlement: set[int], frontier: set[int]) -> set[int]:
    nodes = settlement | frontier
    if len(nodes) < 3:
        return set()

    hull = alphashape.alphashape(G.attrs["scaled_coords"](nodes), G.attrs["alpha"])
    if hull.is_empty:  # type: ignore
        return set()
    hull = hull.buffer(0.5)  # type: ignore

    covered_mask = hull.covers(G.attrs["scaled_points"](G.node_indices()))
    covered_indices = G.attrs["node_indices_array"][covered_mask]

    return set(covered_indices) - settlement


def get_new_shortest_paths_nodes(
    G: PyDiGraph,
    settled_terminals: set[int],
    frontier: set[int],
    root: int,
) -> set[int]:
    # Root to frontier terminals
    path_nodes = set()
    frontier_terminals = set()
    for i in frontier:
        if G[i]["is_terminal"]:
            frontier_terminals.add(i)
            path_nodes.update(G.attrs["shortest_paths"][(root, i)])

    # Inter-terminal
    # all_terminals = list(settled_terminals | frontier_terminals)
    all_terminals = list(frontier_terminals)
    for u in all_terminals:
        for v in all_terminals:
            if u >= v:
                continue
            if u in settled_terminals and v in settled_terminals:
                continue
            path_nodes.update(G.attrs["shortest_paths"][(u, v)])

    return path_nodes


def extract_parents(families: dict[int, set[int]], nodes: set[int]):
    return {i for i in nodes if i in families}


def expand_frontier_via_hull(
    G: PyDiGraph,
    data: dict[str, Any],
    root: int,
    frontier: set[int],
    settlement: set[int],
    settled_terminals: set[int],
):
    """Expands frontier until hull + inter-terminal connectivity stabalizes."""
    families = data["families"]

    # Frontier most likely consists of the parents of newly settled terminals
    # but not the terminals themselves.
    terminal_parents = extract_parents(families, frontier)
    terminals_to_settle = set()
    for parent in terminal_parents:
        terminals_to_settle.update(families[parent])
    # frontier.update(terminals_to_settle)

    # new_path_nodes = get_new_shortest_paths_nodes(G, settled_terminals, frontier, root)
    new_path_nodes = get_new_shortest_paths_nodes(G, terminal_parents, frontier, root)
    frontier.update(new_path_nodes)
    frontier = frontier.difference(settlement)

    # settle until hull + inter-terminal connectivity stabalizes
    while True:
        hull_frontier = find_hull_based_frontier(G, settlement, frontier)
        if hull_frontier == frontier:
            break

        terminal_parents = extract_parents(families, hull_frontier - frontier)
        if not terminal_parents:
            break
        frontier.update(hull_frontier)

        # terminals_to_settle = set()
        for parent in terminal_parents:
            terminals_to_settle.update(families[parent])
        # frontier.update(terminals_to_settle)

        new_path_nodes = get_new_shortest_paths_nodes(G, settled_terminals, frontier, root)
        frontier.update(new_path_nodes)

    frontier.update(terminals_to_settle)
    return frontier


def weight_fn(e):
    return e["need_exploration_point"]


def find_frontier_nodes(G: PyDiGraph, settlement: set[int]) -> set[int]:
    """Finds and returns nodes not in settlement with neighbors in settlement."""
    frontier = set()
    for v in settlement:
        frontier.update(G.neighbors(v))
    frontier.difference_update(settlement)
    return frontier


def expand_frontier_to_new_parents(G: PyDiGraph, data: dict[str, Any], settlement: set[int]) -> set[int]:
    """Expands frontier until parents of new terminals are found.
    Returns the set of newly settled nodes.
    """
    orig_settlement = set(settlement)
    frontier = set()
    families = data["families"]

    frontier_nodes = find_frontier_nodes(G, orig_settlement)
    while frontier_nodes and not any(i in families for i in frontier_nodes):
        frontier.update(frontier_nodes)
        orig_settlement.update(frontier_nodes)
        frontier_nodes = find_frontier_nodes(G, orig_settlement)
    frontier.update(frontier_nodes)

    return frontier


def prune_NTD1(graph: PyDiGraph, non_removables: set[int] | None = None):
    """Prunes non terminal nodes of degree 1."""
    if non_removables is None:
        non_removables = set(
            graph.attrs.get("root_indices", [i for i in graph.node_indices() if graph[i]["is_town"]])
        )
        non_removables.update(
            set(
                graph.attrs.get(
                    "terminal_indices", [i for i in graph.node_indices() if graph[i]["is_terminal"]]
                )
            )
        )
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


def get_root_connected_subgraph(graph: PyDiGraph, root: int, settlement: set[int]):
    """Prunes disconnected nodes from settlement returning the set of remaining nodes."""
    subG = subgraph_stable(graph, settlement)
    assert isinstance(subG, PyDiGraph)
    components = weakly_connected_components(subG)
    if len(components) == 1:
        return subG

    root_component = [i for i, c in enumerate(components) if root in c][0]
    return subgraph_stable(subG, components[root_component])


def generate_root_coverage_ring_subgraphs(G: PyDiGraph, data: dict[str, Any], root: int, terminals: set[int]):
    """Generates root coverage rings for a given graph, root and set of terminals."""
    # NOTE: Starting at root build eccentric rings by expanding frontier until no new terminals are found
    # each ring consists of the previous ring's nodes and newly settled nodes for a row of the dataframe.
    ring_subgraphs = []
    remaining_terminals = set(terminals)
    settlement = {root}
    settled_terminals = set()

    # Do frontier expansion without terminals in the graph and add them later
    subG = G.copy()
    subG.attrs = G.attrs
    subG.remove_nodes_from([i for i in G.node_indices() if G[i]["is_terminal"]])

    while remaining_terminals:
        # Frontier nodes are guaranteed to be connected to root, but the
        # inter-terminal shortest paths may not be yet be covered by the hull...
        frontier = expand_frontier_to_new_parents(subG, data, settlement)
        if not frontier:
            break

        # ...so expand frontier until hull + inter-terminal connectivity stabalizes.
        frontier = expand_frontier_via_hull(G, data, root, frontier, settlement, settled_terminals)

        newly_settled_terminals = {i for i in frontier if i in remaining_terminals}
        remaining_terminals.difference_update(newly_settled_terminals)
        settlement.update(frontier)
        settled_terminals.update(newly_settled_terminals)

        pruned_settlement = get_root_connected_subgraph(G, root, settlement)
        assert isinstance(pruned_settlement, PyDiGraph)
        pruned_settlement.attrs = G.attrs
        prune_NTD1(pruned_settlement)
        ring_subgraphs.append(pruned_settlement)

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

    terminals_in_ring = [i for i in nodes_in_ring if ring[i]["is_terminal"]]
    terminal_value_in_ring = sum(round(G[i]["prizes"][r_idx] / 1e6, 3) for i in terminals_in_ring)

    terminals_in_hull = [i for i in ring.node_indices() if G[i]["is_terminal"]]
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
            "hull_nodes": bin_rings[-1]["hull_nodes"],
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
        "value_range": "0.0-100.0",
        "hull_nodes": ring_analysis[-1]["hull_nodes"],
    }
    value_bins = pd.concat([value_bins, pd.DataFrame([full_row])], ignore_index=True)

    return value_bins


# Simply a helper for figuring out alpha and buffer
def average_non_terminal_edge_length(G: rx.PyGraph | rx.PyDiGraph) -> float:
    """
    Returns the average Euclidean distance of edges where
    NEITHER endpoint is a terminal node.
    Uses scaled coordinates (graph.attrs["scaled_coords"]).
    """
    if "scaled_coords" not in G.attrs:
        raise ValueError("Graph missing scaled_coords function")

    scaled = G.attrs["scaled_coords"](G.node_indices())
    scaled = {idx: p for idx, p in zip(G.node_indices(), scaled)}
    total_length = 0.0
    count = 0

    for u, v in G.edge_list():
        u_node = G[u]
        v_node = G[v]

        # Skip if either endpoint is a terminal
        if u_node.get("is_terminal", False) or v_node.get("is_terminal", False):
            continue

        p1 = scaled[u]
        p2 = scaled[v]
        length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        total_length += length
        count += 1

    return total_length / count if count > 0 else 0.0


def run_analysis(config: dict[str, Any], data: dict[str, Any], bin_count: int = 5):
    G = data["G"]

    terminal_indices = G.attrs["terminal_indices"]
    root_indices = G.attrs["root_indices"]

    if not has_edge_weights(G):
        populate_edge_weights(G)

    # avg_len = average_non_terminal_edge_length(G)
    # logger.info(f"Average internal edge length (non-terminal): {avg_len:.1f} map units")

    import time

    start = time.perf_counter()

    all_analysis = {}
    for r_idx in root_indices:
        terminal_entries = [
            {
                "t_idx": t,
                "r_idx": r_idx,
                "value": round(G[t]["prizes"][r_idx] / 1e6, 3),
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
            G, data, r_idx, {t["t_idx"] for t in terminal_entries}
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

    end = time.perf_counter()
    logger.info(f"    analysis took {end - start:.3f} seconds")

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
                "Terminal": row["terminal_count"],
                "CumTerminal": row["CumTerminal"],
                "%Term": row["%Term"],
                "Cum%Term": row["Cum%Term"],
                "Prize": row["terminal_value"],
                "CumPrize": row["CumPrize"],
                "%Prize": row["%Prize"],
                "Cum%Prize": row["Cum%Prize"],
                "hull_nodes": str(row["hull_nodes"]),
            })

        bin_df = value_bins.copy()
        # Compute % before rename
        bin_df["%Term"] = bin_df["num_terms"] / total_terminals * 100
        bin_df["Cum%Term"] = bin_df["cum_terms"] / total_terminals * 100
        bin_df["%Prize"] = bin_df["sum_prize"] / total_prize * 100
        bin_df["Cum%Prize"] = bin_df["cum_prize"] / total_prize * 100

        # Single rename at end
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
                "hull_nodes": str(row["hull_nodes"]),
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
    G = data["G"]
    G.remove_node(G.attrs["node_key_by_index"].inv[OQUILLAS_EYE_KEY])
    prune_NTD1(G)
    G.attrs["shortest_paths"] = get_all_pairs_shortest_paths(G)
    add_scaled_coords_to_graph(G)

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
