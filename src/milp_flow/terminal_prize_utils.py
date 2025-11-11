# terminal_prize_utils.py
"""Utilities for terminal prize data ranking."""

from enum import Enum
from collections import defaultdict
import math
from typing import Any

from loguru import logger
from rustworkx import PyDiGraph
from scipy.stats import percentileofscore


class SortMetric(str, Enum):
    VALUE = "value"
    NET_T_COST = "net_t_cost"  # value / t_cost
    NET_PATH_COST = "net_path_cost"  # value / path_cost


def sort_key_value(entry):
    return entry["value"]


def sort_key_net_t_cost(entry):
    return entry["value"] / entry["t_cost"]


def sort_key_net_path_cost(entry):
    return entry["value"] / entry["path_cost"]


def get_terminal_families(G: PyDiGraph) -> dict[int, list[int]]:
    """Returns a dict of terminal families by parent index."""
    terminal_indices = G.attrs["terminal_indices"]
    families = {}
    for i in terminal_indices:
        parents = list(G.predecessor_indices(i))
        if parents:
            parent = parents[0]
            families.setdefault(parent, []).append(i)

    return families


def is_family_protected(
    solver_graph: PyDiGraph, entry: dict[str, Any], rank_key: str, threshold: float
) -> bool:
    # Family safeguard: if any sibling in this family is strong for this root, skip prune
    families = solver_graph.attrs["families"]
    prizes_by_root = solver_graph.attrs["prizes_by_root_view"]

    t_idx = entry["t_idx"]
    r_idx = entry["r_idx"]
    parents = list(solver_graph.predecessor_indices(t_idx))
    if parents:
        parent = parents[0]
        family = families.get(parent, [])
        if t_idx in family and len(family) > 1:
            peers_root = len(prizes_by_root[r_idx])
            for fam_t in family:
                fam_entry = next((e for e in prizes_by_root[r_idx] if e["t_idx"] == fam_t), None)
                if fam_entry and fam_entry[rank_key] <= threshold * peers_root:
                    return True
    return False


def get_terminal_root_prizes(
    G: PyDiGraph, all_pairs_path_lengths: dict[tuple[int, int], int]
) -> dict[tuple[int, int], dict[str, Any]]:
    """Returns a dict of terminal root prizes by (t_idx, r_idx)."""
    terminal_indices = G.attrs["terminal_indices"]
    terminal_root_prizes = {}
    for t_idx in terminal_indices:
        for r_idx, value in G[t_idx]["prizes"].items():
            terminal_root_prizes[(t_idx, r_idx)] = {
                "t_idx": t_idx,
                "r_idx": r_idx,
                "value": value,
                "t_cost": G[t_idx]["need_exploration_point"],
                "path_cost": all_pairs_path_lengths[(t_idx, r_idx)],
            }
            logger.trace(f"Terminal {G[t_idx]['waypoint_key']} root {G[r_idx]['waypoint_key']} value {value}")

    return terminal_root_prizes


def get_prizes_by_root_view(
    G: PyDiGraph, tr_data: dict[tuple[int, int], dict[str, Any]], sort_metrics: list[SortMetric] | None = None
) -> dict[int, list[dict[str, Any]]]:
    """Returns a dict of root to terminal valuation data by (t_idx, r_idx) from terminal_root_prizes."""
    if sort_metrics is None:
        sort_metrics = list(SortMetric)

    prizes_by_root_view = defaultdict(list)
    for (t_idx, r_idx), entry in tr_data.items():
        prizes_by_root_view[r_idx].append(entry)

    for r_idx, entries in prizes_by_root_view.items():
        for metric in sort_metrics:
            key = f"terminal_prize_{metric.value}_rank_by_root_view"
            if metric == SortMetric.VALUE:
                sort_key = sort_key_value
            elif metric == SortMetric.NET_T_COST:
                sort_key = sort_key_net_t_cost
            else:  # NET_PATH_COST
                sort_key = sort_key_net_path_cost

            sorted_entries = sorted(entries, key=sort_key, reverse=True)
            for rank, entry in enumerate(sorted_entries, 1):
                entry[key] = rank
                logger.trace(
                    f"Terminal {G[entry['t_idx']]['waypoint_key']} root {G[r_idx]['waypoint_key']} "
                    f"{metric.value} rank {rank}"
                )

    return prizes_by_root_view


def get_prizes_by_terminal_view(
    G: PyDiGraph, tr_data: dict[tuple[int, int], dict[str, Any]], sort_metrics: list[SortMetric] | None = None
) -> dict[int, list[dict[str, Any]]]:
    """Returns a dict of terminal to root valuation data by (t_idx, r_idx) from terminal_root_prizes."""
    if sort_metrics is None:
        sort_metrics = list(SortMetric)

    prizes_by_terminal_view = defaultdict(list)
    for (t_idx, r_idx), entry in tr_data.items():
        prizes_by_terminal_view[t_idx].append(entry)

    for t_idx, entries in prizes_by_terminal_view.items():
        for metric in sort_metrics:
            key = f"root_prize_{metric.value}_rank_by_terminal_view"  # Note: roots ranked, but prize is terminal's
            if metric == SortMetric.VALUE:
                sort_key = sort_key_value
            elif metric == SortMetric.NET_T_COST:
                sort_key = sort_key_net_t_cost
            else:  # NET_PATH_COST
                sort_key = sort_key_net_path_cost

            sorted_entries = sorted(entries, key=sort_key, reverse=True)
            for rank, entry in enumerate(sorted_entries, 1):
                entry[key] = rank
                logger.trace(
                    f"Terminal {G[t_idx]['waypoint_key']} root {G[entry['r_idx']]['waypoint_key']} "
                    f"{metric.value} rank {rank}"
                )

    return prizes_by_terminal_view


def get_full_prize_ranks(
    G: PyDiGraph,
    tr_data: dict[tuple[int, int], dict[str, Any]],
    prizes_by_root_view: dict[int, list[dict[str, Any]]],
    prizes_by_terminal_view: dict[int, list[dict[str, Any]]],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Returns a dict of rank percentiles for all terminal prizes."""
    # Ensure all metrics computed (ranks already in views)
    for entry in tr_data.values():
        for metric in SortMetric:
            if metric == SortMetric.VALUE:
                net = entry["value"]
            elif metric == SortMetric.NET_T_COST:
                net = entry["value"] / entry["t_cost"]
            else:  # NET_PATH_COST
                net = entry["value"] / entry["path_cost"]
            entry[f"terminal_prize_{metric.value}_net"] = net

    # Percentiles by view
    for r_idx, entries in prizes_by_root_view.items():
        for metric in SortMetric:
            metric_nets = [e[f"terminal_prize_{metric.value}_net"] for e in entries]
            pct_key = f"terminal_prize_{metric.value}_pct_by_root_view"
            for entry in entries:
                entry[pct_key] = percentileofscore(
                    metric_nets, entry[f"terminal_prize_{metric.value}_net"], kind="rank"
                )

    for t_idx, entries in prizes_by_terminal_view.items():
        for metric in SortMetric:
            metric_nets = [e[f"terminal_prize_{metric.value}_net"] for e in entries]
            pct_key = f"root_prize_{metric.value}_pct_by_terminal_view"  # Roots ranked, prize terminal's
            for entry in entries:
                entry[pct_key] = percentileofscore(
                    metric_nets, entry[f"terminal_prize_{metric.value}_net"], kind="rank"
                )

    # Global percentiles
    all_values = {m: [e[f"terminal_prize_{m.value}_net"] for e in tr_data.values()] for m in SortMetric}
    for entry in tr_data.values():
        for metric in SortMetric:
            entry[f"terminal_prize_{metric.value}_global_pct"] = percentileofscore(
                all_values[metric], entry[f"terminal_prize_{metric.value}_net"], kind="rank"
            )

    # Aggregate to prize_ranks
    prize_ranks = defaultdict(dict)
    for (t, r), entry in tr_data.items():
        prize_ranks[t][r] = {}
        for metric in SortMetric:
            prize_ranks[t][r][f"terminal_prize_{metric.value}_rank_by_root_view"] = entry.get(
                f"terminal_prize_{metric.value}_rank_by_root_view"
            )
            prize_ranks[t][r][f"terminal_prize_{metric.value}_pct_by_root_view"] = entry.get(
                f"terminal_prize_{metric.value}_pct_by_root_view"
            )
            prize_ranks[t][r][f"root_prize_{metric.value}_rank_by_terminal_view"] = entry.get(
                f"root_prize_{metric.value}_rank_by_terminal_view"
            )
            prize_ranks[t][r][f"root_prize_{metric.value}_pct_by_terminal_view"] = entry.get(
                f"root_prize_{metric.value}_pct_by_terminal_view"
            )
            prize_ranks[t][r][f"terminal_prize_{metric.value}_global_pct"] = entry.get(
                f"terminal_prize_{metric.value}_global_pct"
            )

    return prize_ranks


def get_pruning_threshold(budget: int, capacity_mode: str, key: str) -> float:
    """Return the pruning threshold for the given key."""
    if key == "top_n":
        # ---------------------------------------------------------------
        # Budget-dependent dynamic pruning threshold
        # NOTE: Prunable entries are those ranked below this threshold
        # TODO: Account for bonus lodging capacities - capacity_pressure
        # ---------------------------------------------------------------
        density_center = 100.0
        divisor = 7.0
        expected_ts = min(budget / divisor, density_center)
        density = max(0.0, min(1.0, expected_ts / density_center))
        # Dynamic range: small budgets prune aggressively, large budgets gently
        threshold = 0.20 + 0.70 * density
        logger.info(f"    pruning with threshold {threshold:.3f} (density={density:.3f})")
        return threshold
    elif key == "global":
        # ---------------------------------------------------------------
        # Budget-dependent dynamic global pruning threshold
        # NOTE: Prunable entries are those with global percentile below this threshold
        # TODO: Account for bonus lodging capacities - capacity_pressure
        # ---------------------------------------------------------------
        # From Eureqa solver:
        #   For cap max unprotected: p = 1 - 0.05*ceil(0.978380561715232 + 0.000172951717438112*b + 1.96122942706322*floor(0.00330584990265254*b))
        #   For cap max protected: p = 0.85 + 0.15*floor(5.74823974003092/log(b)) - 0.0499999999999999*ceil(0.762310778720475 + 0.00105235546552624*b)
        #   For cap min unprotected: p = 1 - 0.05*ceil(0.828299998336353 + 7.34705470271906e-6*b^2)

        # Unprotected coeffs (cubic: a*b^3 + b*b^2 + c*b + d)
        if capacity_mode == "max":
            coeffs = [1.84858620e-09, -1.32354602e-06, -1.01324762e-04, 9.63495200e-01]
        else:  # min
            # Constrained cubic fit for unprotected min (<= target, min abs error)
            coeffs = [-1.02335329e-09, 9.06326991e-07, -4.79354896e-04, 9.72101497e-01]

        b3, b2, b = budget**3, budget**2, budget
        threshold = coeffs[0] * b3 + coeffs[1] * b2 + coeffs[2] * b + coeffs[3]
        threshold = max(0.8, min(0.95, threshold))  # Table range cap
        logger.info(f"      global pruning (unprotected baseline) threshold {threshold * 100:.1f}%")
        return threshold
    raise ValueError(f"Unknown pruning threshold key: {key}")


def get_protected_threshold(budget: int, capacity_mode: str, key: str) -> float:
    if key == "global":
        if capacity_mode == "max":
            # Eureqa symbolic for protected max
            import math

            term1 = (
                0.978380561715232
                + 0.000172951717438112 * budget
                + 1.96122942706322 * math.floor(0.00330584990265254 * budget)
            )
            threshold = 1 - 0.05 * math.ceil(term1)
        else:
            # Quadratic fit for protected min
            coeffs = [1.23e-7, -0.00078, 0.95]  # a*b^2 + c*b + d
            threshold = coeffs[0] * budget**2 + coeffs[1] * budget + coeffs[2]
        threshold = max(0.50, min(0.95, threshold))
        logger.info(f"      protected global threshold {threshold * 100:.1f}%")
        return threshold
    raise ValueError(f"Unknown key: {key}")


def nullify_prize_entries(
    solver_graph: PyDiGraph, data: dict[str, Any], prunable_entries: list[dict[str, int]]
):
    """Prune terminal-root pairs with low terminal prizes by nullifying their prizes."""
    logger.info("    nullifying (terminal, root) pairs with low terminal prizes...")
    town_to_region_map = {int(town): region for region, town in data["affiliated_town_region"].items()}
    for entry in prunable_entries:
        t_idx = entry["t_idx"]
        r_idx = entry["r_idx"]
        node = solver_graph[t_idx]
        if prize := node["prizes"].get(r_idx) is not None:
            node["prizes"][r_idx] = 0
            town = solver_graph[r_idx]["waypoint_key"]
            region = town_to_region_map[int(town)]
            node_weight = node["need_exploration_point"]
            path_weight = data["all_pairs_path_lengths"].get((t_idx, r_idx), 2**31)
            logger.debug(
                f"removed prize: {node['waypoint_key']:>5} {region:>5} {town:>5} {prize:>5} {node_weight:>5} {path_weight:>5}"
            )


def prune_null_prize_entries(solver_graph: PyDiGraph, prunable_entries: list[dict[str, int]]):
    """Prune terminal-root pairs with zero value terminal prizes from terminal prizes.

    NOTE: Only removes from graph node payload!
    """
    logger.info("    pruning (terminal, root) pairs with low terminal prizes...")

    terminal_indices = solver_graph.attrs["terminal_indices"]

    total_pairs_pre = sum(len(solver_graph[i]["prizes"]) for i in terminal_indices)

    for entry in prunable_entries:
        node = solver_graph[entry["t_idx"]]
        node["prizes"].pop(entry["r_idx"], None)

    total_pairs_post = sum(len(solver_graph[i]["prizes"]) for i in terminal_indices)

    logger.info(f"      pruned prizes: {total_pairs_pre} → {total_pairs_post} pairs")


def prune_prize_drained_terminals(solver_graph: PyDiGraph):
    """Prune terminal nodes with no prizes from the solver graph, and attrs["terminal_indices"]."""
    logger.info("    pruning terminal nodes with no prizes...")

    terminal_indices = solver_graph.attrs["terminal_indices"].copy()
    removed = 0
    for t in terminal_indices:
        if not solver_graph[t]["is_super_terminal"] and len(solver_graph[t]["prizes"]) == 0:
            removed += remove_solver_graph_node(solver_graph, t)
    solver_graph.attrs["terminal_indices"] = terminal_indices
    logger.info(f"      removed {removed} prize-drained terminals")


# def rebuild_prize_data(solver_graph: PyDiGraph, data: dict[str, Any]):
#     """Rebuild solver_graph prize data based on graph terminal updates during node/prize pruning."""
#     logger.debug("    rebuilding prize data...")

#     families = get_terminal_families(solver_graph)
#     tr_data = get_terminal_root_prizes(solver_graph, data["all_pairs_path_lengths"])
#     prizes_by_terminal_view = get_prizes_by_terminal_view(solver_graph, tr_data)
#     prizes_by_root_view = get_prizes_by_root_view(solver_graph, tr_data)

#     solver_graph.attrs["families"] = families
#     solver_graph.attrs["terminal_root_prizes"] = tr_data
#     solver_graph.attrs["prizes_by_terminal_view"] = prizes_by_terminal_view
#     solver_graph.attrs["prizes_by_root_view"] = prizes_by_root_view

#     # NOTE: The rank-based pruning thresholds must be based on the post-topN filtered data
#     # so if top_n filtered results break the rank-based pruning, this will need to be updated
#     # back to using these for the view arguments to get_full_prize_ranks
#     #     data["prizes_by_root_view"],
#     #     data["prizes_by_terminal_view"],
#     solver_graph.attrs["prize_ranks"] = get_full_prize_ranks(
#         solver_graph,
#         tr_data,
#         prizes_by_root_view,
#         prizes_by_terminal_view,
#     )


def compute_family_dominants_and_protected(
    G: PyDiGraph, tr_data: dict[tuple[int, int], dict[str, Any]], families: dict[int, list[int]]
) -> dict[int, dict[int, int]]:
    """Compute family dominants per root and set is_protected flags in tr_data."""
    family_dominants = defaultdict(dict)
    root_indices = G.attrs.get("root_indices", [])

    # Initialize all entries to False
    for entry in tr_data.values():
        entry["is_protected"] = False

    protected_count = 0
    for r_idx in root_indices:
        for parent, family in families.items():
            if len(family) > 1:
                family_entries = [tr_data[(t, r_idx)] for t in family if (t, r_idx) in tr_data]
                logger.debug(
                    f"Root {r_idx}, parent {parent}: family size {len(family)}, entries found {len(family_entries)}"
                )
                if family_entries:
                    dominant = max(family_entries, key=lambda e: e["value"] / e["path_cost"])
                    family_dominants[r_idx] = dominant["t_idx"]
                    for entry in family_entries:
                        entry["is_protected"] = entry["t_idx"] != dominant["t_idx"]
                        if entry["is_protected"]:
                            protected_count += 1
    logger.info(f"      total protected entries set: {protected_count}")
    return family_dominants


def rebuild_prize_data(solver_graph: PyDiGraph, data: dict[str, Any]):
    """Rebuild solver_graph prize data based on graph terminal updates during node/prize pruning."""
    logger.debug("    rebuilding prize data...")

    families = get_terminal_families(solver_graph)
    tr_data = get_terminal_root_prizes(solver_graph, data["all_pairs_path_lengths"])
    prizes_by_terminal_view = get_prizes_by_terminal_view(solver_graph, tr_data)
    prizes_by_root_view = get_prizes_by_root_view(solver_graph, tr_data)

    solver_graph.attrs["families"] = families
    solver_graph.attrs["terminal_root_prizes"] = tr_data
    solver_graph.attrs["prizes_by_terminal_view"] = prizes_by_terminal_view
    solver_graph.attrs["prizes_by_root_view"] = prizes_by_root_view

    family_dominants = compute_family_dominants_and_protected(solver_graph, tr_data, families)
    solver_graph.attrs["family_dominants"] = family_dominants

    # NOTE: The rank-based pruning thresholds must be based on the post-topN filtered data
    # so if top_n filtered results break the rank-based pruning, this will need to be updated
    # back to using these for the view arguments to get_full_prize_ranks
    #     data["prizes_by_root_view"],
    #     data["prizes_by_terminal_view"],
    solver_graph.attrs["prize_ranks"] = get_full_prize_ranks(
        solver_graph,
        tr_data,
        prizes_by_root_view,
        prizes_by_terminal_view,
    )


def remove_solver_graph_node(solver_graph: PyDiGraph, i: int) -> bool:
    """Remove a node from the solver graph and indices lists.

    NOTE: related (terminal, root) prize data is not updated here!

    returns:
        bool: True if the node was a terminal or root, indicating rank data rebuild may be required
    """
    is_terminal = i in solver_graph.attrs["terminal_indices"]
    is_super_terminal = i in solver_graph.attrs["super_terminal_indices"]
    is_root = i in solver_graph.attrs["root_indices"]
    solver_graph.remove_node(i)

    if is_terminal:
        solver_graph.attrs["terminal_indices"].remove(i)
    if is_super_terminal:
        solver_graph.attrs["super_terminal_indices"].remove(i)
    if is_root:
        solver_graph.attrs["root_indices"].remove(i)

    return is_terminal or is_root
