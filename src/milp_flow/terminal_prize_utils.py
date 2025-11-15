# terminal_prize_utils.py
"""Utilities for terminal prize data ranking."""

from enum import Enum
from collections import Counter, defaultdict
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


class Category(str, Enum):
    ONLY_CHILD = "only_child"
    DOMINANT = "dominant"
    PROTECTED = "protected"


class PCTThresholdLookup:
    """Lookup for per-category PCT thresholds by budget and cap_mode."""

    def __init__(
        self, thresholds_data: dict[str, dict[int, dict[str, float]]], threshold_factors: dict[str, int]
    ):
        self.thresholds = thresholds_data
        self.factors = threshold_factors
        logger.info(f"      pruning threshold factors: {self.factors}")
        self._validate()

    def _validate(self):
        valid_cats = [c.value for c in Category]
        for cap_mode, budget_dict in self.thresholds.items():
            if cap_mode not in ["min", "max"]:
                raise ValueError(f"Invalid cap_mode: {cap_mode}")
            for budget, cat_dict in budget_dict.items():
                if budget < 0:
                    raise ValueError(f"Invalid budget: {budget}")
                for cat, pct in cat_dict.items():
                    if cat not in valid_cats:
                        raise ValueError(f"Invalid category: {cat}")
                    if not 0 <= pct <= 100:
                        raise ValueError(f"Invalid pct for {cat}: {pct}")

    def get_threshold(
        self, budget: int, cap_mode: str, category: str, fallback_next_highest: bool = True
    ) -> float:
        if cap_mode not in self.thresholds:
            raise ValueError(f"Unknown cap_mode: {cap_mode}")
        budget_dict = self.thresholds[cap_mode]
        if budget in budget_dict and category in budget_dict[budget]:
            return budget_dict[budget][category]

        if fallback_next_highest:
            # Next highest budget >= input
            candidates = [b for b in sorted(budget_dict.keys()) if b >= budget]
            if candidates:
                nearest_higher = min(candidates)
                if category in budget_dict[nearest_higher]:
                    logger.debug(f"Used next highest budget {nearest_higher} for {budget}")
                    return budget_dict[nearest_higher][category]

        logger.debug(
            f"No threshold for budget {budget}, cap {cap_mode}, category {category} using 1/2 600 Budget threshold"
        )
        return 0.5 * budget_dict[600][category]


# Thresholds for using the net_path_cost pruning
_THRESHOLDS_DATA: dict[str, dict[int, dict[str, float]]] = {
    "min": {
        50: {Category.ONLY_CHILD: 98, Category.DOMINANT: 98, Category.PROTECTED: 97},
        75: {Category.ONLY_CHILD: 98, Category.DOMINANT: 98, Category.PROTECTED: 97},
        100: {Category.ONLY_CHILD: 98, Category.DOMINANT: 97, Category.PROTECTED: 97},
        125: {Category.ONLY_CHILD: 98, Category.DOMINANT: 95, Category.PROTECTED: 95},
        150: {Category.ONLY_CHILD: 95, Category.DOMINANT: 95, Category.PROTECTED: 95},
        175: {Category.ONLY_CHILD: 93, Category.DOMINANT: 95, Category.PROTECTED: 95},
        200: {Category.ONLY_CHILD: 93, Category.DOMINANT: 95, Category.PROTECTED: 95},
        225: {Category.ONLY_CHILD: 93, Category.DOMINANT: 95, Category.PROTECTED: 95},
        250: {Category.ONLY_CHILD: 93, Category.DOMINANT: 95, Category.PROTECTED: 90},
        275: {Category.ONLY_CHILD: 93, Category.DOMINANT: 90, Category.PROTECTED: 89},
        300: {Category.ONLY_CHILD: 93, Category.DOMINANT: 90, Category.PROTECTED: 89},
        325: {Category.ONLY_CHILD: 89, Category.DOMINANT: 90, Category.PROTECTED: 75},
        350: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        375: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        400: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        425: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        450: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        475: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        500: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        525: {Category.ONLY_CHILD: 79, Category.DOMINANT: 85, Category.PROTECTED: 75},
        550: {Category.ONLY_CHILD: 79, Category.DOMINANT: 84, Category.PROTECTED: 75},
        575: {Category.ONLY_CHILD: 79, Category.DOMINANT: 84, Category.PROTECTED: 75},
        600: {Category.ONLY_CHILD: 79, Category.DOMINANT: 84, Category.PROTECTED: 75},
    },
    "max": {
        50: {Category.ONLY_CHILD: 99, Category.DOMINANT: 99, Category.PROTECTED: 99},
        75: {Category.ONLY_CHILD: 99, Category.DOMINANT: 95, Category.PROTECTED: 97},
        100: {Category.ONLY_CHILD: 98, Category.DOMINANT: 95, Category.PROTECTED: 95},
        125: {Category.ONLY_CHILD: 95, Category.DOMINANT: 93, Category.PROTECTED: 94},
        150: {Category.ONLY_CHILD: 95, Category.DOMINANT: 93, Category.PROTECTED: 94},
        175: {Category.ONLY_CHILD: 95, Category.DOMINANT: 93, Category.PROTECTED: 89},
        200: {Category.ONLY_CHILD: 95, Category.DOMINANT: 93, Category.PROTECTED: 89},
        225: {Category.ONLY_CHILD: 95, Category.DOMINANT: 91, Category.PROTECTED: 82},
        250: {Category.ONLY_CHILD: 93, Category.DOMINANT: 91, Category.PROTECTED: 72},
        275: {Category.ONLY_CHILD: 93, Category.DOMINANT: 90, Category.PROTECTED: 72},
        300: {Category.ONLY_CHILD: 93, Category.DOMINANT: 90, Category.PROTECTED: 72},
        325: {Category.ONLY_CHILD: 93, Category.DOMINANT: 90, Category.PROTECTED: 72},
        350: {Category.ONLY_CHILD: 93, Category.DOMINANT: 90, Category.PROTECTED: 72},
        375: {Category.ONLY_CHILD: 93, Category.DOMINANT: 89, Category.PROTECTED: 72},
        400: {Category.ONLY_CHILD: 89, Category.DOMINANT: 87, Category.PROTECTED: 72},
        425: {Category.ONLY_CHILD: 89, Category.DOMINANT: 87, Category.PROTECTED: 72},
        450: {Category.ONLY_CHILD: 89, Category.DOMINANT: 87, Category.PROTECTED: 72},
        475: {Category.ONLY_CHILD: 88, Category.DOMINANT: 84, Category.PROTECTED: 72},
        500: {Category.ONLY_CHILD: 88, Category.DOMINANT: 84, Category.PROTECTED: 58},
        525: {Category.ONLY_CHILD: 86, Category.DOMINANT: 84, Category.PROTECTED: 58},
        550: {Category.ONLY_CHILD: 82, Category.DOMINANT: 84, Category.PROTECTED: 58},
        575: {Category.ONLY_CHILD: 82, Category.DOMINANT: 84, Category.PROTECTED: 58},
        600: {Category.ONLY_CHILD: 82, Category.DOMINANT: 84, Category.PROTECTED: 58},
    },
}

# Thresholds for using the raw_value pruning
# _THRESHOLDS_DATA: dict[str, dict[int, dict[str, float]]] = {
#     "min": {
#         50: {Category.ONLY_CHILD: 88, Category.DOMINANT: 69, Category.PROTECTED: 70},
#         75: {Category.ONLY_CHILD: 88, Category.DOMINANT: 69, Category.PROTECTED: 67},
#         100: {Category.ONLY_CHILD: 88, Category.DOMINANT: 69, Category.PROTECTED: 67},
#         125: {Category.ONLY_CHILD: 88, Category.DOMINANT: 69, Category.PROTECTED: 61},
#         150: {Category.ONLY_CHILD: 87, Category.DOMINANT: 69, Category.PROTECTED: 61},
#         175: {Category.ONLY_CHILD: 83, Category.DOMINANT: 69, Category.PROTECTED: 61},
#         200: {Category.ONLY_CHILD: 83, Category.DOMINANT: 69, Category.PROTECTED: 61},
#         225: {Category.ONLY_CHILD: 83, Category.DOMINANT: 69, Category.PROTECTED: 58},
#         250: {Category.ONLY_CHILD: 78, Category.DOMINANT: 69, Category.PROTECTED: 41},
#         275: {Category.ONLY_CHILD: 78, Category.DOMINANT: 69, Category.PROTECTED: 41},
#         300: {Category.ONLY_CHILD: 76, Category.DOMINANT: 69, Category.PROTECTED: 41},
#         325: {Category.ONLY_CHILD: 76, Category.DOMINANT: 69, Category.PROTECTED: 39},
#         350: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         375: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         400: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         425: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         450: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         475: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         500: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         525: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         550: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         575: {Category.ONLY_CHILD: 53, Category.DOMINANT: 62, Category.PROTECTED: 39},
#         600: {Category.ONLY_CHILD: 53, Category.DOMINANT: 61, Category.PROTECTED: 39},
#     },
#     "max": {
#         50: {Category.ONLY_CHILD: 88, Category.DOMINANT: 77, Category.PROTECTED: 72},
#         75: {Category.ONLY_CHILD: 88, Category.DOMINANT: 74, Category.PROTECTED: 72},
#         100: {Category.ONLY_CHILD: 88, Category.DOMINANT: 74, Category.PROTECTED: 72},
#         125: {Category.ONLY_CHILD: 88, Category.DOMINANT: 66, Category.PROTECTED: 63},
#         150: {Category.ONLY_CHILD: 88, Category.DOMINANT: 66, Category.PROTECTED: 63},
#         175: {Category.ONLY_CHILD: 78, Category.DOMINANT: 66, Category.PROTECTED: 62},
#         200: {Category.ONLY_CHILD: 78, Category.DOMINANT: 66, Category.PROTECTED: 61},
#         225: {Category.ONLY_CHILD: 78, Category.DOMINANT: 65, Category.PROTECTED: 58},
#         250: {Category.ONLY_CHILD: 76, Category.DOMINANT: 65, Category.PROTECTED: 54},
#         275: {Category.ONLY_CHILD: 76, Category.DOMINANT: 65, Category.PROTECTED: 51},
#         300: {Category.ONLY_CHILD: 76, Category.DOMINANT: 63, Category.PROTECTED: 51},
#         325: {Category.ONLY_CHILD: 76, Category.DOMINANT: 63, Category.PROTECTED: 51},
#         350: {Category.ONLY_CHILD: 75, Category.DOMINANT: 63, Category.PROTECTED: 51},
#         375: {Category.ONLY_CHILD: 75, Category.DOMINANT: 49, Category.PROTECTED: 49},
#         400: {Category.ONLY_CHILD: 75, Category.DOMINANT: 49, Category.PROTECTED: 49},
#         425: {Category.ONLY_CHILD: 75, Category.DOMINANT: 49, Category.PROTECTED: 49},
#         450: {Category.ONLY_CHILD: 75, Category.DOMINANT: 49, Category.PROTECTED: 49},
#         475: {Category.ONLY_CHILD: 58, Category.DOMINANT: 49, Category.PROTECTED: 39},
#         500: {Category.ONLY_CHILD: 58, Category.DOMINANT: 37, Category.PROTECTED: 39},
#         525: {Category.ONLY_CHILD: 58, Category.DOMINANT: 37, Category.PROTECTED: 39},
#         550: {Category.ONLY_CHILD: 57, Category.DOMINANT: 37, Category.PROTECTED: 39},
#         575: {Category.ONLY_CHILD: 53, Category.DOMINANT: 37, Category.PROTECTED: 29},
#         600: {Category.ONLY_CHILD: 53, Category.DOMINANT: 37, Category.PROTECTED: 29},
#     },
# }


def assign_terminal_categories(G: PyDiGraph, families: dict[int, list[int]]) -> None:
    """Assign global category ('only_child', 'dominant', 'protected') to each terminal node."""
    logger.info("    assigning terminal categories...")
    terminal_categories = {}
    for parent, family_ts in families.items():
        family_size = len(family_ts)
        if family_size == 1:
            category = Category.ONLY_CHILD.value
            G[family_ts[0]]["category"] = category
            terminal_categories[family_ts[0]] = category
            continue

        # Dominant: max top ranked prize value across all terminals
        dominant_t = max(family_ts, key=lambda t: max(G[t]["prizes"].values()))
        for t in family_ts:
            category = Category.DOMINANT.value if t == dominant_t else Category.PROTECTED.value
            terminal_categories[t] = category
            G[t]["category"] = category
            G[t]["protector"] = dominant_t
    logger.info(f"      category counts: {dict(Counter(terminal_categories.values()))}")


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
