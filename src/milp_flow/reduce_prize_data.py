# reduce_prize_data.py

from copy import deepcopy
from typing import Any

from loguru import logger
from rustworkx import PyDiGraph

import terminal_prize_utils as tpu
from terminal_prize_utils import Category


# def apply_top_n_filtering(solver_graph: PyDiGraph, data: dict[str, Any]):
#     """Limits the number of prize entries per terminal."""
#     top_n = data["config"]["top_n"]
#     logger.info(f"      applying top-n filtering for the top {top_n} prizes per terminal...")

#     starting_prize_count = len(solver_graph.attrs["terminal_root_prizes"])
#     for t_idx in solver_graph.attrs["terminal_indices"]:
#         prizes = solver_graph[t_idx]["prizes"]
#         prizes = sorted(
#             prizes.items(),
#             key=lambda x: x[1],
#             reverse=True,
#         )
#         solver_graph[t_idx]["prizes"] = dict(prizes[:top_n])

#         # Remove entries from tr_data for all entries beyond top_n
#         tr_data = solver_graph.attrs["terminal_root_prizes"]
#         for entry in prizes[top_n:]:
#             tr_data.pop((t_idx, entry[0]))

#     ending_prize_count = len(solver_graph.attrs["terminal_root_prizes"])

#     logger.info(
#         f"        removed {starting_prize_count - ending_prize_count} terminal-root pairs\n"
#         f"        remaining terminal-root pairs: {ending_prize_count}"
#     )


def apply_top_n_filtering(solver_graph: PyDiGraph, data: dict[str, Any]):
    """Limits the number of prize entries per terminal by net path cost."""
    top_n = data["config"]["top_n"]
    logger.info(f"      applying top-n filtering for the top {top_n} prizes per terminal (net path cost)...")

    all_pairs = data["all_pairs_path_lengths"]
    starting_prize_count = len(solver_graph.attrs["terminal_root_prizes"])
    for t_idx in solver_graph.attrs["terminal_indices"]:
        prizes = solver_graph[t_idx]["prizes"]

        # # Sort by Raw Value - this captures the optimal (albeit not as fast as VPC)
        # prizes = sorted(prizes.items(), key=lambda item: item[1], reverse=True)
        # solver_graph[t_idx]["prizes"] = dict(prizes[:top_n])

        # # Remove entries from tr_data for all entries beyond top_n
        # tr_data = solver_graph.attrs["terminal_root_prizes"]
        # for entry in prizes[top_n:]:
        #     tr_data.pop((t_idx, entry[0]))

        # # Sort by VPC => value / path_cost - this captures the optimal
        # prizes = sorted(prizes.items(), key=lambda item: item[1] / all_pairs[(t_idx, item[0])], reverse=True)
        # solver_graph[t_idx]["prizes"] = dict(prizes[:top_n])

        # # Remove entries from tr_data for all entries beyond top_n
        # tr_data = solver_graph.attrs["terminal_root_prizes"]
        # for entry in prizes[top_n:]:
        #     tr_data.pop((t_idx, entry[0]))

        # Union of both sort approaches
        raw_sorted = sorted(prizes.items(), key=lambda item: item[1], reverse=True)
        vpc_sorted = sorted(
            prizes.items(), key=lambda item: item[1] / all_pairs.get((t_idx, item[0])), reverse=True
        )
        raw_top = set(r for r, _ in raw_sorted[:top_n])
        vpc_top = set(r for r, _ in vpc_sorted[:top_n])
        union_top = raw_top.union(vpc_top)

        solver_graph[t_idx]["prizes"] = {r: prizes[r] for r in union_top}

        # Remove entries from tr_data for all entries beyond top_n
        tr_data = solver_graph.attrs["terminal_root_prizes"]
        for r in prizes:
            if r not in union_top:
                tr_data.pop((t_idx, r))

    ending_prize_count = len(solver_graph.attrs["terminal_root_prizes"])

    logger.info(
        f"        removed {starting_prize_count - ending_prize_count} terminal-root pairs\n"
        f"        remaining terminal-root pairs: {ending_prize_count}"
    )


def apply_global_rank_prize_filtering(solver_graph: PyDiGraph, data: dict[str, Any]) -> list[dict[str, int]]:
    """Identify prunable terminal-root prize entries based on global rank."""
    logger.info("      applying global rank filtering...")

    budget = data["config"]["budget"]
    capacity_mode = data["config"]["capacity_mode"]

    tr_data = solver_graph.attrs["terminal_root_prizes"]
    global_pct_key = "terminal_prize_net_path_cost_global_pct"
    # global_pct_key = "terminal_prize_value_global_pct"
    lookup = solver_graph.attrs["pct_thresholds_lookup"]

    only_child_threshold = (
        lookup.get_threshold(budget, capacity_mode, Category.ONLY_CHILD)
        * lookup.factors[capacity_mode][Category.ONLY_CHILD]
    )
    dominant_threshold = (
        lookup.get_threshold(budget, capacity_mode, Category.DOMINANT)
        * lookup.factors[capacity_mode][Category.DOMINANT]
    )
    # protected_threshold = lookup.get_threshold(budget, capacity_mode, Category.PROTECTED)

    # Only_child and dominant prunables
    prunables = []
    dominant_survivors = set()
    for entry in tr_data.values():
        t_idx = entry["t_idx"]
        category = solver_graph[t_idx]["category"]

        if (
            category == Category.ONLY_CHILD
            and entry[global_pct_key] < only_child_threshold * 0.75
            or category == Category.DOMINANT
            and entry[global_pct_key] < dominant_threshold
        ):
            prunables.append(entry)
            continue

        if category == Category.DOMINANT:
            protector_t = solver_graph[t_idx]["protector"]
            dominant_survivors.add(t_idx)

    # Protected entries by surviving dominants
    saved = 0
    for entry in tr_data.values():
        t_idx = entry["t_idx"]
        category = solver_graph[t_idx]["category"]
        if category != Category.PROTECTED:
            continue

        protector_t = solver_graph[t_idx]["protector"]
        if protector_t in dominant_survivors:
            saved += 1
            continue

        prunables.append(entry)

    logger.info(
        f"      checked {len(tr_data)}; prunables {len(prunables)}; saved {saved} protected;"
        f" dominant thresh {dominant_threshold:.1f}%"
    )

    return prunables


def reduce_prize_data(data: dict[str, Any]) -> None:
    """Apply pruning and reduction logic to terminal prize entries.

    SAFETY: The solver_graph is modified in-place! Be sure the reference graph `data["G"]`
    is not a shallow copy to either the graph or underlying data by using `deepcopy(G.copy())`.
    """
    solver_graph = data.get("solver_graph")
    assert isinstance(solver_graph, PyDiGraph)

    logger.info("    reducing prize data...")

    # 'data' members are nested structures and must be deepcopied
    solver_graph.attrs["families"] = deepcopy(data["families"])
    solver_graph.attrs["terminal_root_prizes"] = deepcopy(data["terminal_root_prizes"])
    solver_graph.attrs["prizes_by_terminal_view"] = deepcopy(data["prizes_by_terminal_view"])
    solver_graph.attrs["prizes_by_root_view"] = deepcopy(data["prizes_by_root_view"])
    solver_graph.attrs["prize_ranks"] = deepcopy(data["prize_ranks"])
    solver_graph.attrs["all_pairs_shortest_path_lengths"] = deepcopy(data["all_pairs_path_lengths"])
    solver_graph.attrs["pct_thresholds_lookup"] = deepcopy(data["pct_thresholds_lookup"])

    top_n = data["config"]["top_n"]
    if top_n is not None and top_n > 0:
        apply_top_n_filtering(solver_graph, data)

    if data["config"]["prune_prizes"]:
        prunable_entries = []
        prunable_entries = apply_global_rank_prize_filtering(solver_graph, data)
        tpu.nullify_prize_entries(solver_graph, data, prunable_entries)
        tpu.prune_null_prize_entries(solver_graph, prunable_entries)
        tpu.prune_prize_drained_terminals(solver_graph)

    terminal_indices = [i for i in solver_graph.node_indices() if solver_graph[i]["is_terminal"]]
    solver_graph.attrs["terminal_indices"] = terminal_indices

    logger.debug("Active t-r pairs:")
    town_to_region_map = {int(town): region for region, town in data["affiliated_town_region"].items()}
    for t_idx in sorted(terminal_indices):
        node = solver_graph[t_idx]
        prizes = node["prizes"]
        for r_idx, prize in prizes.items():
            town = solver_graph[r_idx]["waypoint_key"]
            region = town_to_region_map[int(town)]
            node_weight = node["need_exploration_point"]
            path_weight = data["all_pairs_path_lengths"].get((t_idx, r_idx), 2**31)
            logger.debug(
                f"{node['waypoint_key']:>5} {region:>5} {town:>5} {prize:>5} {node_weight:>5} {path_weight:>5}"
            )
