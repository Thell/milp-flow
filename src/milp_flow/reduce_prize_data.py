# reduce_prize_data.py

from copy import deepcopy
from typing import Any

from loguru import logger
from rustworkx import PyDiGraph

import terminal_prize_utils as tpu


def apply_top_n_filtering(solver_graph: PyDiGraph, top_n: int):
    """Limits the number of prize entries per terminal."""
    logger.info(f"      applying top-n filtering for the top {top_n} remaining prizes...")

    starting_prize_count = len(solver_graph.attrs["terminal_root_prizes"])
    for t_idx in solver_graph.attrs["terminal_indices"]:
        prizes = solver_graph[t_idx]["prizes"]
        prizes = sorted(
            prizes.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        solver_graph[t_idx]["prizes"] = dict(prizes[:top_n])

        # Remove entries from tr_data for all entries beyond top_n
        tr_data = solver_graph.attrs["terminal_root_prizes"]
        for entry in prizes[top_n:]:
            tr_data.pop((t_idx, entry[0]))

    ending_prize_count = len(solver_graph.attrs["terminal_root_prizes"])

    logger.info(
        f"        removed {starting_prize_count - ending_prize_count} terminal-root pairs\n"
        f"        remaining terminal-root pairs: {ending_prize_count}"
    )


def apply_global_rank_prize_filtering(
    solver_graph: PyDiGraph, budget: int, capacity_mode: str
) -> list[dict[str, int]]:
    """Identify prunable terminal-root prize entries based on global rank."""
    logger.info("      applying global rank filtering...")

    tr_data = solver_graph.attrs["terminal_root_prizes"]
    threshold = tpu.get_pruning_threshold(budget, capacity_mode, "global")
    global_pct_key = "terminal_prize_net_path_cost_global_pct"

    # prunables = [entry for entry in tr_data.values() if entry[global_pct_key] < threshold * 100]

    # saved = 0
    # protected_threshold = tpu.get_protected_threshold(budget, capacity_mode, "global")
    # family_dominants = solver_graph.attrs["family_dominants"]
    # for entry in prunables[:]:
    #     if entry["is_protected"]:
    #         r_idx = entry["r_idx"]
    #         dominant_t = family_dominants.get(r_idx)
    #         save = False
    #         if dominant_t:
    #             dominant_entry = tr_data.get((dominant_t, r_idx))
    #             if dominant_entry and dominant_entry[global_pct_key] >= threshold * 100:
    #                 save = True
    #         if not save and entry[global_pct_key] >= protected_threshold * 100:
    #             save = True
    #         if save:
    #             prunables.remove(entry)
    #             saved += 1

    # Testing a simple LUT for thresholds
    threshold = 0
    protected_threshold = 0

    # fmt: off
    if capacity_mode == "max":
        THRESHOLD_LUT = {25: 99.0, 50: 99.0, 75: 99.0, 100: 98.0, 125: 95.0, 150: 95.0, 175: 94.0, 200: 94.0, 225: 94.0, 250: 92.0, 275: 92.0, 300: 92.0, 325: 92.0, 350: 92.0, 375: 92.0, 400: 89.0, 425: 89.0, 450: 89.0, 475: 88.0, 500: 88.0, 525: 86.0, 550: 82.0, 575: 82.0, 600: 82.0}
        PROTECTED_THRESHOLD_LUT = {25: 99.0, 50: 99.0, 75: 96.0, 100: 95.0, 125: 94.0, 150: 94.0, 175: 89.0, 200: 89.0, 225: 82.0, 250: 82.0, 275: 73.0, 300: 73.0, 325: 73.0, 350: 73.0, 375: 73.0, 400: 73.0, 425: 73.0, 450: 73.0, 475: 73.0, 500: 73.0, 525: 58.0, 550: 58.0, 575: 58.0, 600: 58.0}
    else:  # min
        THRESHOLD_LUT = {25: 98.0, 50: 98.0, 75: 98.0, 100: 98.0, 125: 98.0, 150: 94.0, 175: 92.0, 200: 92.0, 225: 92.0, 250: 92.0, 275: 92.0, 300: 92.0, 325: 89.0, 350: 80.0, 375: 80.0, 400: 80.0, 425: 80.0, 450: 80.0, 475: 80.0, 500: 80.0, 525: 80.0, 550: 80.0, 575: 80.0, 600: 80.0}
        PROTECTED_THRESHOLD_LUT = {25: 98.0, 50: 98.0, 75: 98.0, 100: 97.0, 125: 95.0, 150: 95.0, 175: 95.0, 200: 95.0, 225: 95.0, 250: 92.0, 275: 89.0, 300: 89.0, 325: 75.0, 350: 75.0, 375: 75.0, 400: 75.0, 425: 75.0, 450: 75.0, 475: 75.0, 500: 75.0, 525: 75.0, 550: 75.0, 575: 75.0, 600: 75.0}
    # fmt: on

    OFFSET = -10

    # Clamp budget to nearest increment of 25 that is >= budget
    budget = (budget + 24) // 25 * 25
    threshold = THRESHOLD_LUT[budget] + OFFSET
    protected_threshold = PROTECTED_THRESHOLD_LUT[budget] + OFFSET

    prunables = [entry for entry in tr_data.values() if entry[global_pct_key] < threshold]

    saved = 0
    family_dominants = solver_graph.attrs["family_dominants"]
    for entry in prunables[:]:
        if entry["is_protected"]:
            r_idx = entry["r_idx"]
            dominant_t = family_dominants.get(r_idx)
            save = False
            if dominant_t:
                dominant_entry = tr_data.get((dominant_t, r_idx))
                if dominant_entry and dominant_entry[global_pct_key] >= threshold:
                    save = True
            if not save and entry[global_pct_key] >= protected_threshold:
                save = True
            if save:
                prunables.remove(entry)
                saved += 1

    logger.info(
        f"      checked {len(tr_data)}; raw prunables {len(prunables) + saved}; saved {saved} protected (offset ~{protected_threshold:.1f}%); final {len(prunables)}"
    )

    return prunables


def apply_post_topN_filtered_pruning(
    solver_graph: PyDiGraph, budget: int, capacity_mode: str
) -> list[dict[str, int]]:
    """Identify prunable terminal-root prize entries after applying top_n filtering."""
    logger.info("      applying post-topN filtering...")

    tr_data = solver_graph.attrs["terminal_root_prizes"]
    prizes_by_root = solver_graph.attrs["prizes_by_root_view"]
    prizes_by_terminal = solver_graph.attrs["prizes_by_terminal_view"]

    t_rank_key = "terminal_prize_net_path_cost_rank_by_root_view"
    r_rank_key = "root_prize_net_path_cost_rank_by_terminal_view"
    threshold = tpu.get_pruning_threshold(budget, capacity_mode, "top_n")

    prunables = []
    saved = 0
    for entry in tr_data.values():
        if tpu.is_family_protected(solver_graph, entry, t_rank_key, threshold):
            saved += 1
            continue

        t_rank_cutoff = threshold * len(prizes_by_root[entry["r_idx"]])
        r_rank_cutoff = threshold * len(prizes_by_terminal[entry["t_idx"]])
        if entry[t_rank_key] > t_rank_cutoff and entry[r_rank_key] > r_rank_cutoff:
            prunables.append(entry)

    # for entry in prunables[:]:  # Copy to avoid mod during iter
    #     if tpu.is_family_protected(solver_graph, entry, t_rank_key, threshold):
    #         prunables.remove(entry)
    #         saved += 1

    logger.info(
        f"      post top_n: using ranks for {len(prizes_by_root)} roots, {len(prizes_by_terminal)} terminals\n"
        f"      checked {len(tr_data)} pairs; raw prunables {len(prunables) + saved}; saved {saved} family-low-per-r;\n"
        f"      found {len(prunables)} prunable terminal-root pairs"
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
    solver_graph.attrs["family_dominants"] = deepcopy(data["family_dominants"])
    solver_graph.attrs["all_pairs_shortest_path_lengths"] = deepcopy(data["all_pairs_path_lengths"])

    prunable_entries = []
    apply_top_n_filtering(solver_graph, data["config"]["top_n"])

    # prunable_entries = apply_global_rank_prize_filtering(
    #     solver_graph, data["config"]["budget"], data["config"]["capacity_mode"]
    # )

    # TODO: Fix post topN filtering, it is cutting out optimal on LTA min runs
    # NOTE: pruning ranks must be based on the post-topN filtered data
    # tpu.rebuild_prize_data(solver_graph, data)

    # prunable_entries.extend(
    #     apply_post_topN_filtered_pruning(
    #         solver_graph, data["config"]["budget"], data["config"]["capacity_mode"]
    #     )
    # )

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
