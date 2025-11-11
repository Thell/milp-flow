# optimize.py

from collections import Counter
from highspy import Highs
from loguru import logger
from rustworkx import PyDiGraph


from api_highs_model import create_model, get_highs, solve

SUPER_ROOT = 99999


def load_and_extract_worker_data():
    import tkinter as tk
    from tkinter import filedialog
    import json

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Worker Data JSON File", filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
    )
    if not file_path:
        print("File selection cancelled. Exiting.")
        return

    worker_data_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_workers = data["userWorkers"]
    for worker in user_workers:
        tnk_value = worker.get("tnk")  # town id (root waypoint_key)
        job_data = worker.get("job", {})
        assert job_data
        pzk_value = job_data.get("pzk")  # plantzone id (terminal waypoint_key)
        worker_data_list.append({"r": tnk_value, "t": pzk_value})

    return worker_data_list


def optimize(
    data: dict,
    cr_pairs=False,
    prize_scale=False,
    prev_terminals_sets: None | dict = None,
) -> tuple[Highs, dict]:
    solver_graph: PyDiGraph = data["solver_graph"]
    model = get_highs(data["config"]["solver"])
    model, vars = create_model(
        model,
        solver_graph,
        data["config"],
        cr_pairs=cr_pairs,
        prize_scale=prize_scale,
        prev_terminals_sets=prev_terminals_sets,
    )

    logger.info(
        f"Solving graph with {solver_graph.num_nodes()} nodes and {solver_graph.num_edges()} edges containing {len(solver_graph.attrs['terminal_indices'])} terminals."
    )

    model = solve(model, data["config"]["solver"], controller=None)

    col_values = model.getSolution().col_value
    active_graph_indices = [i for i, x_var in vars["x"].items() if int(round(col_values[x_var.index])) == 1]

    cost = sum([solver_graph[i]["need_exploration_point"] for i in active_graph_indices])
    logger.info(f"node costs = {cost}")
    logger.info(f"nodes = {[v['waypoint_key'] for v in [solver_graph[i] for i in active_graph_indices]]}")

    terminal_sets = {}
    key_list = list(vars["x_t_r"].keys())
    cumulative_value = 0
    for i, t_var in enumerate(vars["x_t_r"].values()):
        if int(round(col_values[t_var.index])) == 1:
            terminal, root = key_list[i]
            terminal_sets[terminal] = root
            cumulative_value += solver_graph[terminal]["prizes"][root]
    logger.info(f"terminal value = {cumulative_value}")

    root_used_capacities = Counter(terminal_sets.values())
    logger.debug(
        f"root_capacities = { {solver_graph[r]['waypoint_key']: c for r, c in root_used_capacities.items()} }"
    )
    capacity_costs = {
        root: solver_graph[root]["capacity_cost"][root_used_capacities[root]] for root in root_used_capacities
    }
    logger.info(
        f"capacity_costs = { {solver_graph[r]['waypoint_key']: c for r, c in capacity_costs.items()} }"
    )

    logger.info(
        f"terminal_sets = { {solver_graph[i]['waypoint_key']: solver_graph[root]['waypoint_key'] for i, root in terminal_sets.items()} }"
    )

    logger.debug("terminal values")
    root_prizes = {}
    for r in solver_graph.attrs["root_indices"]:
        prizes = {}
        for t in solver_graph.attrs["terminal_indices"]:
            if r not in solver_graph[t]["prizes"]:
                continue
            prizes[(t, r)] = solver_graph[t]["prizes"][r]
        root_prizes[r] = dict(sorted(prizes.items(), key=lambda x: x[1], reverse=True))

    for t, r in terminal_sets.items():
        root_rank_in_terminal = list(solver_graph[t]["prizes"].keys()).index(r) + 1
        terminal_rank_in_root = list(root_prizes[r].keys()).index((t, r)) + 1
        logger.debug(
            f"{solver_graph[t]['waypoint_key']} -> {solver_graph[r]['waypoint_key']} ({solver_graph[r]['region_key']}): {solver_graph[t]['prizes'][r]} (root rank: {root_rank_in_terminal}, terminal rank: {terminal_rank_in_root})"
        )

    total_cost = cost + sum(capacity_costs.values())
    logger.info(f"{total_cost=}")
    return model, terminal_sets
