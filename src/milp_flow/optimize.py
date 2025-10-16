# optimize.py

from collections import Counter
from highspy import Highs, ObjSense
from loguru import logger
from rustworkx import PyDiGraph


# from generate_graph_data import Arc, GraphData, Node, NodeType as NT
from optimize_par import solve_par

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


def create_model(
    G: PyDiGraph,
    config: dict,
    get_test_set: None | bool = None,
    cr_pairs: None | bool = None,
    prize_scale: None | bool = None,
    prev_terminals_sets: None | dict = None,
) -> tuple[Highs, dict]:
    """Model with assignment and dynamic costs with flow from terminals to roots."""

    if get_test_set:
        worker_data = load_and_extract_worker_data()
        config["worker_data"] = worker_data

    model = Highs()

    roots_indices = G.attrs["root_indices"]
    terminal_indices = G.attrs["terminals"]
    super_root_index = None
    super_terminal_indices = []
    for i in G.node_indices():
        if G[i].get("is_super_terminal", False):
            super_terminal_indices.append(i)
        if G[i]["waypoint_key"] == SUPER_ROOT:
            super_root_index = i

    # Variables
    # Node selection for each node in graph to determine if node is in forest.
    x = model.addBinaries(G.node_indices())

    # Root assignment selector for each terminal (terminal, root) pair
    x_t_r = {}
    for t in terminal_indices:
        for r in G[t]["prizes"]:
            x_t_r[(t, r)] = model.addBinary()
    for t in super_terminal_indices:
        x_t_r[(t, super_root_index)] = model.addBinary()

    # For testing/validation/debugging
    if "worker_data" in config:
        # Worker assignment selector for each terminal, root pair
        # Needs to be translated from waypoint to G index
        index_map = {G[i]["waypoint_key"]: i for i in G.node_indices()}
        assigned_value = 0
        for w in config["worker_data"]:
            t_index = index_map[w["t"]]
            r_index = index_map[w["r"]]
            if t_index not in terminal_indices:
                logger.error(f"terminal {t_index} not in terminal_indices")
            if r_index not in roots_indices:
                logger.error(f"root {r_index} not in roots_indices")
            if r_index not in G[t_index]["prizes"]:
                logger.error(f"root {r_index} not in G[t]['prizes']")
            value = G[t_index]["prizes"].get(r_index, 0)
            assigned_value += value
            logger.debug(f"assigning: {w} -> {t_index}, {r_index}, {value}")
            model.addConstr(x_t_r[(t_index, r_index)] == 1)
            model.addConstr(x[t_index] == 1)
            model.addConstr(x[r_index] == 1)
        logger.debug(f"assigned_value = {assigned_value}")

    # Previous model: allows for iterative optimization (growth of forest)
    # from a given set of terminal, root pairs
    if prev_terminals_sets:
        logger.info("using prev_model")
        for t, r in prev_terminals_sets.items():
            model.addConstr(x_t_r[(t, r)] == 1)
            model.addConstr(x[t] == 1)
            model.addConstr(x[r] == 1)

    # SOS1 - Capacity assignment selector for each root
    # Capacity has step-wise costs and must equal the terminal count assigned to the root.
    # capacity_cost: list[int] => {0, cost_1, cost_2, ...} for 0, 1, 2, ...
    # the zeroth index is the "empty" (non-selected) state for a given root
    c_r = {}
    for r in roots_indices:
        for c in range(len(G[r]["capacity_cost"])):
            c_r[(c, r)] = model.addBinary()
    if cr_pairs:
        # Skip adjacent (e.g., no constr for 0&1, but yes for 0&2)
        for r in roots_indices:
            n = len(G[r]["capacity_cost"])
            for c1 in range(n):
                for c2 in range(c1 + 2, n):
                    model.addConstr(c_r[(c1, r)] + c_r[(c2, r)] <= 1)

    # Flow for root on arc (i,j); when root can transit from i to j
    f_r = {}
    for i, j in G.edge_list():
        common_rs = set(G[i]["transit_bounds"]) & set(G[j]["transit_bounds"])
        for r in common_rs:
            ub = min(G[i]["transit_bounds"][r], G[j]["transit_bounds"][r])
            f_r[(r, i, j)] = model.addVariable(lb=0, ub=ub)

    # Objective
    if prize_scale:
        prizes = model.qsum(
            x_t_r[(t, r)] * (prize / 1e6) for t in terminal_indices for r, prize in G[t]["prizes"].items()
        )
    else:
        prizes = model.qsum(
            x_t_r[(t, r)] * int(prize) for t in terminal_indices for r, prize in G[t]["prizes"].items()
        )
    model.setObjective(prizes, sense=ObjSense.kMaximize)

    # Hard Budget Constraint
    capacity_cost = model.qsum(
        c_r[(c, r)] * cost for r in roots_indices for c, cost in enumerate(G[r]["capacity_cost"])
    )
    node_cost = model.qsum(x[i] * G[i]["need_exploration_point"] for i in G.node_indices())
    model.addConstr(capacity_cost + node_cost <= config["budget"])

    # (Terminal, root) assignment constraints
    for r in roots_indices:
        assigned = model.qsum(x_t_r[(t, r)] for t in terminal_indices if (t, r) in x_t_r)
        # Any terminal assigned to root selects root
        model.addConstr(assigned <= G[r]["ub"] * x[r])
    for t in terminal_indices:
        assigned = model.qsum(x_t_r[(t, r)] for r in G[t]["prizes"])
        # Any root assigned by terminal selects terminal
        model.addConstr(x[t] >= assigned)
        # Terminals may be assigned to at most a single root
        model.addConstr(assigned <= 1)
    if super_root_index is not None:
        # Super root must be selected
        model.addConstr(x[super_root_index] == 1)
        # All super terminals must be assigned to super root
        for t in super_terminal_indices:
            model.addConstr(x_t_r[(t, super_root_index)] == 1)
            model.addConstr(x[t] == 1)

    # Ranked basins cuts
    basins = G.attrs.get("basins", {})
    for r, (basin_ts, cut_value, cut_nodes) in basins.items():
        outside_assigned = model.qsum(
            x_t_r[(t, r)] for t in terminal_indices if t not in basin_ts and (t, r) in x_t_r
        )
        model.addConstr(
            outside_assigned <= cut_value * model.qsum(x[b] for b in cut_nodes if b in G.node_indices())
        )

    # # Aggregated cut - empirical testing suggests this is not necessary
    # total_assigned = model.qsum(x_t_r[(t, r)] for t in terminal_indices for r in G[t]["prizes"])
    # total_cap = model.qsum(
    #     model.qsum(c * c_r[(c, r)] for c in range(len(G[r]["capacity_cost"]))) for r in roots_indices
    # )
    # model.addConstr(total_assigned <= total_cap)
    # if super_root_index is not None:
    #     model.addConstr(total_assigned + len(super_terminal_indices) <= total_cap + G[super_root_index]["ub"])

    # Node/flow based constraints
    # All flow is accumulated from terminals to roots
    flow_roots = roots_indices + ([super_root_index] if super_root_index is not None else [])

    for i in G.node_indices():
        predecessors = G.predecessor_indices(i)
        successors = G.successor_indices(i)
        neighbors = set(predecessors) | set(successors)
        x_neighbors = [x[j] for j in neighbors]

        # Neighbor selection: redundant for selection but imposes a transitive
        # property to selected nodes to improve solution runtime.
        if i in terminal_indices or i in flow_roots:
            # A selected terminal or root must have at least one selected neighbor
            model.addConstr(model.qsum(x_neighbors) >= 1 * x[i])
        else:
            # A selected intermediate must have at least two selected neighbors
            model.addConstr(model.qsum(x_neighbors) >= 2 * x[i])

        for r in flow_roots:
            if r not in G[i]["transit_bounds"] or G[i]["transit_bounds"][r] == 0:
                continue

            r_ub = G[r]["ub"]
            in_flow = model.qsum(f_r[(r, j, i)] for j in predecessors if (r, j, i) in f_r)
            out_flow = model.qsum(f_r[(r, i, j)] for j in successors if (r, i, j) in f_r)

            # Node selection: any flow selects node
            model.addConstr(in_flow <= r_ub * x[i])
            model.addConstr(out_flow <= r_ub * x[i])

            # Flow
            if i == r:
                if r != super_root_index:
                    # Flow at root: all assigned terminals must be accounted for
                    model.addConstr(out_flow == 0)
                    model.addConstr(
                        in_flow == model.qsum(x_t_r[(t, r)] for t in terminal_indices if (t, r) in x_t_r)
                    )
                    # Capacity at root: capacity cost list has a no-cost 'empty' zeroth index with capacity 0
                    capacity_costs = G[r]["capacity_cost"]
                    # While this could be `== 1` empirical testing suggests `== x[r]` is more effective
                    model.addConstr(model.qsum(c_r[(c, r)] for c in range(len(capacity_costs))) == x[r])
                    model.addConstr(
                        in_flow == model.qsum(c * c_r[(c, r)] for c in range(len(capacity_costs)))
                    )
                else:
                    # Flow at SUPER_ROOT is only allowed for incoming SUPER_TERMINAL transit
                    # Super root has no capacity limit or capacity cost.
                    model.addConstr(out_flow == 0)
                    model.addConstr(in_flow == len(super_terminal_indices))
                    model.addConstr(in_flow == model.qsum(x_t_r[(t, r)] for t in super_terminal_indices))

            elif i in terminal_indices and (i, r) in x_t_r:
                # Flow at terminal: assigned terminals have one unit of out_flow.
                # Terminals are leaf nodes so in_flow is zero
                model.addConstr(out_flow == x_t_r[(i, r)])
                model.addConstr(in_flow == 0)

            elif i in super_terminal_indices and r == super_root_index:
                # Flow at super terminal
                model.addConstr(out_flow - in_flow == 1)

            else:
                # Flow at intermediate node
                model.addConstr(out_flow - in_flow == 0)

    return model, {"x": x, "x_t_r": x_t_r, "c_r": c_r, "f_r": f_r}


def optimize(
    data: dict,
    cr_pairs=False,
    prize_scale=False,
    prev_terminals_sets: None | dict = None,
) -> tuple[Highs, dict]:
    solver_graph = data["solver_graph"]
    model, vars = create_model(
        solver_graph,
        data["config"],
        False,
        cr_pairs,
        prize_scale,
        prev_terminals_sets=prev_terminals_sets,
    )

    options = {k: v for k, v in data["config"]["solver"].items()}
    for option_name, option_value in options.items():
        # Non-standard HiGHS options need filtering...
        if option_name not in ["num_threads", "mip_improvement_timeout", "do_not_use_mip_heuristic"]:
            model.setOptionValue(option_name, option_value)

    print("Solving mip problem...")
    options = {k: v for k, v in data["config"]["solver"].items() if k != "num_threads"}
    model = solve_par(model, options)

    active_graph_indices = [
        i for i, x_var in vars["x"].items() if int(round(model.variableValue(x_var))) == 1
    ]

    cost = sum([solver_graph[i]["need_exploration_point"] for i in active_graph_indices])
    logger.info(f"node costs = {cost}")
    logger.info(f"nodes = {[v['waypoint_key'] for v in [solver_graph[i] for i in active_graph_indices]]}")

    terminal_sets = {}
    key_list = list(vars["x_t_r"].keys())
    cumulative_value = 0
    for i, t_var in enumerate(vars["x_t_r"].values()):
        if int(round(model.variableValue(t_var))) == 1:
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
        for t in solver_graph.attrs["terminals"]:
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
