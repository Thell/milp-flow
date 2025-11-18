# api_highs_model.py

from dataclasses import dataclass
from math import inf
import queue
from random import randint, seed
import re
from threading import Event
from threading import Lock
from threading import Thread
import time

from highspy import Highs, HighsLinearObjective, ObjSense
from highspy._core import HighsOptions
from loguru import logger
import numpy as np
from rustworkx import PyDiGraph

from api_common import SUPER_ROOT

TIME_AND_NEWLINE_PATTERN = re.compile(r"(\d+\.\d+s)\n$")


class SolverController:
    """Ties the solver threads to the UI for managing interrupts."""

    def __init__(self):
        self._interrupt_event = Event()

    def stop(self):
        self._interrupt_event.set()

    def is_interrupted(self) -> bool:
        return self._interrupt_event.is_set()


class TimeoutTimer(Thread):
    """Monitors for reset signals, if not received within the timeout period, callback is called."""

    def __init__(self, timeout_seconds: float, callback):
        super().__init__(daemon=True)
        self.timeout_seconds = timeout_seconds
        self.callback = callback
        self.reset_event = Event()
        self.running = True

    def run(self):
        while self.running:
            reset_triggered = self.reset_event.wait(self.timeout_seconds)
            if not self.running:
                break

            if reset_triggered:
                self.reset_event.clear()
            else:
                self.callback()
                self.running = False

    def reset(self):
        if self.running:
            self.reset_event.set()

    def shutdown(self):
        self.running = False
        self.reset_event.set()


@dataclass
class Incumbent:
    id: int
    lock: Lock
    value: float
    solution: np.ndarray
    provided: list[bool]


def get_highs(solver_config: dict) -> Highs:
    """Returns a configured HiGHS instance using HiGHS options in 'solver_config'."""

    highs = Highs()
    highs_options: HighsOptions = highs.getOptions()

    valid_options: set[str] = {n for n in dir(highs_options) if not n.startswith("_")}
    if not solver_config:
        logger.info("No solver options found in config")
        return highs
    else:
        logger.info(f"Configuring HiGHS with options: {solver_config}")

    for option_name, option_value in solver_config.items():
        if option_name not in valid_options:
            continue
        try:
            highs.setOptionValue(option_name, option_value)
        except Exception as e:
            logger.error(
                f"HiGHS Error: Could not set option '{option_name}' to '{option_value}'.\nDetails: {e}"
            )

    if solver_config.get("threads", 1) > 1:
        highs.resetGlobalScheduler(True)

    return highs


def create_model(
    model: Highs,
    G: PyDiGraph,
    config: dict,
    cr_pairs: None | bool = None,
    prize_scale: None | bool = None,
    prev_terminals_sets: None | dict = None,
) -> tuple[Highs, dict]:
    """Model with assignment and dynamic costs with flow from terminals to roots."""

    roots_indices = G.attrs["root_indices"]
    terminal_indices = G.attrs["terminal_indices"]
    super_root_index = None
    super_terminal_indices = []
    for i in G.node_indices():
        if G[i].get("is_super_terminal", False):
            super_terminal_indices.append(i)
        if G[i]["waypoint_key"] == SUPER_ROOT:
            super_root_index = i

    # Identify families (post-terminal_indices setup)
    families = {}
    for t in terminal_indices:
        if t in super_terminal_indices:
            continue  # Skip if irrelevant
        parents = G.predecessor_indices(t)
        if not parents:
            continue
        parent = parents[0]  # Assume unique parent
        if parent not in families:
            families[parent] = []
        families[parent].append(t)

    # Variables
    # Node selection for each node in graph to determine if node is in forest.
    x = model.addBinaries(G.node_indices())

    # Root assignment selector for (terminal, root) pairs
    x_t_r = {}
    for t in terminal_indices:
        for r in G[t]["prizes"]:
            x_t_r[(t, r)] = model.addBinary()
    for t in super_terminal_indices:
        x_t_r[(t, super_root_index)] = model.addBinary()

    # # Iterative optimization by forcing assignment from a given set of (terminal, root) pairs
    expected_value = 0
    if prev_terminals_sets:
        logger.warning("using prev_model")
        node_key_by_index = G.attrs["node_key_by_index"]
        for t, r in prev_terminals_sets.items():
            t_index = node_key_by_index.inv[t]
            r_index = node_key_by_index.inv[r]
            model.addConstr(x_t_r[(t_index, r_index)] == 1)
            model.addConstr(x[t_index] == 1)
            model.addConstr(x[r_index] == 1)
            expected_value += G[t_index]["prizes"][r_index]
        logger.warning(f"expected value = {expected_value}")

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

    # # Single-Objective
    # if prize_scale:
    #     prizes = model.qsum(
    #         x_t_r[(t, r)] * (int(prize) / 1e6)
    #         for t in terminal_indices
    #         for r, prize in G[t]["prizes"].items()
    #     )
    # else:
    #     prizes = model.qsum(
    #         x_t_r[(t, r)] * prize for t in terminal_indices for r, prize in G[t]["prizes"].items()
    #     )
    # model.setObjective(prizes, sense=ObjSense.kMaximize)

    # Muli-Objectives
    model.setOptionValue("blend_multi_objectives", False)
    num_col = model.getNumCol()  # AFTER all variables are added

    # Maximize Profit (prizes)
    profit_coeffs = [0.0] * num_col
    for t in terminal_indices:
        for r, prize in G[t]["prizes"].items():
            col_idx = x_t_r[(t, r)].index
            profit_coeffs[col_idx] = int(prize) / 1e6 if prize_scale else prize
    profit_obj = HighsLinearObjective()
    profit_obj.weight = -1.0  # maximize
    profit_obj.offset = 0.0
    profit_obj.coefficients = profit_coeffs
    profit_obj.abs_tolerance = 0.0001
    profit_obj.rel_tolerance = 0.0001
    profit_obj.priority = 1
    print("Max x_t_r index:", max(v.index for v in x_t_r.values()))
    print("Num columns:", model.getNumCol())
    print("Some sample coefficients:", profit_coeffs[:10])
    model.addLinearObjective(profit_obj)

    # Minimize Transit Cost (non capacity cost, non terminal, non root, non super_terminal)
    # Node selection cost
    cost_coeffs = [0.0] * num_col
    non_transit_indices = set(roots_indices + terminal_indices + super_terminal_indices)
    for i in G.node_indices():
        if i not in non_transit_indices:
            col_idx = x[i].index  # type: ignore
            cost_coeffs[col_idx] = G[i]["need_exploration_point"]

    cost_obj = HighsLinearObjective()
    cost_obj.weight = 1.0  # minimize
    cost_obj.offset = 0.0
    cost_obj.coefficients = cost_coeffs
    cost_obj.abs_tolerance = 0.0
    cost_obj.rel_tolerance = 0.0
    cost_obj.priority = 0
    model.addLinearObjective(cost_obj)

    # Max cost budget constraint
    capacity_cost = model.qsum(
        c_r[(c, r)] * cost for r in roots_indices for c, cost in enumerate(G[r]["capacity_cost"])
    )
    node_cost = model.qsum(x[i] * G[i]["need_exploration_point"] for i in G.node_indices())
    model.addConstr(capacity_cost + node_cost <= config["budget"], name="budget")

    # Minimum and Maximum upper bound on number of terminals constraint
    t_count_lb = config["terminal_count_min_limit"]
    if t_count_lb is not None and t_count_lb > 0:
        logger.info(f"*** setting terminal_count_min_limit = {t_count_lb}")
        model.addConstr(model.qsum(x[t] for t in terminal_indices) >= t_count_lb)
    t_count_ub = config["terminal_count_max_limit"]
    if t_count_ub is not None and t_count_ub > 0:
        logger.info(f"*** setting terminal_count_max_limit = {t_count_ub}")
        model.addConstr(model.qsum(x[t] for t in terminal_indices) <= t_count_ub)

    # (Terminal, Root) assignment constraints
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
        # Terminals may not be selected if not assigned
        model.addConstr(x[t] <= assigned)

    # (Super Terminal, Super Root) assignment constraints
    if super_root_index is not None:
        # Super root must be selected
        model.addConstr(x[super_root_index] == 1)
        # All super terminals must be assigned to super root
        for t in super_terminal_indices:
            model.addConstr(x_t_r[(t, super_root_index)] == 1)
            model.addConstr(x[t] == 1)

    # Per-root family ordering
    # A root should only select lower valued siblings if higher value siblings are already selected
    for r in roots_indices:
        for parent, family in families.items():
            eligible = [t for t in family if (t, r) in x_t_r]
            if len(eligible) < 2:
                continue

            # Cost check - unequal costs causes constraint blocking and infeasibility
            base_cost = G[eligible[0]]["need_exploration_point"]
            if not all(G[t]["need_exploration_point"] == base_cost for t in eligible):
                continue

            # Roots check - unequal roots causes constraint blocking and infeasibility
            sorted_keys = list(G[eligible[0]]["prizes"].keys())
            if not all(list(G[t]["prizes"].keys()) == sorted_keys for t in eligible):
                continue

            # Allow any root to take the dominant prize unlocking all siblings
            # (Chaining of siblings is not waranted)
            sorted_ts = sorted(eligible, key=lambda t: int(G[t]["prizes"][r]))
            high = sorted_ts[-1]
            for i in range(len(sorted_ts) - 1):
                low = sorted_ts[i]
                model.addConstr(x_t_r[(low, r)] <= x[high])

    # Node/flow based constraints: flow from terminals to roots
    flow_roots = roots_indices + ([super_root_index] if super_root_index is not None else [])
    terminal_and_roots = set(terminal_indices) | set(super_terminal_indices) | set(flow_roots)

    for i in G.node_indices():
        predecessors = G.predecessor_indices(i)
        successors = G.successor_indices(i)
        neighbors = set(predecessors) | set(successors)
        x_neighbors = [x[j] for j in neighbors]

        # Neighbor selection: redundant for selection but imposes a transitive
        # property to selected nodes to improve solution runtime.
        if i in terminal_and_roots:
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
                    # While this could be `== 1` empirical testing suggests `== x[r]` is more performant
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


def solve(model: Highs, config: dict, controller: SolverController | None = None) -> Highs:
    """Solve the MIP model using either a single process or multiple processes."""
    logger.info("Solving MIP problem...")

    # User interrupt
    if controller is None:
        controller = SolverController()

    # User provided timeout
    timeout_controller = None
    mip_improvement_timeout = config.get("mip_improvement_timeout", None)
    if mip_improvement_timeout is not None and mip_improvement_timeout != inf and mip_improvement_timeout > 0:
        timeout_controller = TimeoutTimer(mip_improvement_timeout, controller.stop)
        timeout_controller.start()

    # Single solve
    num_processes = config.get("num_processes", 1)
    if num_processes == 1:

        def cbMIPInterruptHandler(e):
            nonlocal controller
            if controller.is_interrupted():  # pyright: ignore[reportOptionalMemberAccess]
                e.interrupt()

        model.HandleUserInterrupt = True
        model.enableCallbacks()
        model.cbMipInterrupt.subscribe(cbMIPInterruptHandler)
        model.solve()
        return model

    # Parallel solve

    # Queues for inter-thread communication
    incumbent_queue = queue.Queue()
    logging_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = Event()

    solution_buffers = [np.empty(model.getNumCol(), dtype=float) for _ in range(num_processes)]
    do_solution_report_capture = [False] * num_processes
    solution_reports = [[] for _ in range(num_processes)]
    obj_sense = model.getObjectiveSense()[1]

    # Model clones for each thread
    clones = [model] + [Highs() for _ in range(num_processes - 1)]
    clones[0].HandleUserInterrupt = True
    clones[0].enableCallbacks()

    if config["random_seed"] > 0:
        seed(config["random_seed"])
    for i in range(1, num_processes):
        clones[i].passOptions(clones[0].getOptions())
        clones[i].passModel(clones[0].getModel())
        if config["random_seed"] == 0:
            clones[i].setOptionValue("random_seed", i)
        else:
            clones[i].setOptionValue("random_seed", randint(0, 2**31 - 1))
        clones[i].HandleUserInterrupt = True
        clones[i].enableCallbacks()

    incumbent = Incumbent(
        id=0,
        lock=Lock(),
        value=2**31 if obj_sense == ObjSense.kMinimize else -(2**31),
        solution=np.zeros(clones[0].getNumCol()),
        provided=[False] * num_processes,
    )

    # Utility functions
    if obj_sense == ObjSense.kMinimize:

        def is_better(a, b):
            return a < b

    else:

        def is_better(a, b):
            return a > b

    # External thread functions
    def logging_manager():
        """Consume logging events and capture final reports into clone_solution_report."""
        # NOTE: Highs will not write anything to the console when the logging callback is enabled.
        # The final report is captured here but written to stdout prior to exiting the main function.
        while not stop_event.is_set():
            try:
                e = logging_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            clone_id = int(e.user_data)

            is_solution_report = e.message.startswith("\nSolving report")
            if is_solution_report:
                # NOTE: all future messages from this thread are part of its solving report
                do_solution_report_capture[clone_id] = True

            if do_solution_report_capture[clone_id]:
                message = e.message.replace("Solving report", f"Solving report for thread {clone_id}")
                solution_reports[clone_id].append(message)
            elif clone_id == incumbent.id:
                # Only print messages for the incumbent thread
                replacement = r"\1 " + str(clone_id) + r"\n"
                print(TIME_AND_NEWLINE_PATTERN.sub(replacement, e.message), end="")

    def incumbent_manager():
        """Process improved-solution events and update the incumbent state."""
        nonlocal incumbent, solution_buffers
        while not stop_event.is_set():
            try:
                value, clone_id = incumbent_queue.get(timeout=0.025)
            except queue.Empty:
                continue

            if is_better(value, incumbent.value):
                with incumbent.lock:
                    incumbent.value = value
                    np.copyto(incumbent.solution, solution_buffers[clone_id], casting="no")
                    incumbent.provided = [False] * num_processes
                    incumbent.provided[clone_id] = True
                    incumbent.id = clone_id

                if timeout_controller is not None:
                    timeout_controller.reset()

    # Callback handler functions
    def cbLoggingHandler(e):
        logging_queue.put_nowait(e)

    def cbMIPImprovedSolutionHandler(e):
        solution = e.data_out.mip_solution
        if solution is None or solution.size == 0:
            return
        # Only a specific clone can update its solution buffer
        np.copyto(solution_buffers[int(e.user_data)], solution, casting="no")
        incumbent_queue.put_nowait((float(e.data_out.objective_function_value), int(e.user_data)))

    def cbMIPInterruptHandler(e):
        nonlocal controller
        if controller.is_interrupted():  # pyright: ignore[reportOptionalMemberAccess]
            e.interrupt()

    def cbMIPUserSolutionHandler(e):
        """Update clone solution buffer to best solution found so far..."""
        clone_id = int(e.user_data)
        if not incumbent.provided[clone_id] and is_better(
            incumbent.value, e.data_out.objective_function_value
        ):
            # Skip rather than block: the callback will be called again
            if incumbent.lock.acquire(blocking=False):
                e.data_in.user_has_solution = True
                np.copyto(e.data_in.user_solution, incumbent.solution, casting="no")
                incumbent.provided[clone_id] = True
                incumbent.lock.release()

    # Callback subscriptions
    for i in range(num_processes):
        clones[i].cbLogging.subscribe(cbLoggingHandler, i)
        clones[i].cbMipImprovingSolution.subscribe(cbMIPImprovedSolutionHandler, i)
        clones[i].cbMipInterrupt.subscribe(cbMIPInterruptHandler, i)
        if config.get("mip_share_incumbent_solution", False):
            clones[i].cbMipUserSolution.subscribe(cbMIPUserSolutionHandler, i)

    ### Main Section

    # Manager threads must be started first
    Thread(target=logging_manager, daemon=True).start()
    Thread(target=incumbent_manager, daemon=True).start()

    # Worker threads
    def worker_task(clone: Highs, i: int):
        clone.solve()
        value = clone.getObjectiveValue()
        if value == incumbent.value or is_better(value, incumbent.value):
            result_queue.put(i)

    for i in range(num_processes):
        Thread(target=worker_task, args=(clones[i], i), daemon=True).start()
        time.sleep(0.1)

    # Result watcher - only a valid incumbent or better solution will be
    # placed in the result queue
    first_to_finish = None
    while first_to_finish is None:
        try:
            # NOTE: timeout allows for Highs interrupt handling.
            first_to_finish = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

    # Cancel the other threads, without any further output
    for i in range(num_processes):
        if i != first_to_finish:
            clones[i].silent()
        clones[i].cancelSolve()

    # Allow managers to consume final events before stopping
    time.sleep(0.1)
    stop_event.set()

    # Print solution report
    for message in solution_reports[first_to_finish]:
        print(message, end="")
    print()

    if timeout_controller is not None:
        timeout_controller.shutdown()

    # No need to wait for manager threads to finish
    # cleanup is handled by the main thread exit.

    return clones[first_to_finish]
