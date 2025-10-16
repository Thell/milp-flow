from dataclasses import dataclass
import psutil
import queue
import re
from threading import Lock, Thread
import time

from highspy import Highs, kHighsInf, ObjSense
import numpy as np

TIME_AND_NEWLINE_PATTERN = re.compile(r"(\d+\.\d+s)\n$")


@dataclass
class Incumbent:
    id: int
    lock: Lock
    value: float
    solution: np.ndarray
    provided: list[bool]
    provided_time: list[float]


def solve_par(model: Highs, config: dict) -> Highs:
    mip_improvement_timer = time.time()

    physical_cpu_count = psutil.cpu_count(logical=False)
    physical_cpu_count = 1 if physical_cpu_count is None else max(2, physical_cpu_count) - 1
    num_threads = config.get("num_threads", physical_cpu_count)
    if num_threads < 1:
        num_threads = 1

    result_queue = queue.Queue()

    clones = [model] + [Highs() for _ in range(num_threads - 1)]
    clones[0].HandleUserInterrupt = True
    clones[0].enableCallbacks()

    for i in range(1, num_threads):
        clones[i].passOptions(clones[0].getOptions())
        clones[i].passModel(clones[0].getModel())
        clones[i].setOptionValue("random_seed", i)
        if config.get("mip_heuristic_run_root_reduced_cost", True) is False:
            if i == 4:
                print(
                    f"  enabling mip_heuristic_run_root_reduced_cost with default mip_heuristic effort on clone {i}"
                )
                clones[i].setOptionValue("mip_heuristic_run_root_reduced_cost", True)
            if i == 5:
                print(
                    f"  enabling mip_heuristic_run_root_reduced_cost with higher mip_heuristic effort 0.5 on clone {i}"
                )
                clones[i].setOptionValue("mip_heuristic_run_root_reduced_cost", True)
                clones[i].setOptionValue("mip_heuristic_effort", 0.5)
            if i == 6:
                print(
                    f"  enabling mip_heuristic_run_root_reduced_cost with higher mip_heuristic effort 1.0 on clone {i}"
                )
                clones[i].setOptionValue("mip_heuristic_run_root_reduced_cost", True)
                clones[i].setOptionValue("mip_heuristic_effort", 1.0)
        clones[i].HandleUserInterrupt = True
        clones[i].enableCallbacks()

    obj_sense = clones[0].getObjectiveSense()[1]

    # The incumbent lock is only utilized during write access since the
    # incumbent.value is immutable afterwards
    incumbent = Incumbent(
        id=0,
        lock=Lock(),
        value=2**31 if obj_sense == ObjSense.kMinimize else -(2**31),
        solution=np.zeros(clones[0].getNumCol()),
        provided=[False] * num_threads,
        provided_time=[0] * num_threads,
    )

    clone_capture_report = [False] * num_threads
    clone_solution_report = [[] for _ in range(num_threads)]

    if obj_sense == ObjSense.kMinimize:

        def is_better(a, b):
            return a < b

    else:

        def is_better(a, b):
            return a > b

    def normalize_objective_value(value: float) -> float:
        if value in (kHighsInf, -kHighsInf):
            return value
        value = float(value)
        if abs(value) < 10000:
            return round(value, 6)
        else:
            return round(value, 0)

    def cbLogging(e):
        """Follow and emit the output log of the incumbent..."""
        # Highs will not write to the console when this callback is enabled
        # so any output comes from our own logging.
        # The final report is captured here but written to stdout after all
        # threads have finished prior to exiting the main function.
        nonlocal incumbent, clone_capture_report, clone_solution_report

        clone_id = int(e.user_data)
        is_solution_report = e.message.startswith("\nSolving report")
        clone_capture_report[clone_id] = clone_capture_report[clone_id] or is_solution_report

        if clone_capture_report[clone_id]:
            message = e.message.replace(r"Solving report", "Solving report for thread " + str(clone_id))
            clone_solution_report[clone_id].append(message)
        elif clone_id == incumbent.id:
            replacement = r"\1 " + str(clone_id) + r"\n"
            print(TIME_AND_NEWLINE_PATTERN.sub(replacement, e.message), end="")
        return

    def cbMIPImprovedSolutionHandler(e):
        """Update incumbent to best solution found so far..."""
        nonlocal incumbent, clones, mip_improvement_timer

        value = normalize_objective_value(e.data_out.objective_function_value)
        if is_better(value, incumbent.value):
            with incumbent.lock:
                clone_id = int(e.user_data)
                mip_improvement_timer = time.time()
                incumbent.value = value
                incumbent.solution[:] = e.data_out.mip_solution
                incumbent.provided = [False] * num_threads
                incumbent.provided_time = [0] * num_threads
                incumbent.provided[clone_id] = True
                incumbent.provided_time[clone_id] = time.time()
                incumbent.id = clone_id
            return

    def cbMIPUserSolutionHandler(e):
        """Update clones to best solution found so far..."""
        nonlocal incumbent

        if incumbent.value in (kHighsInf, -kHighsInf):
            return

        value = normalize_objective_value(e.data_out.objective_function_value)
        if is_better(incumbent.value, value):
            clone_id = int(e.user_data)

            # When incumbent.value has been provided to a clone but the clone is still
            # valued lower we re-provide the incumbent.value if it has been more than 10 seconds
            provide_solution = (incumbent.provided[clone_id] is False) or (
                incumbent.provided[clone_id] and time.time() - incumbent.provided_time[clone_id] > 10
            )
            if provide_solution:
                with incumbent.lock:
                    e.data_in.user_has_solution = True
                    e.data_in.user_solution[:] = incumbent.solution
                    incumbent.provided[clone_id] = True
                    incumbent.provided_time[clone_id] = time.time()
                return

    # Register callbacks
    for i in range(num_threads):
        clones[i].cbMipImprovingSolution.subscribe(cbMIPImprovedSolutionHandler, i)
        clones[i].cbMipUserSolution.subscribe(cbMIPUserSolutionHandler, i)
        clones[i].cbLogging.subscribe(cbLogging, i)

    # Worker function
    def task(clone: Highs, i: int):
        clone.solve()
        value = clone.getObjectiveValue()
        value = normalize_objective_value(value)
        if value == incumbent.value or is_better(value, incumbent.value):
            result_queue.put(i)

    # Start threads
    for i in range(num_threads):
        Thread(target=task, args=(clones[i], i), daemon=True).start()
        time.sleep(0.1)

    # Wait for the first thread to finish, timeout allows for Highs interrupt handling
    first_to_finish = None
    while first_to_finish is None:
        try:
            first_to_finish = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

    # Cancel the other threads, without any further output
    for i in range(num_threads):
        if i != first_to_finish:
            clones[i].silent()
        clones[i].cancelSolve()

    # Print solution report
    for message in clone_solution_report[first_to_finish]:
        print(message, end="")
    print()

    return clones[first_to_finish]
