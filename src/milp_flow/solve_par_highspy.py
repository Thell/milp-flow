# solve_par_highspy.py

import queue
import time
from dataclasses import dataclass
from multiprocessing import cpu_count
from threading import Lock, Thread

import numpy as np
from highspy import Highs, ObjSense, kHighsInf


@dataclass
class Incumbent:
    id: int
    lock: Lock
    value: int
    solution: np.ndarray
    provided: list[bool]


def solve_par(model: Highs, config: dict) -> Highs:
    num_threads = config.get("solver", {}).get("jobs", cpu_count())
    result_queue = queue.Queue()

    clones = [model] + [Highs() for _ in range(num_threads - 1)]
    clones[0].HandleUserInterrupt = True
    clones[0].enableCallbacks()

    for i in range(1, num_threads):
        clones[i].setOptionValue("random_seed", i)
        clones[i].HandleUserInterrupt = True
        clones[i].passModel(clones[0].getModel())
        clones[i].enableCallbacks()

    obj_sense = clones[0].getObjectiveSense()[1]
    incumbent = Incumbent(
        id=0,
        lock=Lock(),
        value=2**31 if obj_sense == ObjSense.kMinimize else -(2**31),
        solution=np.zeros(clones[0].getNumCol()),
        provided=[False] * num_threads,
    )

    solution_gap = 0.03
    solution_gap_abs = 2
    thread_log_capture = [[]] * num_threads

    if obj_sense == ObjSense.kMinimize:

        def is_better(a, b):
            return a < b

    else:

        def is_better(a, b):
            return a > b

    def capture_logs(e):
        nonlocal incumbent, thread_log_capture, clones
        id = int(e.user_data)
        if thread_log_capture[id] or e.message.startswith("\nSolving report"):
            with incumbent.lock:
                thread_log_capture[id].append(e.message)
                for clone_id in range(num_threads):
                    if clone_id != id:
                        clones[clone_id].silent()
        elif id == incumbent.id and "=>" not in e.message:
            print(e.message, end="")

    def cbMIPImprovedSolutionHandler(e):
        nonlocal incumbent, clones, solution_gap, solution_gap_abs
        value = e.data_out.objective_function_value
        value = int(value) if value != kHighsInf else value
        if is_better(value, incumbent.value) and (
            abs(value - incumbent.value) / incumbent.value >= solution_gap
            or abs(value - incumbent.value) >= solution_gap_abs
        ):
            id = int(e.user_data)
            with incumbent.lock:
                incumbent.value = value
                incumbent.solution[:] = e.data_out.mip_solution
                incumbent.provided = [False] * num_threads
                incumbent.provided[id] = True
                incumbent.id = id
                # print(f"Incumbent supplanted by thread {clone_id} with {e_objective_value}")
                return

    def cbMIPUserSolutionHandler(e):
        nonlocal incumbent, solution_gap, solution_gap_abs
        if incumbent.value == 2**31:
            return
        value = e.data_out.objective_function_value
        value = int(value) if value != kHighsInf else value
        id = int(e.user_data)
        if (
            incumbent.provided[id] is False
            and is_better(incumbent.value, value)
            and (
                abs(value - incumbent.value) / incumbent.value >= solution_gap
                or abs(value - incumbent.value) >= solution_gap_abs
            )
        ):
            with incumbent.lock:
                e.data_in.user_has_solution = True
                e.data_in.user_solution[:] = incumbent.solution
                incumbent.provided[id] = True
                # print(f"Provided incumbent to thread {clone_id} with {e_objective_value}")
                return

    for i in range(num_threads):
        clones[i].cbMipImprovingSolution.subscribe(cbMIPImprovedSolutionHandler, i)
        clones[i].cbMipUserSolution.subscribe(cbMIPUserSolutionHandler, i)
        clones[i].cbLogging.subscribe(capture_logs, i)

    def task(clone: Highs, i: int):
        clone.solve()
        result_queue.put(i)

    threads = []
    for i in range(num_threads):
        t = Thread(target=task, args=(clones[i], i), daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.1)

    first_to_finish = None
    while first_to_finish is None:
        try:
            first_to_finish = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

    for i in range(num_threads):
        clones[i].cancelSolve()

    for message in thread_log_capture[first_to_finish]:
        print(message, end="")

    return clones[first_to_finish]
