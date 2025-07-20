from multiprocessing import cpu_count
from random import randint
from threading import Lock, Thread
import queue
import time

from highspy import Highs, kHighsInf, ObjSense
import numpy as np


def solve_par(model: Highs, config: dict) -> Highs:
    num_threads = config.get("num_processes", cpu_count() // 2)
    log_to_console = config.get("log_to_console", False)
    result_queue = queue.Queue()

    clones = [model] + [Highs() for _ in range(num_threads - 1)]
    clones[0].silent(log_to_console is False)
    clones[0].setOptionValue("random_seed", randint(0, 2**31-1))
    clones[0].setOptionValue("mip_rel_gap", 1e-4)
    clones[0].setOptionValue("mip_feasibility_tolerance", 1e-4)
    clones[0].setOptionValue("primal_feasibility_tolerance", 1e-4)
    clones[0].HandleUserInterrupt = True
    clones[0].enableCallbacks()

    for i in range(1, num_threads):
        clones[i].silent()
        clones[i].setOptionValue("random_seed", randint(0, 2**31-1))
        clones[i].setOptionValue("mip_rel_gap", 1e-4)
        clones[i].setOptionValue("mip_feasibility_tolerance", 1e-4)
        clones[i].setOptionValue("primal_feasibility_tolerance", 1e-4)
        clones[i].HandleUserInterrupt = True
        clones[i].passModel(clones[0].getModel())
        clones[i].enableCallbacks()

    obj_sense = clones[0].getObjectiveSense()[1]
    __incumbent_lock = Lock()
    __incumbent_provided = [False] * num_threads
    __incumbent_solution = np.zeros(clones[0].getNumCol())  # objective, solution
    __incumbent_value = 2**31 if obj_sense == ObjSense.kMinimize else -(2**31)
    # __user_solution_gap = 0.03
    # __user_solution_gap_abs = 2

    if obj_sense == ObjSense.kMinimize:

        def is_better(a, b):
            return a < b
    else:

        def is_better(a, b):
            return a > b

    def cbMIPImprovedSolutionHandler(e):
        nonlocal __incumbent_lock, __incumbent_provided, __incumbent_value, __incumbent_solution
        e_objective_value = e.data_out.objective_function_value
        if e_objective_value != kHighsInf:
            e_objective_value = int(e_objective_value)
        clone_id = int(e.user_data)
        # if is_better(e_objective_value, __incumbent_value) and (
        #     abs(e_objective_value - __incumbent_value) / max(1,__incumbent_value) >= __user_solution_gap
        #     or abs(e_objective_value - __incumbent_value) >= __user_solution_gap_abs
        # ):
        if is_better(e_objective_value, __incumbent_value):
            with __incumbent_lock:
                __incumbent_value = e_objective_value
                __incumbent_solution[:] = e.data_out.mip_solution
                __incumbent_provided = [False] * num_threads
                __incumbent_provided[clone_id] = True
                return
                # print(f"Incumbent supplanted by thread {clone_id} with {e_objective_value}")

    def cbMIPUserSolutionHandler(e):
        nonlocal __incumbent_lock, __incumbent_provided, __incumbent_solution, __incumbent_value
        if __incumbent_value == 2**31:
            return
        e_objective_value = e.data_out.objective_function_value
        if e_objective_value != kHighsInf:
            e_objective_value = int(e_objective_value)
        clone_id = int(e.user_data)
        # if (
        #     __incumbent_provided[clone_id] is False
        #     and is_better(__incumbent_value, e_objective_value)
        #     and (
        #         abs(e_objective_value - __incumbent_value) / max(1, __incumbent_value) >= __user_solution_gap
        #         or abs(e_objective_value - __incumbent_value) >= __user_solution_gap_abs
        #     )
        # ):
        if (__incumbent_provided[clone_id] is False and is_better(__incumbent_value, e_objective_value)):
            with __incumbent_lock:
                e.data_in.user_has_solution = True
                e.data_in.user_solution[:] = __incumbent_solution
                __incumbent_provided[clone_id] = True
                return
                # print(
                #     f"Provided incumbent solution with value {__incumbent_value} to ",
                #     f"thread {clone_id} with objective value {e_objective_value}",
                # )

    for i in range(num_threads):
        clones[i].cbMipImprovingSolution.subscribe(cbMIPImprovedSolutionHandler, i)
        clones[i].cbMipUserSolution.subscribe(cbMIPUserSolutionHandler, i)

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

    clones[first_to_finish].cbMipUserSolution.unsubscribe(cbMIPUserSolutionHandler)
    clones[first_to_finish].cbMipImprovingSolution.unsubscribe(cbMIPImprovedSolutionHandler)

    for i in range(num_threads):
        if i is not first_to_finish:
            clones[i].cancelSolve()

    return clones[first_to_finish]
