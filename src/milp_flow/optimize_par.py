import multiprocessing
from multiprocessing import Queue
from random import randint
from time import sleep

from pulp import HiGHS, LpProblem


def solve_par_worker(
    prob: LpProblem, options_dict: dict, queue: Queue, results: list, process_index: int
) -> None:
    options_dict["random_seed"] = randint(0, 2147483647)
    if options_dict["log_file"]:
        options_dict["log_file"] = (
            options_dict["log_file"].with_suffix(f".{process_index}.log").as_posix()
        )
    print(f"Process {process_index} starting using {options_dict}")
    sleep(1)
    solver = HiGHS()
    solver.optionsDict = options_dict
    prob.solve(solver)
    results[process_index] = prob.to_dict()
    queue.put(process_index)
    return


def solve_par(prob: LpProblem, options_dict: dict, num_processes: int) -> LpProblem:
    manager = multiprocessing.Manager()
    processes = []
    queue = multiprocessing.Queue()
    results = manager.list(range(num_processes))

    for i in range(num_processes):
        p = multiprocessing.Process(
            target=solve_par_worker, args=(prob, options_dict, queue, results, i)
        )
        processes.append(p)
        p.start()

    first_process = queue.get()
    for i, process in enumerate(processes):
        if process.is_alive():
            print(f"Terminating process: {i}")
            process.terminate()
            if i != first_process and options_dict["log_file"]:
                print(f"Clearing log from process: {i}")
                options_dict["log_file"].with_suffix(f".{i}.log").unlink()
        process.join()
    print(f"Using results from process {first_process}")
    result = prob.from_dict(results[first_process])
    if options_dict["log_file"]:
        options_dict["log_file"].with_suffix(f".{first_process}.log").rename(
            options_dict["log_file"]
        )
    return result[1]
