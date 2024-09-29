# optimize_par_poi.py

import multiprocessing
from multiprocessing import Queue
from random import randint
from time import sleep

from pyoptinterface import highs


def solve_par_worker(
    model: highs.Model, options: dict, queue: Queue, results: list, process_index: int
) -> None:
    options["random_seed"] = randint(0, 2147483647)
    if options["log_file"]:
        options["log_file"] = options["log_file"].with_suffix(f".{process_index}.log").as_posix()
    print(f"Process {process_index} starting using {options}")
    sleep(1)
    for k, v in options.items():
        model.set_raw_parameter(k, v)
    start_vars = model.mip_start_values.copy()
    model.optimize()
    queue.put(process_index)
    sol_results = [model.get_value(k) for k in start_vars.keys()]
    results[process_index] = sol_results
    return


def solve_par(model: highs.Model, options: dict, num_processes: int) -> highs.Model:
    manager = multiprocessing.Manager()
    processes = []
    queue = multiprocessing.Queue()
    results = manager.list(range(num_processes))

    for i in range(num_processes):
        p = multiprocessing.Process(
            target=solve_par_worker, args=(model, options, queue, results, i)
        )
        processes.append(p)
        p.start()

    first_process = queue.get()
    for i, process in enumerate(processes):
        if process.is_alive():
            print(f"Terminating process: {i}")
            process.terminate()
            if i != first_process and options["log_file"]:
                print(f"Clearing log from process: {i}")
                options["log_file"].with_suffix(f".{i}.log").unlink()
        process.join()
    print(f"Using results from process {first_process}")
    result = results[first_process]
    if options["log_file"]:
        options["log_file"].with_suffix(f".{first_process}.log").rename(options["log_file"])
    return result
