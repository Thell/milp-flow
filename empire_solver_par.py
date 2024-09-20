# import concurrent.futures
import os
from pathlib import Path
import random
import subprocess
import time

from typing import List
from pulp import LpSolver_CMD, HiGHS_CMD, PulpSolverError, constants


class HiGHS_CMD_PAR(LpSolver_CMD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = kwargs.get("num_processes", 8)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError("PuLP: cannot execute " + self.path)
        lp.checkDuplicateVars()

        tmpMps, tmpSol, tmpOptions, tmpLog, tmpMst = self.create_tmp_files(
            lp.name, "mps", "sol", "HiGHS", "HiGHS_log", "mst"
        )
        lp.writeMPS(tmpMps, with_objsense=True)

        file_options: List[str] = []
        file_options.append(f"solution_file={tmpSol}")
        file_options.append("write_solution_to_file=true")
        file_options.append(f"write_solution_style={HiGHS_CMD.SOLUTION_STYLE}")
        if not self.msg:
            file_options.append("log_to_console=false")
        if "threads" in self.optionsDict:
            file_options.append(f"threads={self.optionsDict['threads']}")
        if "gapRel" in self.optionsDict:
            file_options.append(f"mip_rel_gap={self.optionsDict['gapRel']}")
        if "gapAbs" in self.optionsDict:
            file_options.append(f"mip_abs_gap={self.optionsDict['gapAbs']}")
        if "logPath" in self.optionsDict:
            highs_log_file = self.optionsDict["logPath"]
        else:
            highs_log_file = tmpLog
        # file_options.append(f"log_file={highs_log_file}")

        command: List[str] = []
        command.append(self.path)
        command.append(tmpMps)
        # command.append(f"--options_file={tmpOptions}")
        if self.timeLimit is not None:
            command.append(f"--time_limit={self.timeLimit}")
        if not self.mip:
            command.append("--solver=simplex")
        if "threads" in self.optionsDict:
            command.append("--parallel=on")
        if self.optionsDict.get("warmStart", False):
            self.writesol(tmpMst, lp)
            command.append(f"--read_solution_file={tmpMst}")

        options = iter(self.options)
        for option in options:
            # assumption: all cli and file options require an argument which is provided after the equal sign (=)
            if "=" not in option:
                option += f"={next(options)}"

            # identify cli options by a leading dash (-) and treat other options as file options
            if option.startswith("-"):
                command.append(option)
            else:
                file_options.append(option)

        # Execute processes using a different random seeds passed via the options file.
        process_options = []
        process_commands = []
        for i in range(self.num_processes):
            p_options = file_options.copy()
            # Each process needs its own log file.
            p_options.append(f"log_file={highs_log_file.replace(".HiGHS_log", f"_{i}.log")}")
            # Each process needs it's own random_seed.
            p_options.append(f"random_seed={random.randint(0, 2147483647)}")
            # Each process needs it's own options file.
            options_filename = tmpOptions.replace(".HiGHS", f"_{i}.HiGHS")
            process_options.append(p_options)
            with open(options_filename, "w") as options_file:
                options_file.write("\n".join(p_options))
            # Each process needs its own command to pass the options file.
            process_command = command.copy()
            process_command.append(f"--options_file={options_filename}")
            process_commands.append(process_command)

        process_list = []
        for i in range(self.num_processes):
            print("Starting HiGHS instance:", i)
            process = subprocess.Popen(process_commands[i])
            process_list.append((process, i))

        # Monitor for the first process that completes
        first_process = None
        while not first_process:
            for process, index in process_list:
                if process.poll() is not None:
                    first_process = (process, index)
                    break
            time.sleep(0.1)

        print(f"Process {first_process[1]} completed...\n")

        # Terminate the remaining processes.
        for process, index in process_list:
            if process != first_process[0] and process.poll() is None:
                process.terminate()
                print(f"Terminated process {index}")

        # Set the process and highs_log_file to use for further pulp processing
        process = first_process[0]
        log_file_entry = [
            option for option in process_options[first_process[1]] if "log_file=" in option
        ]
        if log_file_entry:
            highs_log_file = log_file_entry[0].split("=")[1]
        else:
            raise ValueError("Log file option not found in command.")

        # HiGHS return code semantics (see: https://github.com/ERGO-Code/HiGHS/issues/527#issuecomment-946575028)
        # - -1: error
        # -  0: success
        # -  1: warning
        # process = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
        if process.wait() == -1:
            raise PulpSolverError(
                "Pulp: Error while executing HiGHS, use msg=True for more details" + self.path
            )

        with open(highs_log_file, "r") as log_file:
            print(log_file.read())

        with open(highs_log_file, "r") as log_file:
            lines = log_file.readlines()

        lines = [line.strip().split() for line in lines]

        # LP
        model_line = [line for line in lines if line[:2] == ["Model", "status"]]
        if len(model_line) > 0:
            model_status = " ".join(model_line[0][3:])  # Model status: ...
        else:
            # ILP
            model_line = [line for line in lines if "Status" in line][0]
            model_status = " ".join(model_line[1:])
        sol_line = [line for line in lines if line[:2] == ["Solution", "status"]]
        sol_line = sol_line[0] if len(sol_line) > 0 else ["Not solved"]
        sol_status = sol_line[-1]
        if model_status.lower() == "optimal":  # optimal
            status, status_sol = (
                constants.LpStatusOptimal,
                constants.LpSolutionOptimal,
            )
        elif sol_status.lower() == "feasible":  # feasible
            # Following the PuLP convention
            status, status_sol = (
                constants.LpStatusOptimal,
                constants.LpSolutionIntegerFeasible,
            )
        elif model_status.lower() == "infeasible":  # infeasible
            status, status_sol = (
                constants.LpStatusInfeasible,
                constants.LpSolutionInfeasible,
            )
        elif model_status.lower() == "unbounded":  # unbounded
            status, status_sol = (
                constants.LpStatusUnbounded,
                constants.LpSolutionUnbounded,
            )
        else:  # no solution
            status, status_sol = (
                constants.LpStatusNotSolved,
                constants.LpSolutionNoSolutionFound,
            )

        if not os.path.exists(tmpSol) or os.stat(tmpSol).st_size == 0:
            status_sol = constants.LpSolutionNoSolutionFound
            values = None
        elif status_sol in {
            constants.LpSolutionNoSolutionFound,
            constants.LpSolutionInfeasible,
            constants.LpSolutionUnbounded,
        }:
            values = None
        else:
            values = self.readsol(tmpSol)

        self.delete_tmp_files(tmpMps, tmpSol, tmpOptions, tmpLog, tmpMst)
        for i, options in enumerate(process_options):
            if i == first_process[1]:
                continue
            log_file_entry = [option for option in options if "log_file=" in option]
            if log_file_entry:
                log_file = log_file_entry[0].split("=")[1]
                Path.unlink(log_file)

        lp.assignStatus(status, status_sol)

        if status == constants.LpStatusOptimal:
            lp.assignVarsVals(values)

        return status, highs_log_file

    def writesol(self, filename, lp):
        """Writes a HiGHS solution file"""

        variable_rows = []
        for var in lp.variables():  # zero variables must be included
            variable_rows.append(f"{var.name} {var.varValue or 0}")

        # Required preamble for HiGHS to accept a solution
        all_rows = [
            "Model status",
            "None",
            "",
            "# Primal solution values",
            "Feasible",
            "",
            f"# Columns {len(variable_rows)}",
        ]
        all_rows.extend(variable_rows)

        with open(filename, "w") as file:
            file.write("\n".join(all_rows))

    def readsol(self, filename):
        """Read a HiGHS solution file"""
        with open(filename) as file:
            lines = file.readlines()

        begin, end = None, None
        for index, line in enumerate(lines):
            if begin is None and line.startswith("# Columns"):
                begin = index + 1
            if end is None and line.startswith("# Rows"):
                end = index
        if begin is None or end is None:
            raise PulpSolverError("Cannot read HiGHS solver output")

        values = {}
        for line in lines[begin:end]:
            name, value = line.split()
            values[name] = float(value)
        return values


# def solve_with_seed(problem: LpProblem, config: Dict[str, Any]):
#     solver = HiGHS_CMD_PAR(random_seed=seed)
#     return problem.solve(solver), config["random_seed"]


# def parallel_solve(problem: LpProblem, seeds: list):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as executor:
#         futures = {executor.submit(solve_with_seed, problem, seed): seed for seed in seeds}

#         for future in concurrent.futures.as_completed(futures):
#             result, seed = future.result()
#             print(f"Solved with seed {seed}. Cancelling other jobs.")
#             # Cancel other futures
#             for f in futures:
#                 if not f.done():
#                     f.cancel()
#             break


# def parallel_solve(problem: LpProblem, configs: list):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
#         futures = {executor.submit(solve_with_seed, problem, config): config for config in configs}

#         for future in concurrent.futures.as_completed(futures):
#             result, seed = future.result()
#             print(f"Solved with seed {seed}. Cancelling other jobs.")
#             # Cancel other futures
#             for f in futures:
#                 if not f.done():
#                     f.cancel()
#             break
