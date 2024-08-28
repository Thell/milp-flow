"""Flow Model of a Budget-Constrained Prize Collecting Steiner Forest with Cost Based Capacity"""

import json
import logging
from pathlib import Path
from typing import List


import natsort
from pulp import LpVariable, lpSum, LpProblem
import pulp

from generate_empire_data import (
    NodeType as NT,
    Node,
    Arc,
    generate_empire_data,
    GraphData,
)

logger = logging.getLogger(__name__)


def create_problem(config, G):
    """Create the problem and add the variables and constraints."""

    def link_in_out(prob: LpProblem, v: Node, in_arcs: List[Arc], out_arcs: List[Arc]):
        f_vars = []
        ƒ = v.pulp_vars["ƒ"]
        for 𝓻 in v.𝓻:
            ƒ𝓻_key = f"ƒ𝓻_{𝓻.id}"
            ins = [
                var
                for a in in_arcs
                for key, var in a.pulp_vars.items()
                if key.startswith("ƒ𝓻_") and (key == ƒ𝓻_key or v.type is NT.lodging)
            ]
            f_vars.append(ins)
            outs = [
                var
                for a in out_arcs
                for key, var in a.pulp_vars.items()
                if key.startswith("ƒ𝓻_") and (key == ƒ𝓻_key or v.type is NT.lodging)
            ]
            prob += lpSum(ins) == lpSum(outs), f"out_{ƒ𝓻_key}_at_{v.name()}"
        prob += ƒ == lpSum(f_vars), f"ƒ_{v.name()}"
        prob += ƒ <= v.𝓬 * v.pulp_vars["𝒙"], f"↧_{v.name()}"

    prob = LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    # Variables

    # Create cost ∈ ℕ₀ 0 <= cost <= budget
    cost = LpVariable("cost", lowBound=0, upBound=config["budget"], cat="Integer")

    # Create node variables.
    for v in G["V"].values():
        # 𝒙 ∈ {0,1} for each node indicating if node is in solution and cost calculation.
        v.pulp_vars["𝒙"] = LpVariable(f"𝒙_{v.name()}", cat="Binary")
        # ƒ ∈ ℕ₀ for each node such that 0 <= ƒ <= 𝓬 for cost calculation and performance.
        v.pulp_vars["ƒ"] = LpVariable(f"ƒ_{v.name()}", lowBound=0, upBound=v.𝓬, cat="Integer")

    # Create edge variables.
    for a in G["E"].values():
        # 𝓡 group specific ƒ ∈ ℕ₀ vars for each arc 0 <= ƒ <= 𝓬
        for 𝓻 in set(a.source.𝓻).intersection(set(a.destination.𝓻)):
            key = f"ƒ𝓻_{𝓻.id}"
            ub = a.𝓬 if a.source.type in [NT.𝓡, NT.𝓢, NT.𝓣, NT.lodging] else 𝓻.𝓬
            cat = "Binary" if a.source.type in [NT.𝓢, NT.t] else "Integer"
            a.pulp_vars[key] = LpVariable(f"{key}_on_{a.name()}", 0, ub, cat)

    # Objective

    # Maximize total prizes ∑v(p)𝒙 for 𝓡 group specific values in all binary terminal inflows.
    prizes = []
    for v in G["V"].values():
        if v.isTerminal:
            for 𝓻 in v.𝓻:
                for a in v.inbound_arcs:
                    prizes.append(round(v.𝓻_prizes[𝓻.id]["value"], 2) * a.pulp_vars[f"ƒ𝓻_{𝓻.id}"])
    prob += lpSum(prizes), "ObjectiveFunction"

    # Constraints

    # Cost var is defined with ub = budget so this is ∑v(𝑐)𝒙 <= budget
    prob += cost == lpSum(v.cost * v.pulp_vars["𝒙"] for v in G["V"].values()), "TotalCost"

    # A single lodging within each 𝓡 group.
    for 𝒓 in G["R"].values():
        vars = []
        for v in G["V"].values():
            if v.name().startswith(f"lodging_{r.id}_"):
                vars.append(v.pulp_vars["𝒙"])
        prob += lpSum(vars) <= 1, f"Lodge_{𝒓.id}"

    # 𝓡 group specific ƒ⁻ == ƒ⁺
    for v in G["V"].values():
        if v.type not in [NT.𝓢, NT.𝓣]:
            link_in_out(prob, v, v.inbound_arcs, v.outbound_arcs)

    # 𝓡 group specific ƒ⁻𝓣 == ƒ⁺𝓢
    link_in_out(prob, G["V"]["𝓣"], G["V"]["𝓣"].inbound_arcs, G["V"]["𝓢"].outbound_arcs)

    return prob


def config_solver(config, outfile_path, filename):
    mips_tol = config["solver"]["mips_tol"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = config["budget"] / 10_000
    time_limit = config["solver"]["time_limit"]

    options = [
        "parallel=on",
        "threads=16",
        f"time_limit={time_limit}",
        f"mip_rel_gap={mips_gap}",
        "mip_heuristic_effort=0.5",
        f"mip_feasibility_tolerance={mips_tol}",
        f"primal_feasibility_tolerance={mips_tol}",
        "mip_improving_solution_save=on",
        f"mip_improving_solution_file={outfile_path}/improvements/{filename}.sol",
        # f"write_presolved_model_to_file=True",
    ]
    options = [o for o in options if "default" not in o]

    solver = pulp.HiGHS_CMD(
        # path="/home/thell/.local/bin/highs",
        path="/home/thell/HiGHS/build/bin/highs",
        keepFiles=True,
        options=options,
        logPath=f"{outfile_path}/logs/{filename}.log",
    )
    return solver


def config_filename(config):
    budget = config["budget"]
    lodging_bonus = config["lodging_bonus"]
    top_n = config["top_n"]
    nearest_n = config["nearest_n"]
    waypoint_capacity = config["waypoint_capacity"]
    prefix = config["solver"]["file_prefix"]
    suffix = config["solver"]["file_suffix"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = budget / 10_000
    mips_gap = str(mips_gap).replace("0.", "")
    if prefix:
        prefix = f"{prefix}_"
    if suffix:
        suffix = f"_{suffix}"

    filename = f"{prefix}mc{budget}_lb{lodging_bonus}_tn{top_n}_nn{nearest_n}_wc{waypoint_capacity}_g{mips_gap}{suffix}"
    return filename


def print_solving_info_header(config):
    budget = config["budget"]
    lodging_bonus = config["lodging_bonus"]
    top_n = config["top_n"]
    nearest_n = config["nearest_n"]
    waypoint_capacity = config["waypoint_capacity"]
    mips_tol = config["solver"]["mips_tol"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = budget / 10_000

    logging.info(
        f"\n===================================================="
        f"\nSolving: budget: {budget}, Lodging_bonus: {lodging_bonus} top_n: {top_n},"
        f" nearest_n: {nearest_n}, capacity={waypoint_capacity}"
        f"\nUsing threads=16 and mip_feasibility_tolerance={mips_tol}, gap: {mips_gap}"
    )


def extract_solution(config, prob, graph_data):
    solution_vars = {}
    gt0lt1_vars = set()

    for v in prob.variables():
        if v.varValue is not None and v.varValue > 0:
            if v.varValue < 1:
                gt0lt1_vars.add(v.name)
            rounded_value = round(v.varValue)
            if rounded_value >= 1:
                solution_vars[v.name] = {
                    "value": rounded_value,
                    "lowBound": v.lowBound,
                    "upBound": v.upBound,
                }

    calculated_cost = 0
    outputs = []
    arc_loads = []
    waypoint_loads = []
    for k, v in solution_vars.items():
        if k.startswith("ƒ_"):
            kname = k.replace("ƒ_", "")
            if "_to_" in k:
                # An arc
                𝓢, destination = kname.split("_to_")
                arc_key = (𝓢, destination)
                arc = graph_data["E"].get(arc_key, None)
                outputs.append(f"{arc}, {v}")
                if arc.source.type is NT.waypoint or arc.destination.type is NT.waypoint:
                    arc_loads.append((arc, v["value"]))
            else:
                # A node
                node = graph_data["V"].get(kname, None)
                outputs.append(f"{node}, {v}")
                calculated_cost = calculated_cost + node.cost
                if node.type is NT.waypoint:
                    waypoint_loads.append((node, v["value"]))
    outputs = natsort.natsorted(outputs)

    solver_cost = 0
    if "cost" in solution_vars.keys():
        solver_cost = int(solution_vars["cost"]["value"])
    solver_value = round(prob.objective.value())

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        logging.debug("Detailed Solution:")
        for output in outputs:
            logging.debug(output)

    logging.info(
        f"\n"
        # f"                 ƒ_𝓢: {solution_vars["ƒ_𝓢"]["value"]}\n"
        # f"                 ƒ_𝓣: {solution_vars["ƒ_𝓣"]["value"]}\n"
        f"                 |𝒕|: {len([x for x in outputs if x.startswith("Node(name: t_")])}\n"
        f"              budget: {config["budget"]}\n"
        f"           Actual ∑𝑐: {calculated_cost}\n"
        f"               LP ∑𝑐: {solver_cost}\n"
        f"              ∑prize: {round(solver_value)}\n"
        # f"      Max waypoint ƒ: {max(waypoint_loads, key=itemgetter(1))}\n"
        # f"           Max arc ƒ: {max(arc_loads, key=itemgetter(1))}\n"
    )
    if gt0lt1_vars:
        logging.warning(f"WARNING: 0 < x < 1 vars count: {len(gt0lt1_vars)}")

    data = {"config": config, "objective": solver_value, "solution_vars": solution_vars}
    return data


def empire_solver(config):
    graph_data: GraphData = generate_empire_data(config)
    print("nodes:", len(graph_data["V"]))
    print("arcs:", len(graph_data["E"]))

    prob = create_problem(config, graph_data)

    filename = config_filename(config)
    outfile_path = "/home/thell/milp-flow/highs_output"
    prob.writeLP(f"{outfile_path}/models/{filename}.lp")

    print_solving_info_header(config)
    solver = config_solver(config, outfile_path, filename)
    solver.solve(prob)

    old_path = f"{outfile_path}/../MaximizeEmpireValue-pulp.sol"
    new_path = f"{outfile_path}/solutions/{filename}.sol"
    Path(old_path).rename(new_path)
    print("Solution moved to:", new_path)

    out_data = extract_solution(config, prob, graph_data)
    filepath = f"{outfile_path}/solution-vars/{filename}_{out_data["objective"]}.json"
    with open(filepath, "w") as file:
        json.dump(out_data, file, indent=4)
    print("Solution vars written to:", filepath)

    # assert (
    #     prob.variablesDict()["ƒ_𝓢"].value() == prob.variablesDict()["ƒ_𝓣"].value()
    # ), "Load value mismatch: 𝓢 != 𝓣."


def main(config):
    """
    top_n: count of supply nodes per root by value (index based, zero=1)
    nearest_n: count of nearest roots available on waypoint nodes (index based, zero=1)
    waypoint_capacity: max loads on a waypoint
    """

    # for budget in [10, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]:
    #  5 =>          0       0         0   0.00%   13953400.80399  13900824.11        0.38%        4      0      0      1220     1.2s
    # 10 =>  L       0       0         0   0.00%   25014394.85418  24954483.03        0.24%     2046    151     61      2667     1.9s
    # 20 =>        713       4       306  71.02%   48433521.05607  44334018.51        9.25%     1543    263   9896    294384    78.8s
    # 30 =>       2389      14      1126  96.80%   61193200.58428  58540725           4.53%     1793    421   9567     1034k   302.6s
    # 50 =>       7290       5      3502  99.44%   87363482.57106  84340234.96002     3.58%     2267    617   9819     2901k   913.6s

    for budget in [501]:
        config["budget"] = budget
        config["top_n"] = 4
        config["nearest_n"] = 5
        config["waypoint_capacity"] = 25
        config["solver"]["file_prefix"] = "TMPREWORK-FlowRework"
        config["solver"]["file_suffix"] = "t18k"
        config["solver"]["mips_gap"] = "default"
        config["solver"]["time_limit"] = "18000"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
