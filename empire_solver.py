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

    def filter_arcs(groupflow, arcs):
        return [
            var
            for a in arcs
            for key, var in a.vars.items()
            if key.startswith("groupflow_") and (key == groupflow or v.isLodging)
        ]

    def link_in_out_by_group(prob: LpProblem, v: Node, in_arcs: List[Arc], out_arcs: List[Arc]):
        all_inflows = []
        f = v.vars["f"]
        for group in v.groups:
            groupflow_key = f"groupflow_{group.id}"
            inflows = filter_arcs(groupflow_key, in_arcs)
            outflows = filter_arcs(groupflow_key, out_arcs)
            prob += lpSum(inflows) == lpSum(outflows), f"balance_{groupflow_key}_at_{v.name()}"
            all_inflows.append(inflows)
        prob += f == lpSum(all_inflows), f"flow_{v.name()}"
        prob += f <= v.ub * v.vars["x"], f"x_{v.name()}"

    prob = LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    # Variables
    # Create cost ∈ ℕ₀ 0 <= cost <= budget
    cost = LpVariable("cost", 0, config["budget"], "Integer")

    # Create node variables.
    for v in G["V"].values():
        # x ∈ {0,1} for each node indicating if node is in solution and cost calculation.
        v.vars["x"] = LpVariable(f"x_{v.name()}", 0, 1, "Binary")
        # f ∈ ℕ₀ for each node such that 0 <= f <= ub for cost calculation and performance.
        v.vars["f"] = LpVariable(f"flow_{v.name()}", 0, v.ub, "Integer")

    # Create arc variables.
    for arc in G["E"].values():
        # Group specific f ∈ ℕ₀ vars for each arc 0 <= f <= ub
        for group in set(arc.source.groups).intersection(set(arc.destination.groups)):
            key = f"groupflow_{group.id}"
            ub = arc.ub if arc.source.type in [NT.group, NT.𝓢, NT.𝓣, NT.lodging] else group.ub
            cat = "Binary" if arc.source.type in [NT.𝓢, NT.plant] else "Integer"
            arc.vars[key] = LpVariable(f"{key}_on_{arc.name()}", 0, ub, cat)

    # Objective: Maximize prizes ∑v(p)x for group specific values for all binary plant inflows.
    prize_values = [
        round(p.group_prizes[group.id]["value"], 2) * arc.vars[f"groupflow_{group.id}"]
        for p in G["P"].values()
        for group in p.groups
        for arc in p.inbound_arcs
    ]
    prob += lpSum(prize_values), "ObjectiveFunction"

    # Constraints
    # Cost var is defined with ub = budget so this is ∑v(𝑐)x <= budget
    prob += cost == lpSum(v.cost * v.vars["x"] for v in G["V"].values()), "TotalCost"

    # Group specific plant exclusivity enforced by plant's 0<=f<=ub bounds at variable creation.

    # Group specific lodging exclusivity, each lodging's 0<=f<=ub enforces correct selection.
    for group in G["G"].values():
        vars = [lodge.vars["x"] for lodge in G["L"].values() if lodge.groups[0] == group]
        prob += lpSum(vars) <= 1, f"lodging_{group.id}"

    # Group specific f⁻ == f⁺
    for v in G["V"].values():
        if v.type not in [NT.𝓢, NT.𝓣]:
            link_in_out_by_group(prob, v, v.inbound_arcs, v.outbound_arcs)

    # Group specific f⁻𝓣 == f⁺𝓢
    link_in_out_by_group(prob, G["V"]["𝓣"], G["V"]["𝓣"].inbound_arcs, G["V"]["𝓢"].outbound_arcs)

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


def extract_solution(config, prob, G):
    solution_vars = {}

    v_inSol = {k: v for k, v in G["V"].items() if v.inSolution()}

    plant_count = len([v for v in v_inSol.values() if v.isPlant])
    plant_cost = sum([v.cost for v in v_inSol.values() if v.isPlant])
    waypoint_count = len([v for v in v_inSol.values() if v.isWaypoint])
    waypoint_cost = sum([v.cost for v in v_inSol.values() if v.isWaypoint])
    lodging_count = sum([round(v.vars["f"].varValue) for v in v_inSol.values() if v.isLodging])
    lodging_cost = sum([v.cost for v in v_inSol.values() if v.isLodging])
    calculated_cost = plant_cost + waypoint_cost + lodging_cost

    logging.info(
        "\n"
        f"              Plants: {plant_count} for {plant_cost}\n"
        f"           Waypoints: {waypoint_count} for {waypoint_cost}\n"
        f"            Lodgings: {lodging_count} for {lodging_cost}\n"
    )

    for v in prob.variables():
        if v.varValue is not None and v.varValue > 0:
            rounded_value = round(v.varValue)
            if rounded_value >= 1:
                solution_vars[v.name] = {
                    "value": rounded_value,
                    "lowBound": v.lowBound,
                    "upBound": v.upBound,
                }

    outputs = []
    for k, v in solution_vars.items():
        if k.startswith("flow_"):
            kname = k.replace("flow_", "")
            if "_to_" in k:
                # An arc
                𝓢, destination = kname.split("_to_")
                arc_key = (𝓢, destination)
                arc = G["E"].get(arc_key, None)
                outputs.append(f"{arc}, {v}")
            else:
                # A node
                node = G["V"].get(kname, None)
                outputs.append(f"{node}, {v}")
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
        f"              budget: {config["budget"]}\n"
        f"           Actual ∑𝑐: {calculated_cost}\n"
        f"               LP ∑𝑐: {solver_cost}\n"
        f"              ∑prize: {round(solver_value)}\n"
    )

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


def main(config):
    """
    top_n: count of supply nodes per root by value (index based, zero=1)
    nearest_n: count of nearest roots available on waypoint nodes (index based, zero=1)
    waypoint_capacity: max loads on a waypoint
    """

    # for budget in [10, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]:
    #  5 =>          0       0         0   0.00%   13953400.80399  13900824.11        0.38%        4      0      0      1220     1.3s
    # 10 =>  L       0       0         0   0.00%   25014394.85418  24954483.03        0.24%     2046    151     61      2667     2.0s
    # 20 =>        713       4       306  71.02%   48433521.05607  44334018.51        9.25%     1543    263   9896    294384    79.0s
    # 30 =>       2389      14      1126  96.80%   61193200.58428  58540725           4.53%     1793    421   9567     1034k   304.1s
    # 50 =>       7290       5      3502  99.44%   87363482.57106  84340234.96002     3.58%     2267    617   9819     2901k   914.8s

    for budget in [10]:
        config["budget"] = budget
        config["top_n"] = 4
        config["nearest_n"] = 5
        config["waypoint_capacity"] = 25
        config["solver"]["file_prefix"] = "unicode_cleanup"
        config["solver"]["file_suffix"] = "fixedlodgingtest"
        config["solver"]["mips_gap"] = "default"
        config["solver"]["time_limit"] = "22000"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
