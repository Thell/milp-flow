"""Flow Model of a Budget-Constrained Prize Collecting Steiner Forest with Cost Based Capacity"""

import json
import logging
import os
from pathlib import Path
from typing import List

from pulp import LpVariable, lpSum, LpProblem, LpMaximize, HiGHS_CMD

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
            for arc in arcs
            for key, var in arc.vars.items()
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

    prob = LpProblem("MaximizeEmpireValue", LpMaximize)

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
        round(plant.group_prizes[group.id]["value"], 2) * arc.vars[f"groupflow_{group.id}"]
        for plant in G["P"].values()
        for group in plant.groups
        for arc in plant.inbound_arcs
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


def config_filename(config):
    parts = [
        config["solver"]["file_prefix"],
        f"b{config["budget"]}",
        f"lb{config["lodging_bonus"]}",
        f"tn{config["top_n"]}",
        f"nn{config["nearest_n"]}",
        f"wc{config["waypoint_capacity"]}",
        f"g{config["solver"]["mips_gap"].replace("0.", "")}",
        config["solver"]["file_suffix"],
    ]
    return "_".join(filter(None, parts))


def config_solver(config):
    filename = config_filename(config)
    filepath = Path(os.path.dirname(__file__), "highs_output")

    mips_tol = config["solver"]["mips_tol"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = config["budget"] / 10_000
    time_limit = config["solver"]["time_limit"]

    options = [
        "parallel=on",
        "threads=16",
        f"time_limit={time_limit}",
        f"mip_feasibility_tolerance={mips_tol}",
        "mip_heuristic_effort=0.5",
        f"mip_rel_gap={mips_gap}",
        f"primal_feasibility_tolerance={mips_tol}",
        # "mip_improving_solution_save=on",
        # f"mip_improving_solution_file={filepath.joinpath("improvements", f'{filename}.sol')}",
        # f"write_model_file={filepath.joinpath("models", f"{filename}.lp")}",
        # f"write_presolved_model_to_file=True",
    ]
    options = [o for o in options if "default" not in o]

    solver = HiGHS_CMD(
        path="/home/thell/HiGHS/build/bin/highs",
        keepFiles=True,
        options=options,
        logPath=filepath.joinpath("logs", f"{filename}.log"),
    )
    return solver


def print_solving_info_header(config, G: GraphData):
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
        f"\nSolving:    graph with {len(G['V'])} nodes and {len(G['E'])} arcs"
        f"\n  Using:    budget: {budget}, lodging_bonus: {lodging_bonus}, top_n: {top_n},"
        f" nearest_n: {nearest_n}, capacity: {waypoint_capacity}"
        f"\n   With:    threads: 16, mip_feasibility_tolerance: {mips_tol}, gap: {mips_gap}"
    )


def print_solving_info_trailer(config, prob, G):
    v_inSol = {k: v for k, v in G["V"].items() if v.inSolution()}
    plant_count = len([v for v in v_inSol.values() if v.isPlant])
    plant_cost = sum([v.cost for v in v_inSol.values() if v.isPlant])
    waypoint_count = len([v for v in v_inSol.values() if v.isWaypoint])
    waypoint_cost = sum([v.cost for v in v_inSol.values() if v.isWaypoint])
    lodging_count = sum([round(v.vars["f"].varValue) for v in v_inSol.values() if v.isLodging])
    lodging_cost = sum([v.cost for v in v_inSol.values() if v.isLodging])

    calculated_cost = plant_cost + waypoint_cost + lodging_cost
    solver_cost = prob.variablesDict()["cost"].varValue
    solver_value = prob.objective.value()

    logging.info(
        "\n"
        f"              Plants: {plant_count} for {plant_cost}\n"
        f"           Waypoints: {waypoint_count} for {waypoint_cost}\n"
        f"            Lodgings: {lodging_count} for {lodging_cost}\n"
        f"\n"
        f"              budget: {config["budget"]}\n"
        f"           Actual ∑𝑐: {calculated_cost}\n"
        f"               LP ∑𝑐: {solver_cost}\n"
        f"              ∑prize: {round(solver_value)}\n"
    )


def save_reproduction_data(config, prob, G):
    obj_value = round(prob.objective.value())
    filename = f"{config_filename(config)}_{obj_value}"
    filepath = Path(os.path.dirname(__file__), "highs_output", "solutions")

    new_path = filepath.joinpath(f"{filename}.lp")
    prob.writeLP(new_path)
    print("         LP model saved:", new_path)

    new_path = filepath.joinpath(f"{filename}_pulp.json")
    prob.to_json(new_path)
    print("Solved pulp model saved:", new_path)

    old_path = filepath.parent.parent.joinpath("MaximizeEmpireValue-pulp.sol")
    new_path = filepath.joinpath(f"{filename}_highs.sol")
    Path(old_path).rename(new_path)
    print("   HiGHS solution saved:", new_path)

    new_path = filepath.joinpath(f"{filename}_cfg.json")
    with open(new_path, "w") as file:
        json.dump(config, file, indent=4)
    print("   Problem config saved:", new_path)

    print(f"     Stem for workerman: {new_path.stem.replace("_cfg", "")}")


def empire_solver(config):
    G: GraphData = generate_empire_data(config)
    solver = config_solver(config)
    prob = create_problem(config, G)

    print_solving_info_header(config, G)
    solver.solve(prob)
    print_solving_info_trailer(config, prob, G)

    save_reproduction_data(config, prob, G)


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

    for budget in [5, 10, 20, 30, 50]:
        config["budget"] = budget
        config["top_n"] = 4
        config["nearest_n"] = 5
        config["waypoint_capacity"] = 25
        config["solver"]["file_prefix"] = ""
        config["solver"]["file_suffix"] = ""
        config["solver"]["mips_gap"] = "default"
        config["solver"]["time_limit"] = "22000"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
