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

    prob = LpProblem(config["name"], LpMaximize)

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
    # Group specific plant exclusivity enforced by plant's 0<=f<=ub bounds at variable creation.

    # Cost var is defined with ub = budget so this is ∑v(𝑐)x <= budget
    prob += cost == lpSum(v.cost * v.vars["x"] for v in G["V"].values()), "TotalCost"

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

    prob += G["V"]["𝓢"].vars["x"] == 1, "x_source"

    # Enforce that all nodes maintain a minimum degree of 2.
    for node in G["V"].values():
        if node.type in [NT.S, NT.T]:
            continue

        in_neighbors = [arc.source.vars["x"] for arc in node.inbound_arcs]
        out_neighbors = [arc.destination.vars["x"] for arc in node.outbound_arcs]

        # Waypoint and plants always have 1 or more inbound neighbors than outbound neighbors
        # which is sortof a misnomer since all waypoints have anti-parallel edges but for some
        # reason this constraint improves performance. I think it is because of the zone edges.
        if node.isWaypoint or node.isPlant:
            prob += lpSum(in_neighbors) >= lpSum(out_neighbors)

        # All nodes should be 2-degree.
        if node.isWaypoint:
            prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
        else:
            prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0

        # Every active node must have at least one active outbound neighbor
        prob += lpSum(out_neighbors) >= node.vars["x"]

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

    mips_tol = config["solver"].get("mips_tol", 1e-6)
    mips_gap = config["solver"].get("mips_gap", 1e-5)
    if mips_gap == "auto":
        mips_gap = config["budget"] / 10_000
    time_limit = config["solver"]["time_limit"]

    options = [
        "parallel=on",
        "threads=16",
        f"time_limit={time_limit}",
        "mip_feasibility_tolerance=1e-4",  # 1e-6 from config file
        "primal_feasibility_tolerance=1e-4",  # 1e-6 from config file
        # f"mip_rel_gap={mips_gap}",  # 1e-5 from this main()'s config
        "mip_pool_soft_limit=5000",  # Leaving near default seems best. 892.66 on 501 w/7500 904 w/5000 Both w/1e-6 feas and default feas
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

    old_path = filepath.parent.parent.joinpath(f"{config["name"]}-pulp.sol")
    new_path = filepath.joinpath(f"{filename}_highs.sol")
    Path(old_path).rename(new_path)
    print("   HiGHS solution saved:", new_path)

    new_path = filepath.joinpath(f"{filename}_cfg.json")
    with open(new_path, "w") as file:
        json.dump(config, file, indent=4)
    print("   Problem config saved:", new_path)

    print(f"     Stem for workerman: {new_path.stem.replace("_cfg", "")}")


def empire_solver(config):
    print("\n====================================================")

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
    #   5 =>         1       0         1 100.00%   13900824.11     13900824.11        0.00%       42      4     15       539     1.3s
    #  10 =>         1       0         1 100.00%   24954483.03     24954483.03        0.00%      293     18     16      1136     0.7s
    #  20 =>         1       0         1 100.00%   44334018.51     44334018.51        0.00%       48      5     18      2665     1.1s
    #  30 =>         1       0         1 100.00%   58540725        58540725           0.00%       71     48    255     21461     8.7s
    #  50 =>         7       0         4 100.00%   84340234.96     84340234.96        0.00%      175     87    730     44348    16.6s
    # 100 =>      1865       0       814 100.00%   137568512.6988  137562435.03       0.00%     1311    222   4884    401490   134.5s
    # 150 =>     12797       0      2652 100.00%   183029741.7658  183012954.78       0.01%     1540    335   4880     1971k   674.9s
    # 200 =>     39435       0     18036 100.00%   222778830.6818  222756620.54       0.01%     2411    433   4846     9074k  2782.7s
    # 250 =>     38701       0     15709 100.00%   260995924.6606  260969949.12       0.01%     2886    486   4868     7755k  2635.1s
    # 300 =>     31674       0     11645 100.00%   297920432.924   297890668.77       0.01%     2253    463   4902     5684k  1804.5s
    # 350 =>     25784       0     10578 100.00%   332070168.0292  332036997.45       0.01%     2154    338   5020     4910k  1630.9s
    # 400 =>     19413       0      8167 100.00%   361375079.5188  361339029          0.01%     2528    501   4957     3624k  1233.6s
    # 450 =>     37922       0     14616 100.00%   385985522.281   385946932.51       0.01%     2199    459   4973     7307k  2382.9s
    # 501 =>     16940       0      6857 100.00%   409742237.0891  409701270.72       0.01%     2405    413   5303     3636k  1288.2s

    # for budget in [5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]:
    for budget in [5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]:
        config["name"] = "EmpireSolver"
        config["budget"] = budget
        config["top_n"] = 4
        config["nearest_n"] = 5
        config["waypoint_capacity"] = 25
        config["solver"]["file_prefix"] = "NewBestYetAgain"
        config["solver"]["file_suffix"] = ""
        config["solver"]["mips_gap"] = "0.00005"
        config["solver"]["time_limit"] = "46800"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
