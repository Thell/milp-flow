"""Flow Model of a Budget-Constrained Prize Collecting Steiner Forest with Cost Based Capacity"""

import json
import logging
import os
from pathlib import Path
import random
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

    # Active node neighborhood constraints.
    for node in G["V"].values():
        if node.type in [NT.S, NT.T]:
            continue

        in_neighbors = [arc.source.vars["x"] for arc in node.inbound_arcs]
        out_neighbors = [arc.destination.vars["x"] for arc in node.outbound_arcs]

        if node.type in [NT.plant, NT.waypoint, NT.group]:
            prob += lpSum(in_neighbors) >= lpSum(out_neighbors)
        if node.isWaypoint:
            prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
        else:
            prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0
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
        f"random_seed={config["solver"]["random_seed"]}",
        "mip_feasibility_tolerance=1e-4",  # 1e-6 from config file
        "primal_feasibility_tolerance=1e-4",  # 1e-6 from config file
        # f"mip_rel_gap={mips_gap}",  # 1e-5 from this main()'s config
        "mip_pool_soft_limit=5000",
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
    filepath = Path(os.path.dirname(__file__), "highs_output")

    new_path = filepath.joinpath("models", f"{filename}.lp")
    prob.writeLP(new_path)
    print("         LP model saved:", new_path)

    new_path = filepath.joinpath("solutions", f"{filename}_pulp.json")
    prob.to_json(new_path)
    print("Solved pulp model saved:", new_path)

    old_path = filepath.parent.parent.joinpath(f"{config["name"]}-pulp.sol")
    new_path = filepath.joinpath("solutions", f"{filename}_highs.sol")
    Path(old_path).rename(new_path)
    print("   HiGHS solution saved:", new_path)

    new_path = filepath.joinpath("solutions", f"{filename}_cfg.json")
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

    # Var6 results
    #   5 =>         1       0         1 100.00%   13900824.11     13900824.11        0.00%      107      9      5       563     1.6s
    #  10 =>         1       0         1 100.00%   24954483.03     24954483.03        0.00%      217     11      4      1091     0.8s
    #  20 =>         1       0         1 100.00%   44336605.295    44334018.51        0.01%       39      6      6      3157     1.5s
    #  30 =>         1       0         1 100.00%   58540725        58540725           0.00%       39     22    305     11595     5.6s
    #  50 =>         5       0         3 100.00%   84340234.96     84340234.96        0.00%       81     44    627     31409    11.8s
    # 100 =>      1286       0       535 100.00%   137562435.03    137562435.03       0.00%      996    194   4867    277984    92.4s
    # 150 =>      9449       0      3799 100.00%   183030044.3538  183012954.78       0.01%     1666    307   4873     1509k   572.9s
    # 200 =>     34960       0     14080 100.00%   222778887.4344  222756620.54       0.01%     2092    449   4937     7365k  2749.1s
    # 250 =>     36662       0     10922 100.00%   260995999.6362  260969949.12       0.01%     2650    496   5154     6371k  2559.2s
    # 300 =>     30356       0     13247 100.00%   297920443.466   297890668.77       0.01%     2353    536   5328     5283k  1901.0s
    # 350 =>     17919       0      8522 100.00%   332070195.4507  332036997.45       0.01%     1925    437   5151     3753k  1281.3s
    # 400 =>     15475       0      5356 100.00%   361375133.5839  361339029          0.01%     2037    353   4947     2932k  1023.1s
    # 450 =>     17917       0      6861 100.00%   385985476.6622  385946932.51       0.01%     2290    529   4941     3904k  1343.9s
    # 501 =>     10849       0      3955 100.00%   409742166.3493  409701270.72       0.01%     2448    406   5017     2109k   767.0s

    # for budget in [5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]:
    # for budget in [375, 395, 415, 435, 455, 475, 495, 515]:
    for budget in [300, 350, 400, 450, 501]:
        config["name"] = "EmpireSolverTest1"
        config["budget"] = budget
        config["top_n"] = 4
        config["nearest_n"] = 5
        config["waypoint_capacity"] = 25
        config["solver"]["file_prefix"] = "Var1"
        config["solver"]["file_suffix"] = ""
        config["solver"]["mips_gap"] = "0.0001"
        config["solver"]["time_limit"] = "46800"
        config["solver"]["random_seed"] = random.randint(0, 2147483647)
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
