"""Flow Model of a Budget-Constrained Prize Collecting Steiner Forest with Cost Based Capacity"""

import json
import logging
import os
from pathlib import Path
import random
from typing import List

from pulp import LpVariable, lpSum, LpProblem, LpMaximize, HiGHS_CMD
from empire_solver_par import HiGHS_CMD_PAR

from generate_empire_data import (
    NodeType as NT,
    Node,
    Arc,
    generate_empire_data,
    GraphData,
)

logger = logging.getLogger(__name__)


def filter_arcs(v, groupflow, arcs):
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
        inflows = filter_arcs(v, groupflow_key, in_arcs)
        outflows = filter_arcs(v, groupflow_key, out_arcs)
        prob += lpSum(inflows) == lpSum(outflows), f"balance_{groupflow_key}_at_{v.name()}"
        all_inflows.append(inflows)
    prob += f == lpSum(all_inflows), f"flow_{v.name()}"
    prob += f <= v.ub * v.vars["x"], f"x_{v.name()}"


def create_problem(config, G):
    """Create the problem and add the variables and constraints."""

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

        # All nodes should be 2-degree.
        if node.isWaypoint:
            # Waypoint in/out neighbors are the same.
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

    options = [
        f"time_limit={config["solver"]["time_limit"]}",
        f"random_seed={config["solver"]["random_seed"]}",
        f"mip_rel_gap={config["solver"]["mips_gap"]}",
        f"mip_feasibility_tolerance={config["solver"]["mips_tol"]}",
        f"primal_feasibility_tolerance={config["solver"]["mips_tol"]}",
        "mip_pool_soft_limit=5000",
    ]
    options = [o for o in options if "default" not in o]

    if config["solver"]["num_processes"] == 1:
        solver = HiGHS_CMD(
            path="/home/thell/.local/bin/HiGHS",
            msg=True,
            keepFiles=True,
            options=options,
            logPath=filepath.joinpath("logs", f"{filename}.log"),
        )
    else:
        solver = HiGHS_CMD_PAR(
            path="/home/thell/.local/bin/HiGHS",
            msg=False,
            keepFiles=True,
            options=options,
            num_processes=config["solver"]["num_processes"],
        )

    return solver


def print_solving_info_header(config, G: GraphData):
    budget = config["budget"]
    lodging_bonus = config["lodging_bonus"]
    top_n = config["top_n"]
    nearest_n = config["nearest_n"]
    waypoint_capacity = config["waypoint_capacity"]
    num_proc = config["solver"]["num_processes"]
    mips_tol = config["solver"]["mips_tol"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = budget / 10_000

    logging.info(
        f"\nSolving:  graph with {len(G['V'])} nodes and {len(G['E'])} arcs"
        f"\n  Using:  budget: {budget}, lodging_bonus: {lodging_bonus}, top_n: {top_n},"
        f" nearest_n: {nearest_n}, capacity: {waypoint_capacity}"
        f"\n   With:  feasibility_tolerance: {mips_tol}, gap: {mips_gap}, num_processes: {num_proc}"
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


def save_reproduction_data(log_file, config, prob, G):
    obj_value = round(prob.objective.value())
    filename = f"{config_filename(config)}_{obj_value}"
    filepath = Path(os.path.dirname(__file__), "highs_output")

    old_path = filepath.parent.joinpath(log_file)
    new_path = filepath.joinpath("logs", f"{filename}.log")
    Path(old_path).rename(new_path)
    print("         log file saved:", new_path)

    new_path = filepath.joinpath("models", f"{filename}.lp")
    prob.writeLP(new_path)
    print("         LP model saved:", new_path)

    new_path = filepath.joinpath("solutions", f"{filename}_pulp.json")
    prob.to_json(new_path)
    print("Solved pulp model saved:", new_path)

    old_path = filepath.parent.joinpath(f"{config["name"]}-pulp.sol")
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
    if config["solver"]["num_processes"] == 1:
        solver.solve(prob)
        log_file = solver.optionsDict["logPath"]
    else:
        _, log_file = prob.solve(solver)
    print_solving_info_trailer(config, prob, G)

    save_reproduction_data(log_file, config, prob, G)


def main(config):
    """
    top_n: count of supply nodes per root by value (index based, zero=1)
    nearest_n: count of nearest roots available on waypoint nodes (index based, zero=1)
    waypoint_capacity: max loads on a waypoint
    """

    # Final results
    #   5 =>         1       0         1 100.00%   13900824.11     13900824.11        0.00%       39      6      3       522     1.6s
    #  10 =>         1       0         1 100.00%   24954483.03     24954483.03        0.00%      210      9      0      1037     0.9s
    #  20 =>         1       0         1 100.00%   44334018.51     44334018.51        0.00%        3      0      0      2741     1.5s
    #  30 =>         1       0         1 100.00%   58540725        58540725           0.00%      121     13     72      9847     4.3s
    #  50 =>         6       0         3 100.00%   84340234.96     84340234.96        0.00%      103     54    463     37903    16.2s
    # 100 =>      1054       0       476 100.00%   137569210.5721  137562435.03       0.00%     1099    167   4681    238685    97.8s (72)
    # 150 =>      6204       0      2988 100.00%   183030694.898   183012954.78       0.01%     2028    338   5231     1253k   552.5s (541)
    # 200 =>     22618       0      8945 100.00%   222778742.6519  222756620.54       0.01%     2036    411   4909     4677k  1954.2s (1957)
    # 250 =>     24453       0      9140 100.00%   260996027.1117  260969949.12       0.01%     2699    405   4866     4228k  1922.6s (2365)
    # 300 =>     23752       0      5666 100.00%   297920443.5496  297890668.77       0.01%     2198    400   4952     3979k  1762.8s (2230)
    # 350 =>     21409       0      7579 100.00%   332061172.113   332028018.72       0.01%     2339    387   4930     3415k  1452.3s (1431)
    # 400 =>     11294       0      3753 100.00%   361374766.5588  361339029          0.01%     2111    400   4987     1931k   895.6s (1127)
    # 450 =>     11857       0      3466 100.00%   385949540.2905  385910949.29       0.01%     2259    299   4983     2152k   962.8s (1220)
    # 501 =>      7639       0      3226 100.00%   409742164.9647  409701270.72       0.01%     2089    456   5134     1820k   804.5s (836)

    test_set = [5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]
    bench_set = [375, 395, 415, 435, 455, 475, 495, 515]
    for budget in test_set + bench_set:
        config["name"] = "Empire"
        config["budget"] = budget
        config["lodging_bonus"] = 5
        config["top_n"] = 6
        config["nearest_n"] = 7
        config["waypoint_capacity"] = 35
        config["solver"]["num_processes"] = 6
        config["solver"]["file_prefix"] = ""
        config["solver"]["file_suffix"] = ""
        config["solver"]["mips_gap"] = "default"
        config["solver"]["mips_tol"] = 1e-4
        config["solver"]["time_limit"] = 46800
        config["solver"]["random_seed"] = random.randint(0, 2147483647)
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    for path in ["logs", "models", "solutions"]:
        os.makedirs(f"highs_output/{path}", exist_ok=True)
    main(config)
