"""Flow Model of a Budget-Constrained Prize Collecting Steiner Forest with Cost Based Capacity"""

import json
import logging
from pathlib import Path
from typing import List

import natsort
from pulp import LpVariable, lpSum, LpProblem
import pulp

from tmp_gen_data import (
    NodeType as NT,
    Node,
    Arc,
    generate_empire_data,
    GraphData,
)

logger = logging.getLogger(__name__)


def get_fr_vars(v, arcs, var_key):
    return [
        var
        for a in arcs
        for k, var in a.vars.items()
        if k.startswith("fzone_") and (k == var_key or v.isLodging)
    ]


def link_in_out_flows(prob: LpProblem, v: Node, in_arcs: List[Arc], out_arcs: List[Arc]):
    f_vars = []
    for zone in v.zones:
        key = f"fzone_{zone.id}"
        inflows = get_fr_vars(v, in_arcs, key)
        f_vars.append(inflows)
        outflows = get_fr_vars(v, out_arcs, key)
        prob += lpSum(inflows) == lpSum(outflows), f"FlowBalanceZone_{key}_at_{v.name()}"

    f = v.vars["f"]
    prob += f == lpSum(f_vars), f"f_{v.name()}"
    prob += f <= v.ub * v.vars["x"], f"↧_{v.name()}"


def create_problem(config, G):
    """Create the problem and add the variables and constraints."""

    prob = LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    # Variables
    # Cost var ∈ ℕ₀ 0 <= cost <= budget
    cost = LpVariable("cost", 0, config["budget"], "Integer")

    # Node vars.
    for v in G["V"].values():
        # x ∈ {0,1} for each node indicating if node is in solution and cost calculation.
        v.vars["x"] = LpVariable(f"x_{v.name()}", cat="Binary")
        # f ∈ ℕ₀ for each node such that 0 <= f <= ub for cost calculation and performance.
        v.vars["f"] = LpVariable(f"f_{v.name()}", 0, v.ub, "Integer")

    # Edge vars.
    for a in G["E"].values():
        # 𝓡 group specific f ∈ ℕ₀ vars for each arc 0 <= f <= ub
        for zone in set(a.source.zones).intersection(set(a.destination.zones)):
            key = f"fzone_{zone.id}"
            ub = a.ub if a.source.type in [NT.𝓡, NT.𝓢, NT.𝓣, NT.lodge] else zone.ub
            cat = "Binary" if a.source.type in [NT.𝓢, NT.plant] else "Integer"
            a.vars[key] = LpVariable(f"{key}_on_{a.name()}", 0, ub, cat)

    # Objective
    # Maximize total value ∑v(v)x for 𝓡 group specific values in all binary plant inflows.
    values = []
    for v in G["V"].values():
        if v.isPlant:
            for zone in v.zones:
                for a in v.inbound_arcs:
                    values.append(
                        round(v.zone_values[zone.id]["value"], 2) * a.vars[f"fzone_{zone.id}"]
                    )
    prob += lpSum(values), "ObjectiveFunction"

    # Constraints

    # Cost var is defined with ub = budget so this is ∑v(𝑐)x <= budget
    prob += cost == lpSum(v.cost * v.vars["x"] for v in G["V"].values()), "TotalCost"

    # A single lodge within each 𝓡 group.
    for 𝒓 in G["R"].values():
        vars = []
        for v in G["V"].values():
            if v.name().startswith(f"lodge_{r.id}_"):
                vars.append(v.vars["x"])
        prob += lpSum(vars) <= 1, f"Lodge_{𝒓.id}"

    # 𝓡 group specific f⁻ == f⁺
    for v in G["V"].values():
        if v.type not in [NT.𝓢, NT.𝓣]:
            link_in_out_flows(prob, v, v.inbound_arcs, v.outbound_arcs)

    # 𝓡 group specific f⁻𝓣 == f⁺𝓢
    link_in_out_flows(prob, G["V"]["𝓣"], G["V"]["𝓣"].inbound_arcs, G["V"]["𝓢"].outbound_arcs)

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
    waypoint_ub = config["waypoint_ub"]
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

    filename = f"{prefix}mc{budget}_lb{lodging_bonus}_tn{top_n}_nn{nearest_n}_wc{waypoint_ub}_g{mips_gap}{suffix}"
    return filename


def print_solving_info_header(config):
    budget = config["budget"]
    lodging_bonus = config["lodging_bonus"]
    top_n = config["top_n"]
    nearest_n = config["nearest_n"]
    waypoint_ub = config["waypoint_ub"]
    mips_tol = config["solver"]["mips_tol"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = budget / 10_000

    logging.info(
        f"\n===================================================="
        f"\nSolving: budget: {budget}, Lodging_bonus: {lodging_bonus} top_n: {top_n},"
        f" nearest_n: {nearest_n}, ub={waypoint_ub}"
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
    calculated_value = 0
    outputs = []
    arc_loads = []
    waypoint_loads = []
    for k, v in solution_vars.items():
        if k.startswith("fzone") and "_on_plant_" in k:
            tokens = k.split("_")
            zone, plant = (tokens[1], tokens[4])
            calculated_value += round(graph_data["V"][f"plant_{plant}"].zone_values[zone]["value"])
        if k.startswith("f_"):
            kname = k.replace("f_", "")
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
        f"            # plants: {len([x for x in outputs if x.startswith("Node(name: plant_")])}\n"
        f"              budget: {config["budget"]}\n"
        f"             LP cost: {solver_cost}\n"
        f"         Actual cost: {calculated_cost}\n"
        f"            LP value: {round(solver_value)}\n"
        f"        Actual value: {calculated_value}\n"
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


def main(config):
    """
    top_n: count of supply nodes per root by value (index based, zero=1)
    nearest_n: count of nearest roots available on waypoint nodes (index based, zero=1)
    waypoint_ub: max loads on a waypoint
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
        config["waypoint_ub"] = 25
        config["solver"]["file_prefix"] = "FlowReworked"
        config["solver"]["file_suffix"] = "t25k"
        config["solver"]["mips_gap"] = "default"
        config["solver"]["time_limit"] = "25000"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
