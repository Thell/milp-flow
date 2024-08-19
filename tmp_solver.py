"""Flow Model of a Budget-Constrained Prize Collecting Steiner Forest with Cost Based Capacity"""

import json
import logging
from operator import itemgetter
from pathlib import Path
from typing import List


import natsort
from pulp import LpVariable, lpSum
import pulp

from tmp_gen_data import (
    NodeType as NT,
    Node,
    Arc,
    generate_empire_data,
    GraphData,
)

logger = logging.getLogger(__name__)


def create_problem(config, G):
    """Create the problem and add the variables and constraints."""

    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    # Create 𝒙 ∈ {0,1} for each node indicating if node is in solution.
    for v in G["V"].values():
        v.pulp_vars["𝒙"] = LpVariable(f"𝒙_{v.name()}", cat="Binary")

    # Create 𝒙 ∈ {0,1} for each 𝓻 indicating if arc(𝓻, lodging) is used in solution.
    for a in G["E"].values():
        if a.source.type is NT.R:
            a.pulp_vars["𝒙"] = LpVariable(f"𝒙_{a.name()}", cat="Binary")

    # Create ƒ ∈ ℕ₀ for each node such that 0 <= ƒ <= 𝓬
    for v in G["V"].values():
        v.pulp_vars["ƒ"] = LpVariable(f"ƒ_{v.name()}", lowBound=0, upBound=v.𝓬, cat="Integer")

    # Create ƒ ∈ ℕ₀ for each ƒ⁺ arc(𝓻, lodging) such that 0 <= ƒ <= 𝓬
    for a in G["E"].values():
        if a.source.type is NT.R:
            a.pulp_vars["ƒ"] = LpVariable(f"ƒ_{a.name()}", lowBound=0, upBound=a.𝓬, cat="Integer")

    # Create 𝓻 specific ƒ ∈ ℕ₀ vars for each node 0 <= ƒ <= 𝓬
    for v in G["V"].values():
        for 𝓻 in v.𝓻:
            v.pulp_vars[f"ƒ𝓻_{𝓻.id}"] = LpVariable(
                f"ƒ𝓻_{𝓻.id}_at_{v.name()}",
                lowBound=0,
                upBound=v.𝓬 if v.type in [NT.𝒕, NT.lodging] else 𝓻.𝓬,
                cat="Binary" if v.isTerminal else "Integer",
            )

    # Create 𝓻 specific ƒ ∈ ℕ₀ vars for each arc 0 <= ƒ <= 𝓬
    for a in G["E"].values():
        for 𝓻 in set(a.source.𝓻).intersection(set(a.destination.𝓻)):
            a.pulp_vars[f"ƒ𝓻_{𝓻.id}"] = LpVariable(
                f"ƒ𝓻_{𝓻.id}_on_{a.name()}",
                lowBound=0,
                upBound=a.𝓬,  # should this be r.c instead? It'd be tighter.
                cat="Binary" if a.source.type is NT.𝓢 else "Integer",
            )

    # Create cost ∈ ℕ₀ 0 <= cost <= budget
    cost = LpVariable("cost", lowBound=0, upBound=config["budget"], cat="Integer")

    # Objective... Maximize total prizes.
    prizes = [
        v.𝓻_prizes[𝓻.id]["value"] * v.pulp_vars[f"ƒ𝓻_{𝓻.id}"]
        for v in G["V"].values()
        if v.isTerminal
        for 𝓻 in v.𝓻
    ]
    prob += lpSum(prizes), "ObjectiveFunction"

    # Constraints

    # ∑ƒ⁺𝓢𝒓 == ∑ƒ⁻𝓣𝒓
    prob += G["V"]["𝓢"].pulp_vars["ƒ"] == G["V"]["𝓣"].pulp_vars["ƒ"], "ƒ𝓢_to_ƒ𝓣"

    # cost == ∑v(𝑐), cost var is defined with ub = budget
    prob += cost == lpSum(v.cost * v.pulp_vars["𝒙"] for v in G["V"].values()), "TotalCost"

    # A single arc(𝒓, lodging) within each 𝓡 group.
    for 𝒓 in G["R"].values():
        prob += lpSum(a.pulp_vars["𝒙"] for a in 𝒓.outbound_arcs) <= 1, f"ƒ⁺𝓡_{𝒓.id}_to_ƒ⁻𝓣"

    # 𝒙 == 1 if ƒ >= 1 else 0 for all nodes
    for v in G["V"].values():
        ƒ = v.pulp_vars["ƒ"]
        𝒙 = v.pulp_vars["𝒙"]

        ƒ_vars = [v for k, v in v.pulp_vars.items() if k.startswith("ƒ𝓻_")]
        ϵ = 1e-5
        M = v.𝓬 + ϵ

        prob += ƒ == lpSum(ƒ_vars), f"ƒ_{v.name()}"
        prob += ƒ >= ϵ - v.𝓬 * (1 - 𝒙), f"↥_{v.name()}"
        prob += ƒ <= v.𝓬 * 𝒙, f"↧_{v.name()}"

    # 𝒙 == 1 if ƒ >= 1 else 0 for all arc(𝒓, lodging) arcs within each 𝓡 group
    for a in G["E"].values():
        if a.source.type is NT.R:
            ƒ = a.pulp_vars["ƒ"]
            𝒙 = a.pulp_vars["𝒙"]

            ƒ_vars = [v for k, v in a.pulp_vars.items() if k.startswith("ƒ𝓻_")]
            ϵ = 1e-5
            M = a.𝓬

            prob += ƒ == lpSum(ƒ_vars), f"ƒ_{a.name()}"
            prob += ƒ >= ϵ - M * (1 - 𝒙), f"↥_{a.name()}"
            prob += ƒ <= M * 𝒙, f"↧_{a.name()}"

    # ƒ𝓻⁻ == 𝒗(ƒ𝓻) == ƒ𝓻⁺
    def link_node_to_arcs(prob: pulp.LpProblem, v: Node, direction: str, arcs: List[Arc]):
        for ƒ𝓻_key, ƒ𝓻_var in v.pulp_vars.items():
            if ƒ𝓻_key.startswith("ƒ𝓻_"):
                prob += (
                    ƒ𝓻_var
                    == lpSum(
                        var
                        for a in arcs
                        for key, var in a.pulp_vars.items()
                        if key.startswith("ƒ𝓻_") and (key == ƒ𝓻_key or v.type is NT.lodging)
                    ),
                    f"{direction}_{ƒ𝓻_key}_at_{v.name()}",
                )

    for v in G["V"].values():
        if v.type is not NT.𝓢:
            link_node_to_arcs(prob, v, "ƒ⁻", v.inbound_arcs)
        if v.type is not NT.𝓣:
            link_node_to_arcs(prob, v, "ƒ⁺", v.outbound_arcs)

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
        f"                 ƒ_𝓢: {solution_vars["ƒ_𝓢"]["value"]}\n"
        f"                 ƒ_𝓣: {solution_vars["ƒ_𝓣"]["value"]}\n"
        f"                 |𝒕|: {len([x for x in outputs if x.startswith("Node(name: t_")])}\n"
        f"              budget: {config["budget"]}\n"
        f"           Actual ∑𝑐: {calculated_cost}\n"
        f"               LP ∑𝑐: {solver_cost}\n"
        f"              ∑prize: {round(solver_value)}\n"
        f"      Max waypoint ƒ: {max(waypoint_loads, key=itemgetter(1))}\n"
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

    assert (
        prob.variablesDict()["ƒ_𝓢"].value() == prob.variablesDict()["ƒ_𝓣"].value()
    ), "Load value mismatch: 𝓢 != 𝓣."


def main(config):
    """
    top_n: count of supply nodes per root by value (index based, zero=1)
    nearest_n: count of nearest roots available on waypoint nodes (index based, zero=1)
    waypoint_capacity: max loads on a waypoint
    """

    # for budget in [10, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]:
    #  5 =>          0       0         0   0.00%   15634089.23775  13900824.10784    12.47%        4      4      6      1452     1.4s
    # 10 =>          0       0         0   0.00%   25104041.22931  24954483.03307     0.60%       22      0      0      5054     2.4s
    # 20 =>  B      15       0         7  98.44%   45427294.96655  44334018.50963     2.47%      761    242   1771    118969    38.2s
    # 30 =>       5145       5      2522 100.00%   61602751.05139  58540724.99758     5.23%     1842    417   9993     2542k  1270.5s
    # 50 =>       9894       4      4807  99.95%   86395249.99527  84340234.96242     2.44%     2442    713  10038     5310k  2912.3s

    for budget in [501]:
        config["budget"] = budget
        config["top_n"] = 4
        config["nearest_n"] = 5
        config["waypoint_capacity"] = 25
        config["solver"]["file_prefix"] = "TMPREWORK"
        config["solver"]["file_suffix"] = "NoCostObj"
        config["solver"]["mips_gap"] = "default"
        config["solver"]["time_limit"] = "20000"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
