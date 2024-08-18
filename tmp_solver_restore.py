"""Flow Model of a Budget-Constrained Prize Collecting Steiner Forest with Cost Based Capacity"""

# Fastest on budget 501
# mc501_lb0_tn4_nn5_wc25_gdefault_NoCostObj
#  L   38092    2616      9098  52.56%   412883925.561   409701270.7536     0.78%     2527    746   9899    12787k  6600.0s
#      62555       6     21868  99.11%   410280372.0681  409701270.7536     0.14%     2244    757  10261    38753k 17633.3s

# OK on budget 50 but definitely not the fastest at about 10 minutes slower than the fastest
# mc50_lb0_tn4_nn5_wc25_gdefault_NoCostObj
#  L   10145     274      3261  91.67%   89095609.95957  84340234.96241     5.64%     1834    471   9941     5181k  2516.0s
#      11002       9      3824  99.40%   86792044.05214  84340234.96241     2.91%     2165    588  10074     6827k  3233.2s

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
    """Create the problem and add the varaibles and constraints."""

    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    # Testing performance based on ordering of vars.

    # First we'll create the indicator vars for nodes.
    for v in G["V"].values():
        𝒙 = LpVariable(f"𝒙_{v.name()}", cat="Binary")
        v.pulp_vars["𝒙"] = 𝒙
    # Edges don't have indicator vars except...
    for arc in G["E"].values():
        if arc.source.type is NT.R:
            𝒙 = LpVariable(f"𝒙_{arc.name()}", cat="Binary")
            arc.pulp_vars["𝒙"] = 𝒙

    # Then we'll create the individual node f vars
    for v in G["V"].values():
        ƒ = LpVariable(f"ƒ_{v.name()}", lowBound=0, upBound=v.𝓬, cat="Integer")
        v.pulp_vars["ƒ"] = ƒ
    # Edges dont have individual f vars except...
    for arc in G["E"].values():
        if arc.source.type is NT.R:
            ƒ = LpVariable(f"ƒ_{arc.name()}", lowBound=0, upBound=arc.𝓬, cat="Integer")
            arc.pulp_vars["ƒ"] = ƒ

    # Next we'll create the specific fr vars for each node
    for v in G["V"].values():
        for 𝓻 in v.𝓻:
            ƒ_key = f"ƒ𝓻_{𝓻.id}"
            ƒ_var = LpVariable(
                f"{ƒ_key}_at_{v.name()}",
                lowBound=0,
                upBound=v.𝓬 if v.type in [NT.𝒕, NT.lodging] else 𝓻.𝓬,
                cat="Binary" if v.isTerminal else "Integer",
            )
            v.pulp_vars[ƒ_key] = ƒ_var
    # and the specific fr vars for each edge
    for arc in G["E"].values():
        for 𝓻 in set(arc.source.𝓻).intersection(set(arc.destination.𝓻)):
            ƒ_key = f"ƒ𝓻_{𝓻.id}"
            ƒ_var = LpVariable(
                f"{ƒ_key}_on_{arc.name()}",
                lowBound=0,
                upBound=arc.𝓬,
                cat="Binary" if arc.source.type is NT.𝓢 else "Integer",
            )
            arc.pulp_vars[ƒ_key] = ƒ_var

    # The only remaining var is the total cost var
    cost = LpVariable("cost", lowBound=0, upBound=config["budget"])

    # Next we'll add the constraints...
    # We'll start with the main 'controlling' constraints...

    # Ensure ∑ƒ⁺𝓢𝒓 == ∑ƒ⁻𝓣𝒓
    prob += G["V"]["𝓢"].pulp_vars["ƒ"] == G["V"]["𝓣"].pulp_vars["ƒ"], "ƒ𝓢_to_ƒ𝓣"

    # Ensure cost <= budget
    prob += cost == lpSum(v.cost * v.pulp_vars["𝒙"] for v in G["V"].values()), "TotalCost"

    # Ensure a single 𝓡 -> lodging -> 𝓣 path for each 𝒓.
    for 𝒓 in G["R"].values():
        prob += lpSum(arc.pulp_vars["𝒙"] for arc in 𝒓.outbound_arcs) <= 1, f"ƒ⁺𝓡_{𝒓.id}_to_ƒ⁻𝓣"

    # Now the node indicator constraints...
    for v in G["V"].values():
        ƒ = v.pulp_vars["ƒ"]
        ƒ_vars = [v for k, v in v.pulp_vars.items() if k.startswith("ƒ𝓻_")]
        𝒙 = v.pulp_vars["𝒙"]

        ϵ = 1e-5
        M = v.𝓬 + ϵ

        prob += ƒ == lpSum(ƒ_vars), f"ƒ_{v.name()}"
        prob += ƒ >= ϵ - M * (1 - 𝒙), f"↥_{v.name()}"
        prob += ƒ <= M * 𝒙, f"↧_{v.name()}"

    # And the few edge indicator constraints...
    for arc in G["E"].values():
        if arc.source.type is not NT.R:
            continue
        ƒ = arc.pulp_vars["ƒ"]
        ƒ_vars = [v for k, v in arc.pulp_vars.items() if k.startswith("ƒ𝓻_")]
        𝒙 = arc.pulp_vars["𝒙"]

        ϵ = 1e-5
        M = arc.𝓬 + ϵ

        prob += ƒ == lpSum(ƒ_vars), f"ƒ_{arc.name()}"
        prob += ƒ >= ϵ - M * (1 - 𝒙), f"↥_{arc.name()}"
        prob += ƒ <= M * 𝒙, f"↧_{arc.name()}"

    # Then we need to constrain the node's specific fr vars to the ƒ⁻ and ƒ⁺ values by linking them.
    def link_node_to_arcs(prob: pulp.LpProblem, v: Node, direction: str, arcs: List[Arc]):
        for ƒ𝓻_key, ƒ𝓻_var in v.pulp_vars.items():
            if ƒ𝓻_key.startswith("ƒ𝓻_"):
                prob += (
                    ƒ𝓻_var
                    == lpSum(
                        var
                        for arc in arcs
                        for key, var in arc.pulp_vars.items()
                        if key.startswith("ƒ𝓻_") and (key == ƒ𝓻_key or v.type is NT.lodging)
                    ),
                    f"{direction}_{ƒ𝓻_key}_at_{v.name()}",
                )

    for v in G["V"].values():
        if v.type is not NT.𝓢:
            link_node_to_arcs(prob, v, "ƒ⁻", v.inbound_arcs)
        if v.type is not NT.𝓣:
            link_node_to_arcs(prob, v, "ƒ⁺", v.outbound_arcs)

    # Lastly... our objective... Maximize prizes.
    prizes = [
        v.𝓻_prizes[𝓻.id]["value"] * v.pulp_vars[f"ƒ𝓻_{𝓻.id}"]
        for v in G["V"].values()
        if v.isTerminal
        for 𝓻 in v.𝓻
    ]
    prob += lpSum(prizes), "ObjectiveFunction"

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
    #   5 =>  L       0       0         0   0.00%   14860480.95834  13900824.10784     6.90%      266     50     54      1284     2.1s
    #  10 =>          0       0         0   0.00%   25796097.74482  24954483.03307     3.37%       50     48      6      8987     4.1s
    #  20 =>  L     519       2       250  50.00%   46101466.08544  44334018.50963     3.99%     1306    214   7005    272299    98.1s
    #  30 =>       3699       1      1790  99.89%   61469477.04108  58540724.99759     5.00%     1470    384   9919     1551k   735.4s
    #  50 =>       9890       1      4779  99.95%   87323477.82678  84340234.96241     3.54%     1978    658   9883     7382k  3444.6s
    # 501 =>      76391    3059     31328  69.72%   412192681.5614  409701270.7039     0.61%     2090    758   9678    43406k 19998.4s
    for budget in [5, 10, 20, 30, 50]:
        config["budget"] = budget
        config["top_n"] = 4
        config["nearest_n"] = 5
        config["waypoint_capacity"] = 25
        config["solver"]["file_prefix"] = "TMPREWORK"
        config["solver"]["file_suffix"] = "Restored"
        config["solver"]["mips_gap"] = "default"
        config["solver"]["time_limit"] = "20000"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
