"""MILP Worker Node Empire Problem
(See `generate_empire_data.py` for node and arc generation details.)

Goal: Maximize supply value - cost

The model is based on flow where source outbound load == sink inbound load per supply destination.

* All arcs and nodes have a load per supply warehouse equal to the sum of the inbound/source loads.
* All arcs and nodes loads must be <= source supply. This is more robust for result verification
  but requires more time. Using the max warehouse load as the load limit works as well.
* All nodes are subject to inbound == outbound.

WARNING: Solver integer tolerance is extremely important in this problem. Solvers _will_ use
the tolerance to eliminate node costs which are calculated using the binary load indicators.
With HiGHS the default of 1e-6 is not tight enough, using `mip_feasibility_tolerance=1e-9`
results in proper binary assignments but takes much longer to solve. Using a small epsilon
and rounding the results improves performance while still obtaining proper results.
"""

import json
import logging
from operator import itemgetter
from pathlib import Path
from typing import List


import natsort
import pulp

from generate_empire_data import (
    NodeType,
    Node,
    Arc,
    generate_empire_data,
    GraphData,
)

logger = logging.getLogger(__name__)


def create_load_hasload_vars(prob: pulp.LpProblem, obj: Node | Arc, load_vars: List[Node]):
    assert load_vars, "Each node must have at least one warehouse load variable."

    load_var = pulp.LpVariable(f"Load_{obj.name()}", lowBound=0, upBound=obj.capacity, cat="Integer")
    obj.pulp_vars["Load"] = load_var
    prob += load_var == pulp.lpSum(load_vars), f"TotalLoad_{obj.name()}"

    has_load_var = pulp.LpVariable(f"HasLoad_{obj.name()}", cat="Binary")
    obj.pulp_vars["HasLoad"] = has_load_var

    # Usage of epsilon reduces impact of fractionals; improves HiGHS performance.
    eps = 1e-5
    M = obj.capacity + eps
    prob += load_var >= eps - M * (1 - has_load_var), f"ClampLoadMin_{obj.name()}"
    prob += load_var <= M * has_load_var, f"ClampLoadMax_{obj.name()}"


def create_node_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """Create and add solver vars for nodes.
    `HasLoad`: binary indicator of a load.
    `LoadForWarehouse_{warehouse}`: integer value of a destination bound load.
    """
    for node in graph_data["nodes"].values():
        warehouse_load_vars = []
        for LoadForWarehouse in node.LoadForWarehouse:
            load_key = f"LoadForWarehouse_{LoadForWarehouse.id}"
            load_var = pulp.LpVariable(
                f"{load_key}_at_{node.name()}",
                lowBound=0,
                upBound=node.capacity
                if node.type in [NodeType.origin, NodeType.lodging]
                else LoadForWarehouse.capacity,
                cat="Binary" if node.type is NodeType.origin else "Integer",
            )
            node.pulp_vars[load_key] = load_var
            warehouse_load_vars.append(load_var)

        create_load_hasload_vars(prob, node, warehouse_load_vars)


def create_arc_vars(prob: pulp.LpProblem, graph_data: GraphData, max_cost: int):
    """Create and add solver vars for arcs.
    `HasLoad` is a binary indicator of a load.
    `LoadForWarehouse_{warehouse}`: integer value of a destination bound load.

    LoadForWarehouse is an intersection of the source and destination LoadForWarehouse lists.
    Arcs can not transport a load the source doesn't send or a load the destination can't recieve.
    """

    for arc in graph_data["arcs"].values():
        warehouse_load_vars = []
        for LoadForWarehouse in set(arc.source.LoadForWarehouse).intersection(
            set(arc.destination.LoadForWarehouse)
        ):
            load_key = f"LoadForWarehouse_{LoadForWarehouse.id}"
            load_var = pulp.LpVariable(
                f"{load_key}_on_{arc.name()}",
                lowBound=0,
                upBound=arc.capacity,
                cat="Binary" if arc.source.type is NodeType.source else "Integer",
            )
            arc.pulp_vars[load_key] = load_var
            warehouse_load_vars.append(load_var)

        create_load_hasload_vars(prob, arc, warehouse_load_vars)


def link_node_and_arc_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure load per warehouse flow conservation for all nodes.
    Link the inbound arc load sum to the node load to the outbound arc load sum.
    Special handling for source and sink nodes.
    """

    def get_arc_vars(node: Node, node_var_key: str, arcs: List[Arc]) -> List[pulp.LpVariable]:
        """Get the relevant arc variables for a given node and key."""
        arc_vars = []
        for arc in arcs:
            for key, var in arc.pulp_vars.items():
                if key.startswith("LoadForWarehouse_") and (
                    key == node_var_key or node.type is NodeType.lodging
                ):
                    arc_vars.append(var)
        return arc_vars

    def link_node_to_arcs(prob: pulp.LpProblem, node: Node, direction: str, arcs: List[Arc]):
        """Link a node's load variable to the sum of its arc variables."""
        for key, var in node.pulp_vars.items():
            if key.startswith("LoadForWarehouse_"):
                arc_vars = get_arc_vars(node, key, arcs)
                prob += var == pulp.lpSum(arc_vars), f"Link{direction}_{key}_at_{node.name()}"

    for node in graph_data["nodes"].values():
        if node.type is not NodeType.source:
            link_node_to_arcs(prob, node, "Inbound", node.inbound_arcs)
        if node.type is not NodeType.sink:
            link_node_to_arcs(prob, node, "Outbound", node.outbound_arcs)


def create_value_var(prob: pulp.LpProblem, graph_data: GraphData):
    """Sum variable of value at supply nodes with load."""
    value_var = pulp.LpVariable("value", lowBound=0)
    prob += (
        value_var
        == pulp.lpSum(
            node.warehouse_values[warehouse.id]["value"]
            * node.pulp_vars[f"LoadForWarehouse_{warehouse.id}"]
            for node in graph_data["nodes"].values()
            for warehouse in node.LoadForWarehouse
            if node.type is NodeType.origin
        ),
        "TotalValue",
    )
    return value_var


def create_cost_var(prob: pulp.LpProblem, graph_data: GraphData, max_cost: int):
    """Sum variable of cost at nodes with load."""
    cost_var = pulp.LpVariable("cost", lowBound=0, upBound=max_cost)
    prob += (
        cost_var
        == pulp.lpSum(
            node.cost * node.pulp_vars["HasLoad"]
            for node in graph_data["nodes"].values()
            if node.type is not NodeType.source and node.type is not NodeType.sink
        ),
        "TotalCost",
    )
    return cost_var


def add_source_to_warehouse_constraints(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure each warehouse receives its total origin supply."""
    for warehouse in graph_data["warehouse_nodes"].values():
        source_load = graph_data["nodes"]["source"].pulp_vars[f"LoadForWarehouse_{warehouse.id}"]
        prob += (
            warehouse.pulp_vars["Load"] == source_load,
            f"BalanceSupplyLoadForWarehouse_{warehouse.id}",
        )


def add_warehouse_to_lodging_constraints(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure a single warehouse -> lodging -> sink path for each warehouse."""
    for warehouse in graph_data["warehouse_nodes"].values():
        prob += pulp.lpSum(arc.pulp_vars["HasLoad"] for arc in warehouse.outbound_arcs) <= 1


def add_source_to_sink_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure source outbound load equals sink inbound load."""
    prob += (
        graph_data["nodes"]["source"].pulp_vars["Load"]
        == graph_data["nodes"]["sink"].pulp_vars["Load"],
        "BalanceSourceSink",
    )


def create_problem(config, graph_data):
    """Create the problem and add the varaibles and constraints."""
    max_cost = config["max_cost"]

    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    create_node_vars(prob, graph_data)
    create_arc_vars(prob, graph_data, max_cost)
    link_node_and_arc_vars(prob, graph_data)

    total_value_var = create_value_var(prob, graph_data)
    total_cost_var = create_cost_var(prob, graph_data, max_cost)

    prob += total_value_var - total_cost_var, "ObjectiveFunction"
    prob += total_cost_var <= max_cost, "MaxCost"

    add_source_to_warehouse_constraints(prob, graph_data)
    add_warehouse_to_lodging_constraints(prob, graph_data)
    add_source_to_sink_constraint(prob, graph_data)

    return prob


def config_solver(config, outfile_path, filename):
    mips_tol = config["solver"]["mips_tol"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = config["max_cost"] / 10_000
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
    ]
    options = [o for o in options if "default" not in o]

    solver = pulp.HiGHS_CMD(
        path="/home/thell/.local/bin/highs",
        keepFiles=True,
        options=options,
        logPath=f"{outfile_path}/logs/{filename}.log",
    )
    return solver


def config_filename(config):
    max_cost = config["max_cost"]
    lodging_bonus = config["lodging_bonus"]
    top_n = config["top_n"]
    nearest_n = config["nearest_n"]
    waypoint_capacity = config["waypoint_capacity"]
    prefix = config["solver"]["file_prefix"]
    suffix = config["solver"]["file_suffix"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = max_cost / 10_000
    mips_gap = str(mips_gap).replace("0.", "")
    if prefix:
        prefix = f"{prefix}_"
    if suffix:
        suffix = f"_{suffix}"

    filename = f"{prefix}mc{max_cost}_lb{lodging_bonus}_tn{top_n}_nn{nearest_n}_wc{waypoint_capacity}_g{mips_gap}{suffix}"
    return filename


def print_solving_info_header(config):
    max_cost = config["max_cost"]
    lodging_bonus = config["lodging_bonus"]
    top_n = config["top_n"]
    nearest_n = config["nearest_n"]
    waypoint_capacity = config["waypoint_capacity"]
    mips_tol = config["solver"]["mips_tol"]
    mips_gap = config["solver"]["mips_gap"]
    if mips_gap == "auto":
        mips_gap = max_cost / 10_000

    logging.info(
        f"\n===================================================="
        f"\nSolving: max_cost: {max_cost}, Lodging_bonus: {lodging_bonus} top_n: {top_n},"
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
        if k.startswith("Load_"):
            kname = k.replace("Load_", "")
            if "_to_" in k:
                # An arc
                source, destination = kname.split("_to_")
                arc_key = (source, destination)
                arc = graph_data["arcs"].get(arc_key, None)
                outputs.append(f"{arc}, {v}")
                if arc.source.type is NodeType.waypoint or arc.destination.type is NodeType.waypoint:
                    arc_loads.append((arc, v["value"]))
            else:
                # A node
                node = graph_data["nodes"].get(kname, None)
                outputs.append(f"{node}, {v}")
                calculated_cost = calculated_cost + node.cost
                if node.type is NodeType.waypoint:
                    waypoint_loads.append((node, v["value"]))
    outputs = natsort.natsorted(outputs)

    solver_cost = 0
    if "cost" in solution_vars.keys():
        solver_cost = int(solution_vars["cost"]["value"])
    solver_value = round(solution_vars["value"]["value"])

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        logging.debug("Detailed Solution:")
        for output in outputs:
            logging.debug(output)

    logging.info(
        f"\n"
        f"         Load_source: {solution_vars["Load_source"]["value"]}\n"
        f"           Load_sink: {solution_vars["Load_sink"]["value"]}\n"
        f"     Calculated cost: {calculated_cost}\n"
        f"         Solver cost: {calculated_cost}\n"
        f"            Max cost: {config["max_cost"]}\n"
        f"               Value: {round(solver_value)}\n"
        f"          Value/Cost: {round(solver_value / max(1, solver_cost, calculated_cost))}\n"
        f"    Num origin nodes: {len([x for x in outputs if x.startswith("Node(name: origin_")])}\n"
        f"   Max waypoint load: {max(waypoint_loads, key=itemgetter(1))}\n"
        f"       Max arc load: {max(arc_loads, key=itemgetter(1))}\n"
    )
    if gt0lt1_vars:
        logging.warning(f"WARNING: 0 < x < 1 vars count: {len(gt0lt1_vars)}")

    data = {"config": config, "solution_vars": solution_vars}
    return data


def empire_solver(config):
    graph_data: GraphData = generate_empire_data(config)
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
    value = round(out_data["solution_vars"]["value"]["value"])
    filepath = f"{outfile_path}/solution-vars/{filename}_{value}.json"
    with open(filepath, "w") as file:
        json.dump(out_data, file, indent=4)
    print("Solution vars written to:", filepath)

    assert (
        prob.variablesDict()["Load_source"].value() == prob.variablesDict()["Load_sink"].value()
    ), "Load value mismatch between source and sink."


def main(config):
    """
    top_n: count of supply nodes per warehouse by value (index based, zero=1)
    nearest_n: count of nearest warehouses available on waypoint nodes (index based, zero=1)
    waypoint_capacity: max loads on a waypoint
    """
    # for max_cost in [10, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]:
    for max_cost in [200, 250, 300, 350, 400, 450, 501]:
        config["max_cost"] = max_cost
        config["solver"]["file_prefix"] = "Full"
        config["solver"]["mips_gap"] = "default"
        config["solver"]["time_limit"] = "default"
        empire_solver(config)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
