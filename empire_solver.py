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

from pathlib import Path
from typing import List

import json
import pulp

from print_utils import print_empire_solver_output
from generate_empire_data import (
    NodeType,
    Node,
    Arc,
    generate_empire_data,
    get_reference_data,
    GraphData,
)


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


def create_problem(graph_data, max_cost):
    """Create the problem and add the varaibles and constraints."""
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


def main(max_cost=400, lodging_bonus=1, top_n=2, nearest_n=4, waypoint_capacity=15):
    """
    top_n: count of supply nodes per warehouse by value (index based, zero=1)
    nearest_n: count of nearest warehouses available on waypoint nodes (index based, zero=1)
    waypoint_capacity: max loads on a waypoint
    """
    graph_data: GraphData = generate_empire_data(lodging_bonus, top_n, nearest_n, waypoint_capacity)
    prob = create_problem(graph_data, max_cost)

    prefix = "RemoveSupply"
    suffix = "no_gap"
    mips_tol = 1e-6
    mips_gap = max_cost / 10_000

    if prefix:
        prefix = f"{prefix}_"
    if suffix:
        suffix = f"_{suffix}"

    filename = f"{prefix}mc{max_cost}_lb{lodging_bonus}_tn{top_n}_nn{nearest_n}_wc{waypoint_capacity}{suffix}"
    outfile_path = "/home/thell/milp-flow/highs_output"

    prob.writeLP(f"{outfile_path}/models/{filename}.lp")

    solver = pulp.HiGHS_CMD(
        path="/home/thell/.local/bin/highs",
        keepFiles=True,
        threads=16,
        logPath=f"{outfile_path}/logs/{filename}.log",
        options=[
            "parallel=on",
            "threads=16",
            f"mip_rel_gap={mips_gap}",
            "mip_heuristic_effort=0.5",
            f"mip_feasibility_tolerance={mips_tol}",
            f"primal_feasibility_tolerance={mips_tol}",
            "mip_improving_solution_save=on",
            f"mip_improving_solution_file={outfile_path}/improvements/{filename}.sol",
        ],
    )

    print()
    print(
        f"=== Solving: max_cost: {max_cost}, Lodging_bonus: {lodging_bonus} top_n: {top_n},"
        f" nearest_n: {nearest_n}, capacity={waypoint_capacity}"
        f"\nUsing threads=16 and mip_feasibility_tolerance={mips_tol}, gap: {mips_gap}"
    )
    print()

    solver.solve(prob)

    ref_data = get_reference_data(lodging_bonus, top_n)
    out_data = print_empire_solver_output(
        prob, graph_data, ref_data, max_cost, top_n, detailed=False
    )
    value = round(out_data["solution_vars"]["value"]["value"])
    filepath = f"{outfile_path}/solution-vars/{filename}_{value}.json"
    with open(filepath, "w") as file:
        json.dump(out_data, file, indent=4)

    print("Solution vars saved to:", filepath)
    old_path = f"{outfile_path}/../MaximizeEmpireValue-pulp.sol"
    new_path = f"{outfile_path}/solutions/{filename}.sol"
    Path(old_path).rename(new_path)

    assert (
        prob.variablesDict()["Load_source"].value() == prob.variablesDict()["Load_sink"].value()
    ), "Load value mismatch between source and sink."


if __name__ == "__main__":
    main()
