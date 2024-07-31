"""MILP Worker Node Empire Problem
(See `generate_empire_data.py` for node and arc generation details.)

Goal: Maximize supply value - cost

The model is based on flow where source outbound load == sink inbound load per supply destination.

* All arcs and nodes have a load per supply warehouse equal to the sum of the inbound/source loads.
* All arcs and nodes loads must be <= source supply. This is more robust for result verification
  but requires more time. Using the max warehouse load as the load limit works as well.

Nodes are subject to:
  - inbound == outbound
  - single outbound arc per load
arcs are subject to:
  - arc and reverse arc load mutual exclusivity

WARNING: Solver integer tolerance is extremely important in this problem. Solvers _will_ use
the tolerance to eliminate node costs which are calculated using the binary load indicators.
With HiGHS the default of 1e-6 is not tight enough, using `mip_feasibility_tolerance=1e-9`
results in proper binary assignments but takes much longer to solve.
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

    # Usage of epsilon ensures reduces impact of fractionals; improves HiGHS performance.
    eps = 1e-5
    M = obj.capacity + eps
    prob += load_var >= eps - M * (1 - has_load_var), f"ClampLoadMin_{obj.name()}"
    prob += load_var <= M * has_load_var, f"ClampLoadMax_{obj.name()}"

    # where obj.type==Arc and source.type==waypoint and destination.type==waypoint
    # let
    #   y=LoadForWarehouse_{} of source
    #   x=bLoadForWarehouse_{} binary load indicator on arc
    #   z=LoadForWarehouse_{} of arc
    #   M=LoadForWarehouse_{} capacity (from warehouse)
    # then z=yx is => z<=xM; z>=0; z<=y; z>=y-(1-x)M
    # or
    #   LoadForWarehouse_{w.id}_on_{arc.id} <= bLoadForWarehouse_{w.id}_on_{arc.id} * Warehouse_{w.id}_capacity
    #   LoadForWarehouse_{w.id}_on_{arc.id} <= LoadForWarehouse_{w.id}_at_{arc.source.id}
    #   LoadForWarehouse_{w.id}_on_{arc.id} >= 0
    #   LoadForWarehouse_{w.id}_on_{arc.id} >= LoadForWarehouse_{w.id}_at_{arc.source.id} - (1 - bLoadForWarehouse_{w.id}_on_{arc.id}) * Warehouse_{w.id}_capacity
    #
    # then for any given waypoint node its' LoadForWarehouse is
    #   LoadForWarehouse = Sum(LoadForWarehouse_{w}_on_{arc.id} for arc in Node.inboundArcs)
    # and its' HasLoad binary variable is
    #   HasLoad>=0
    #   HasLoad<=Sum(LoadForWarehouse_{w.id}_at_{node.id} for w in node.LoadForWarehouse)


def create_node_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """has_load and load vars for nodes.
    `has_load`: binary indicator of a load.
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
                if node.type in [NodeType.supply, NodeType.origin, NodeType.lodging]
                else LoadForWarehouse.capacity,
                cat="Integer",
            )
            node.pulp_vars[load_key] = load_var
            warehouse_load_vars.append(load_var)

        create_load_hasload_vars(prob, node, warehouse_load_vars)


def create_arc_vars(prob: pulp.LpProblem, graph_data: GraphData, max_cost: int):
    """has_load and load vars for arcs.
    `has_load` is a binary indicator of a load.
    `LoadForWarehouse_{warehouse}`: integer value of a destination bound load.

    arcs can not transport a load the source doesn't send or a load the destination can't recieve.
    LoadForWarehouse is an intersection of the source and destination LoadForWarehouse lists.
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
                cat="Integer",
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
        is_source = node.type is NodeType.source
        arc_vars = []
        for arc in arcs:
            for key, var in arc.pulp_vars.items():
                if key.startswith("LoadForWarehouse_") and (
                    key == node_var_key or node.type is NodeType.lodging
                ):
                    arc_vars.append(arc.pulp_vars["HasLoad"] if is_source else var)
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
            node.value * node.pulp_vars["HasLoad"]
            for node in graph_data["nodes"].values()
            if node.type is NodeType.supply
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


def add_reverse_arc_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure reverse arcs on waypoint/town nodes don't return loads."""

    # Only waypoint/town arcs have reverse arcs.
    def is_transit_arc(arc):
        return (arc.source.type in [NodeType.waypoint, NodeType.town]) and (
            arc.destination.type in [NodeType.waypoint, NodeType.town]
        )

    seen_arcs = []
    for arc in graph_data["arcs"].values():
        if not is_transit_arc(arc):
            continue

        reverse_arc = graph_data["arcs"][(arc.destination.name(), arc.source.name())]
        if arc.key in seen_arcs or reverse_arc.key in seen_arcs:
            continue
        seen_arcs.append(arc.key)
        seen_arcs.append(reverse_arc.key)

        shared_keys = list(set(arc.pulp_vars.keys()).intersection(reverse_arc.pulp_vars.keys()))
        for var_key in shared_keys:
            if var_key.startswith("LoadForWarehouse"):
                # Forward load indicator.
                b_forward = pulp.LpVariable(f"b{var_key}_on_{arc.name()}", cat="Binary")
                arc.pulp_vars[f"b{var_key}"] = b_forward
                prob += (
                    arc.pulp_vars[var_key] <= arc.destination.capacity * b_forward,
                    f"Linkb{var_key}_on_{arc.name()}",
                )

                # Reverse load indicator.
                b_reverse = pulp.LpVariable(f"b{var_key}_on_{reverse_arc.name()}", cat="Binary")
                reverse_arc.pulp_vars[f"b{var_key}"] = b_reverse
                prob += (
                    reverse_arc.pulp_vars[var_key] <= arc.source.capacity * b_reverse,
                    f"Linkb{var_key}_on_{reverse_arc.name()}",
                )

                # Mutual exclusion.
                prob += (
                    b_forward + b_reverse <= 1,
                    f"MutualExclusion{var_key}_{arc.name()}_{reverse_arc.name()}",
                )


def add_waypoint_single_outbound_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure transit nodes do not split a load to multiple outbound arcs.
    NOTE: do not call this prior to calling add_reverse_arc_constraint() because
    transit arcs have the binary indicators added in add_reverse_arc_constraint().
    The binary indicators are in the arc's pulp_vars keyed with `bLoadForWarehouse_{}`
    """

    def is_transit_arc(arc):
        return (arc.source.type in [NodeType.waypoint, NodeType.town]) and (
            arc.destination.type in [NodeType.waypoint, NodeType.town]
        )

    for node in graph_data["nodes"].values():
        if node.type not in [NodeType.waypoint, NodeType.town]:
            continue

        outbound_indicator_vars = {}
        for arc in node.outbound_arcs:
            if is_transit_arc(arc):
                for key, var in arc.pulp_vars.items():
                    if key.startswith("bLoadForWarehouse_"):
                        if key not in outbound_indicator_vars:
                            outbound_indicator_vars[key] = []
                        outbound_indicator_vars[key].append(var)

        for vars in outbound_indicator_vars.values():
            if len(vars) > 1:
                prob += pulp.lpSum(vars) <= 1


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

    # NOTE: Solutions without these are optimum value within cost constraint but in the transit
    # (waypoint/town) nodes unnecessary loads that don't contribute to the optimal solution are
    # being added because there isn't a constraint to stop inter-waypoint cycles.

    # The below constraints ensure no direct A <-> B cycles and limit outbound routes. They still
    # wouldn't specifically stop A -> B -> C -> A pantom cycles and the only thing that would stop
    # that is linking the capacity to the dynamic source load for each transit node which creates
    # a more stochastic problem that hinders solving efficiently.
    # These cycles only exist on nodes that already have a load incurring a cost, so the only reason
    # to use these is result validation of route cost minimization or to improve performance.

    # Results with default mip_feasibility_tolerance and load/has_load link with epsilon 1e-5.
    # without both: 10 => 4.5s, 30 => 4m48s, 50 =>  8m02s, 100 => 37m03s (phantoms: 50, 100)
    # with reverse: 10 => 7.9s, 30 => 3m04s, 50 => 18m16s, 100 => 74m25s (phantoms: 50)
    #  with single: 10 => 4.4s, 30 => 4m41s, 50 =>  8m04s, 100 => 37m22s (phantoms: 50, 100)
    #    with both: 10 => 5.3s, 30 => 8m38s, 50 =>  9m25s, 100 => 44m55s (no phantoms)
    #
    # Revised mcXXX_lb1_tn2_nn4_wc15_eps (rounding results) Heuristics=.5 Tol=1e-6
    # without both: 10 => 4.6s, 30 => 4m48s, 50 =>  5m07s, 100 => 28m15s, 200 => Term @ 9hr
    #
    # Time constraint max_cost=200 tests:
    #                    Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work
    #                 Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time
    # without both: (completed with fractionals - cost and value correct if round(v["value"]) is used)
    # improvement L   81867    4316     29858  44.58%   230220574.5478  222220439.3912     3.60%     1892   1096   9920    30632k 13970.5s
    #             T  143839    5983     54759  60.74%   228390723.635   222220439.4249     2.78%      948    911   4540    43144k 19885.5s
    # completed =>   187120       8     78506  99.99%   223731390.2573  222220439.4249     0.68%     2206   1104   9948    67924k 29567.6s
    #    with both:
    # improvement =>  26505    1688      7076  56.65%   232322232.2251  222142404.6765     4.58%     4724   1421   9123     9946k  8687.4s
    # termination => 148626   15771     57904  81.81%   229059269.8077  222142404.6765     3.11%     5592   1613   9639    48973k 40236.1s

    # Uncomment the following lines to add constraints aimed at preventing inter-waypoint cycles:
    # add_reverse_arc_constraint(prob, graph_data)
    # add_waypoint_single_outbound_constraint(prob, graph_data)

    return prob


def main(max_cost=300, lodging_bonus=1, top_n=2, nearest_n=4, waypoint_capacity=15):
    """
    top_n: count of supply nodes per warehouse by value (index based, zero=1)
    nearest_n: count of nearest warehouses available on waypoint nodes (index based, zero=1)
    waypoint_capacity: max loads on a waypoint
    """
    graph_data: GraphData = generate_empire_data(lodging_bonus, top_n, nearest_n, waypoint_capacity)
    prob = create_problem(graph_data, max_cost)

    prefix = "RoundingResults"
    suffix = "eps"
    filename = f"{prefix}_mc{max_cost}_lb{lodging_bonus}_tn{top_n}_nn{nearest_n}_wc{waypoint_capacity}_{suffix}"
    outfile_path = "/home/thell/milp-flow/highs_output"

    prob.writeLP(f"{outfile_path}/models/{filename}.lp")

    mips_tol = 1e-6
    mips_gap = 0.025
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
        f"\nUsing threads=16 and mip_feasibility_tolerance={mips_tol}"
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
