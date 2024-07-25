"""MILP Worker Node Empire Problem
(See `generate_empire_data.py` for node and edge generation details.)

Goal: Maximize supply value - cost

The model is based on flow where source outbound load == sink inbound load per supply destination.

* All edges and nodes have a load which is calculated per supply warehouse.
* All edges and nodes have a capacity which the calculated load must not exceed.
"""

from typing import List

import pulp

from print_utils import print_empire_solver_output
from generate_empire_data import (
    NodeType,
    Node,
    Edge,
    generate_empire_data,
    get_reference_data,
    GraphData,
)


def create_load_hasload_vars(prob: pulp.LpProblem, obj: Node | Edge, load_vars: List[Node]):
    assert load_vars, "Each node must have at least one warehouse load variable."

    load_var = pulp.LpVariable(f"Load_{obj.name()}", lowBound=0, upBound=obj.capacity, cat="Integer")
    obj.pulp_vars["Load"] = load_var
    prob += load_var == pulp.lpSum(load_vars), f"TotalLoad_{obj.name()}"

    has_load_var = pulp.LpVariable(f"HasLoad_{obj.name()}", cat="Binary")
    obj.pulp_vars["HasLoad"] = has_load_var

    obj_is_node = isinstance(obj, Node)
    test_type = obj.type if obj_is_node else obj.destination.type
    if test_type in [NodeType.waypoint, NodeType.town]:
        # Linking constraint to ensure TotalLoad is either 0 or within [1, capacity]
        # prob += load_var >= has_load_var, f"ClampLoadMin_{obj.name()}"
        pass
    elif test_type is NodeType.lodging:
        # Linking constraint to ensure TotalLoad is either 0 or within [min_capacity, capacity]
        min_capacity = obj.min_capacity if obj_is_node else obj.destination.min_capacity
        prob += load_var >= has_load_var * min_capacity, f"ClampLoadMin_{obj.name()}"

    # Linking constraint to ensure TotalLoad is less than capacity.
    prob += load_var <= has_load_var * obj.capacity, f"ClampLoadMax_{obj.name()}"

    # Ensure has_load_var is 1 if load_var is non-zero and 0 if load_var is zero
    prob += load_var <= 1e-5 + (obj.capacity * has_load_var), f"ZeroOrPositiveLoad_{obj.name()}"
    prob += load_var - (has_load_var * obj.capacity) <= 0, f"ForceZeroLoad_{obj.name()}"

    # Ensure consistency between TotalLoad and HasLoad
    prob += has_load_var <= load_var, f"LinkLoadHasLoad_{obj.name()}"


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
                # # TODO: Benchmark this on larger samples, I think it is superceded by ClampBySource
                # # supply, origin and lodging nodes are load limited.
                if node.type in [NodeType.supply, NodeType.origin, NodeType.lodging]
                else LoadForWarehouse.capacity,
                cat="Integer",
            )
            node.pulp_vars[load_key] = load_var
            warehouse_load_vars.append(load_var)

            # if node.type in [NodeType.waypoint, NodeType.town]:
            #     source_load = graph_data["nodes"]["source"].pulp_vars[load_key]
            #     constraint_key = f"ClampBySource_{LoadForWarehouse.id}_at_{node.name()}"
            #     prob += load_var <= source_load, constraint_key

        create_load_hasload_vars(prob, node, warehouse_load_vars)


def create_edge_vars(prob: pulp.LpProblem, graph_data: GraphData, max_cost: int):
    """has_load and load vars for edges.
    `has_load` is a binary indicator of a load.
    `LoadForWarehouse_{warehouse}`: integer value of a destination bound load.

    Edges can not transport a load the source doesn't send or a load the destination can't recieve.
    LoadForWarehouse is an intersection of the source and destination LoadForWarehouse lists.
    """

    for edge in graph_data["edges"].values():
        warehouse_load_vars = []
        for LoadForWarehouse in set(edge.source.LoadForWarehouse).intersection(
            set(edge.destination.LoadForWarehouse)
        ):
            load_key = f"LoadForWarehouse_{LoadForWarehouse.id}"
            load_var = pulp.LpVariable(
                f"{load_key}_on_{edge.name()}",
                lowBound=0,
                upBound=edge.capacity,
                cat="Integer",
            )
            edge.pulp_vars[load_key] = load_var
            warehouse_load_vars.append(load_var)

            # if edge.destination.type in [NodeType.waypoint, NodeType.town]:
            #     source_load = graph_data["nodes"]["source"].pulp_vars[load_key]
            #     constraint_key = f"ClampBySource_{LoadForWarehouse.id}_at_{edge.name()}"
            #     prob += load_var <= source_load, constraint_key

        create_load_hasload_vars(prob, edge, warehouse_load_vars)


def link_node_and_edge_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure load per warehouse flow conservation for all nodes.
    Link the inbound edges load sum to the node load to the outbound edges load sum.
    Special handling for source and sink nodes.
    """

    def get_edge_vars(node: Node, node_var_key: str, edges: List[Edge]) -> List[pulp.LpVariable]:
        """Get the relevant edge variables for a given node and key."""
        is_source = node.type is NodeType.source
        edge_vars = []
        for edge in edges:
            for key, var in edge.pulp_vars.items():
                if key.startswith("LoadForWarehouse_") and (
                    key == node_var_key or node.type is NodeType.lodging
                ):
                    edge_vars.append(edge.pulp_vars["HasLoad"] if is_source else var)
        return edge_vars

    def link_node_to_edges(prob: pulp.LpProblem, node: Node, direction: str, edges: List[Edge]):
        """Link a node's load variable to the sum of its edge variables."""
        for key, var in node.pulp_vars.items():
            if key.startswith("LoadForWarehouse_"):
                edge_vars = get_edge_vars(node, key, edges)
                prob += var == pulp.lpSum(edge_vars), f"Link{direction}_{key}_at_{node.name()}"

    for node in graph_data["nodes"].values():
        if node.type is not NodeType.source:
            link_node_to_edges(prob, node, "Inbound", node.inbound_edges)
        if node.type is not NodeType.sink:
            link_node_to_edges(prob, node, "Outbound", node.outbound_edges)


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


def add_warehouse_load_balance_constraints(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure each warehouse receives its total origin supply."""
    for warehouse in graph_data["warehouse_nodes"].values():
        source_load = graph_data["nodes"]["source"].pulp_vars[f"LoadForWarehouse_{warehouse.id}"]
        prob += (
            warehouse.pulp_vars["Load"] == source_load,
            f"BalancesupplyLoadForWarehouse_{warehouse.id}",
        )


def add_waypoint_clamp_by_source_constraints(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure each waypoint/town node's and edge's LoadForWarehouse is <= its' source load."""

    for node in graph_data["nodes"].values():
        if node.type in [NodeType.waypoint, NodeType.town]:
            for var_key, var in node.pulp_vars.items():
                if var_key.startswith("LoadForWarehouse_"):
                    source_load = graph_data["nodes"]["source"].pulp_vars[var_key]
                    var_name = f"ClampBySource{var_key}_at_{node.name()}"
                    prob += var <= source_load, var_name

    for edge in graph_data["edges"].values():
        if edge.destination.type in [NodeType.waypoint, NodeType.town]:
            for var_key, var in edge.pulp_vars.items():
                if var_key.startswith("LoadForWarehouse_"):
                    source_load = graph_data["nodes"]["source"].pulp_vars[var_key]
                    var_name = (f"ClampBySource{var_key}_on_{edge.name()}",)
                    prob += var <= source_load, var_name


def add_warehouse_outbound_count_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure a single warehouse -> lodging -> sink path for each warehouse."""
    for warehouse in graph_data["warehouse_nodes"].values():
        prob += pulp.lpSum(edge.pulp_vars["HasLoad"] for edge in warehouse.outbound_edges) <= 1


def add_source_to_sink_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure source outbound load equals sink inbound load."""
    prob += (
        graph_data["nodes"]["source"].pulp_vars["Load"]
        == graph_data["nodes"]["sink"].pulp_vars["Load"],
        "BalanceSourceSink",
    )


def add_reverse_edge_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure reverse edges on waypoint/town nodes don't return loads."""

    def is_transit_node(node):
        return node.type in [NodeType.waypoint, NodeType.town]

    for edge in graph_data["edges"].values():
        if not is_transit_node(edge.source) or not is_transit_node(edge.destination):
            continue

        reverse_edge = graph_data["edges"][(edge.destination.name(), edge.source.name())]
        shared_keys = list(set(edge.pulp_vars.keys()).intersection(reverse_edge.pulp_vars.keys()))
        for var_key in shared_keys:
            if var_key.startswith("LoadForWarehouse"):
                # Make them mutally exclusive
                # Binary variables indicating if there's a load on the respective edges
                b_forward = pulp.LpVariable(f"bForward_{edge.name()}_{var_key}", cat="Binary")
                b_reverse = pulp.LpVariable(
                    f"bReverse_{reverse_edge.name()}_{var_key}", cat="Binary"
                )

                forward_var = edge.pulp_vars[var_key]
                reverse_var = reverse_edge.pulp_vars[var_key]

                prob += (
                    forward_var <= edge.destination.capacity * b_forward,
                    f"ForwardLoad_{edge.name()}_{var_key}",
                )
                prob += (
                    reverse_var <= edge.source.capacity * b_reverse,
                    f"ReverseLoad_{reverse_edge.name()}_{var_key}",
                )

                # Ensure that the load on the reverse edge is zero if there's
                # load on the forward edge and vice versa
                prob += (
                    b_forward + b_reverse <= 1,
                    f"MutualExclusivity_{edge.name()}_{reverse_edge.name()}_{var_key}",
                )


def create_problem(graph_data, max_cost):
    """Create the problem and add the varaibles and constraints."""
    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    create_node_vars(prob, graph_data)
    create_edge_vars(prob, graph_data, max_cost)
    link_node_and_edge_vars(prob, graph_data)

    total_value_var = create_value_var(prob, graph_data)
    total_cost_var = create_cost_var(prob, graph_data, max_cost)

    # prob += total_value_var, "ObjectiveFunction"
    prob += total_value_var - total_cost_var, "ObjectiveFunction"
    prob += total_cost_var <= max_cost, "MaxCost"

    # add_reverse_edge_constraint(prob, graph_data)
    add_warehouse_outbound_count_constraint(prob, graph_data)
    add_warehouse_load_balance_constraints(prob, graph_data)
    # add_waypoint_clamp_by_source_constraints(prob, graph_data)
    add_source_to_sink_constraint(prob, graph_data)

    return prob


def main(max_cost=30, lodging_bonus=1, top_n=2, nearest_n=4, waypoint_capacity=15):
    """
    top_n: count of supply nodes per warehouse by value
    nearest_n: count of nearest warehouses available on waypoint nodes
    waypoint_capacity: max loads on a waypoint
    """
    graph_data: GraphData = generate_empire_data(lodging_bonus, top_n, nearest_n, waypoint_capacity)
    prob = create_problem(graph_data, max_cost)
    prob.writeLP(f"last_{max_cost}.lp")

    solver = pulp.HiGHS_CMD(
        path="/home/thell/.local/bin/highs",
        keepFiles=True,
        threads=16,
        logPath=f"/home/thell/milp-flow/highs_log_{max_cost}.txt",
        options=[
            "parallel=on",
            "threads=16",
            # "mip_feasibility_tolerance=1e-6",
            "mip_improving_solution_save=on",
            f"mip_improving_solution_file=highs_improved_solution_{max_cost}.sol",
        ],
    )
    solver.solve(prob)
    # prob.solve(solver=pulp.HiGHS())

    assert prob.variablesDict()["Load_source"].value() == prob.variablesDict()["Load_sink"].value()

    ref_data = get_reference_data(lodging_bonus, top_n)
    print_empire_solver_output(prob, graph_data, ref_data, max_cost, top_n, detailed=True)


if __name__ == "__main__":
    main()
