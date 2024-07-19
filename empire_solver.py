"""MILP Worker Node Empire Problem
(See `generate_empire_data.py` for node and edge generation details.)

Goal: Maximize demand value - cost

The model is based on flow where source outbound load == sink inbound load per demand destination.

* All edges and nodes have a load which is calculated per demand warehouse.
* All edges and nodes have a capacity which the calculated load must not exceed.
"""

from typing import Dict, TypedDict

import pulp

from print_utils import print_empire_solver_output
from generate_empire_data import NodeType, Node, Edge, generate_empire_data, get_reference_data

M = 1e6


class GraphData(TypedDict):
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    warehouse_nodes: Dict[str, Node]


def create_node_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """has_load and load vars for nodes.
    `has_load`: binary indicator of a load.
    `load_for_{warehouse}`: integer value of a destination bound load.
    """
    for node in graph_data["nodes"].values():
        match node.type:
            case NodeType.demand:
                # Demand nodes only have their own load.
                node.pulp_vars[f"LoadForWarehouse_{node.demand_destination_id}"] = pulp.LpVariable(
                    f"LoadForWarehouse_{node.demand_destination_id}_at_{node.name()}",
                    lowBound=0,
                    upBound=node.capacity,
                    cat="Integer",
                )
            case NodeType.origin:
                # Origins only have loads for the 'top_n' demand destinations.
                for source, _ in node.inbound_edge_ids:
                    demand_destination_id = graph_data["nodes"][source].demand_destination_id
                    node.pulp_vars[f"LoadForWarehouse_{demand_destination_id}"] = pulp.LpVariable(
                        f"LoadForWarehouse_{demand_destination_id}_at_{node.name()}",
                        lowBound=0,
                        upBound=node.capacity,
                        cat="Integer",
                    )
            case NodeType.warehouse:
                # Warehouses should only have their own loads.
                node.pulp_vars[f"LoadForWarehouse_{node.id}"] = pulp.LpVariable(
                    f"LoadForWarehouse_{node.id}_at_{node.name()}",
                    lowBound=0,
                    upBound=node.capacity,
                    cat="Integer",
                )
            case NodeType.lodging:
                # Lodgings should only have their own loads.
                node.pulp_vars[f"LoadForWarehouse_{node.name()}"] = pulp.LpVariable(
                    f"LoadForWarehouse_{node.id}_at_{node.name()}",
                    lowBound=0,
                    upBound=node.capacity,
                    cat="Integer",
                )
            case NodeType.source | NodeType.waypoint | NodeType.town | NodeType.sink:
                # Source, Waypoint and Sink nodes have a LoadForWarehouse var for each warehouse.
                for warehouse in graph_data["warehouse_nodes"].values():
                    node.pulp_vars[f"LoadForWarehouse_{warehouse.id}"] = pulp.LpVariable(
                        f"LoadForWarehouse_{warehouse.id}_at_{node.name()}",
                        lowBound=0,
                        upBound=warehouse.capacity,
                        cat="Integer",
                    )
            case NodeType.INVALID:
                assert node.type is not NodeType.INVALID, "INVALID node type."
                return  # Unreachable: Stops pyright unbound error reporting.

        # The load is simply the sum of the warehouse loads at the node.
        load_var = pulp.LpVariable(
            f"Load_{node.name()}", lowBound=0, upBound=node.capacity, cat="Integer"
        )
        prob += (
            load_var
            == pulp.lpSum(
                var for key, var in node.pulp_vars.items() if key.startswith("LoadForWarehouse")
            ),
            f"TotalLoad_{node.name()}",
        )
        node.pulp_vars["Load"] = load_var

        # Load activates has_load and if activated there must be a load.
        has_load_var = pulp.LpVariable(f"HasLoad_{node.name()}", cat="Binary")
        prob += load_var <= has_load_var * M, f"LoadActivation_{node.name()}"
        prob += has_load_var <= load_var, f"LinkLoadHasLoad_{node.name()}"
        node.pulp_vars["HasLoad"] = has_load_var


def create_edge_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """has_load and load vars for edges.
    `has_load` is a binary indicator of a load.
    `load_for_{warehouse}` is an integer value of a destination warehouse bound load.
    """

    for edge in graph_data["edges"].values():
        match edge.type:
            case (NodeType.source, NodeType.demand):
                edge.pulp_vars[f"LoadForWarehouse_{edge.destination.demand_destination_id}"] = (
                    pulp.LpVariable(
                        f"LoadForWarehouse_{edge.destination.demand_destination_id}_on_{edge.name()}",
                        lowBound=0,
                        upBound=edge.capacity,
                        cat="Integer",
                    )
                )
            case (NodeType.demand, NodeType.origin):
                edge.pulp_vars[f"LoadForWarehouse_{edge.source.demand_destination_id}"] = (
                    pulp.LpVariable(
                        f"LoadForWarehouse_{edge.source.demand_destination_id}_on_{edge.name()}",
                        lowBound=0,
                        upBound=edge.capacity,
                        cat="Integer",
                    )
                )
            case (NodeType.warehouse, NodeType.lodging):
                # There is one edge for each lodging capacity for the warehouse.
                edge.pulp_vars[f"LoadForWarehouse_{edge.source.id}"] = pulp.LpVariable(
                    f"LoadForWarehouse_{edge.source.id}_on_{edge.name()}",
                    lowBound=0,
                    upBound=edge.capacity,
                    cat="Integer",
                )
            case (NodeType.lodging, NodeType.sink):
                edge.pulp_vars[f"LoadForWarehouse_{edge.source.demand_destination_id}"] = (
                    pulp.LpVariable(
                        f"LoadForWarehouse_{edge.source.demand_destination_id}_on_{edge.name()}",
                        lowBound=0,
                        upBound=edge.capacity,
                        cat="Integer",
                    )
                )
            case _:
                for warehouse in graph_data["warehouse_nodes"].values():
                    edge.pulp_vars[f"LoadForWarehouse_{warehouse.id}"] = pulp.LpVariable(
                        f"LoadForWarehouse_{warehouse.id}_on_{edge.name()}",
                        lowBound=0,
                        upBound=warehouse.capacity,
                        cat="Integer",
                    )

        # Load is the sum of all LoadForWarehouse vars on the edge.
        load_var = pulp.LpVariable(
            f"Load_{edge.name()}", lowBound=0, upBound=edge.capacity, cat="Integer"
        )
        prob += (
            load_var
            == pulp.lpSum(
                var for key, var in edge.pulp_vars.items() if key.startswith("LoadForWarehouse")
            ),
            f"TotalLoad_{edge.name()}",
        )
        edge.pulp_vars["Load"] = load_var

        # Load activates has_load and if activated there must be a load.
        has_load_var = pulp.LpVariable(f"HasLoad_{edge.name()}", cat="Binary")
        prob += load_var <= has_load_var * M, f"LoadActivation_{edge.name()}"
        prob += has_load_var <= load_var, f"LinkLoadHasLoad_{edge.name()}"
        edge.pulp_vars["HasLoad"] = has_load_var


def link_node_and_edge_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure load per warehouse flow conservation for all nodes.
    Link the inbound edges load sum to the node load to the outbound edges load sum.
    Special handling for source and sink nodes.
    """
    for node in graph_data["nodes"].values():
        if node.type is NodeType.source:
            # Source load is linked to the outgoing edges summed by demand destination.
            # Since each demand node has a cap of 1 just sum the HasLoad of each edge per warehouse.
            outbound_edges = [graph_data["edges"][edge] for edge in node.outbound_edge_ids]
            for warehouse in graph_data["warehouse_nodes"].values():
                prob += (
                    node.pulp_vars[f"LoadForWarehouse_{warehouse.id}"]
                    == pulp.lpSum(
                        edge.pulp_vars["HasLoad"]
                        for edge in outbound_edges
                        if edge.destination.demand_destination_id == warehouse.id
                    ),
                    f"LinkSourceDemandLoadForWarhouse_{warehouse.id}_at_{node.name()}",
                )
            continue

        # Ensure the each node's LoadForWarehouse equals the sum of the related inbound vars.
        in_edges = [graph_data["edges"][edge] for edge in node.inbound_edge_ids]
        for node_var_key, node_var in node.pulp_vars.items():
            if not node_var_key.startswith("LoadForWarehouse_"):
                continue
            prob += (
                node_var
                == pulp.lpSum(
                    var
                    for edge in in_edges
                    for var_key, var in edge.pulp_vars.items()
                    if var_key == node_var_key
                    or (node.type is NodeType.lodging and var_key.startswith("LoadForWarehouse_"))
                ),
                f"LinkInboundLoadForWarehouse_{node_var_key}_at_{node.name()}",
            )

        # Outbound is not needed for the sink.
        if node.type is NodeType.sink:
            continue

        # Ensure the each node's LoadForWarehouse equals the sum of the related outbound vars.
        out_edges = [graph_data["edges"][edge] for edge in node.outbound_edge_ids]
        for node_var_key, node_var in node.pulp_vars.items():
            if not node_var_key.startswith("LoadForWarehouse_"):
                continue
            prob += (
                node_var
                == pulp.lpSum(
                    var
                    for edge in out_edges
                    for var_key, var in edge.pulp_vars.items()
                    if var_key == node_var_key
                    or (node.type is NodeType.lodging and var_key.startswith("LoadForWarehouse_"))
                ),
                f"LinkOutboundLoadForWarehouse_{node_var_key}_at_{node.name()}",
            )


def create_value_var(prob: pulp.LpProblem, graph_data: GraphData):
    """Sum variable of value at demand nodes with load."""
    value_var = pulp.LpVariable("value", lowBound=0)
    prob += (
        value_var
        == pulp.lpSum(
            node.value * node.pulp_vars["HasLoad"]
            for node in graph_data["nodes"].values()
            if node.type is NodeType.demand
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
    """Ensure each warehouse receives its total origin demand."""
    for warehouse in graph_data["warehouse_nodes"].values():
        prob += (
            warehouse.pulp_vars["Load"]
            == graph_data["nodes"]["source"].pulp_vars[f"LoadForWarehouse_{warehouse.id}"],
            f"BalanceDemandLoadForWarehouse_{warehouse.id}",
        )


def add_warehouse_outbound_count_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure a single warehouse -> lodging -> sink path for each warehouse."""
    for warehouse in graph_data["warehouse_nodes"].values():
        prob += (
            pulp.lpSum(
                graph_data["edges"][edge_key].pulp_vars["HasLoad"]
                for edge_key in warehouse.outbound_edge_ids
            )
            <= 1
        )


def add_source_to_sink_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure source outbound load equals sink inbound load."""
    # If all warehouses are balanced with demand then I don't think this should be required.
    prob += (
        pulp.lpSum(
            graph_data["nodes"]["source"].pulp_vars["Load"]
            == graph_data["nodes"]["sink"].pulp_vars["Load"]
        ),
        "BalanceSourceSink",
    )


def create_problem(graph_data, max_cost):
    """Create the problem and add the varaibles and constraints."""
    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    create_node_vars(prob, graph_data)
    create_edge_vars(prob, graph_data)
    link_node_and_edge_vars(prob, graph_data)

    total_value_var = create_value_var(prob, graph_data)
    total_cost_var = create_cost_var(prob, graph_data, max_cost)
    prob += total_value_var - total_cost_var, "ObjectiveFunction"
    prob += total_cost_var <= max_cost, "MaxCost"

    add_warehouse_outbound_count_constraint(prob, graph_data)
    add_warehouse_load_balance_constraints(prob, graph_data)

    nullLoadTypes = set()
    for obj in graph_data["nodes"].values():
        count = len([v for v in obj.pulp_vars.keys() if v.startswith("LoadForWarehouse")])
        nullLoadTypes.add((obj.type, count))
    print(nullLoadTypes)

    # Enabling this conflicts with some other constraint resulting in a zero value solution.
    # add_source_to_sink_constraint(prob, graph_data)

    prob.writeLP("test.lp")
    return prob


def main(max_cost=15, top_n=0):
    graph_data = generate_empire_data(top_n)
    prob = create_problem(graph_data, max_cost)

    solver = pulp.HiGHS_CMD(
        path="/home/thell/.local/bin/highs",
        keepFiles=True,
        threads=16,
        logPath="/home/thell/milp-flow/highs_log_{max_cost}.txt",
        options=["parallel=on", "threads=16"],
    )
    solver.solve(prob)

    ref_data = get_reference_data(top_n)
    print_empire_solver_output(prob, graph_data, ref_data, max_cost, detailed=True)


if __name__ == "__main__":
    main()
