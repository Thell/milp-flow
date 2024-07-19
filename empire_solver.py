"""MILP Worker Node Empire Problem
(See `generate_empire_data.py` for node and edge generation details.)

Goal: Maximize demand value - cost

The model is based on flow where source outbound load == sink inbound load per demand destination.

* All edges and nodes have a load which is calculated per demand warehouse.
* All edges and nodes have a capacity which the calculated load must not exceed.
"""

from typing import Dict, TypedDict, List

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

                # MARK NODE DEMAND
                # assert node.LoadForWarehouse
                for LoadForWarehouse in node.LoadForWarehouse:
                    node.pulp_vars[f"LoadForWarehouse_{LoadForWarehouse.id}"] = pulp.LpVariable(
                        # MARK NODE DEMAND
                        f"LoadForWarehouse_{LoadForWarehouse.id}_at_{node.name()}",
                        lowBound=0,
                        upBound=node.capacity,
                        cat="Integer",
                    )
            case NodeType.origin:
                # Origins only have loads for the 'top_n' demand destinations.
                for edge in node.inbound_edges:
                    for LoadForWarehouse in edge.source.LoadForWarehouse:
                        node.pulp_vars[f"LoadForWarehouse_{LoadForWarehouse.id}"] = pulp.LpVariable(
                            f"LoadForWarehouse_{LoadForWarehouse.id}_at_{node.name()}",
                            lowBound=0,
                            upBound=node.capacity,
                            cat="Integer",
                        )
            case NodeType.warehouse:
                # Warehouses should only have their own loads.
                # MARK Warehouse self id.
                for LoadForWarehouse in node.LoadForWarehouse:
                    node.pulp_vars[f"LoadForWarehouse_{LoadForWarehouse.id}"] = pulp.LpVariable(
                        f"LoadForWarehouse_{LoadForWarehouse.id}_at_{node.name()}",
                        lowBound=0,
                        upBound=node.capacity,
                        cat="Integer",
                    )
            case NodeType.lodging:
                # Lodgings should only have their own loads.
                node.pulp_vars[f"LoadForWarehouse_{node.id}"] = pulp.LpVariable(
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
                # MARK DEST DEMAND
                for LoadForWarehouse in edge.destination.LoadForWarehouse:
                    edge.pulp_vars[f"LoadForWarehouse_{LoadForWarehouse.id}"] = pulp.LpVariable(
                        # MARK DEST DEMAND
                        f"LoadForWarehouse_{LoadForWarehouse.id}_on_{edge.name()}",
                        lowBound=0,
                        upBound=edge.capacity,
                        cat="Integer",
                    )
            case (NodeType.demand, NodeType.origin):
                # MARK SOURCE DEMAND
                for LoadForWarehouse in edge.source.LoadForWarehouse:
                    edge.pulp_vars[f"LoadForWarehouse_{LoadForWarehouse.id}"] = pulp.LpVariable(
                        # MARK DEST DEMAND
                        f"LoadForWarehouse_{LoadForWarehouse.id}_on_{edge.name()}",
                        lowBound=0,
                        upBound=edge.capacity,
                        cat="Integer",
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
                # MARK SOURCE DEMAND
                for LoadForWarehouse in edge.source.LoadForWarehouse:
                    edge.pulp_vars[f"LoadForWarehouse_{LoadForWarehouse.id}"] = pulp.LpVariable(
                        # MARK DEST DEMAND
                        f"LoadForWarehouse_{LoadForWarehouse.id}_on_{edge.name()}",
                        lowBound=0,
                        upBound=edge.capacity,
                        cat="Integer",
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

    def get_edge_vars(node: Node, node_var_key: str, edges: List[Edge]) -> List[pulp.LpVariable]:
        """Get the relevant edge variables for a given node and key."""
        is_source = node.type == NodeType.source
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
        prob += pulp.lpSum(edge.pulp_vars["HasLoad"] for edge in warehouse.outbound_edges) <= 1


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
    create_edge_vars(prob, graph_data)
    link_node_and_edge_vars(prob, graph_data)

    total_value_var = create_value_var(prob, graph_data)
    total_cost_var = create_cost_var(prob, graph_data, max_cost)
    prob += total_value_var - total_cost_var, "ObjectiveFunction"
    prob += total_cost_var <= max_cost, "MaxCost"

    add_warehouse_outbound_count_constraint(prob, graph_data)
    add_warehouse_load_balance_constraints(prob, graph_data)

    # Enabling this slow convergence down _dramatically_.
    # Using an assert after calling the solver is _much_ faster and gives the same result.
    # add_source_to_sink_constraint(prob, graph_data)

    prob.writeLP("test.lp")
    return prob


def main(max_cost=50, top_n=0):
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
    assert prob.variablesDict()["Load_source"].value() == prob.variablesDict()["Load_sink"].value()

    ref_data = get_reference_data(top_n)
    print_empire_solver_output(prob, graph_data, ref_data, max_cost, detailed=True)


if __name__ == "__main__":
    main()
