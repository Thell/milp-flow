"""Solve MILP Worker Node Empire Sample Problem"""

import pulp

from file_utils import read_empire_json
from print_utils import print_solver_output


def create_edge_flow_vars(data):
    """Create continuous variables for loads on all directed flow edges."""
    edge_flow_vars = {}
    for edge in data["edges"]:
        edge_flow_vars[(edge["source"], edge["destination"])] = pulp.LpVariable(
            f"flow_{edge['source']}_{edge['destination']}", lowBound=0, cat="Continuous"
        )
    return edge_flow_vars


def create_edge_load_vars(data):
    """Create continuous dynamic load variables for each transit edge's active load."""
    edge_load_vars = {}
    for edge in data["edges"]:
        edge_load_vars[(edge["source"], edge["destination"])] = pulp.LpVariable(
            f"load_{edge['source']}_{edge['destination']}", lowBound=0, cat="Continuous"
        )
    return edge_load_vars


def create_node_load_vars(nodes):
    """Create continuous dynamic load variables for each transit node's active load."""
    node_load_vars = {}
    for node in nodes:
        if node.startswith(("transit_", "active_prod")):
            node_load_vars[node] = pulp.LpVariable(f"load_{node}", lowBound=0, cat="Continuous")
    return node_load_vars


def create_node_has_load_vars(nodes):
    """Create binary variables to represent the load condition."""
    node_has_load_vars = {}
    for node in nodes:
        node_has_load_vars[node] = pulp.LpVariable(f"has_load_{node}", cat=pulp.LpBinary)
    return node_has_load_vars


def create_city_active_prod_counts(data, edge_flow_vars):
    """Create active_prod count per city variables for city worker equality."""
    return {
        city: pulp.lpSum(
            edge_flow_vars[(edge["source"], edge["destination"])]
            for edge in data["edges"]
            if edge["source"].startswith(f"city_{city}_prod")
            and edge["destination"].startswith("active_prod")
        )
        for city in data["cities"]
    }


def create_city_worker_counts(data, edge_flow_vars):
    """Create active city worker count variables for active_prod equality."""
    return {
        city: pulp.lpSum(
            edge_flow_vars[(edge["source"], edge["destination"])]
            for edge in data["edges"]
            if edge["source"].startswith(f"warehouse_{city}_worker")
        )
        for city in data["cities"]
    }


def add_objective_function(prob, data, edge_flow_vars, node_has_load_vars):
    """Add the objective function to the problem.
    The objective is to maximize production value and minimize cost.
    Minimizing the cost is mostly done via constraints.
    """
    total_value = pulp.lpSum(
        edge["value"] * edge_flow_vars[(edge["source"], edge["destination"])]
        for edge in data["edges"]
        if "value" in edge and edge["destination"].startswith("active_prod_")
    )
    transit_cost = pulp.lpSum(
        node["cost"] * node_has_load_vars[node["node"]] for node in data["nodes"]
    )
    prob += total_value - transit_cost, "ObjectiveFunction"


def add_cost_constraint(prob, data, node_node_has_load_vars):
    """Add the cost constraint to the problem.
    Constraint should ensure total cost <= max_cost from data.
    """
    prob += (
        pulp.lpSum((node["cost"] * node_node_has_load_vars[node["node"]] for node in data["nodes"]))
        <= data["max_cost"],
        "Max_cost_constraint",
    )


def add_prod_node_activation_constraint(prob, data, edge_flow_vars):
    """Add the prod node activation constraint to the problem.
    Constraint should ensure only a single node can activate a production node.
    """
    for edge in data["edges"]:
        if edge["destination"].startswith("active_prod_"):
            prob += (
                pulp.lpSum(
                    edge_flow_vars[(src["source"], edge["destination"])]
                    for src in data["edges"]
                    if src["destination"] == edge["destination"]
                )
                <= 1,
                f"Inbound_from_{edge['source']}_to_{edge['destination']}",
            )


def add_non_transit_node_flow_constraints(prob, data, nodes, node_load_vars, edge_load_vars):
    """Add transit flow conservation constraints to the problem.
    Constraints should ensure flow conservation for the loads of non transit nodes.
    """
    for node in nodes:
        if not node.startswith("transit_"):
            continue

        incoming_edges = [edge for edge in data["edges"] if edge["destination"] == node]
        outgoing_edges = [edge for edge in data["edges"] if edge["source"] == node]

        # Sum of loads on incoming edges should be equal to sum of loads on outgoing edges
        prob += (
            pulp.lpSum(
                [edge_load_vars[(edge["source"], edge["destination"])] for edge in incoming_edges]
            )
            == pulp.lpSum(
                [edge_load_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
            ),
            f"FlowConservation_Transit_{node}",
        )

        # Additionally, ensure that the flow conservation respects the load constraints
        prob += (
            node_load_vars[node]
            == pulp.lpSum(
                [edge_load_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
            ),
            f"LoadConservation_Transit_{node}",
        )


def add_transit_node_flow_constraints(prob, data, nodes, edge_flow_vars):
    """Add prod flow conservation constraints to the problem.
    Constraints should ensure flow conservation for transit loads.
    """
    for node in nodes:
        if node.startswith("transit_") or node in ["source", "sink"]:
            continue

        incoming_edges = [edge for edge in data["edges"] if edge["destination"] == node]
        outgoing_edges = [edge for edge in data["edges"] if edge["source"] == node]

        # Sum of incoming flows should be equal to sum of outgoing flows
        prob += (
            pulp.lpSum(
                [edge_flow_vars[(edge["source"], edge["destination"])] for edge in incoming_edges]
            )
            == pulp.lpSum(
                [edge_flow_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
            ),
            f"FlowConservation_for_{node}",
        )


def add_edge_load_flow_constraints(prob, data, nodes, edge_flow_vars, node_load_vars):
    """Add prod flow conservation constraints to the problem.
    Constraints should ensure flow conservation for non transit loads.
    """
    for node in nodes:
        if not node.startswith(("transit_", "active_prod")):
            continue

        incoming_edges = [edge for edge in data["edges"] if edge["destination"] == node]
        outgoing_edges = [edge for edge in data["edges"] if edge["source"] == node]

        # Sum of incoming loads should be equal to the node's load variable
        prob += (
            node_load_vars[node]
            == pulp.lpSum(
                [edge_flow_vars[(edge["source"], edge["destination"])] for edge in incoming_edges]
            ),
            f"LoadSumIncoming_{node}",
        )

        # Sum of outgoing loads should be equal to the node's load variable
        prob += (
            node_load_vars[node]
            == pulp.lpSum(
                [edge_flow_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
            ),
            f"LoadSumOutgoing_{node}",
        )


def add_loaded_edge_activation_constraints(prob, data, edge_flow_vars, node_has_load_vars):
    """Add loaded edge constraint to the problem.
    Constraint should ensure an edge is activated when both endpoints are under load.
    This is a cost saving constraint to eliminate parallel paths when a partial alternate exists.
    """
    for edge in data["edges"]:
        source = edge["source"]
        destination = edge["destination"]
        edge_name = (source, destination)
        if source.startswith("transit_") and destination.startswith("transit_"):
            prob += (
                node_has_load_vars[source]
                + node_has_load_vars[destination]
                - edge_flow_vars[edge_name]
                <= 1,
                f"Loaded_edge_endpoints_{edge_name}",
            )


def add_loaded_node_activation_constraints(prob, nodes, node_load_vars, node_has_load_vars):
    """Add loaded node constraint to the problem.
    Constraint should ensure an node is activated when under load.
    """
    for node in nodes:
        if not node.startswith("transit"):
            continue
        prob += (
            node_load_vars[node] <= node_has_load_vars[node] * 1e6,
            f"HasLoadConstraintUpper_{node}",
        )
        prob += (
            node_load_vars[node] >= node_has_load_vars[node] * 1e-6,
            f"HasLoadConstraintLower_{node}",
        )


def add_load_has_load_link_constraints(prob, data, nodes, node_load_vars, node_has_load_vars):
    """Add link constraint to the problem.
    Constraint should ensure binary has_load_vars is linked to continuous load_vars > 0.
    """
    for node in nodes:
        if node in node_load_vars:
            prob += (
                node_load_vars[node]
                <= (data["num_production_nodes"] + 1) * node_has_load_vars[node],
                f"Link_load_to_has_load_{node}_upper",
            )


def add_city_inbound_constraints(prob, data, node_node_load_vars, city_active_prod_count):
    """Add city inbound constraint to the problem.
    Constraint should ensure load at city transit nodes equals the city's active prod count.
    """
    for city in data["cities"]:
        transit_node = f"transit_{city}"
        prob += (
            node_node_load_vars[transit_node] >= city_active_prod_count[city],
            f"Load_at_{transit_node}_must_be_at_least_active_prod_count",
        )


def add_city_worker_balance_constraints(prob, data, city_active_prod_count, city_worker_count):
    """Add production workers constraint to ensure each city's active prod node has an active worker.
    Constraint should ensure each city's worker node count equals the city's active prod count.
    """
    for city in data["cities"]:
        prob += (
            city_active_prod_count[city] == city_worker_count[city],
            f"City_{city}_prod_worker_balance",
        )


def add_city_max_worker_constraints(prob, data, city_worker_count):
    """Add production workers constraint to ensure each city's active prod node has an active worker.
    Constraint should ensure each city's worker node count <= city's max_workers.
    """
    for city in data["cities"]:
        prob += (
            city_worker_count[city] <= data["max_workers"][str(city)],
            f"City_{city}_max_workers",
        )


def add_source_to_sink_constraint(prob, data, nodes, edge_flow_vars):
    """Add source to sink constraint through transit nodes.
    Constraint should ensure overall total source to sink connectivity.
    """
    for node in nodes:
        if node.startswith("transit_"):
            prob += (
                pulp.lpSum(
                    edge_flow_vars[(edge["source"], edge["destination"])]
                    for edge in data["edges"]
                    if edge["destination"] == node
                )
                >= pulp.lpSum(
                    edge_flow_vars[(edge["source"], edge["destination"])]
                    for edge in data["edges"]
                    if edge["source"] == node
                ),
                f"Flow_through_transit_{node}",
            )


# Notes: Make all edges and nodes follow the same inbound constraints except source and make all
# edges and nodes follow the same outbound constraints except sink. Allow the graph structure to
# dictate the solution instead of the 'layered' constraints.


def main():
    data = read_empire_json("sample_empire.json")
    edges = data["edges"]

    nodes = set()
    for edge in edges:
        nodes.add(edge["source"])
        nodes.add(edge["destination"])

    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    edge_flow_vars = create_edge_flow_vars(data)
    node_load_vars = create_node_load_vars(nodes)
    node_has_load_vars = create_node_has_load_vars(nodes)
    city_active_prod_counts = create_city_active_prod_counts(data, edge_flow_vars)
    city_worker_counts = create_city_worker_counts(data, edge_flow_vars)

    add_objective_function(prob, data, edge_flow_vars, node_has_load_vars)

    add_cost_constraint(prob, data, node_has_load_vars)
    add_prod_node_activation_constraint(prob, data, edge_flow_vars)
    add_transit_node_flow_constraints(prob, data, nodes, edge_flow_vars)
    add_edge_load_flow_constraints(prob, data, nodes, edge_flow_vars, node_load_vars)
    add_loaded_edge_activation_constraints(prob, data, edge_flow_vars, node_has_load_vars)
    add_loaded_node_activation_constraints(prob, nodes, node_load_vars, node_has_load_vars)
    add_load_has_load_link_constraints(prob, data, nodes, node_load_vars, node_has_load_vars)
    add_city_worker_balance_constraints(prob, data, city_active_prod_counts, city_worker_counts)

    # Eliminate these
    edge_load_vars = create_edge_load_vars(data)
    add_non_transit_node_flow_constraints(prob, data, nodes, node_load_vars, edge_load_vars)
    add_city_inbound_constraints(prob, data, node_load_vars, city_active_prod_counts)
    add_city_max_worker_constraints(prob, data, city_worker_counts)

    add_source_to_sink_constraint(prob, data, nodes, edge_flow_vars)

    prob.solve()
    print_solver_output(prob, data, edge_flow_vars, doFull=True)


if __name__ == "__main__":
    main()
