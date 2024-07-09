import json
import os
import pulp

from generate_output import print_solver_output


def read_sample_empire_from_json(filename="sample_empire.json"):
    """Read the sample empire data from a JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def create_edge_vars(data):
    edge_vars = {}
    for edge in data["edges"]:
        edge_vars[(edge["source"], edge["destination"])] = pulp.LpVariable(
            f"flow_{edge['source']}_{edge['destination']}", lowBound=0, cat="Continuous"
        )
    return edge_vars


def create_edge_load_vars(data):
    edge_load_vars = {}
    for edge in data["edges"]:
        edge_load_vars[(edge["source"], edge["destination"])] = pulp.LpVariable(
            f"load_{edge['source']}_{edge['destination']}", lowBound=0, cat="Continuous"
        )
    return edge_load_vars


def create_load_vars(nodes):
    """Create continuous dynamic load variables for each transit node's active load."""
    load_vars = {}
    for node in nodes:
        if node.startswith(("transit_", "active_prod")):
            load_vars[node] = pulp.LpVariable(f"load_{node}", lowBound=0, cat="Continuous")
    return load_vars


def create_has_load_vars(nodes):
    """Create binary variables to represent the load condition."""
    has_load_vars = {}
    for node in nodes:
        has_load_vars[node] = pulp.LpVariable(f"has_load_{node}", cat=pulp.LpBinary)
    return has_load_vars


def create_active_prod_cost_var(data, edge_vars):
    """Create active_prod node costs variable."""
    return pulp.lpSum(
        edge["cost"] * edge_vars[(edge["source"], edge["destination"])]
        for edge in data["edges"]
        if edge["source"].startswith(tuple([f"city_{city}_prod" for city in data["cities"]]))
        and edge["destination"].startswith("active_prod")
    )


def create_city_active_prod_counts(data, edge_vars):
    """Create active_prod count per city variables for city worker equality."""
    return {
        city: pulp.lpSum(
            edge_vars[(edge["source"], edge["destination"])]
            for edge in data["edges"]
            if edge["source"].startswith(f"city_{city}_prod")
            and edge["destination"].startswith("active_prod")
        )
        for city in data["cities"]
    }


def create_city_warehouse_worker_counts(data, edge_vars):
    """Create active worker per city count variables for active_prod equality."""
    return {
        city: pulp.lpSum(
            edge_vars[(edge["source"], edge["destination"])]
            for edge in data["edges"]
            if edge["source"].startswith(f"warehouse_{city}_worker")
        )
        for city in data["cities"]
    }


def add_transit_node_flow_conservation_constraints(
    prob, data, nodes, edge_vars, load_vars, edge_load_vars
):
    """Add transit flow conservation constraints to the problem.
    Constraints should ensure flow conservation for the loads of transit nodes.
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
            load_vars[node]
            == pulp.lpSum(
                [edge_load_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
            ),
            f"LoadConservation_Transit_{node}",
        )


def add_objective_function(prob, data, edge_vars, active_prod_cost, has_load_vars):
    """Add the objective function to the problem.
    The objective is to maximize production value and minimize cost.
    Minimizing the cost is mostly done via constraints.
    """
    total_value = pulp.lpSum(
        edge["value"] * edge_vars[(edge["source"], edge["destination"])]
        for edge in data["edges"]
        if "value" in edge and edge["destination"].startswith("active_prod_")
    )
    transit_cost = pulp.lpSum(has_load_vars[node["node"]] for node in data["nodes"])
    prob += total_value - transit_cost, "ObjectiveFunction"


def add_cost_constraint(prob, data, active_prod_cost, has_load_vars):
    """Add the cost constraint to the problem.
    Constraint should ensure total cost <= max_cost from data.
    """
    prob += (
        pulp.lpSum((node["cost"] * has_load_vars[node["node"]] for node in data["nodes"]))
        <= data["max_cost"],
        "Max_cost_constraint",
    )


def add_prod_node_activation_constraint(prob, data, edge_vars):
    """Add the prod node activation constraint to the problem.
    Constraint should ensure only a single node can activate a production node.
    """
    for edge in data["edges"]:
        if edge["destination"].startswith("active_prod_"):
            prob += (
                pulp.lpSum(
                    edge_vars[(src["source"], edge["destination"])]
                    for src in data["edges"]
                    if src["destination"] == edge["destination"]
                )
                <= 1,
                f"Inbound_from_{edge['source']}_to_{edge['destination']}",
            )


# def add_flow_conservation_constraints(prob, nodes, edge_vars, data, load_vars):
#     """Add flow conservation constraints to the problem.
#     Constraints should ensure flow conservation for active prod and transit nodes.
#     """
#     for node in nodes:
#         inbound_edges = [edge for edge in data["edges"] if edge["destination"] == node]
#         outbound_edges = [edge for edge in data["edges"] if edge["source"] == node]

#         if node.startswith("transit_"):
#             # Load Bearing Flow Constraints.
#             prob += (
#                 pulp.lpSum(
#                     edge_vars[(edge["source"], edge["destination"])] for edge in inbound_edges
#                 )
#                 <= data["num_production_nodes"],
#                 f"Dynamic_load_conservation_for_{node}_inflow",
#             )
#             prob += (
#                 pulp.lpSum(
#                     edge_vars[(edge["source"], edge["destination"])] for edge in outbound_edges
#                 )
#                 == load_vars[node],
#                 f"Dynamic_load_conservation_for_{node}_outflow",
#             )
#         elif node not in ["source", "sink"]:
#             # Non-Load Bearing Flow Constraints.
#             inflow = pulp.lpSum(
#                 edge_vars[(edge["source"], edge["destination"])] for edge in inbound_edges
#             )
#             outflow = pulp.lpSum(
#                 edge_vars[(edge["source"], edge["destination"])] for edge in outbound_edges
#             )
#             prob += (
#                 inflow == outflow,
#                 f"Flow_conservation_for_{node}",
#             )


def add_prod_node_flow_conservation_constraints(prob, data, nodes, edge_vars):
    """Add prod flow conservation constraints to the problem.
    Constraints should ensure flow conservation for active prod nodes.
    """
    for node in nodes:
        if node.startswith("transit_") or node in ["source", "sink"]:
            continue

        incoming_edges = [edge for edge in data["edges"] if edge["destination"] == node]
        outgoing_edges = [edge for edge in data["edges"] if edge["source"] == node]

        # Sum of incoming flows should be equal to sum of outgoing flows
        prob += (
            pulp.lpSum([edge_vars[(edge["source"], edge["destination"])] for edge in incoming_edges])
            == pulp.lpSum(
                [edge_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
            ),
            f"FlowConservation_for_{node}",
        )


# def add_transit_node_flow_conservation_constraints(prob, data, nodes, edge_vars, load_vars):
#     """Add transit flow conservation constraints to the problem.
#     Constraints should ensure flow conservation for the loads of transit nodes.
#     """
#     for node in nodes:
#         if not node.startswith("transit_"):
#             continue

#         incoming_edges = [edge for edge in data["edges"] if edge["destination"] == node]
#         outgoing_edges = [edge for edge in data["edges"] if edge["source"] == node]

#         # # Sum of incoming flows should be equal to sum of outgoing flows
#         # prob += (
#         #     pulp.lpSum([edge_vars[(edge["source"], edge["destination"])] for edge in incoming_edges])
#         #     <= data["num_production_nodes"],
#         #     f"FlowConservation_Transit_{node}",
#         # )

#         # Additionally, ensure that the flow conservation respects the load constraints
#         prob += (
#             load_vars[node]
#             == pulp.lpSum(
#                 [edge_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
#             ),
#             f"LoadConservation_Transit_{node}",
#         )


def add_link_constraint(prob, nodes, load_vars, has_load_vars, data):
    """Add link constraint to the problem.
    Constraint should ensure binary has_load_vars is linked to continuous load_vars > 0.
    """
    for node in nodes:
        if node in load_vars:
            prob += (
                load_vars[node] <= (data["num_production_nodes"] + 1) * has_load_vars[node],
                f"Link_load_to_has_load_{node}_upper",
            )


def add_loaded_edge_activation_constraint(prob, data, edge_vars, has_load_vars):
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
                has_load_vars[source] + has_load_vars[destination] - edge_vars[edge_name] <= 1,
                f"Loaded_edge_endpoints_{edge_name}",
            )


def add_load_constraints(prob, data, nodes, edge_vars, load_vars):
    """Add constraints to ensure load_vars are calculated correctly."""
    for node in nodes:
        if not node.startswith(("transit_", "active_prod")):
            continue

        incoming_edges = [edge for edge in data["edges"] if edge["destination"] == node]
        outgoing_edges = [edge for edge in data["edges"] if edge["source"] == node]

        # Sum of incoming loads should be equal to the node's load variable
        prob += (
            load_vars[node]
            == pulp.lpSum(
                [edge_vars[(edge["source"], edge["destination"])] for edge in incoming_edges]
            ),
            f"LoadSumIncoming_{node}",
        )

        # Sum of outgoing loads should be equal to the node's load variable
        prob += (
            load_vars[node]
            == pulp.lpSum(
                [edge_vars[(edge["source"], edge["destination"])] for edge in outgoing_edges]
            ),
            f"LoadSumOutgoing_{node}",
        )


def add_has_load_constraints(prob, nodes, load_vars, has_load_vars):
    for node in nodes:
        if not node.startswith("transit"):
            continue
        prob += (load_vars[node] <= has_load_vars[node] * 1e6, f"HasLoadConstraintUpper_{node}")
        prob += (load_vars[node] >= has_load_vars[node] * 1e-6, f"HasLoadConstraintLower_{node}")


def add_city_inbound_constraint(prob, data, load_vars, city_active_prod_count):
    """Add city inbound constraint to the problem.
    Constraint should ensure load at city transit nodes equals the city's active prod count.
    """
    for city in data["cities"]:
        transit_node = f"transit_{city}"
        prob += (
            load_vars[transit_node] >= city_active_prod_count[city],
            f"Load_at_{transit_node}_must_be_at_least_active_prod_count",
        )


def add_city_worker_balance_constraint(
    prob, city_active_prod_count, city_warehouse_worker_count, data
):
    """Add production workers constraint to ensure each city's active prod node has an active worker.
    Constraint should ensure each city's worker node count equals the city's active prod count.
    """
    for city in data["cities"]:
        prob += (
            city_active_prod_count[city] == city_warehouse_worker_count[city],
            f"City_{city}_balance",
        )


def add_city_max_worker_constraint(prob, city_active_prod_count, city_warehouse_worker_count, data):
    """Add production workers constraint to ensure each city's active prod node has an active worker.
    Constraint should ensure each city's worker node count <= city's max_workers.
    """
    for city in data["cities"]:
        prob += (
            city_active_prod_count[city] <= data["max_workers"][str(city)],
            f"City_{city}_max_workers",
        )


def add_source_to_sink_constraint(prob, nodes, edge_vars, data):
    """Add source to sink constraint through transit nodes.
    Constraint should ensure overall total source to sink connectivity.
    """
    for node in nodes:
        if node.startswith("transit_"):
            prob += (
                pulp.lpSum(
                    edge_vars[(edge["source"], edge["destination"])]
                    for edge in data["edges"]
                    if edge["destination"] == node
                )
                >= pulp.lpSum(
                    edge_vars[(edge["source"], edge["destination"])]
                    for edge in data["edges"]
                    if edge["source"] == node
                ),
                f"Flow_through_transit_{node}",
            )


def main():
    data = read_sample_empire_from_json()
    edges = data["edges"]

    # Identify all nodes involved in edges.
    nodes = set()
    for edge in edges:
        nodes.add(edge["source"])
        nodes.add(edge["destination"])

    # Create the LP problem
    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    # # Create solver variables.
    # # Binary variables for each edge flow condition.
    # edge_vars = create_edge_vars(data)

    # edge_load_vars = create_edge_load_vars(data)

    # # Binary variables to represent the load condition.
    # has_load_vars = create_has_load_vars(nodes)
    # # Continuous variables for dynamic load of each transit node's active load.
    # load_vars = create_load_vars(nodes)
    # # Active_prod count per city variables for city worker equality.
    # city_active_prod_count = create_city_active_prod_counts(data, edge_vars)
    # # Active worker per city count variables for active_prod equality.
    # city_warehouse_worker_count = create_city_warehouse_worker_counts(data, edge_vars)
    # # Active_prod node costs variable for objective function cost accounting.
    # active_prod_cost = create_active_prod_cost_var(data, edge_vars)

    # # Add the objective function to the problem.
    # add_objective_function(prob, data, edge_vars, active_prod_cost, has_load_vars)

    # # Add constraints to the problem.
    # # Constraint to ensure total cost <= max_cost from data.
    # add_cost_constraint(prob, data, active_prod_cost, has_load_vars)
    # # Constraint to ensure only a single node can activate a production node.
    # add_prod_node_activation_constraint(prob, data, edge_vars)

    # # Constraints to ensure flow conservation for active prod and transit nodes.
    # # add_flow_conservation_constraints(prob, nodes, edge_vars, data, load_vars)
    # add_transit_node_flow_conservation_constraints(
    #     prob, data, nodes, edge_vars, load_vars, edge_load_vars
    # )

    # # Constraints to ensure flow conservation for non-load bearing nodes.
    # add_prod_node_flow_conservation_constraints(prob, data, nodes, edge_vars)

    # # Constraints to ensure flow conservation for load bearing transit nodes.
    # # add_transit_node_flow_conservation_constraints(prob, data, nodes, edge_vars, load_vars)

    # # Constraint to ensure binary has_load_vars is linked to continuous load_vars > 0.
    # # add_link_constraint(prob, nodes, load_vars, has_load_vars, data)
    # add_has_load_constraints(prob, nodes, load_vars, has_load_vars)
    # add_load_constraints(prob, data, nodes, edge_vars, load_vars)

    # # Constraint to ensure load at city transit nodes equals the city's active prod count.
    # add_city_inbound_constraint(prob, data, load_vars, city_active_prod_count)

    # # Constraint to ensure a load at both endpoints means an active edge.
    # add_loaded_edge_activation_constraint(prob, data, edge_vars, has_load_vars)

    # # Constraint to ensure each city's worker node count equals the city's active prod count.
    # add_city_worker_balance_constraint(
    #     prob, city_active_prod_count, city_warehouse_worker_count, data
    # )
    # # Constraint to ensure overall total source to sink connectivity.
    # add_source_to_sink_constraint(prob, nodes, edge_vars, data)

    # Create solver variables.
    edge_vars = create_edge_vars(data)
    edge_load_vars = create_edge_load_vars(data)
    has_load_vars = create_has_load_vars(nodes)
    load_vars = create_load_vars(nodes)
    city_active_prod_count = create_city_active_prod_counts(data, edge_vars)
    city_warehouse_worker_count = create_city_warehouse_worker_counts(data, edge_vars)
    active_prod_cost = create_active_prod_cost_var(data, edge_vars)

    # Add the objective function to the problem.
    add_objective_function(prob, data, edge_vars, active_prod_cost, has_load_vars)

    # Add constraints to the problem.
    add_cost_constraint(prob, data, active_prod_cost, has_load_vars)
    add_prod_node_activation_constraint(prob, data, edge_vars)
    add_prod_node_flow_conservation_constraints(prob, data, nodes, edge_vars)
    add_transit_node_flow_conservation_constraints(
        prob, data, nodes, edge_vars, load_vars, edge_load_vars
    )
    add_link_constraint(prob, nodes, load_vars, has_load_vars, data)
    add_city_inbound_constraint(prob, data, load_vars, city_active_prod_count)
    add_loaded_edge_activation_constraint(prob, data, edge_vars, has_load_vars)
    add_city_worker_balance_constraint(
        prob, city_active_prod_count, city_warehouse_worker_count, data
    )
    add_source_to_sink_constraint(prob, nodes, edge_vars, data)
    add_has_load_constraints(prob, nodes, load_vars, has_load_vars)
    add_load_constraints(prob, data, nodes, edge_vars, load_vars)
    add_city_max_worker_constraint(prob, city_active_prod_count, city_warehouse_worker_count, data)

    # Solve it
    prob.solve()
    print_solver_output(prob, data, edge_vars)


if __name__ == "__main__":
    main()
