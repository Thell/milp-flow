import json
import os
import pulp

from output_helper import output_helper


def read_sample_empire_from_json(filename="sample_empire.json"):
    filename = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def solve_milp(data):
    # Create the LP problem
    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    # Create variables for each edge flow
    edge_vars = {}
    for edge in data["edges"]:
        var_name = f"flow_{edge['source']}_{edge['destination']}"
        edge_vars[(edge["source"], edge["destination"])] = pulp.LpVariable(var_name, cat="Binary")

    # Objective function: Maximize the total value of active production flows.
    prob += pulp.lpSum(
        edge["value"] * edge_vars[(edge["source"], edge["destination"])]
        for edge in data["edges"]
        if "value" in edge and edge["destination"].startswith("active_prod_")
    )

    # Inbound to active production nodes constraint: only one city can enable a prod node.
    for edge in data["edges"]:
        if edge["destination"].startswith("active_prod_"):
            prob += (
                pulp.lpSum(
                    edge_vars[(src["source"], edge["destination"])]
                    for src in data["edges"]
                    if src["destination"] == edge["destination"]
                )
                <= 1,
                f"Inbound from {edge['source']} to {edge['destination']}",
            )

    # Production workers constraint: ensure each city's active prod node has an active worker
    city_active_prod_count = {
        city: pulp.lpSum(
            edge_vars[(edge["source"], edge["destination"])]
            for edge in data["edges"]
            if edge["source"].startswith(f"city_{city}_prod")
            and edge["destination"].startswith("active_prod")
        )
        for city in data["cities"]
    }

    city_warehouse_worker_count = {
        city: pulp.lpSum(
            edge_vars[(edge["source"], edge["destination"])]
            for edge in data["edges"]
            if edge["source"].startswith(f"warehouse_{city}_worker")
        )
        for city in data["cities"]
    }

    for city in data["cities"]:
        prob += (
            city_active_prod_count[city] == city_warehouse_worker_count[city],
            f"City_{city}_balance",
        )

    # Dynamic load conservation constraints
    load_vars = {}

    # Identify all nodes involved in edges
    nodes = set()
    for edge in data["edges"]:
        nodes.add(edge["source"])
        nodes.add(edge["destination"])

    # Create load variables for each transit node
    for node in nodes:
        if node.startswith("transit_"):
            load_vars[node] = pulp.LpVariable(f"load_{node}", lowBound=0, cat="Continuous")
            inbound_edges = [edge for edge in data["edges"] if edge["destination"] == node]
            outbound_edges = [edge for edge in data["edges"] if edge["source"] == node]

            # Inflow constraint: sum of loads (inflows) from incoming edges <= capacity of transit node
            prob += (
                pulp.lpSum(
                    edge_vars[(edge["source"], edge["destination"])] for edge in inbound_edges
                )
                <= data["num_production_nodes"],
                f"Dynamic_load_conservation_for_{node}_inflow",
            )

            # Outflow constraint: sum of loads (outflows) from outgoing edges == load at the node
            prob += (
                pulp.lpSum(
                    edge_vars[(edge["source"], edge["destination"])] for edge in outbound_edges
                )
                == load_vars[node],
                f"Dynamic_load_conservation_for_{node}_outflow",
            )
        elif node not in ["source", "sink"]:
            inflow = pulp.lpSum(
                edge_vars[(edge["source"], edge["destination"])]
                for edge in data["edges"]
                if edge["destination"] == node
            )
            outflow = pulp.lpSum(
                edge_vars[(edge["source"], edge["destination"])]
                for edge in data["edges"]
                if edge["source"] == node
            )
            prob += (
                inflow == outflow,
                f"Flow conservation for {node}",
            )

    # Cost constraint: Ensure total cost does not exceed a max_cost.
    prob += (
        pulp.lpSum(
            edge["cost"] * edge_vars[(edge["source"], edge["destination"])]
            for edge in data["edges"]
            if "cost" in edge
        )
        <= data["max_cost"],
        "Max_cost_constraint",
    )

    # # Create binary variables to represent the condition load_vars[node] > 0
    # has_load_vars = {}
    # for datum in data["nodes"]:
    #     node = datum["node"]
    #     has_load_vars[node] = pulp.LpVariable(f"has_load_{node}", cat=pulp.LpBinary)

    # # Link has_load_vars[node] to load_vars[node] > 0
    # M = 1000  # A large constant
    # for node in data["nodes"]:
    #     curr_node = node["node"]
    #     if curr_node in load_vars:
    #         prob += load_vars[curr_node] <= M * has_load_vars[curr_node], f"Link_load_to_has_load_{curr_node}_upper"
    #         prob += load_vars[curr_node] >= 0.001 * has_load_vars[curr_node], f"Link_load_to_has_load_{curr_node}_lower"

    # # Cost constraint: Ensure total cost does not exceed a max_cost.
    # prob += (
    #     pulp.lpSum(node["cost"] * has_load_vars[node["node"]] for node in data["nodes"]) <= data["max_cost"],
    #     "Max_cost_constraint",
    # )

    # # City Inbound Constraint: Ensure transit nodes receive enough flow from active production nodes
    # for city in data["cities"]:
    #     transit_node = f"transit_{city}"

    #     # Count the number of active production nodes (active_prod) connected to this transit node
    #     active_prod_count = pulp.lpSum(
    #         edge_vars[(edge["source"], edge["destination"])]
    #         for edge in data["edges"]
    #         if edge["source"].startswith(f"city_{city}_prod")
    #         and edge["destination"].startswith("active_prod")
    #     )

    #     # Ensure the load reaching the transit node is at least equal to the count of active production nodes
    #     prob += (
    #         load_vars[transit_node] >= active_prod_count,
    #         f"Load_at_{transit_node}_must_be_at_least_active_prod_count",
    #     )

    # Source to Sink Constraint through Transit Nodes
    for transit_node in nodes:
        if transit_node.startswith("transit_"):
            prob += (
                pulp.lpSum(
                    edge_vars[(edge["source"], edge["destination"])]
                    for edge in data["edges"]
                    if edge["destination"] == transit_node
                )
                >= pulp.lpSum(
                    edge_vars[(edge["source"], edge["destination"])]
                    for edge in data["edges"]
                    if edge["source"] == transit_node
                ),
                f"Flow_through_transit_{transit_node}",
            )

    prob.solve()
    output_helper(prob, data, edge_vars)
    return prob


# Example usage
data = read_sample_empire_from_json()
solve_milp(data)
