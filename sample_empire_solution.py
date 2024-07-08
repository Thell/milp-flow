import json
import os
import pulp

from output_helper import output_helper as output_details_and_summary


def read_sample_empire_from_json(filename="sample_empire.json"):
    filename = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(filename, "r") as file:
        data = json.load(file)
    return data


data = read_sample_empire_from_json()


# Identify all nodes involved in edges.
nodes = set()
for edge in data["edges"]:
    nodes.add(edge["source"])
    nodes.add(edge["destination"])

# Create the LP problem
prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

# Create binary variables for each edge flow condition.
edge_vars = {}
for edge in data["edges"]:
    var_name = f"flow_{edge['source']}_{edge['destination']}"
    edge_vars[(edge["source"], edge["destination"])] = pulp.LpVariable(var_name, cat="Binary")

# Create binary variables to represent the load condition.
has_load_vars = {}
for node in nodes:
    has_load_vars[node] = pulp.LpVariable(f"has_load_{node}", cat=pulp.LpBinary)

# Create continuous dynamic load variables for each transit node's active load.
load_vars = {}
for node in nodes:
    if node.startswith("transit_"):
        load_vars[node] = pulp.LpVariable(f"load_{node}", lowBound=0, cat="Continuous")

# Create active_prod node costs variable.
# active_prod nodes don't have a load for conditional cost accounting so use the edge state.
# Safety: This should be safe as only one city can activate a prod node at a time.
active_prod_cost = pulp.lpSum(
    edge["cost"] * edge_vars[(edge["source"], edge["destination"])]
    for edge in data["edges"]
    if edge["source"].startswith(tuple([f"city_{city}_prod" for city in data["cities"]]))
    and edge["destination"].startswith("active_prod")
)

# Create active_prod count per city variables for city worker equality.
city_active_prod_count = {
    city: pulp.lpSum(
        edge_vars[(edge["source"], edge["destination"])]
        for edge in data["edges"]
        if edge["source"].startswith(f"city_{city}_prod")
        and edge["destination"].startswith("active_prod")
    )
    for city in data["cities"]
}

# Create active worker per city count variables for active_prod equality.
city_warehouse_worker_count = {
    city: pulp.lpSum(
        edge_vars[(edge["source"], edge["destination"])]
        for edge in data["edges"]
        if edge["source"].startswith(f"warehouse_{city}_worker")
    )
    for city in data["cities"]
}

# Objective function: Maximize the total value of active production flows.
prob += (
    pulp.lpSum(
        edge["value"] * edge_vars[(edge["source"], edge["destination"])]
        for edge in data["edges"]
        if "value" in edge and edge["destination"].startswith("active_prod_")
    )
    - active_prod_cost
    - pulp.lpSum(node["cost"] * 100 * has_load_vars[node["node"]] for node in data["nodes"])
)

# Cost constraint: Ensure total cost does not exceed a max_cost.
prob += (
    pulp.lpSum(
        active_prod_cost + (node["cost"] * has_load_vars[node["node"]] for node in data["nodes"])
    )
    <= data["max_cost"],
    "Max_cost_constraint",
)

# Prod Node Activation Constraint: Ensure only one city can activate a prod node.
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

# Flow Conservation Constraints.
for node in nodes:
    inbound_edges = [edge for edge in data["edges"] if edge["destination"] == node]
    outbound_edges = [edge for edge in data["edges"] if edge["source"] == node]

    if node.startswith("transit_"):
        # Load Bearing Flow Constraints.
        # Inflow: sum of loads from incoming edges <= capacity of transit node.
        prob += (
            pulp.lpSum(edge_vars[(edge["source"], edge["destination"])] for edge in inbound_edges)
            <= data["num_production_nodes"],
            f"Dynamic_load_conservation_for_{node}_inflow",
        )
        # Transit Outflow Constraint: sum of loads from outgoing edges == load at the node.
        prob += (
            pulp.lpSum(edge_vars[(edge["source"], edge["destination"])] for edge in outbound_edges)
            == load_vars[node],
            f"Dynamic_load_conservation_for_{node}_outflow",
        )
    elif node not in ["source", "sink"]:
        # Non-Load Bearing Flow Constraints.
        inflow = pulp.lpSum(
            edge_vars[(edge["source"], edge["destination"])] for edge in inbound_edges
        )
        outflow = pulp.lpSum(
            edge_vars[(edge["source"], edge["destination"])] for edge in outbound_edges
        )
        prob += (
            inflow == outflow,
            f"Flow conservation for {node}",
        )

# Link Constraint: Ensure binary has_load_vars[node] is linked to continuous load_vars[node] > 0.
for node in nodes:
    if node in load_vars:
        prob += (
            load_vars[node] <= (data["num_production_nodes"] + 1) * has_load_vars[node],
            f"Link_load_to_has_load_{node}_upper",
        )

# City Inbound Constraint: Ensure load city transit nodes receive are >= city active prod nodes.
for city in data["cities"]:
    transit_node = f"transit_{city}"
    prob += (
        load_vars[transit_node] >= city_active_prod_count[city],
        f"Load_at_{transit_node}_must_be_at_least_active_prod_count",
    )

# Production workers constraint: ensure each city's active prod node has an active worker
for city in data["cities"]:
    prob += (
        city_active_prod_count[city] == city_warehouse_worker_count[city],
        f"City_{city}_balance",
    )

# Source to Sink Constraint through Transit Nodes
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

prob.solve()
output_details_and_summary(prob, data, edge_vars)
