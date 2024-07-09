### Sample Node Empire Edges Generation

# We will need flow edges for:
#
# Source -> Production Nodes
#  - supply set to # production nodes to ensure all production node flows are enabled.
#  - capacity of 1 since each production node can only supply 1 city.
#  - cost of 0 since these are virtual nodes used for just for flow.
#  - value of 0 since only city supply value nodes have value.
#  - MILP constraint that ∑ of enabled `source_prod_{}` edges = num_production_nodes ensuring supply.
#
# Production Node -> City Supply Value Node
#  - capacity of 1 since only one city can be supplied by a production node.
#  - cost of 0 since these are virtual nodes used for flow selection based on maximizing value.
#  - value of 0 since we store the value on the City Supply Node's outgoing edge.
#
# City Supply Value Node -> Active Production Nodes
# City Supply Value Node -> Unused Production Nodes
#  - capacity of 1 since supply either goes to 1 city or is unused.
#  - cost is > 0 for active production nodes and 0 for unused production nodes.
#  - value is >= 0 the value of the supply to each city with 0 meaning supply cant reach the city
#    because of logistical constraints or is unused.
#  - MILP constraint that a single Production Node only enables one of its outgoing edges.
#
# Active Production Nodes -> Transit Nodes (entry into the interconnected transit node network)
#  - capacity of 1 since each production node enters transit on its own edge.
#  - cost is > 0 since each production node's transit entry incurs a cost.
#  - value of 0 since only city supply value nodes have value.
#  - MILP constraint should not be needed because of the previous Active Production Nodes constraint.
#
# <-> Transit Interconnect Nodes
#  - capacity equal to the number of production nodes so all supplies can move accross all flows.
#  - cost is >= 0 since each transit node incurs a cost except transit across a city.
#  - edge cost is dependent on the destination node since it is assumed the source node's cost has
#    already been counted on the flow leading to it.
#  - edge cost is counted only once and once counted it is freely usable for any other transit.
#    If both edges in the following example are used there should be a single cost of 2:
#      Edge(active_prod_131 -> transit_21, cap: 1, cost: 2, value: 0)
#      Edge(active_prod_132 -> transit_21, cap: 1, cost: 2, value: 0)

# Transit Interconnect Nodes for Cities -> City Warehouse Workers
#  - each city has a distinct capacity of workers and rather than a single node with a single
#    capacity and some variable for determining how much capacity has been used between the transit
#    and the warehouse we instead are using one node per warehouse worker for simple summation.
#  - cost of 0 since these are virtual nodes used for just for flow.
#  - value of 0 since only city supply value nodes have value.

# City Warehouse Workers -> City Warehouse
#  - capacity of the city warehouse's maximum worker limit.
#  - cost of 0 since these are virtual nodes used for just for flow.
#  - value of 0 since only city supply value nodes have value.

# City Warehouse -> Sink
# Unused Production Nodes -> Unused Production Nodes Aggregator
#  - capacity of 1 since each unused production node can only be either active or unused once.
#  - cost of 0 since these are virtual nodes used for just for flow.
#  - value of 0 since only city supply value nodes have value.

# Unused Production Node Aggregator -> Sink
#  - capacity of num_production_nodes
#  - cost of 0 since these are virtual nodes used for just for flow.
#  - value of 0 since only city supply value nodes have value.

# Sink with demand set to # production nodes.
#  - Flow should be conserved from source to sink.

# The goal will be to conserve the flow from source to sink and maximize the value of production
# node supplies reaching the warehouses with a hard limit on the upper bounds of total cost.

import json
import os


def read_data_json(data_path, filename):
    filepath = os.path.join(os.path.dirname(__file__), "data", data_path, filename)
    with open(filepath, "r") as file:
        return json.load(file)


def read_user_json(filename):
    return read_data_json("user", filename)


def read_workerman_json(filename):
    return read_data_json("workerman", filename)


def read_sample_json(filename):
    return read_data_json("sample", filename)


DECK_LINKS = read_workerman_json("deck_links.json")
EXPLORATION_NODE_DATA = read_workerman_json("exploration.json")
TOWN_NODE_TRANSLATE = read_workerman_json("town_node_translate.json")

SAMPLE_CITY_NODES = [1, 61]
SAMPLE_MAX_WORKERS = {"1": 11, "61": 19}
SAMPLE_NETWORK_NODES = read_sample_json("sample_filter_nodes.json")["network_nodes"]
SAMPLE_PRODUCTION_NODES = read_sample_json("sample_filter_nodes.json")["production_nodes"]
SAMPLE_PRODUCTION_NODE_VALUES = read_sample_json("sample_node_values_per_city.json")

SAMPLE_PRODUCTION_NODE_PARENTS = {}
for node in SAMPLE_PRODUCTION_NODES:
    for link in DECK_LINKS:
        if node == link[1]:
            SAMPLE_PRODUCTION_NODE_PARENTS[str(node)] = link[0]


def save_sample_empire_to_json(
    edges,
    nodes,
    production_nodes,
    sink_demand,
    cities,
    max_workers,
    num_production_nodes,
    max_cost,
    filename="sample_empire.json",
):
    # Convert edges to a dictionary format
    edges_data = [
        {
            "source": edge.source,
            "destination": edge.destination,
            "capacity": edge.capacity,
            "cost": edge.cost,
            "value": edge.value,
            "load": edge.load,
        }
        for edge in edges
    ]

    node_costs = [{"node": node.node, "cost": node.cost} for node in nodes]

    data = {
        "edges": edges_data,
        "nodes": node_costs,
        "production_nodes": production_nodes,
        "sink_demand": sink_demand,
        "cities": cities,
        "max_cost": max_cost,
        "max_workers": max_workers,
        "num_production_nodes": num_production_nodes,
    }

    filename = os.path.join(os.path.dirname(__file__), "data", filename)
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Data saved to {filename}")


class Node:
    def __init__(self, node, cost=0):
        self.node = node
        self.cost = cost

    def __repr__(self):
        return f"Node({self.node} cost: {self.cost})"


class Edge:
    def __init__(self, source, destination, capacity, cost=0, value=0, load=0):
        self.source = source
        self.destination = destination
        self.capacity = capacity
        self.cost = cost
        self.value = round(value * 1000000)
        self.load = load

    def __repr__(self):
        return f"Edge({self.source} -> {self.destination}, cap: {self.capacity}, cost: {self.cost}, value: {self.value} load: {self.load})"


edges = []
num_production_nodes = len(SAMPLE_PRODUCTION_NODES)

# Source to production nodes
for prod_node in SAMPLE_PRODUCTION_NODES:
    node = EXPLORATION_NODE_DATA[str(prod_node)]
    edges.append(Edge("source", f"prod_{node["key"]}", 1))

# Production nodes to city-specific production value nodes
for city in SAMPLE_CITY_NODES:
    for prod_node in SAMPLE_PRODUCTION_NODES:
        edges.append(Edge(f"prod_{prod_node}", f"city_{city}_prod_{prod_node}", 1))

# City-specific production nodes to active production nodes
for city in SAMPLE_CITY_NODES:
    town = TOWN_NODE_TRANSLATE["tnk2tk"][str(city)]
    for prod_node, value_data in SAMPLE_PRODUCTION_NODE_VALUES[str(town)].items():
        cost = EXPLORATION_NODE_DATA[str(prod_node)]["CP"]
        edges.append(
            Edge(
                f"city_{city}_prod_{prod_node}",
                f"active_prod_{prod_node}",
                1,
                cost=cost,
                value=value_data["value"],
                load=1,
            )
        )

# City-specific production nodes to unused production nodes
for city in SAMPLE_CITY_NODES:
    for prod_node in SAMPLE_PRODUCTION_NODES:
        edges.append(Edge(f"city_{city}_prod_{prod_node}", f"unused_prod_{prod_node}", 1))

# Active production nodes to transit network nodes
for prod_node in SAMPLE_PRODUCTION_NODES:
    prod_node_parent = SAMPLE_PRODUCTION_NODE_PARENTS[str(prod_node)]
    prod_transit_node = EXPLORATION_NODE_DATA[str(prod_node_parent)]
    cost = prod_transit_node["CP"]
    edges.append(
        Edge(
            f"active_prod_{prod_node}",
            f"transit_{prod_node_parent}",
            1,
            cost=cost,
        )
    )
    edges.append(
        Edge(
            f"transit_{prod_node_parent}",
            f"active_prod_{prod_node}",
            1,
            cost=0,
        )
    )

# Transit network node interconnects
for transit_node in SAMPLE_NETWORK_NODES:
    for link in DECK_LINKS:
        if not all(node in SAMPLE_NETWORK_NODES for node in link):
            continue
        if transit_node in link:
            node_a = link[0]
            node_b = link[1]
            cost_a_to_b = EXPLORATION_NODE_DATA[str(node_b)]["CP"]
            cost_b_to_a = EXPLORATION_NODE_DATA[str(node_a)]["CP"]
            edges.append(
                Edge(
                    f"transit_{node_a}",
                    f"transit_{node_b}",
                    num_production_nodes,
                    cost=cost_a_to_b,
                )
            )
            edges.append(
                Edge(
                    f"transit_{node_b}",
                    f"transit_{node_a}",
                    num_production_nodes,
                    cost=cost_b_to_a,
                )
            )


# Transit city aggregator nodes to city warehouse nodes (workers)
for city_node in SAMPLE_CITY_NODES:
    for worker in range(SAMPLE_MAX_WORKERS[str(city_node)]):
        edges.append(
            Edge(f"transit_{city_node}", f"warehouse_{city_node}_worker_{worker}", 1, cost=0)
        )

# Town warehouse worker nodes to sink
for city_node in SAMPLE_CITY_NODES:
    for worker in range(SAMPLE_MAX_WORKERS[str(city_node)]):
        edges.append(Edge(f"warehouse_{city_node}_worker_{worker}", "sink", 1, cost=0))

# Unused production nodes to sink
for prod_node in SAMPLE_PRODUCTION_NODES:
    edges.append(
        Edge(
            f"unused_prod_{prod_node}",
            "sink",
            1,
            cost=0,
        )
    )

edges_seen = set()
edges_deduped = []
for edge in edges:
    source = edge.source
    destination = edge.destination
    edge_var = f"{source} -> {destination}"
    if edge_var in edges_seen:
        continue
    edges_deduped.append(edge)
    edges_seen.add(edge_var)
edges = edges_deduped

nodes = []
seen_nodes = set()
for edge in edges:
    if edge.destination not in seen_nodes:
        seen_nodes.add(edge.destination)
        nodes.append(Node(node=edge.destination, cost=edge.cost))

# Sink node with demand set to number of production nodes
sink_demand = num_production_nodes

# This will be the max cost constraint in the MILP model.
max_cost = 50

save_sample_empire_to_json(
    edges,
    nodes,
    SAMPLE_PRODUCTION_NODES,
    sink_demand,
    SAMPLE_CITY_NODES,
    SAMPLE_MAX_WORKERS,
    num_production_nodes,
    max_cost,
)

# Print edges for verification
for edge in edges:
    print(edge)
print("Total edges:", len(edges))
for node in nodes:
    print(node)
print(f"Total nodes {len(nodes)}")


## Visualize
# import networkx as nx
# G = nx.DiGraph()
# for edge in edges:
#     G.add_edge(
#         edge.source, edge.destination, capacity=edge.capacity, cost=edge.cost, value=edge.value
#     )
# nx.write_gml(G, "node_empire.gml")
# print("Graph exported to sample_node_empire.gml")
