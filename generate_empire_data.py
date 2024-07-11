"""Empire Data Generation
#
# * The model is based on flow conservation where the source outbound load == sink inbound load.
# * All nodes except source have the same "base" inbount constraints.
# * All nodes except sink have the same "base" outbound constraints.
# * Every edge has a reverse edge with one way flows denoted with a reverse edge capacity of 0.
# * All edges share the same "base" inbound and outbound constraints.
# * All edges and nodes have a capacity which the calculated load must not exceed.
#

## Sample of generated edges.
# Source -> Demand
#   Edge(source -> demand_plantzone_439_at_warehouse_107, cap: 1, load: 0)
#   Edge(demand_plantzone_439_at_warehouse_107 -> source, cap: 0, load: 0)
# Demand -> Origin
#   Edge(demand_plantzone_855_at_warehouse_229 -> origin_855, cap: 1, load: 0)
#   Edge(origin_855 -> demand_plantzone_855_at_warehouse_229, cap: 0, load: 0)
# Origin -> Waypoint
#   Edge(origin_1553 -> waypoint_1336, cap: 1, load: 0)
#   Edge(waypoint_1336 -> origin_1553, cap: 0, load: 0)
# Waypoint -> Waypoint
#   Edge(waypoint_1133 -> waypoint_1154, cap: 274, load: 0)
#   Edge(waypoint_1154 -> waypoint_1133, cap: 274, load: 0)
# Origin -> Town
#   Edge(origin_1827 -> town_1795, cap: 1, load: 0)
#   Edge(town_1795 -> origin_1827, cap: 0, load: 0)
# Waypoint -> Town
#   Edge(waypoint_1013 -> town_1002, cap: 274, load: 0)
#   Edge(town_1002 -> waypoint_1013, cap: 274, load: 0)
# Town -> Town
#   Edge(town_1781 -> town_1795, cap: 274, load: 0)
#   Edge(town_1795 -> town_1781, cap: 274, load: 0)
# Town -> Warehouse
#   Edge(town_601 -> warehouse_77, cap: 94, load: 0)
#   Edge(warehouse_77 -> town_601, cap: 0, load: 0)
# Warehouse -> Lodging
#   Edge(warehouse_221 -> lodging_221_for_6, cap: 6, load: 0)
#   Edge(lodging_221_for_6 -> warehouse_221, cap: 0, load: 0)
# Lodging -> Sink
#   Edge(lodging_5_for_2 -> sink, cap: 2, load: 0)
#   Edge(sink -> lodging_202_for_3, cap: 0, load: 0)

# Node: source
#  - capacity: num_production_nodes
#  - cost: 0
#  - load: num_production_nodes
#  - value: 0
#
# Edges: flow source to plantzone_{node} for town_{town}
#  - capacity: 1
#  - load: continuous calculated
#  * reverse capacity: 0
#
# Nodes: plantzone_{node}
#  - capacity: 1
#  - cost: 0
#  - load: continuous calculated
#  - value: >= 0
#
# Edges: flow_demand_plantzone_{node}_at_warehouse_{town}_to_plantzone_{plantzone}
#  - capacity: 1
#  - load: continuous calculated
#  * reverse capacity: 0
#
# Nodes: waypoint_{waypoint}
#  - capacity: 1
#  - cost: >= 0
#  - load: continuous calculated
#  - value: 0
#
# **waypoint Interconnects**
# Edges: flow_waypoint_{waypoint}_to_waypoint_{waypoint}
#  - capacity: num_production_nodes
#  - load: continuous calculated
#  * reverse capacity: num_production_nodes
#
# **waypoint Exits**
# Edges: flow_waypoint_{waypoint}_to_warehouse_{town}
#  - capacity: num_production_nodes
#  - load: continuous calculated
#  * reverse capacity: 0
#
# Nodes: warehouse_{warehouse}
#  - capacity: data["town_max_workers"][town]
#  - cost: 0
#  - load: continuous calculated
#  - value: 0
#  * constrain to a single outbound flow_warehouse_{town}_to_lodging_{town}
#
# Edges: flow_warehouse_{town}_to_lodging_{town}
#  - capacity: equal to capacity of destination lodging_{town} node.
#  - load: continuous calculated
#  * reverse capacity: 0
#
# Nodes: lodging_{town}
#  - capacity: equal to capacity of inbound edge
#  - cost: >= data["town_lodging"][town][capacity]
#  - load: continuous calculated
#  - value: 0
#
# Edges: flow_lodging_{town}_to_sink
#  - capacity: num_prodction_nodes
#  - load: continuous calculated
#  * reverse capacity: 0
#
# Node: sink
#  - capacity: num_production_nodes
#  - cost: 0
#  - load: continuous calculated
#  - value: 0
"""

import json

from file_utils import read_workerman_json, read_user_json, write_user_json


class Node:
    def __init__(self, id, capacity, cost=0, value=0):
        self.id = id
        self.capacity = capacity
        self.cost = cost
        self.value = value * 1_000_000

    def __repr__(self):
        return (
            f"Node(id: {self.id}, capacity: {self.capacity}, cost: {self.cost}, value: {self.value})"
        )

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash((self.id))


class Edge:
    def __init__(self, source, destination, capacity, cost=0, value=0):
        self.source = source
        self.destination = destination
        self.capacity = capacity

    def __repr__(self):
        return f"Edge({self.source} -> {self.destination}, cap: {self.capacity})"

    def __eq__(self, other):
        return (self.source, self.destination) == (other.source, other.destination)

    def __hash__(self):
        return hash((self.source + self.destination))


def town_waypoint_to_town_warehouse(id):
    return str(read_workerman_json("town_node_translate.json")["tnk2tk"].get(str(id), 0))


def town_warehouse_to_town_waypoint(id):
    return str(read_workerman_json("town_node_translate.json")["tk2tnk"].get(str(id), 0))


def waypoint_is_town(id):
    return town_waypoint_to_town_warehouse(id) != 0


def generate_source_to_demand_data(nodes, edges, plantzone_values_at_towns):
    """Add source node and per town demand valued plantzone nodes and edges."""
    town_warehouses = plantzone_values_at_towns.keys()
    num_production_nodes = len(plantzone_values_at_towns[list(town_warehouses)[0]])

    source = "source"
    nodes.add(Node(id=source, capacity=num_production_nodes))

    for town_warehouse in town_warehouses:
        for plantzone, data in plantzone_values_at_towns[town_warehouse].items():
            destination = f"demand_plantzone_{plantzone}_at_warehouse_{town_warehouse}"
            nodes.add(Node(id=destination, capacity=1, value=data["value"]))
            edges.add(Edge(source, destination, capacity=1))
            edges.add(Edge(destination, source, capacity=0))


def generate_demand_to_plantzone_data(nodes, edges, plantzone_values_at_towns):
    """Add plantzone waypoint nodes and demand edges"""
    town_warehouses = plantzone_values_at_towns.keys()
    exploration_node_data = read_workerman_json("exploration.json")

    for town_warehouse in town_warehouses:
        for plantzone in plantzone_values_at_towns[town_warehouse].keys():
            source = f"demand_plantzone_{plantzone}_at_warehouse_{town_warehouse}"

            destination = f"origin_{plantzone}"
            cost = exploration_node_data[plantzone]["CP"]
            nodes.add(Node(id=destination, capacity=1, cost=cost))

            edges.add(Edge(source, destination, capacity=1))
            edges.add(Edge(destination, source, capacity=0))


def generate_waypoint_data(nodes, edges, plantzone_values_at_towns):
    """Add waypoint nodes and edges ignore appending waypoint_plantzone nodes."""
    links = read_workerman_json("deck_links.json")
    exploration_node_data = read_workerman_json("exploration.json")
    town_warehouses = plantzone_values_at_towns.keys()
    town_waypoints = [town_warehouse_to_town_waypoint(warehouse) for warehouse in town_warehouses]
    plantzones = plantzone_values_at_towns[list(town_warehouses)[0]].keys()
    num_production_nodes = len(plantzones)

    for link in links:
        a = str(link[0])
        b = str(link[1])
        a_data = exploration_node_data[a]
        b_data = exploration_node_data[b]

        node_a = None
        node_b = None
        a_is_plantzone = a in plantzones
        b_is_plantzone = b in plantzones
        a_is_town_waypoint = a in town_waypoints
        b_is_town_waypoint = b in town_waypoints

        if a_is_plantzone:
            a = f"origin_{a}"
        elif a_is_town_waypoint:
            a = f"town_{a}"
            node_a = Node(id=a, capacity=num_production_nodes, cost=a_data["CP"])
            nodes.add(node_a)
        else:
            a = f"waypoint_{a}"
            node_a = Node(id=a, capacity=num_production_nodes, cost=a_data["CP"])
            nodes.add(node_a)

        if b_is_plantzone:
            b = f"origin_{b}"
        elif b_is_town_waypoint:
            b = f"town_{b}"
            node_b = Node(id=b, capacity=num_production_nodes, cost=b_data["CP"])
            nodes.add(node_b)
        else:
            b = f"waypoint_{b}"
            node_b = Node(id=b, capacity=num_production_nodes, cost=b_data["CP"])
            nodes.add(node_b)

        if a_is_plantzone:
            edge_a = Edge(a, b, capacity=1)
            edge_b = Edge(b, a, capacity=0)
        elif b_is_plantzone:
            edge_a = Edge(a, b, capacity=0)
            edge_b = Edge(b, a, capacity=1)
        else:
            edge_a = Edge(a, b, capacity=num_production_nodes)
            edge_b = Edge(b, a, capacity=num_production_nodes)

        edges.add(edge_a)
        edges.add(edge_b)


def generate_town_to_warehouse_data(nodes, edges, plantzone_values_at_towns):
    """Add warehouse_{town} nodes and waypoint_{town} to warehouse_{town} edges."""
    town_warehouse_lodging = read_workerman_json("all_lodging_storage.json")
    town_lodging_capacities = {}
    for town_warehouse, lodgings in town_warehouse_lodging.items():
        max_lodging = max([int(k) for k in lodgings.keys()])
        town_lodging_capacities[town_warehouse] = max_lodging

    for town_warehouse in plantzone_values_at_towns.keys():
        town_waypoint = town_warehouse_to_town_waypoint(town_warehouse)
        source = f"town_{town_waypoint}"
        destination = f"warehouse_{town_warehouse}"
        capacity = town_lodging_capacities[town_warehouse]
        nodes.add(Node(id=destination, capacity=capacity))

        edges.add(Edge(source, destination, capacity))
        edges.add(Edge(destination, source, 0))


def generate_warehouse_to_lodging_data(nodes, edges, plantzone_values_at_towns, useP2W):
    """Add per town lodging data."""
    town_lodging = read_workerman_json("all_lodging_storage.json")

    for town_warehouse in plantzone_values_at_towns.keys():
        town_lodging_data = town_lodging[town_warehouse]
        source = f"warehouse_{town_warehouse}"

        for capacity, data in town_lodging_data.items():
            usage_capacity = 1 + (3 * useP2W) + str(int(capacity))
            if town_warehouse in ["1375", "218", "1382"]:
                usage_capacity = "0"

            destination = f"lodging_{town_warehouse}_for_{usage_capacity}"
            nodes.add(Node(id=destination, capacity=int(usage_capacity), cost=data[0]["cost"]))

            edges.add(Edge(source, destination, usage_capacity))
            edges.add(Edge(destination, source, 0))


def generate_lodging_to_sink_data(nodes, edges, plantzone_values_at_towns):
    """Add sink node and per town lodging to sink edge data."""
    town_lodging = read_workerman_json("all_lodging_storage.json")

    sink = "sink"
    nodes.add(Node(id=sink, capacity=len(plantzone_values_at_towns.keys())))

    for town_warehouse in plantzone_values_at_towns.keys():
        town_lodging_data = town_lodging[town_warehouse]
        for capacity, _ in town_lodging_data.items():
            lodging = f"lodging_{town_warehouse}_for_{capacity}"

            edges.add(Edge(lodging, sink, capacity))
            edges.add(Edge(sink, lodging, 0))


def print_sample_nodes(nodes):
    """Print sample nodes."""
    seen_node_prefixes = set()
    for node in nodes:
        prefix = node.id.split("_")[0]
        if prefix in seen_node_prefixes:
            continue
        seen_node_prefixes.add(prefix)
        print(node)


def print_sample_edges(edges):
    """Print sample edges."""
    seen_edge_prefixes = set()
    for edge in edges:
        prefix_pair = (edge.source.split("_")[0], edge.destination.split("_")[0])
        if prefix_pair in seen_edge_prefixes:
            continue
        seen_edge_prefixes.add(prefix_pair)
        print(edge)


def edges_to_json(edges):
    """format edges for json"""
    return [
        {
            "source": edge.source,
            "destination": edge.destination,
            "capacity": edge.capacity,
        }
        for edge in edges
    ]


def nodes_to_json(nodes):
    """format nodes for json"""
    return [
        {"id": node.id, "capacity": node.capacity, "cost": node.cost, "value": node.value}
        for node in nodes
    ]


def main():
    plantzone_values_at_towns = read_user_json("node_values_per_town.json")

    edges = set()
    nodes = set()

    generate_source_to_demand_data(nodes, edges, plantzone_values_at_towns)
    generate_demand_to_plantzone_data(nodes, edges, plantzone_values_at_towns)
    generate_waypoint_data(nodes, edges, plantzone_values_at_towns)
    generate_town_to_warehouse_data(nodes, edges, plantzone_values_at_towns)
    generate_warehouse_to_lodging_data(nodes, edges, plantzone_values_at_towns, useP2W=True)
    generate_lodging_to_sink_data(nodes, edges, plantzone_values_at_towns)

    print_sample_nodes(nodes)
    print_sample_edges(edges)
    print(f"Num edges: {len(edges)}, Num nodes: {len(nodes)}")

    data = {"edges": edges_to_json(edges), "nodes": nodes_to_json(nodes)}
    filepath = write_user_json("full_empire.json", data)
    print(f"Empire data written to {filepath}")


if __name__ == "__main__":
    main()
