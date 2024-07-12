"""Empire Data Generation
Generate the edges and nodes used for MILP optimized node empire solver.

* The model is based on flow conservation where the source outbound load == sink inbound load.
* All nodes except source have the same "base" inbount constraints.
* All nodes except sink have the same "base" outbound constraints.
* Every edge has a reverse edge with one way flows denoted with a reverse edge capacity of 0.
* All edges share the same "base" inbound and outbound constraints.
* All edges and nodes have a capacity which the calculated load must not exceed.
"""

from enum import Enum
from typing import Set, Dict, Any

from file_utils import read_workerman_json, read_user_json, write_user_json


class NodeType(Enum):
    SOURCE = "source"
    DEMAND = "demand"
    ORIGIN = "origin"
    WAYPOINT = "waypoint"
    TOWN = "town"
    WAREHOUSE = "warehouse"
    LODGING = "lodging"
    SINK = "sink"

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


class Node:
    def __init__(self, id: str, type: NodeType, capacity: int, cost: int = 0, value: int = 0):
        self.id = id
        self.type = type
        self.capacity = capacity
        self.cost = cost
        self.value = value

    def name(self):
        if self.type in [NodeType.SOURCE, NodeType.SINK]:
            return self.id
        return f"{self.type}_{self.id}"

    def __repr__(self) -> str:
        return f"Node(name: {self.name()}, capacity: {self.capacity}, cost: {self.cost}, value: {self.value})"

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __hash__(self) -> int:
        return hash((self.id))


class Edge:
    def __init__(self, source: Node, destination: Node, capacity: int, cost: int = 0):
        self.source = source
        self.destination = destination
        self.capacity = capacity

    def __repr__(self) -> str:
        return f"Edge({self.source.name()} -> {self.destination.name()}, capacity: {self.capacity})"

    def __eq__(self, other) -> bool:
        return (self.source, self.destination) == (other.source, other.destination)

    def __hash__(self) -> int:
        return hash((self.source.name() + self.destination.name()))


def add_edges(edges: Set[Edge], node_a: Node, node_b: Node, ref_data: Dict[str, Any]):
    """Add edges between a and b.

    Edge Types:
    - Source to Demand:
        Edge(source -> demand_plantzone, capacity: 1)
        Edge(demand_plantzone -> source, capacity: 0)

    Demand to Origin:
        Edge(demand_plantzone -> origin, capacity: 1)
        Edge(origin -> demand_plantzone, capacity: 0)

    Waypoint Origins:
        Edge(origin -> waypoint, capacity: 1)
        Edge(waypoint -> origin, capacity: 0)

        Edge(origin -> town, capacity: 1)
        Edge(town -> origin, capacity: 0)

    Waypoint Interconnects:
        Edge(waypoint -> waypoint, capacity: max_capacity)
        Edge(waypoint -> town, capacity: max_capacity)
        Edge(town -> waypoint, capacity: max_capacity)
        Edge(town -> town, capacity: max_capacity)

    Waypoint Exits:
        Edge(town -> warehouse, capacity: warehouse_max_capacity)
        Edge(warehouse -> town, capacity: 0)

    Warehouse to Lodging:
        Edge(warehouse -> lodging, capacity: equal to capacity of destination lodging_{warehouse})
        Edge(lodging -> warehouse, capacity: 0)

    Lodging to Sink:
        Edge(lodging -> sink, cap: equal to capacity of source lodging_{warehouse})
        Edge(sink -> lodging, capacity: 0)
    """
    max_capacity = ref_data["max_capacity"]

    edge_configurations = {
        (NodeType.SOURCE, NodeType.DEMAND): (1, 0),
        (NodeType.DEMAND, NodeType.SOURCE): (0, 1),
        (NodeType.DEMAND, NodeType.ORIGIN): (1, 0),
        (NodeType.ORIGIN, NodeType.DEMAND): (0, 1),
        (NodeType.ORIGIN, NodeType.WAYPOINT): (1, 0),
        (NodeType.WAYPOINT, NodeType.ORIGIN): (0, 1),
        (NodeType.ORIGIN, NodeType.TOWN): (1, 0),
        (NodeType.TOWN, NodeType.ORIGIN): (0, 1),
        (NodeType.WAYPOINT, NodeType.WAYPOINT): (max_capacity, max_capacity),
        (NodeType.WAYPOINT, NodeType.TOWN): (max_capacity, max_capacity),
        (NodeType.TOWN, NodeType.WAYPOINT): (max_capacity, max_capacity),
        (NodeType.TOWN, NodeType.TOWN): (max_capacity, max_capacity),
        (NodeType.TOWN, NodeType.WAREHOUSE): (node_b.capacity, 0),
        (NodeType.WAREHOUSE, NodeType.TOWN): (0, node_a.capacity),
        (NodeType.WAREHOUSE, NodeType.LODGING): (node_b.capacity, 0),
        (NodeType.LODGING, NodeType.WAREHOUSE): (0, node_a.capacity),
        (NodeType.LODGING, NodeType.SINK): (node_a.capacity, 0),
        (NodeType.SINK, NodeType.LODGING): (0, node_b.capacity),
    }

    capacity, reverse_capacity = edge_configurations.get((node_a.type, node_b.type), (1, 0))

    edge_a = Edge(node_a, node_b, capacity=capacity)
    edge_b = Edge(node_b, node_a, capacity=reverse_capacity)

    edges.add(edge_a)
    edges.add(edge_b)


def get_node(nodes, node_id: str, node_type: NodeType, ref_data: Dict[str, Any], **kwargs) -> Node:
    """
    Generate, add and return node based on NodeType.

    kwargs `plantzone` and `warehouse` are required for demand nodes.
    kwargs `capacity` is required for warehouse nodes.
    kwargs `capacity` and `cost` are required for lodging nodes.
    """

    match node_type:
        case NodeType.SOURCE:
            capacity, cost, value = ref_data["max_capacity"], 0, 0
        case NodeType.DEMAND:
            warehouse = kwargs.get("warehouse")
            plantzone = kwargs.get("plantzone")
            assert (
                warehouse and plantzone
            ), "Demand nodes require 'warehouse' and 'plantzone' kwargs."
            capacity = 1
            cost = 0
            value = (
                ref_data["plantzone_data"][warehouse][plantzone]["value"]
                if warehouse and plantzone
                else 0
            )
        case NodeType.ORIGIN:
            capacity = 1
            cost = ref_data["waypoint_data"][node_id]["CP"]
            value = 0
        case NodeType.WAYPOINT:
            capacity = ref_data["max_capacity"]
            cost = ref_data["waypoint_data"][node_id]["CP"]
            value = 0
        case NodeType.TOWN:
            capacity = ref_data["max_capacity"]
            cost = ref_data["waypoint_data"][node_id]["CP"]
            value = 0
        case NodeType.WAREHOUSE:
            capacity = kwargs.get("capacity")
            assert capacity is not None, "Warehouse nodes require 'capacity' kwargs."
            cost = 0
            value = 0
        case NodeType.LODGING:
            capacity = kwargs.get("capacity")
            cost = kwargs.get("cost")
            assert (
                capacity is not None and cost is not None
            ), "Lodging nodes require 'capacity' and 'cost' kwargs."
            value = 0
        case NodeType.SINK:
            capacity = ref_data["max_capacity"]
            cost = 0
            value = 0

    node = Node(node_id, node_type, capacity, cost, value)
    nodes.add(node)
    return node


def get_demand_node(nodes, plantzone, warehouse, ref_data):
    """"""
    return get_node(
        nodes,
        f"plantzone_{plantzone}_at_warehouse_{warehouse}",
        NodeType.DEMAND,
        ref_data,
        plantzone=plantzone,
        warehouse=warehouse,
    )


def get_lodging_node(nodes, warehouse, capacity, cost, ref_data):
    usage_capacity = int(capacity) + ref_data["lodging_bonus"]
    if warehouse in ["1375", "218", "1382"]:
        usage_capacity = 0

    return get_node(
        nodes,
        f"{warehouse}_for_{usage_capacity}",
        NodeType.LODGING,
        ref_data,
        capacity=usage_capacity,
        cost=cost,
    )


def get_waypoint_node(nodes, node_id, ref_data):
    if node_id in ref_data["plantzones"]:
        return get_node(nodes, node_id, NodeType.ORIGIN, ref_data)
    elif node_id in ref_data["towns"]:
        return get_node(nodes, node_id, NodeType.TOWN, ref_data)
    else:
        return get_node(nodes, node_id, NodeType.WAYPOINT, ref_data)


def get_reference_data():
    """Read and prepare data from reference json files."""

    ref_data = {
        "lodging_bonus": 4,
        "lodging_data": read_workerman_json("all_lodging_storage.json"),
        "waypoint_data": read_workerman_json("exploration.json"),
        "plantzone_data": read_user_json("node_values_per_town.json"),
        "warehouse_to_town_data": read_workerman_json("town_node_translate.json")["tk2tnk"],
    }
    ref_data["warehouses"] = ref_data["plantzone_data"].keys()
    ref_data["plantzones"] = ref_data["plantzone_data"][list(ref_data["warehouses"])[0]].keys()
    ref_data["towns"] = [ref_data["warehouse_to_town_data"][w] for w in ref_data["warehouses"]]
    ref_data["max_capacity"] = len(ref_data["plantzones"])
    ref_data["warehouse_max_capacity"] = generate_warehouse_capacities(ref_data)
    return ref_data


def generate_warehouse_capacities(ref_data: Dict[str, Any]):
    warehouse_capacities = {}
    for warehouse, lodgings in ref_data["lodging_data"].items():
        if warehouse not in ref_data["warehouses"]:
            continue
        max_lodging = ref_data["lodging_bonus"] + max([int(k) for k in lodgings.keys()])
        warehouse_capacities[warehouse] = max_lodging
    return warehouse_capacities


def generate_source_to_demand_data(nodes: Set[Node], edges: Set[Edge], ref_data: Dict[str, Any]):
    """Add source node and per town demand valued plantzone nodes and edges."""

    source = get_node(nodes, "source", NodeType.SOURCE, ref_data)
    for warehouse in ref_data["warehouses"]:
        for plantzone in ref_data["plantzone_data"][warehouse].keys():
            destination = get_demand_node(nodes, plantzone, warehouse, ref_data)
            add_edges(edges, source, destination, ref_data)


def generate_demand_to_plantzone_data(nodes: Set[Node], edges: Set[Edge], ref_data: Dict[str, Any]):
    """Add plantzone waypoint nodes and demand edges"""

    for warehouse in ref_data["warehouses"]:
        for plantzone in ref_data["plantzone_data"][warehouse].keys():
            source = get_demand_node(nodes, plantzone, warehouse, ref_data)
            destination = get_node(nodes, plantzone, NodeType.ORIGIN, ref_data)
            add_edges(edges, source, destination, ref_data)


def generate_waypoint_data(nodes: Set[Node], edges: Set[Edge], ref_data: Dict[str, Any]):
    """Add waypoint nodes and edges ignore appending waypoint plantzone nodes."""

    for link in read_workerman_json("deck_links.json"):
        a, b = str(link[0]), str(link[1])
        node_a = get_waypoint_node(nodes, a, ref_data)
        node_b = get_waypoint_node(nodes, b, ref_data)
        add_edges(edges, node_a, node_b, ref_data)


def generate_town_to_warehouse_data(nodes: Set[Node], edges: Set[Edge], ref_data: Dict[str, Any]):
    """Add warehouse_{town} nodes and waypoint_{town} to warehouse_{town} edges."""

    for warehouse, capacity in ref_data["warehouse_max_capacity"].items():
        town = ref_data["warehouse_to_town_data"][warehouse]
        source = get_node(nodes, town, NodeType.TOWN, ref_data)
        destination = get_node(nodes, warehouse, NodeType.WAREHOUSE, ref_data, capacity=capacity)
        add_edges(edges, source, destination, ref_data)


def generate_warehouse_to_lodging_data(nodes: Set[Node], edges: Set[Edge], ref_data: Dict[str, Any]):
    """Add per town lodging data."""

    for warehouse in ref_data["warehouses"]:
        warehouse_capacity = ref_data["warehouse_max_capacity"][warehouse]
        source = get_node(
            nodes, warehouse, NodeType.WAREHOUSE, ref_data, capacity=warehouse_capacity
        )

        for lodging_capacity, data in ref_data["lodging_data"][warehouse].items():
            cost = data[0].get("cost")
            destination = get_lodging_node(nodes, warehouse, lodging_capacity, cost, ref_data)
            add_edges(edges, source, destination, ref_data)


def generate_lodging_to_sink_data(nodes: Set[Node], edges: Set[Edge], ref_data: Dict[str, Any]):
    """Add sink node and per town lodging to sink edge data."""

    destination = get_node(nodes, "sink", NodeType.SINK, ref_data)
    for warehouse in ref_data["warehouses"]:
        for lodging_capacity, data in ref_data["lodging_data"][warehouse].items():
            cost = data[0].get("cost")
            source = get_lodging_node(nodes, warehouse, lodging_capacity, cost, ref_data)
            add_edges(edges, source, destination, ref_data)


def print_sample_nodes(nodes: Set[Node]):
    """Print sample nodes."""

    seen_node_prefixes = set()
    for node in nodes:
        prefix = node.name().split("_")[0]
        if prefix in seen_node_prefixes:
            continue
        seen_node_prefixes.add(prefix)
        print(node)


def print_sample_edges(edges: Set[Edge]):
    """Print sample edges."""
    seen_edge_prefixes = set()
    for edge in edges:
        prefix_pair = (edge.source.type.value, edge.destination.type.value)
        if prefix_pair in seen_edge_prefixes:
            continue
        seen_edge_prefixes.add(prefix_pair)
        print(edge)


def edges_to_json(edges: Set[Edge]):
    """format edges for json"""
    data = [
        {
            "source": edge.source.name(),
            "destination": edge.destination.name(),
            "capacity": edge.capacity,
        }
        for edge in edges
    ]
    return sorted(data, key=lambda edge: (edge["source"], edge["destination"]))


def nodes_to_json(nodes: Set[Node]):
    """format nodes for json"""
    data = [
        {"id": node.id, "capacity": node.capacity, "cost": node.cost, "value": node.value}
        for node in nodes
    ]
    return sorted(data, key=lambda node: node["id"])


def main():
    ref_data = get_reference_data()

    edges = set()
    nodes = set()

    generate_source_to_demand_data(nodes, edges, ref_data)
    generate_demand_to_plantzone_data(nodes, edges, ref_data)
    generate_waypoint_data(nodes, edges, ref_data)
    generate_town_to_warehouse_data(nodes, edges, ref_data)
    generate_warehouse_to_lodging_data(nodes, edges, ref_data)
    generate_lodging_to_sink_data(nodes, edges, ref_data)

    print_sample_nodes(nodes)
    print_sample_edges(edges)
    print(f"Num edges: {len(edges)}, Num nodes: {len(nodes)}")

    data = {"edges": edges_to_json(edges), "nodes": nodes_to_json(nodes)}
    filepath = write_user_json("full_empire.json", data)
    print(f"Empire data written to {filepath}")


if __name__ == "__main__":
    main()
