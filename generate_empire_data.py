"""Empire Data Generation
Generate the edges and nodes used for MILP optimized node empire solver.
"""

from __future__ import annotations
from enum import IntEnum, auto
from typing import Dict, Any

from file_utils import read_workerman_json, read_user_json, write_user_json


class NodeType(IntEnum):
    source = auto()
    demand = auto()
    origin = auto()
    waypoint = auto()
    town = auto()
    warehouse = auto()
    lodging = auto()
    sink = auto()
    INVALID = auto()

    def __repr__(self):
        return self.name


class Node:
    def __init__(
        self,
        id: str,
        type: NodeType,
        capacity: int,
        cost: int = 0,
        value: int = 0,
        demand_destination_id=None,
    ):
        self.id = id
        self.type = type
        self.capacity = capacity
        self.cost = cost
        self.value = value
        self.demand_destination_id = demand_destination_id
        self.demand_destination_name = (
            f"warehouse_{demand_destination_id}" if demand_destination_id else None
        )
        self.key = self.name()
        self.inbound_edge_ids = []
        self.outbound_edge_ids = []
        self.pulp_vars = {}

    def name(self):
        if self.type in [NodeType.source, NodeType.sink]:
            return self.id
        return f"{self.type.name}_{self.id}"

    def as_dict(self):
        return {
            "key": self.name(),
            "name": self.name(),
            "id": self.id,
            "type": self.type.name.lower(),
            "capacity": self.capacity,
            "cost": self.cost,
            "value": self.value,
            "demand_destination_id": self.demand_destination_id,
            "demand_destination_name": self.demand_destination_name,
            "inbound_edge_ids": self.inbound_edge_ids,
            "outbound_edge_ids": self.outbound_edge_ids,
        }

    def __repr__(self) -> str:
        return f"Node(name: {self.name()}, capacity: {self.capacity}, cost: {self.cost}, value: {self.value})"

    def __eq__(self, other) -> bool:
        return self.name() == other.name()

    def __hash__(self) -> int:
        return hash((self.name()))


class Edge:
    def __init__(self, source: Node, destination: Node, capacity: int, cost: int = 0):
        self.source = source
        self.destination = destination
        self.capacity = capacity
        self.key = (source.name(), destination.name())
        self.type = (source.type, destination.type)
        self.pulp_vars = {}

    def as_dict(self):
        return {
            "key": self.key,
            "name": self.name(),
            "capacity": self.capacity,
            "type": self.type,
            "source": self.source.as_dict(),
            "destination": self.destination.as_dict(),
        }

    def name(self):
        return f"{self.source.name()}_to_{self.destination.name()}"

    def __repr__(self) -> str:
        return f"Edge({self.source.name()} -> {self.destination.name()}, capacity: {self.capacity})"

    def __eq__(self, other) -> bool:
        return (self.source, self.destination) == (other.source, other.destination)

    def __hash__(self) -> int:
        return hash((self.source.name() + self.destination.name()))


def add_edges(nodes: Dict[str, Node], edges: Dict[tuple, Edge], node_a: Node, node_b: Node):
    """Add edges between a and b.

    Edge Types:
    - Source to Origin:
        Edge(source -> origin, capacity: 1)

    - Waypoint Origins:
        Edge(origin -> waypoint, capacity: 1)
        Edge(origin -> town, capacity: 1)

    - Waypoint Interconnects:
        Edge(waypoint <-> waypoint, capacity: max_capacity)
        Edge(waypoint <-> town, capacity: max_capacity)
        Edge(town <-> town, capacity: max_capacity)

    - Waypoint Exits:
        Edge(town -> warehouse, capacity: warehouse_max_capacity)

    - Town to Warehouse:
        Edge(warehouse -> warehouse, capacity: equal to capacity of destination warehouse

    - Warehouse to Sink:
        Edge(warehouse -> sink, cap: equal to capacity of source warehouse
    """
    # A safety measure to ensure edge direction.
    if node_a.type > node_b.type:
        node_a, node_b = node_b, node_a

    edge_configurations = {
        (NodeType.source, NodeType.demand): (1, 0),
        (NodeType.demand, NodeType.origin): (1, 0),
        (NodeType.origin, NodeType.waypoint): (1, 0),
        (NodeType.origin, NodeType.town): (1, 0),
        (NodeType.waypoint, NodeType.waypoint): (node_b.capacity, node_a.capacity),
        (NodeType.waypoint, NodeType.town): (node_b.capacity, node_a.capacity),
        (NodeType.town, NodeType.town): (node_b.capacity, node_a.capacity),
        (NodeType.town, NodeType.warehouse): (node_b.capacity, 0),
        (NodeType.warehouse, NodeType.lodging): (node_b.capacity, 0),
        (NodeType.lodging, NodeType.sink): (node_a.capacity, 0),
    }

    capacity, reverse_capacity = edge_configurations.get((node_a.type, node_b.type), (1, 0))

    edge_a = Edge(node_a, node_b, capacity=capacity)
    edge_b = Edge(node_b, node_a, capacity=reverse_capacity)

    for edge in [edge_a, edge_b]:
        if edge.key not in edges and edge.capacity > 0:
            edges[edge.key] = edge
            nodes[edge.source.key].outbound_edge_ids.append(edge.key)
            nodes[edge.destination.key].inbound_edge_ids.append(edge.key)


def get_link_node_type(node_id: str, ref_data: Dict[str, Any]):
    """Return the NodeType of the given node_id node.

    - NodeType.INVALID indicates a node that is unused and not added to the graph.
    """
    if node_id in ref_data["towns"]:
        return NodeType.town
    if node_id in ref_data["all_plantzones"]:
        if node_id not in ref_data["origins"]:
            return NodeType.INVALID
        return NodeType.origin
    return NodeType.waypoint


def get_link_nodes(nodes, link, ref_data):
    node_a_id, node_b_id = str(link[1]), str(link[0])
    node_a_type = get_link_node_type(node_a_id, ref_data)
    node_b_type = get_link_node_type(node_b_id, ref_data)

    if NodeType.INVALID in [node_a_type, node_b_type]:
        # Not used in the graph because they are beyond the scope of the optimization.
        return (None, None)

    # Ensure edge node order.
    if node_a_type > node_b_type:
        node_a_id, node_b_id = node_b_id, node_a_id
        node_a_type, node_b_type = node_b_type, node_a_type

    return (
        get_node(nodes, node_a_id, node_a_type, ref_data),
        get_node(nodes, node_b_id, node_b_type, ref_data),
    )


def get_node(nodes, node_id: str, node_type: NodeType, ref_data: Dict[str, Any], **kwargs) -> Node:
    """
    Generate, add and return node based on NodeType.

    kwargs `origin` and `warehouse` are required for demand nodes.
    kwargs `capacity` is required for warehouse nodes.
    kwargs `capacity`, `cost` and `warehouse` are required for lodging nodes.
    """

    demand_destination_id = None

    match node_type:
        case NodeType.source:
            capacity = ref_data["max_capacity"]
            cost = 0
            value = 0
        case NodeType.demand:
            origin = kwargs.get("origin")
            warehouse = kwargs.get("warehouse")
            assert warehouse and origin, "Demand nodes require 'warehouse' and 'origin' kwargs."
            isinstance(origin, Node)
            isinstance(warehouse, Node)
            node_id = f"{origin.key}_for_{warehouse.key}"
            capacity = 1
            cost = 0
            demand_destination_id = warehouse.id
            value = ref_data["origin_values"][origin.id][warehouse.id]["value"]
        case NodeType.origin:
            capacity = 1
            cost = ref_data["waypoint_data"][node_id]["CP"]
            value = 0
        case NodeType.waypoint | NodeType.town:
            capacity = ref_data["max_capacity"]
            cost = ref_data["waypoint_data"][node_id]["CP"]
            value = 0
        case NodeType.warehouse:
            capacity = ref_data["lodging_data"][node_id]["max_capacity"] + ref_data["lodging_bonus"]
            cost = 0
            value = 0
            demand_destination_id = node_id
        case NodeType.lodging:
            capacity = kwargs.get("capacity")
            cost = kwargs.get("cost")
            warehouse = kwargs.get("warehouse")
            assert (
                capacity and cost is not None and warehouse
            ), "Lodging nodes require 'capacity', 'cost' and 'warehouse' kwargs."
            value = 0
            demand_destination_id = warehouse.id
        case NodeType.sink:
            capacity = ref_data["max_capacity"]
            cost = 0
            value = 0
        case NodeType.INVALID:
            assert node_type is not NodeType.INVALID, "INVALID node type."
            return  # Unreachable: Stops pyright unbound error reporting.

    node = Node(node_id, node_type, capacity, cost, value, demand_destination_id)
    if node.key not in nodes:
        nodes[node.key] = node
    return nodes[node.key]


def get_reference_data(top_n):
    """Read and prepare data from reference json files."""
    ref_data = {
        "lodging_bonus": 1,
        "top_n_origin_values": top_n,
        "all_plantzones": read_workerman_json("plantzone.json").keys(),
        "lodging_data": read_workerman_json("all_lodging_storage.json"),
        "origin_values": read_user_json("node_values_per_town.json"),
        "town_to_warehouse": read_workerman_json("town_node_translate.json")["tnk2tk"],
        "warehouse_to_town": read_workerman_json("town_node_translate.json")["tk2tnk"],
        "waypoint_data": read_workerman_json("exploration.json"),
        "waypoint_links": read_workerman_json("deck_links.json"),
    }

    origins = ref_data["origin_values"].keys()
    warehouses = ref_data["origin_values"][list(origins)[0]].keys()
    towns = [ref_data["warehouse_to_town"][w] for w in warehouses]

    ref_data["max_capacity"] = len(origins)
    ref_data["origins"] = origins
    ref_data["warehouses"] = warehouses
    ref_data["towns"] = towns

    for warehouse, lodgings in ref_data["lodging_data"].items():
        if warehouse not in ref_data["warehouses"]:
            continue
        max_lodging = ref_data["lodging_bonus"] + max([int(k) for k in lodgings.keys()])
        ref_data["lodging_data"][warehouse]["max_capacity"] = max_lodging

    return ref_data


def print_sample_nodes(nodes: Dict[str, Node], detailed: bool = False):
    """Print sample nodes."""
    seen_node_types = set()
    for _, node in nodes.items():
        if node.type in seen_node_types:
            continue
        seen_node_types.add(node.type)
        print(node)
        if detailed and node.type not in [NodeType.source, NodeType.sink]:
            print(node.as_dict())
            print()


def print_sample_edges(edges: Dict[tuple, Edge], detailed: bool = False):
    """Print sample edges."""
    seen_edge_prefixes = set()
    for _, edge in edges.items():
        prefix_pair = (edge.source.type.value, edge.destination.type.value)
        if prefix_pair in seen_edge_prefixes:
            continue
        seen_edge_prefixes.add(prefix_pair)
        print(edge)
        if (
            detailed
            and edge.source.type != NodeType.source
            and edge.destination.type != NodeType.sink
        ):
            print(edge.as_dict())
            print()


def process_links(nodes: Dict[str, Node], edges: Dict[tuple, Edge], ref_data: Dict[str, Any]):
    """Process all waypoint links and add the nodes and edges to the graph.

    Calls handlers for origin and town nodes to add origin value nodes and
    warehouse/lodging nodes with their respective source and sink edges.
    """
    for link in ref_data["waypoint_links"]:
        source, destination = get_link_nodes(nodes, link, ref_data)
        if source is None or destination is None:
            continue

        add_edges(nodes, edges, source, destination)

        if source.type == NodeType.origin:
            process_origin(nodes, edges, source, ref_data)
        if destination.type == NodeType.town:
            process_town(nodes, edges, destination, ref_data)


def process_origin(
    nodes: Dict[str, Node], edges: Dict[tuple, Edge], origin: Node, ref_data: Dict[str, Any]
):
    """Add origin demand value nodes and edges between the source and origin nodes."""
    for i, (warehouse_id, value_data) in enumerate(ref_data["origin_values"][origin.id].items()):
        if i > ref_data["top_n_origin_values"]:
            return
        warehouse = get_node(nodes, warehouse_id, NodeType.warehouse, ref_data)
        if value_data["value"] == 0:
            continue

        demand_node = get_node(
            nodes,
            "",
            NodeType.demand,
            ref_data,
            origin=origin,
            warehouse=warehouse,
        )
        add_edges(nodes, edges, nodes["source"], demand_node)
        add_edges(nodes, edges, demand_node, origin)


def process_town(
    nodes: Dict[str, Node], edges: Dict[tuple, Edge], town: Node, ref_data: Dict[str, Any]
):
    """Add town warehouse and lodging nodes and edges between the town and sink nodes."""
    warehouse_id = ref_data["town_to_warehouse"][town.id]
    lodging_data = ref_data["lodging_data"][warehouse_id]
    lodging_bonus = ref_data["lodging_bonus"]

    max_capacity = lodging_data["max_capacity"] + lodging_bonus
    warehouse_node = get_node(
        nodes, warehouse_id, NodeType.warehouse, ref_data, capacity=max_capacity
    )
    add_edges(nodes, edges, town, warehouse_node)

    for capacity, lodging_data in lodging_data.items():
        if capacity == "max_capacity":
            continue
        lodging_node = get_node(
            nodes,
            f"{warehouse_node.id}_for_{int(capacity) + lodging_bonus}",
            NodeType.lodging,
            ref_data,
            capacity=int(capacity) + ref_data["lodging_bonus"],
            cost=lodging_data[0].get("cost"),
            warehouse=warehouse_node,
        )
        add_edges(nodes, edges, warehouse_node, lodging_node)
        add_edges(nodes, edges, lodging_node, nodes["sink"])


def generate_empire_data(top_n):
    """Generate and return a Dict of Node and Edge objects composing the empire data."""

    edges: Dict[tuple, Edge] = {}
    nodes: Dict[str, Node] = {}
    ref_data = get_reference_data(top_n)

    get_node(nodes, "source", NodeType.source, ref_data)
    get_node(nodes, "sink", NodeType.sink, ref_data)
    process_links(nodes, edges, ref_data)
    # populate_node_edge_links(nodes, edges)

    nodes_dict = dict(sorted(nodes.items(), key=lambda item: item[1].type))
    edges_dict = dict(sorted(edges.items(), key=lambda item: item[1].as_dict()["type"]))
    warehouse_nodes = {k: v for k, v in nodes_dict.items() if v.type == NodeType.warehouse}

    nullLoadTypes = set()
    for obj in nodes_dict.values():
        if obj.demand_destination_id is None:
            nullLoadTypes.add(obj.type)
    print(nullLoadTypes)

    data = {
        "nodes": nodes_dict,
        "warehouse_nodes": warehouse_nodes,
        "edges": edges_dict,
    }

    return data


def main(write=True):
    data = generate_empire_data(top_n=1)  # top_n 1 to 24

    print_sample_nodes(data["nodes"], detailed=False)
    print_sample_edges(data["edges"], detailed=False)
    print(f"Num edges: {len(data["edges"])}, Num nodes: {len(data["nodes"])}")

    if write:
        data = {
            "edges": [edge.as_dict() for _, edge in data["edges"].items()],
            "nodes": [node.as_dict() for _, node in data["nodes"].items()],
        }

        filepath = write_user_json("full_empire.json", data)
        print(f"Empire data written to {filepath}")


if __name__ == "__main__":
    main()
