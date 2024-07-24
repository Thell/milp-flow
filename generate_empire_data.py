"""Empire Data Generation
Generate the edges and nodes used for MILP optimized node empire solver.
"""

from __future__ import annotations
from enum import IntEnum, auto
from typing import Any, Dict, List, TypedDict

import networkx as nx

from file_utils import read_workerman_json, read_user_json, write_user_json


class GraphData(TypedDict):
    nodes: Dict[str, Node]
    edges: Dict[tuple[str, str], Edge]
    warehouse_nodes: Dict[str, Node]


class NodeType(IntEnum):
    # Control - use all warehouses in LoadForWarehouse from edge.destination
    source = auto()
    # Bottleneck - use since warehouse in LoadForWarehouse
    demand = auto()
    origin = auto()
    # Transit - use nearest_n warehouses in LoadForWarehouse
    waypoint = auto()
    town = auto()
    # Bottleneck - use single warehouse in LoadForWarehouse
    warehouse = auto()
    lodging = auto()
    # Control - use all warehouses in LoadForWarehouse from edge.source
    sink = auto()

    INVALID = auto()

    def is_node_capacity_type(self):
        """These types of nodes have a different capcity than the LoadForWarehouse node."""
        return self in [NodeType.demand, NodeType.origin, NodeType.lodging]

    def __repr__(self):
        return self.name


class Node:
    def __init__(
        self,
        id: str,
        type: NodeType,
        capacity: int,
        min_capacity: int = 0,
        cost: int = 0,
        value: int = 0,
        LoadForWarehouse: List[Node] = [],
    ):
        self.id = id
        self.type = type
        self.capacity = capacity
        self.min_capacity = min_capacity
        self.cost = cost
        self.value = value
        self.LoadForWarehouse = LoadForWarehouse if LoadForWarehouse else []
        self.key = self.name()
        self.inbound_edges: List[Edge] = []
        self.outbound_edges: List[Edge] = []
        self.pulp_vars = {}

    def name(self) -> str:
        if self.type in [NodeType.source, NodeType.sink]:
            return self.id
        return f"{self.type.name}_{self.id}"

    def as_dict(self) -> Dict[str, Any]:
        obj_dict = {
            "key": self.name(),
            "name": self.name(),
            "id": self.id,
            "type": self.type.name.lower(),
            "capacity": self.capacity,
            "min_capacity": self.capacity,
            "cost": self.cost,
            "value": self.value,
            "LoadForWarehouse": [],
            "inbound_edges": [edge.key for edge in self.inbound_edges],
            "outbound_edges": [edge.key for edge in self.outbound_edges],
        }
        for node in self.LoadForWarehouse:
            if node is self:
                obj_dict["LoadForWarehouse"].append("self")
            else:
                obj_dict["LoadForWarehouse"].append(node.name())
        return obj_dict

    def __repr__(self) -> str:
        return f"Node(name: {self.name()}, capacity: {self.capacity}, min_capacity: {self.min_capacity}, cost: {self.cost}, value: {self.value})"

    def __eq__(self, other) -> bool:
        return self.name() == other.name()

    def __hash__(self) -> int:
        return hash((self.name()))


class Edge:
    def __init__(self, source: Node, destination: Node, capacity: int, cost: int = 0):
        self.source = source
        self.destination = destination
        self.capacity = capacity
        self.cost = cost
        self.key = (source.name(), destination.name())
        self.type = (source.type, destination.type)
        self.pulp_vars = {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name(),
            "capacity": self.capacity,
            "type": self.type,
            "source": self.source.name(),
            "destination": self.destination.name(),
        }

    def name(self) -> str:
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
            nodes[edge.source.key].outbound_edges.append(edge)
            nodes[edge.destination.key].inbound_edges.append(edge)

            if edge.destination.type is NodeType.origin:
                assert len(edge.source.LoadForWarehouse) == 1, "LoadForWarehouse count mismatch."
                edge.destination.LoadForWarehouse.extend(edge.source.LoadForWarehouse)
            elif edge.destination.type is NodeType.lodging:
                edge.destination.LoadForWarehouse = [edge.source]


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

    LoadForWarehouse = []
    min_capacity = 0

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
            LoadForWarehouse = [warehouse]
            value = ref_data["origin_values"][origin.id][warehouse.id]["value"]
        case NodeType.origin:
            capacity = 1
            cost = ref_data["waypoint_data"][node_id]["CP"]
            value = 0
        case NodeType.waypoint | NodeType.town:
            capacity = ref_data["waypoint_capacity"]
            cost = ref_data["waypoint_data"][node_id]["CP"]
            value = 0
        case NodeType.warehouse:
            capacity = ref_data["lodging_data"][node_id]["max_capacity"] + ref_data["lodging_bonus"]
            cost = 0
            value = 0
            # MARK
            LoadForWarehouse = []
        case NodeType.lodging:
            capacity = kwargs.get("capacity")
            min_capacity = kwargs.get("min_capacity")
            warehouse = kwargs.get("warehouse")
            cost = kwargs.get("cost")
            assert (
                capacity and (min_capacity is not None) and (cost is not None) and warehouse
            ), "Lodging nodes require 'capacity', 'min_capacity' 'cost' and 'warehouse' kwargs."
            value = 0
            # MARK
            LoadForWarehouse = [warehouse]
        case NodeType.sink:
            capacity = ref_data["max_capacity"]
            cost = 0
            value = 0
        case NodeType.INVALID:
            assert node_type is not NodeType.INVALID, "INVALID node type."
            return  # Unreachable: Stops pyright unbound error reporting.

    node = Node(node_id, node_type, capacity, min_capacity, cost, value, LoadForWarehouse)
    if node.key not in nodes:
        if node.type is NodeType.warehouse:
            node.LoadForWarehouse = [node]
        nodes[node.key] = node

    return nodes[node.key]


def get_reference_data(lodging_bonus, top_n):
    """Read and prepare data from reference json files."""
    ref_data = {
        "lodging_bonus": lodging_bonus,  # There's always at least 1 free lodging.
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


def print_sample_nodes(graph_data: GraphData, detailed: bool = False):
    """Print sample nodes."""
    seen_node_types = set()
    for _, node in graph_data["nodes"].items():
        if node.type in seen_node_types:
            continue
        seen_node_types.add(node.type)
        print(node)
        if detailed and node.type not in [NodeType.source, NodeType.sink]:
            print(node.as_dict())
            print()


def print_sample_edges(graph_data: GraphData, detailed: bool = False):
    """Print sample edges."""
    seen_edge_prefixes = set()
    for _, edge in graph_data["edges"].items():
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

    min_capacity = 0
    for capacity, lodging_data in lodging_data.items():
        if capacity == "max_capacity":
            continue
        max_capacity = int(capacity) + ref_data["lodging_bonus"]
        lodging_node = get_node(
            nodes,
            f"{warehouse_node.id}_for_{int(capacity) + lodging_bonus}",
            NodeType.lodging,
            ref_data,
            capacity=max_capacity,
            min_capacity=min_capacity,
            cost=lodging_data[0].get("cost"),
            warehouse=warehouse_node,
        )
        min_capacity = max_capacity + 1
        add_edges(nodes, edges, warehouse_node, lodging_node)
        add_edges(nodes, edges, lodging_node, nodes["sink"])


def nearest_n_warehouses(ref_data: Dict[str, Any], graph_data: GraphData, nearest_n: int):
    waypoint_graph = nx.DiGraph()
    for node in graph_data["nodes"].values():
        waypoint_graph.add_node(node.id, type=node.type)

    for edge in graph_data["edges"].values():
        weight = edge.destination.cost
        if "1727" in edge.name():
            weight = 999999
        waypoint_graph.add_edge(edge.source.id, edge.destination.id, weight=weight)

    all_pairs = dict(nx.all_pairs_bellman_ford_path_length(waypoint_graph, weight="weight"))

    nearest_warehouses = {}
    nearest_warehouse_nodes = {}

    for node_id, node in graph_data["nodes"].items():
        if node.type in {NodeType.waypoint, NodeType.town}:
            distances = []
            for warehouse_key, warehouse in graph_data["warehouse_nodes"].items():
                town_id = ref_data["warehouse_to_town"][warehouse.id]
                distances.append((warehouse, all_pairs[node.id][town_id]))
            nearest_warehouses[node_id] = sorted(distances, key=lambda x: x[1])[:nearest_n]
            nearest_warehouse_nodes[node_id] = [w for w, _ in nearest_warehouses[node_id]]

    return nearest_warehouse_nodes


def contract_graph(graph_data: GraphData):
    pass


def generate_empire_data(lodging_bonus, top_n, nearest_n, waypoint_capacity):
    """Generate and return a Dict of Node and Edge objects composing the empire data."""

    edges: Dict[tuple[str, str], Edge] = {}
    nodes: Dict[str, Node] = {}
    ref_data = get_reference_data(lodging_bonus, top_n)
    ref_data["waypoint_capacity"] = waypoint_capacity

    get_node(nodes, "source", NodeType.source, ref_data)
    get_node(nodes, "sink", NodeType.sink, ref_data)
    process_links(nodes, edges, ref_data)

    nodes_dict = dict(sorted(nodes.items(), key=lambda item: item[1].type))
    edges_dict = dict(sorted(edges.items(), key=lambda item: item[1].as_dict()["type"]))
    warehouse_nodes = {k: v for k, v in nodes_dict.items() if v.type == NodeType.warehouse}

    graph_data: GraphData = {
        "nodes": nodes_dict,
        "edges": edges_dict,
        "warehouse_nodes": warehouse_nodes,
    }

    # All warehouse nodes have now been generated, finalize LoadForWarehouse entries
    nearest_warehouses = nearest_n_warehouses(ref_data, graph_data, nearest_n)
    for node in nodes_dict.values():
        if node.type in [NodeType.source, NodeType.sink]:
            node.LoadForWarehouse = [w for w in warehouse_nodes.values()]
        elif node.type in [NodeType.waypoint, NodeType.town]:
            node.LoadForWarehouse = [w for w in nearest_warehouses[node.key]]

    contract_graph(graph_data)

    return graph_data


def main(write=True):
    # n 1 to 24
    graph_data = generate_empire_data(lodging_bonus=4, top_n=2, nearest_n=5, waypoint_capacity=50)

    print_sample_nodes(graph_data, detailed=True)
    print_sample_edges(graph_data, detailed=False)
    print(f"Num edges: {len(graph_data["edges"])}, Num nodes: {len(graph_data["nodes"])}")

    if write:
        data = {
            "edges": [edge.as_dict() for _, edge in graph_data["edges"].items()],
            "nodes": [node.as_dict() for _, node in graph_data["nodes"].items()],
        }

        filepath = write_user_json("full_empire.json", data)
        print(f"Empire data written to {filepath}")


if __name__ == "__main__":
    main()
