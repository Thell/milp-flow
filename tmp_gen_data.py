"""Empire Data Generation
Generate the arcs and nodes used for MILP optimized node empire solver.
"""

from __future__ import annotations
from enum import IntEnum, auto
import json
import logging
from typing import Any, Dict, List, TypedDict

import networkx as nx

from file_utils import read_workerman_json, read_user_json, write_user_json

logger = logging.getLogger(__name__)


class GraphData(TypedDict):
    V: Dict[str, Node]
    E: Dict[tuple[str, str], Arc]
    R: Dict[str, Node]


class NodeType(IntEnum):
    𝓢 = auto()
    𝒕 = auto()
    waypoint = auto()
    town = auto()
    𝓡 = auto()
    lodging = auto()
    𝓣 = auto()

    INVALID = auto()

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
        𝓻: List[Node] = [],
    ):
        self.id = id
        self.type = type
        self.𝓬 = capacity
        self.min_capacity = min_capacity
        self.cost = cost
        self.𝓻_prizes: Dict[str, Dict[str, Any]] = {}
        self.𝓻 = 𝓻 if 𝓻 else []
        self.key = self.name()
        self.inbound_arcs: List[Arc] = []
        self.outbound_arcs: List[Arc] = []
        self.pulp_vars = {}
        self.isTerminal = type == NodeType.t

    def name(self) -> str:
        if self.type in [NodeType.𝓢, NodeType.𝓣]:
            return self.id
        return f"{self.type.name}_{self.id}"

    def as_dict(self) -> Dict[str, Any]:
        obj_dict = {
            "key": self.name(),
            "name": self.name(),
            "id": self.id,
            "type": self.type.name.lower(),
            "capacity": self.𝓬,
            "min_capacity": self.𝓬,
            "cost": self.cost,
            "root_values": self.𝓻_prizes,
            "LoadForRoot": [],
            "inbound_arcs": [arc.key for arc in self.inbound_arcs],
            "outbound_arcs": [arc.key for arc in self.outbound_arcs],
        }
        for node in self.𝓻:
            if node is self:
                obj_dict["LoadForRoot"].append("self")
            else:
                obj_dict["LoadForRoot"].append(node.name())
        return obj_dict

    def __repr__(self) -> str:
        return f"Node(name: {self.name()}, capacity: {self.𝓬}, min_capacity: {self.min_capacity}, cost: {self.cost}, value: {self.𝓻_prizes})"

    def __eq__(self, other) -> bool:
        return self.name() == other.name()

    def __hash__(self) -> int:
        return hash((self.name()))


class Arc:
    def __init__(self, source: Node, destination: Node, capacity: int, cost: int = 0):
        self.source = source
        self.destination = destination
        self.𝓬 = capacity
        self.cost = cost
        self.key = (source.name(), destination.name())
        self.type = (source.type, destination.type)
        self.pulp_vars = {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name(),
            "capacity": self.𝓬,
            "type": self.type,
            "source": self.source.name(),
            "destination": self.destination.name(),
        }

    def allowable_loads(self) -> set[Node]:
        return set(self.source.𝓻).intersection(set(self.destination.𝓻))

    def name(self) -> str:
        return f"{self.source.name()}_to_{self.destination.name()}"

    def __repr__(self) -> str:
        return f"arc({self.source.name()} -> {self.destination.name()}, capacity: {self.𝓬})"

    def __eq__(self, other) -> bool:
        return (self.source, self.destination) == (other.source, other.destination)

    def __hash__(self) -> int:
        return hash((self.source.name() + self.destination.name()))


def add_arcs(nodes: Dict[str, Node], arcs: Dict[tuple, Arc], node_a: Node, node_b: Node):
    """Add arcs between a and b."""
    # A safety measure to ensure arc direction.
    if node_a.type > node_b.type:
        node_a, node_b = node_b, node_a

    arc_configurations = {
        (NodeType.𝓢, NodeType.𝒕): (1, 0),
        (NodeType.𝒕, NodeType.waypoint): (1, 0),
        (NodeType.𝒕, NodeType.town): (1, 0),
        (NodeType.waypoint, NodeType.waypoint): (node_b.𝓬, node_a.𝓬),
        (NodeType.waypoint, NodeType.town): (node_b.𝓬, node_a.𝓬),
        (NodeType.town, NodeType.town): (node_b.𝓬, node_a.𝓬),
        (NodeType.town, NodeType.𝓡): (node_b.𝓬, 0),
        (NodeType.𝓡, NodeType.lodging): (node_b.𝓬, 0),
        (NodeType.lodging, NodeType.𝓣): (node_a.𝓬, 0),
    }

    capacity, reverse_capacity = arc_configurations.get((node_a.type, node_b.type), (1, 0))

    arc_a = Arc(node_a, node_b, capacity=capacity)
    arc_b = Arc(node_b, node_a, capacity=reverse_capacity)

    for arc in [arc_a, arc_b]:
        if arc.key not in arcs and arc.𝓬 > 0:
            arcs[arc.key] = arc
            nodes[arc.source.key].outbound_arcs.append(arc)
            nodes[arc.destination.key].inbound_arcs.append(arc)

            if arc.destination.type is NodeType.lodging:
                arc.destination.𝓻 = [arc.source]


def get_link_node_type(node_id: str, ref_data: Dict[str, Any]):
    """Return the NodeType of the given node_id node.

    - NodeType.INVALID indicates a node that is unused and not added to the graph.
    """
    if node_id in ref_data["towns"]:
        return NodeType.town
    if node_id in ref_data["all_plantzones"]:
        if node_id not in ref_data["terminals"]:
            return NodeType.INVALID
        return NodeType.𝒕
    return NodeType.waypoint


def get_link_nodes(nodes, link, ref_data):
    node_a_id, node_b_id = str(link[1]), str(link[0])
    node_a_type = get_link_node_type(node_a_id, ref_data)
    node_b_type = get_link_node_type(node_b_id, ref_data)

    if NodeType.INVALID in [node_a_type, node_b_type]:
        # Not used in the graph because they are beyond the scope of the optimization.
        return (None, None)

    # Ensure arc node order.
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

    kwargs `terminal` and `root` are required for supply nodes.
    kwargs `capacity` is required for root nodes.
    kwargs `capacity`, `cost` and `root` are required for lodging nodes.
    """

    LoadForRoot = []
    min_capacity = 0

    match node_type:
        case NodeType.𝓢:
            capacity = ref_data["max_capacity"]
            cost = 0
        case NodeType.𝒕:
            capacity = 1
            cost = ref_data["waypoint_data"][node_id]["CP"]
        case NodeType.waypoint | NodeType.town:
            capacity = ref_data["waypoint_capacity"]
            cost = ref_data["waypoint_data"][node_id]["CP"]
        case NodeType.𝓡:
            capacity = ref_data["lodging_data"][node_id]["max_capacity"] + ref_data["lodging_bonus"]
            capacity = min(capacity, ref_data["waypoint_capacity"])
            cost = 0
            LoadForRoot = []
        case NodeType.lodging:
            capacity = kwargs.get("capacity")
            min_capacity = kwargs.get("min_capacity")
            root = kwargs.get("root")
            cost = kwargs.get("cost")
            assert (
                capacity and (min_capacity is not None) and (cost is not None) and root
            ), "Lodging nodes require 'capacity', 'min_capacity' 'cost' and 'root' kwargs."
            LoadForRoot = [root]
        case NodeType.𝓣:
            capacity = ref_data["max_capacity"]
            cost = 0
        case NodeType.INVALID:
            assert node_type is not NodeType.INVALID, "INVALID node type."
            return  # Unreachable: Stops pyright unbound error reporting.

    node = Node(node_id, node_type, capacity, min_capacity, cost, LoadForRoot)
    if node.key not in nodes:
        if node.type is NodeType.𝓡:
            node.𝓻 = [node]
        nodes[node.key] = node

    return nodes[node.key]


# def get_reference_data(lodging_bonus, top_n):
def get_reference_data(config):
    """Read and prepare data from reference json files."""
    ref_data = {
        "lodging_bonus": config["lodging_bonus"],
        "top_n_terminal_values": config["top_n"],
        "all_plantzones": read_workerman_json("plantzone.json").keys(),
        "lodging_data": read_workerman_json("all_lodging_storage.json"),
        "terminal_values": read_user_json("node_values_per_town.json"),
        "town_to_root": read_workerman_json("town_node_translate.json")["tnk2tk"],
        "root_to_town": read_workerman_json("town_node_translate.json")["tk2tnk"],
        "root_to_townname": read_workerman_json("warehouse_to_townname.json"),
        "waypoint_data": read_workerman_json("exploration.json"),
        "waypoint_links": read_workerman_json("deck_links.json"),
    }

    terminals = ref_data["terminal_values"].keys()
    roots = ref_data["terminal_values"][list(terminals)[0]].keys()
    towns = [ref_data["root_to_town"][w] for w in roots]

    ref_data["max_capacity"] = len(terminals)
    ref_data["terminals"] = terminals
    ref_data["roots"] = roots
    ref_data["towns"] = towns

    for root, lodgings in ref_data["lodging_data"].items():
        if root not in ref_data["roots"]:
            continue
        max_lodging = 1 + ref_data["lodging_bonus"] + max([int(k) for k in lodgings.keys()])
        ref_data["lodging_data"][root]["max_capacity"] = max_lodging

    return ref_data


def print_sample_nodes(graph_data: GraphData):
    """Print sample nodes."""
    seen_node_types = set()
    for _, node in graph_data["V"].items():
        if node.type in seen_node_types:
            continue
        seen_node_types.add(node.type)
        logging.info(node)
        if node.type not in [NodeType.𝓢, NodeType.𝓣]:
            logging.debug(node.as_dict())


def print_sample_arcs(graph_data: GraphData):
    """Print sample arcs."""
    seen_arc_prefixes = set()
    for _, arc in graph_data["E"].items():
        prefix_pair = (arc.source.type.value, arc.destination.type.value)
        if prefix_pair in seen_arc_prefixes:
            continue
        seen_arc_prefixes.add(prefix_pair)
        logging.info(arc)
        if arc.source.type != NodeType.𝓢 and arc.destination.type != NodeType.𝓣:
            logging.debug(arc.as_dict())


def process_links(nodes: Dict[str, Node], arcs: Dict[tuple, Arc], ref_data: Dict[str, Any]):
    """Process all waypoint links and add the nodes and arcs to the graph.

    Calls handlers for terminal and town nodes to add terminal value nodes and
    root/lodging nodes with their respective source and sink arcs.
    """
    for link in ref_data["waypoint_links"]:
        source, destination = get_link_nodes(nodes, link, ref_data)
        if source is None or destination is None:
            continue

        add_arcs(nodes, arcs, source, destination)

        if source.type == NodeType.𝒕:
            process_terminal(nodes, arcs, source, ref_data)
        if destination.type == NodeType.town:
            process_town(nodes, arcs, destination, ref_data)


def process_terminal(
    nodes: Dict[str, Node], arcs: Dict[tuple, Arc], terminal: Node, ref_data: Dict[str, Any]
):
    """Add terminal root values and arcs between the source and terminal nodes."""
    for i, (root_id, value_data) in enumerate(ref_data["terminal_values"][terminal.id].items(), 1):
        if i > ref_data["top_n_terminal_values"]:
            break
        if value_data["value"] == 0:
            continue
        value_data["value"]
        terminal.𝓻_prizes[root_id] = value_data

    add_arcs(nodes, arcs, nodes["𝓢"], terminal)


def process_town(
    nodes: Dict[str, Node], arcs: Dict[tuple, Arc], town: Node, ref_data: Dict[str, Any]
):
    """Add town root and lodging nodes and arcs between the town and sink nodes."""
    root_id = ref_data["town_to_root"][town.id]
    lodging_data = ref_data["lodging_data"][root_id]
    lodging_bonus = ref_data["lodging_bonus"]

    max_capacity = 1 + lodging_data["max_capacity"] + lodging_bonus
    root_node = get_node(nodes, root_id, NodeType.𝓡, ref_data, capacity=max_capacity)
    add_arcs(nodes, arcs, town, root_node)

    min_capacity = 0
    for capacity, lodging_data in lodging_data.items():
        if capacity == "max_capacity":
            continue
        max_capacity = 1 + int(capacity) + ref_data["lodging_bonus"]
        if min_capacity > ref_data["waypoint_capacity"]:
            continue
        lodging_node = get_node(
            nodes,
            f"{root_node.id}_for_{1 + int(capacity) + lodging_bonus}",
            NodeType.lodging,
            ref_data,
            capacity=max_capacity,
            min_capacity=min_capacity,
            cost=lodging_data[0].get("cost"),
            root=root_node,
        )
        min_capacity = max_capacity + 1
        add_arcs(nodes, arcs, root_node, lodging_node)
        add_arcs(nodes, arcs, lodging_node, nodes["𝓣"])


def nearest_n_roots(ref_data: Dict[str, Any], graph_data: GraphData, nearest_n: int):
    waypoint_graph = nx.DiGraph()
    for node in graph_data["V"].values():
        waypoint_graph.add_node(node.id, type=node.type)

    for arc in graph_data["E"].values():
        weight = arc.destination.cost
        if "1727" in arc.name():
            weight = 999999
        waypoint_graph.add_edge(arc.source.id, arc.destination.id, weight=weight)

    all_pairs = dict(nx.all_pairs_bellman_ford_path_length(waypoint_graph, weight="weight"))

    nearest_roots_dist = {}
    nearest_roots = {}

    for node_id, node in graph_data["V"].items():
        if node.type in {NodeType.waypoint, NodeType.town}:
            distances = []
            for root in graph_data["R"].values():
                town_id = ref_data["root_to_town"][root.id]
                distances.append((root, all_pairs[node.id][town_id]))
            nearest_roots_dist[node_id] = sorted(distances, key=lambda x: x[1])[:nearest_n]
            nearest_roots[node_id] = [w for w, _ in nearest_roots_dist[node_id]]

    return nearest_roots


# def generate_empire_data(lodging_bonus, top_n, nearest_n, waypoint_capacity):
def generate_empire_data(config):
    """Generate and return a Dict of Node and arc objects composing the empire data."""
    validate_config(config)
    logging.basicConfig(level=config["data_log_level"])

    arcs: Dict[tuple[str, str], Arc] = {}
    nodes: Dict[str, Node] = {}
    ref_data = get_reference_data(config)
    ref_data["waypoint_capacity"] = min(config["waypoint_capacity"], config["budget"])

    get_node(nodes, "𝓢", NodeType.𝓢, ref_data)
    get_node(nodes, "𝓣", NodeType.𝓣, ref_data)
    process_links(nodes, arcs, ref_data)

    nodes_dict = dict(sorted(nodes.items(), key=lambda item: item[1].type))
    arcs_dict = dict(sorted(arcs.items(), key=lambda item: item[1].as_dict()["type"]))
    roots = {k: v for k, v in nodes_dict.items() if v.type == NodeType.𝓡}

    graph_data: GraphData = {
        "V": nodes_dict,
        "E": arcs_dict,
        "R": roots,
    }

    # All root nodes have now been generated, finalize LoadForRoot entries
    nearest_roots = nearest_n_roots(ref_data, graph_data, config["nearest_n"])
    for node in nodes_dict.values():
        if node.type in [NodeType.𝓢, NodeType.𝓣]:
            node.𝓻 = [w for w in roots.values()]
        elif node.type in [NodeType.waypoint, NodeType.town]:
            node.𝓻 = [w for w in nearest_roots[node.key]]
        elif node.type is NodeType.𝒕:
            node.𝓻 = [w for w in roots.values() if w.id in node.𝓻_prizes.keys()]

    return graph_data


def validate_config(config):
    assert config["budget"] >= 1, "A budget is required."
    assert config["lodging_bonus"] >= 0 and config["lodging_bonus"] <= 3, "Limit of 3 bonus lodging."
    assert (
        config["top_n"] >= 1 and config["top_n"] <= 24
    ), "Top n valued towns per worker node are required."
    assert (
        config["nearest_n"] >= config["top_n"] and config["nearest_n"] <= 24
    ), "Waypoints require the nearest n towns."


def main(config):
    validate_config(config)
    logging.basicConfig(level=config["data_log_level"])

    graph_data = generate_empire_data(config)

    print_sample_nodes(graph_data)
    print_sample_arcs(graph_data)
    logging.info(f"Num arcs: {len(graph_data["E"])}, Num nodes: {len(graph_data["V"])}")

    data = {
        "arcs": [arc.as_dict() for _, arc in graph_data["E"].items()],
        "nodes": [node.as_dict() for _, node in graph_data["V"].items()],
    }

    filepath = write_user_json("full_empire.json", data)
    print(f"Empire data written to {filepath}")


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
