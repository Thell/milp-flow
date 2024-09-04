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
    G: Dict[str, Node]
    P: Dict[str, Node]
    L: Dict[str, Node]


class NodeType(IntEnum):
    𝓢 = auto()
    plant = auto()
    waypoint = auto()
    town = auto()
    group = auto()
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
        ub: int,
        lb: int = 0,
        cost: int = 0,
        groups: List[Node] = [],
    ):
        self.id = id
        self.type = type
        self.ub = ub
        self.lb = lb
        self.cost = cost
        self.group_prizes: Dict[str, Dict[str, Any]] = {}
        self.groups = groups if groups else []
        self.key = self.name()
        self.inbound_arcs: List[Arc] = []
        self.outbound_arcs: List[Arc] = []
        self.vars = {}
        self.isPlant = type == NodeType.plant
        self.isLodging = type == NodeType.lodging
        self.isTown = type == NodeType.town
        self.isWaypoint = type == NodeType.waypoint
        self.isGroup = type == NodeType.group

    def name(self) -> str:
        if self.type in [NodeType.𝓢, NodeType.𝓣]:
            return self.id
        return f"{self.type.name}_{self.id}"

    def inSolution(self):
        x_var = self.vars.get("x", None)
        if x_var is not None:
            return x_var.varValue is not None and round(x_var.varValue) >= 1
        else:
            return False

    def as_dict(self) -> Dict[str, Any]:
        obj_dict = {
            "key": self.name(),
            "name": self.name(),
            "id": self.id,
            "type": self.type.name.lower(),
            "ub": self.ub,
            "lb": self.ub,
            "cost": self.cost,
            "group_prizes": self.group_prizes,
            "groups": [],
            "inbound_arcs": [arc.key for arc in self.inbound_arcs],
            "outbound_arcs": [arc.key for arc in self.outbound_arcs],
        }
        for node in self.groups:
            if node is self:
                obj_dict["groups"].append("self")
            else:
                obj_dict["groups"].append(node.name())
        return obj_dict

    def __repr__(self) -> str:
        return f"Node(name: {self.name()}, ub: {self.ub}, lb: {self.lb}, cost: {self.cost}, value: {self.group_prizes})"

    def __eq__(self, other) -> bool:
        return self.name() == other.name()

    def __hash__(self) -> int:
        return hash((self.name()))


class Arc:
    def __init__(self, source: Node, destination: Node, ub: int, cost: int = 0):
        self.source = source
        self.destination = destination
        self.ub = ub
        self.cost = cost
        self.key = (source.name(), destination.name())
        self.type = (source.type, destination.type)
        self.vars = {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name(),
            "ub": self.ub,
            "type": self.type,
            "source": self.source.name(),
            "destination": self.destination.name(),
        }

    def allowable_loads(self) -> set[Node]:
        return set(self.source.groups).intersection(set(self.destination.groups))

    def name(self) -> str:
        return f"{self.source.name()}_to_{self.destination.name()}"

    def __repr__(self) -> str:
        return f"arc({self.source.name()} -> {self.destination.name()}, ub: {self.ub})"

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
        (NodeType.𝓢, NodeType.plant): (1, 0),
        (NodeType.plant, NodeType.waypoint): (1, 0),
        (NodeType.plant, NodeType.town): (1, 0),
        (NodeType.waypoint, NodeType.waypoint): (node_b.ub, node_a.ub),
        (NodeType.waypoint, NodeType.town): (node_b.ub, node_a.ub),
        (NodeType.town, NodeType.town): (node_b.ub, node_a.ub),
        (NodeType.town, NodeType.group): (node_b.ub, 0),
        (NodeType.group, NodeType.lodging): (node_b.ub, 0),
        (NodeType.lodging, NodeType.𝓣): (node_a.ub, 0),
    }

    ub, reverse_ub = arc_configurations.get((node_a.type, node_b.type), (1, 0))

    arc_a = Arc(node_a, node_b, ub=ub)
    arc_b = Arc(node_b, node_a, ub=reverse_ub)

    for arc in [arc_a, arc_b]:
        if arc.key not in arcs and arc.ub > 0:
            arcs[arc.key] = arc
            nodes[arc.source.key].outbound_arcs.append(arc)
            nodes[arc.destination.key].inbound_arcs.append(arc)

            if arc.destination.type is NodeType.lodging:
                arc.destination.groups = [arc.source]


def get_link_node_type(node_id: str, ref_data: Dict[str, Any]):
    """Return the NodeType of the given node_id node.

    - NodeType.INVALID indicates a node that is unused and not added to the graph.
    """
    if node_id in ref_data["towns"]:
        return NodeType.town
    if node_id in ref_data["all_plantzones"]:
        if node_id not in ref_data["plants"]:
            return NodeType.INVALID
        return NodeType.plant
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

    kwargs `plant` and `group` are required for supply nodes.
    kwargs `ub` is required for group nodes.
    kwargs `ub`, `cost` and `group` are required for lodging nodes.
    """

    groups = []
    lb = 0

    match node_type:
        case NodeType.𝓢:
            ub = ref_data["max_ub"]
            cost = 0
        case NodeType.plant:
            ub = 1
            cost = ref_data["waypoint_data"][node_id]["CP"]
        case NodeType.waypoint | NodeType.town:
            ub = ref_data["waypoint_ub"]
            cost = ref_data["waypoint_data"][node_id]["CP"]
        case NodeType.group:
            ub = ref_data["lodging_data"][node_id]["max_ub"] + ref_data["lodging_bonus"]
            ub = min(ub, ref_data["waypoint_ub"])
            cost = 0
        case NodeType.lodging:
            ub = kwargs.get("ub")
            lb = kwargs.get("lb")
            root = kwargs.get("root")
            cost = kwargs.get("cost")
            assert (
                ub and (lb is not None) and (cost is not None) and root
            ), "Lodging nodes require 'ub', 'lb' 'cost' and 'root' kwargs."
            groups = [root]
        case NodeType.𝓣:
            ub = ref_data["max_ub"]
            cost = 0
        case NodeType.INVALID:
            assert node_type is not NodeType.INVALID, "INVALID node type."
            return  # Unreachable: Stops pyright unbound error reporting.

    node = Node(node_id, node_type, ub, lb, cost, groups)
    if node.key not in nodes:
        if node.type is NodeType.group:
            node.groups = [node]
        nodes[node.key] = node

    return nodes[node.key]


def get_reference_data(config):
    """Read and prepare data from reference json files."""
    ref_data = {
        "lodging_bonus": config["lodging_bonus"],
        "top_n_plant_values": config["top_n"],
        "all_plantzones": read_workerman_json("plantzone.json").keys(),
        "lodging_data": read_workerman_json("all_lodging_storage.json"),
        "plant_values": read_user_json("node_values_per_town.json"),
        "town_to_group": read_workerman_json("town_node_translate.json")["tnk2tk"],
        "group_to_town": read_workerman_json("town_node_translate.json")["tk2tnk"],
        "group_to_townname": read_workerman_json("warehouse_to_townname.json"),
        "waypoint_data": read_workerman_json("exploration.json"),
        "waypoint_links": read_workerman_json("deck_links.json"),
    }

    plants = ref_data["plant_values"].keys()
    groups = ref_data["plant_values"][list(plants)[0]].keys()
    towns = [ref_data["group_to_town"][w] for w in groups]

    ref_data["max_ub"] = len(plants)
    ref_data["plants"] = plants
    ref_data["groups"] = groups
    ref_data["towns"] = towns

    for group, lodgings in ref_data["lodging_data"].items():
        if group not in ref_data["groups"]:
            continue
        max_lodging = 1 + ref_data["lodging_bonus"] + max([int(k) for k in lodgings.keys()])
        ref_data["lodging_data"][group]["max_ub"] = max_lodging

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

    Calls handlers for plant and town nodes to add plant value nodes and
    group/lodging nodes with their respective source and sink arcs.
    """
    for link in ref_data["waypoint_links"]:
        source, destination = get_link_nodes(nodes, link, ref_data)
        if source is None or destination is None:
            continue

        add_arcs(nodes, arcs, source, destination)

        if source.isPlant:
            process_plant(nodes, arcs, source, ref_data)
        if destination.isTown:
            process_town(nodes, arcs, destination, ref_data)


def process_plant(
    nodes: Dict[str, Node], arcs: Dict[tuple, Arc], plant: Node, ref_data: Dict[str, Any]
):
    """Add plant group values and arcs between the source and plant nodes."""
    for i, (group_id, value_data) in enumerate(ref_data["plant_values"][plant.id].items(), 1):
        if i > ref_data["top_n_plant_values"]:
            break
        if value_data["value"] == 0:
            continue
        value_data["value"]
        plant.group_prizes[group_id] = value_data

    add_arcs(nodes, arcs, nodes["𝓢"], plant)


def process_town(
    nodes: Dict[str, Node], arcs: Dict[tuple, Arc], town: Node, ref_data: Dict[str, Any]
):
    """Add town group and lodging nodes and arcs between the town and sink nodes."""
    group_id = ref_data["town_to_group"][town.id]
    lodging_data = ref_data["lodging_data"][group_id]
    lodging_bonus = ref_data["lodging_bonus"]

    lodgings = [(1 + lodging_bonus, 0)]
    for ub, lodging_data in lodging_data.items():
        if ub == "max_ub":
            continue
        current = (1 + lodging_bonus + int(ub), lodging_data[0].get("cost"))
        while lodgings and current[1] <= lodgings[-1][1] and current[0] >= lodgings[-1][0]:
            lodgings.pop(-1)
        lodgings.append(current)
        if current[0] + 1 >= ref_data["waypoint_ub"]:
            break

    group_node = get_node(nodes, group_id, NodeType.group, ref_data, ub=lodgings[-1][0])
    add_arcs(nodes, arcs, town, group_node)

    lb = 0
    for ub, cost in lodgings:
        lodging_node = get_node(
            nodes,
            f"{group_node.id}_for_{ub}",
            NodeType.lodging,
            ref_data,
            ub=ub,
            lb=lb,
            cost=cost,
            root=group_node,
        )
        lb = ub + 1
        add_arcs(nodes, arcs, group_node, lodging_node)
        add_arcs(nodes, arcs, lodging_node, nodes["𝓣"])


def nearest_n_towns(ref_data: Dict[str, Any], G: GraphData, nearest_n: int):
    waypoint_graph = nx.DiGraph()
    for arc in G["E"].values():
        weight = 999999 if "1727" in arc.name() else arc.destination.cost
        waypoint_graph.add_edge(arc.source.id, arc.destination.id, weight=weight)
    all_pairs = dict(nx.all_pairs_bellman_ford_path_length(waypoint_graph, weight="weight"))

    nearest_towns_dist = {}
    nearest_towns = {}

    for node_id, node in G["V"].items():
        if node.isWaypoint or node.isTown:
            distances = []
            for group in G["G"].values():
                town_id = ref_data["group_to_town"][group.id]
                distances.append((group, all_pairs[node.id][town_id]))
            nearest_towns_dist[node_id] = sorted(distances, key=lambda x: x[1])[:nearest_n]
            nearest_towns[node_id] = [w for w, _ in nearest_towns_dist[node_id]]

    return nearest_towns


def finalize_groups(ref_data: Dict[str, Any], G: GraphData, nearest_n: int):
    # All group nodes have now been generated, finalize groups entries
    nearest_towns = nearest_n_towns(ref_data, G, nearest_n)
    for v in G["V"].values():
        if v.type in [NodeType.𝓢, NodeType.𝓣]:
            v.groups = [w for w in G["G"].values()]
        elif v.isWaypoint or v.isTown:
            v.groups = [w for w in nearest_towns[v.key]]
        elif v.isPlant:
            v.groups = [w for w in G["G"].values() if w.id in v.group_prizes.keys()]


def generate_empire_data(config):
    """Generate and return a GraphData Dict composing the LP empire data."""
    validate_config(config)
    logging.basicConfig(level=config["data_log_level"])

    ref_data = get_reference_data(config)
    ref_data["waypoint_ub"] = min(config["waypoint_ub"], config["budget"])

    nodes: Dict[str, Node] = {}
    arcs: Dict[tuple[str, str], Arc] = {}

    get_node(nodes, "𝓢", NodeType.𝓢, ref_data)
    get_node(nodes, "𝓣", NodeType.𝓣, ref_data)
    process_links(nodes, arcs, ref_data)

    G: GraphData = {
        "V": dict(sorted(nodes.items(), key=lambda item: item[1].type)),
        "E": dict(sorted(arcs.items(), key=lambda item: item[1].as_dict()["type"])),
        "G": {k: v for k, v in nodes.items() if v.isGroup},
        "P": {k: v for k, v in nodes.items() if v.isPlant},
        "L": {k: v for k, v in nodes.items() if v.isLodging},
    }
    finalize_groups(ref_data, G, config["nearest_n"])

    return G


def validate_config(config):
    assert config["budget"] >= 1, "A budget is required."
    assert config["lodging_bonus"] >= 0 and config["lodging_bonus"] <= 5, "Limit of 3 bonus lodging."
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
