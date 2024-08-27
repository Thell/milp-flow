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
    towns: Dict[str, Node]


class NodeType(IntEnum):
    𝓢 = auto()
    plant = auto()
    waypoint = auto()
    town = auto()
    lodge = auto()
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
        zones: List[Node] = [],
    ):
        self.id = id
        self.type = type
        self.ub = ub
        self.lb = lb
        self.cost = cost
        self.zone_values: Dict[str, Dict[str, Any]] = {}
        self.zones = zones if zones else []
        self.key = self.name()
        self.inbound_arcs: List[Arc] = []
        self.outbound_arcs: List[Arc] = []
        self.vars = {}
        self.isPlant = type == NodeType.plant
        self.isLodging = type == NodeType.lodge

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
            "ub": self.ub,
            "lb": self.ub,
            "cost": self.cost,
            "zone_values": self.zone_values,
            "zones": [],
            "inbound_arcs": [arc.key for arc in self.inbound_arcs],
            "outbound_arcs": [arc.key for arc in self.outbound_arcs],
        }
        for zone in self.zones:
            if zone is self:
                obj_dict["zones"].append("self")
            else:
                obj_dict["zones"].append(zone.name())
        return obj_dict

    def __repr__(self) -> str:
        return f"Node(name: {self.name()}, ub: {self.ub}, lb: {self.lb}, cost: {self.cost}, value: {self.zone_values})"

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
        return set(self.source.zones).intersection(set(self.destination.zones))

    def name(self) -> str:
        return f"{self.source.name()}_to_{self.destination.name()}"

    def __repr__(self) -> str:
        return f"arc({self.source.name()} -> {self.destination.name()}, ub: {self.ub})"

    def __eq__(self, other) -> bool:
        return (self.source, self.destination) == (other.source, other.destination)

    def __hash__(self) -> int:
        return hash((self.source.name() + self.destination.name()))


def add_arcs(nodes: Dict[str, Node], arcs: Dict[tuple, Arc], u: Node, v: Node):
    """Add arcs between a and b."""
    # A safety measure to ensure arc direction.
    if u.type > v.type:
        u, v = v, u

    arc_configurations = {
        (NodeType.𝓢, NodeType.plant): (1, 0),
        (NodeType.plant, NodeType.waypoint): (1, 0),
        (NodeType.plant, NodeType.town): (1, 0),
        (NodeType.waypoint, NodeType.waypoint): (v.ub, u.ub),
        (NodeType.waypoint, NodeType.town): (v.ub, u.ub),
        (NodeType.town, NodeType.town): (v.ub, u.ub),
        (NodeType.town, NodeType.lodge): (v.ub, 0),
        (NodeType.lodge, NodeType.𝓣): (u.ub, 0),
    }

    ub, reverse_ub = arc_configurations.get((u.type, v.type), (1, 0))

    arc_a = Arc(u, v, ub=ub)
    arc_b = Arc(v, u, ub=reverse_ub)

    for a in [arc_a, arc_b]:
        if a.key not in arcs and a.ub > 0:
            arcs[a.key] = a
            nodes[a.source.key].outbound_arcs.append(a)
            nodes[a.destination.key].inbound_arcs.append(a)

            if a.destination.type is NodeType.lodge:
                a.destination.zones = [a.source]


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

    kwargs `plant` and `root` are required for supply nodes.
    kwargs `ub` is required for root nodes.
    kwargs `ub`, `cost` and `root` are required for lodge nodes.
    """

    zones = []
    lb = 0

    match node_type:
        case NodeType.𝓢:
            ub = ref_data["max_capacity"]
            cost = 0
        case NodeType.plant:
            ub = 1
            cost = ref_data["waypoint_data"][node_id]["CP"]
        case NodeType.waypoint | NodeType.town:
            ub = ref_data["waypoint_ub"]
            cost = ref_data["waypoint_data"][node_id]["CP"]
        case NodeType.lodge:
            ub = kwargs.get("ub")
            lb = kwargs.get("lb")
            zone = kwargs.get("zone")
            cost = kwargs.get("cost")
            assert (
                ub and (lb is not None) and (cost is not None) and zone
            ), "Lodging nodes require 'ub', 'lb' 'cost' and 'zone' kwargs."
            zones = [zone]
        case NodeType.𝓣:
            ub = ref_data["max_capacity"]
            cost = 0
        case NodeType.INVALID:
            assert node_type is not NodeType.INVALID, "INVALID node type."
            return  # Unreachable: Stops pyright unbound error reporting.

    node = Node(node_id, node_type, ub, lb, cost, zones)
    if node.key not in nodes:
        nodes[node.key] = node

    return nodes[node.key]


# def get_reference_data(lodge_bonus, top_n):
def get_reference_data(config):
    """Read and prepare data from reference json files."""
    ref_data = {
        "lodging_bonus": config["lodging_bonus"],
        "top_n_plant_values": config["top_n"],
        "all_plantzones": read_workerman_json("plantzone.json").keys(),
        "lodging_data": read_workerman_json("all_lodging_storage.json"),
        "plant_values": read_user_json("node_values_per_town.json"),
        "town_to_root": read_workerman_json("town_node_translate.json")["tnk2tk"],
        "root_to_town": read_workerman_json("town_node_translate.json")["tk2tnk"],
        "root_to_townname": read_workerman_json("warehouse_to_townname.json"),
        "townnames": read_workerman_json("townnames.json"),
        "waypoint_data": read_workerman_json("exploration.json"),
        "waypoint_links": read_workerman_json("deck_links.json"),
    }

    plants = ref_data["plant_values"].keys()
    roots = ref_data["plant_values"][list(plants)[0]].keys()
    towns = [ref_data["root_to_town"][w] for w in roots]

    ref_data["max_capacity"] = len(plants)
    ref_data["plants"] = plants
    ref_data["roots"] = roots
    ref_data["towns"] = towns

    for root, lodges in ref_data["lodging_data"].items():
        if root not in ref_data["roots"]:
            continue
        max_lodging = 1 + ref_data["lodging_bonus"] + max([int(k) for k in lodges.keys()])
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

    Calls handlers for plant and town nodes to add plant value nodes and
    root/lodge nodes with their respective source and sink arcs.
    """
    for link in ref_data["waypoint_links"]:
        source, destination = get_link_nodes(nodes, link, ref_data)
        if source is None or destination is None:
            continue

        add_arcs(nodes, arcs, source, destination)

        if source.type == NodeType.plant:
            process_plant(nodes, arcs, source, ref_data)
        if destination.type == NodeType.town:
            process_town(nodes, arcs, destination, ref_data)


def process_plant(
    nodes: Dict[str, Node], arcs: Dict[tuple, Arc], plant: Node, ref_data: Dict[str, Any]
):
    """Add plant root values and arcs between the source and plant nodes."""
    for i, (root_id, value_data) in enumerate(ref_data["plant_values"][plant.id].items(), 1):
        if i > ref_data["top_n_plant_values"]:
            break
        if value_data["value"] == 0:
            continue
        value_data["value"]
        plant.zone_values[ref_data["root_to_town"][root_id]] = value_data

    add_arcs(nodes, arcs, nodes["𝓢"], plant)


def process_town(
    nodes: Dict[str, Node], arcs: Dict[tuple, Arc], town: Node, ref_data: Dict[str, Any]
):
    """Add town root and lodge nodes and arcs between the town and sink nodes."""
    root_id = ref_data["town_to_root"][town.id]
    lodging_data = ref_data["lodging_data"][root_id]
    lodging_bonus = ref_data["lodging_bonus"]

    lodges = [(1 + lodging_bonus, 0)]
    for ub, lodge_data in lodging_data.items():
        if ub == "max_capacity":
            continue
        current = (1 + lodging_bonus + int(ub), lodge_data[0].get("cost"))
        while lodges and current[1] <= lodges[-1][1] and current[0] >= lodges[-1][0]:
            lodges.pop(-1)
        lodges.append(current)
        if current[0] + 1 >= ref_data["waypoint_ub"]:
            break

    lb = 0
    for ub, cost in lodges:
        lodge_node = get_node(
            nodes,
            f"{town.id}_for_{ub}",
            NodeType.lodge,
            ref_data,
            ub=ub,
            lb=lb,
            cost=cost,
            zone=town,
        )
        lb = ub + 1
        add_arcs(nodes, arcs, town, lodge_node)
        add_arcs(nodes, arcs, lodge_node, nodes["𝓣"])


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

    nearest_towns_dist = {}
    nearest_towns = {}

    for node in graph_data["V"].values():
        if node.type in {NodeType.waypoint, NodeType.town}:
            distances = []
            for town in graph_data["towns"].values():
                distances.append((town, all_pairs[node.id][town.id]))
            nearest_towns_dist[node.id] = sorted(distances, key=lambda x: x[1])[:nearest_n]
            nearest_towns[node.id] = [w for w, _ in nearest_towns_dist[node.id]]

    return nearest_towns


def generate_empire_data(config):
    """Generate and return a Dict of Node and arc objects composing the empire data."""
    validate_config(config)
    logging.basicConfig(level=config["data_log_level"])

    arcs: Dict[tuple[str, str], Arc] = {}
    nodes: Dict[str, Node] = {}
    ref_data = get_reference_data(config)
    ref_data["waypoint_ub"] = min(config["waypoint_ub"], config["budget"])

    get_node(nodes, "𝓢", NodeType.𝓢, ref_data)
    get_node(nodes, "𝓣", NodeType.𝓣, ref_data)
    process_links(nodes, arcs, ref_data)

    nodes_dict = dict(sorted(nodes.items(), key=lambda item: item[1].type))
    arcs_dict = dict(sorted(arcs.items(), key=lambda item: item[1].as_dict()["type"]))
    towns = {k: v for k, v in nodes_dict.items() if v.type == NodeType.town}

    graph_data: GraphData = {
        "V": nodes_dict,
        "E": arcs_dict,
        "towns": towns,
    }

    # All town nodes have now been generated, finalize zones entries.
    nearest_roots = nearest_n_roots(ref_data, graph_data, config["nearest_n"])
    for node in nodes_dict.values():
        if node.type in [NodeType.𝓢, NodeType.𝓣]:
            node.zones = [t for t in towns.values()]
        elif node.type in [NodeType.waypoint, NodeType.town]:
            node.zones = [t for t in nearest_roots[node.id]]
        elif node.type is NodeType.plant:
            node.zones = [t for t in towns.values() if t.id in node.zone_values.keys()]

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
