# api_exploration_graph.py

from collections import Counter, defaultdict

from bidict import bidict
from loguru import logger
import rustworkx as rx

import data_store as ds
from api_common import get_clean_exploration_data


def exploration_graph_nw(data: dict, directed: bool = False) -> rx.PyGraph | rx.PyDiGraph:
    """Generate and return a node weighted PyGraph graph from 'exploration.json'.
    The returned graph will have an attribute [node_key_by_index] containing all
    node indices and exploration keys.

    Indices and keys of the node key map are integer values.
    """
    if directed:
        print("Generating node weighted directed graph...")
    else:
        print("Generating node weighted undirected graph...")

    graph = rx.PyGraph(multigraph=False)

    # Map from exploration key to PyGraph node index (or inverse)
    node_key_by_index = bidict({i: k for k, i in zip(data.keys(), graph.add_nodes_from(data.values()))})

    # Add unweighted undirected edge from each node to its neighbors.
    # NOTE: If edges are added for both (i,j) and (j,i) then they
    #       are distinct edges that are both undirected.
    for i in graph.node_indices():
        neighbors = graph[i]["link_list"]
        for j_key in neighbors:
            j = node_key_by_index.inv[j_key]
            if not graph.has_edge(i, j):
                graph.add_edge(i, j, None)

    if directed:
        graph = graph.to_directed()
    graph.attrs = {"node_key_by_index": node_key_by_index}

    print(f"  generated graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges.")
    return graph


def exploration_graph_ew_directed(data) -> rx.PyGraph | rx.PyDiGraph:
    """Generate and return an edge weighted directed PyDiGraph from 'exploration.json'
    The returned graph will have an attribute [node_key_by_index] containing all
    node indices and exploration keys.

    Edge weighting is done by the process of splitting each node into two nodes, an _in
    and _out node for each original node respectively. All incoming arcs to the original
    node go to the 'in' node and all outgoing arcs come from the 'out' node.

    The payload of the connecting edge is the exploration data for the original node.
    All other edges have a None payload.

    Indices of the node key map are integer values and keys are strings consisting of the
    original node's waypoint_key attribute with a suffix of '_in' or '_out'.
    """
    print("Generating edge weighted directed graph...")

    graph = rx.PyDiGraph(multigraph=False)
    node_key_by_index: bidict[int, str] = bidict({})

    def split_exploration_node(exploration_node: dict):
        waypoint_key = exploration_node["waypoint_key"]

        in_key = f"{waypoint_key}_in"
        out_key = f"{waypoint_key}_out"
        if in_key in node_key_by_index.inv:
            assert out_key in node_key_by_index.inv
            return
        else:
            assert out_key not in node_key_by_index.inv

        tmp = exploration_node.copy()
        tmp["waypoint_key"] = in_key
        in_index = graph.add_node(tmp)

        tmp = exploration_node.copy()
        tmp["waypoint_key"] = out_key
        out_index = graph.add_node(tmp)

        node_key_by_index[in_index] = in_key
        node_key_by_index[out_index] = out_key
        graph.add_edge(in_index, out_index, exploration_node)

    # Split all exploration nodes.
    for exploration_node in data.values():
        split_exploration_node(exploration_node)

    # Add all outbound arcs using each node's "link_list".
    # Each link will go from the exploration nodes '_out' to the destination's '_in'
    # the node indices within the graph are obtained using the node_key_by_index map
    for exploration_key, exploration_data in data.items():
        out_key = f"{exploration_key}_out"
        out_index = node_key_by_index.inv[out_key]

        for destination_key in exploration_data["link_list"]:
            in_key = f"{destination_key}_in"
            in_index = node_key_by_index.inv[in_key]
            graph.add_edge(out_index, in_index, None)

    graph.attrs = {"node_key_by_index": node_key_by_index}
    print(f"  generated graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges.")
    return graph


def get_exploration_graph(config: dict) -> rx.PyGraph | rx.PyDiGraph:
    """Returns a rustworkx graph.

    Arguments:
        directed: boolean (false: PyGraph, true: PyDigraph)
        edge_weighted: boolean (false: no weighting, true: see note)

    Note: When edge_weighted is true this implies directed is true. Nodes
    will be split into an _in and _out node with a single directed arc
    between them with a weight equal to the node's cost.
    """
    directed: bool = config.get("exploration_data", {}).get("directed", False)
    edge_weighted: bool = config.get("exploration_data", {}).get("edge_weighted", False)
    if not isinstance(directed, bool) or not isinstance(edge_weighted, bool):
        raise ValueError("directed and edge_weighted must be booleans")

    data = get_clean_exploration_data(config)

    match (directed, edge_weighted):
        case (False, False):
            graph = exploration_graph_nw(data, False)
        case (True, False):
            graph = exploration_graph_nw(data, True)
        case (False, True):
            graph = exploration_graph_ew_directed(data)
        case (True, True):
            graph = exploration_graph_ew_directed(data)
    return graph


def populate_edge_weights(graph: rx.PyGraph | rx.PyDiGraph):
    for u, v in graph.edge_list():
        graph.update_edge(u, v, {"need_exploration_point": graph[v]["need_exploration_point"]})


def has_edge_weights(graph: rx.PyGraph | rx.PyDiGraph) -> bool:
    has_edge_data = False
    for edge in graph.edges():
        if edge is not None and edge.get("need_exploration_point") is not None:
            has_edge_data = True
            break
    return has_edge_data


def get_all_pairs_shortest_paths(graph: rx.PyGraph | rx.PyDiGraph) -> dict[tuple[int, int], list[int]]:
    if not has_edge_weights(graph):
        populate_edge_weights(graph)
    shortest_paths = rx.all_pairs_dijkstra_shortest_paths(
        graph, edge_cost_fn=lambda edge_data: edge_data["need_exploration_point"]
    )
    shortest_paths = {
        (u, v): list(path)
        for u, paths_from_source in shortest_paths.items()
        for v, path in paths_from_source.items()
    }
    return shortest_paths


def get_all_pairs_path_lengths(graph: rx.PyGraph | rx.PyDiGraph) -> dict[tuple[int, int], int]:
    """Calculates and returns the node weighted shortest paths for all pairs."""
    if not has_edge_weights(graph):
        populate_edge_weights(graph)
    shortest_paths = get_all_pairs_shortest_paths(graph)
    return {
        key: sum(graph[w]["need_exploration_point"] for w in path) for key, path in shortest_paths.items()
    }


def get_neighboring_territories(exploration_data: dict):
    """Processes exploration data to generate and return dict of `{territory: [neighbors]}`.
    Note: territory is included in neighbors

    If it is desired to omit the great ocean nodes then that should be done when creating
    the expxploration data.
    """
    territories = set()
    territory_pairs = set()
    for node_data in exploration_data.values():
        node_territory = node_data["territory_key"]
        territories.add(node_territory)
        for neighbor in node_data["link_list"]:
            neighbor_data = exploration_data.get(neighbor, None)
            if neighbor_data:
                neighbor_territory = neighbor_data["territory_key"]
                territories.add(neighbor_territory)
                territory_pairs.add((node_territory, neighbor_territory))

    results = {t: set() for t in territories}
    for t1, t2 in territory_pairs:
        results[t1].add(t2)
        results[t2].add(t1)
    results = {t: sorted(v) for t, v in results.items()}
    return results


def get_neighboring_region_groups(config: dict) -> dict[int, list[int]]:
    """Processes exploration data to generate and return a dict of `{region_group: [neighbors]}`.
    Note: region_group is included in neighbors
    """
    exploration_data = get_clean_exploration_data(config)
    region_groups = set()
    region_group_pairs = set()
    for node_data in exploration_data.values():
        node_territory = node_data["territory_key"]
        node_region_group = node_data["region_group_key"]
        region_groups.add(node_territory)

        for neighbor in node_data["link_list"]:
            neighbor_data = exploration_data.get(neighbor, None)
            if neighbor_data:
                neighbor_region_group = neighbor_data["region_group_key"]
                region_groups.add(neighbor_region_group)
                region_group_pairs.add((node_region_group, neighbor_region_group))

    results = {t: set() for t in region_groups}
    for t1, t2 in region_group_pairs:
        results[t1].add(t2)
        results[t2].add(t1)
    results = {t: sorted(v) for t, v in results.items()}
    return results


def generate_region_group_neighbors(waypoint_key: int, config: dict):
    exploration_nodes = get_clean_exploration_data(config)
    waypoint_region_group = exploration_nodes[waypoint_key]["region_group_key"]

    region_group_neighbors = get_neighboring_region_groups(config)
    neighbor_region_groups = region_group_neighbors[waypoint_region_group]

    neighbors = set()
    for node in exploration_nodes.values():
        if node["region_group_key"] in neighbor_region_groups:
            neighbors.add(node["waypoint_key"])
    neighbors = sorted(list(neighbors))
    return neighbors


def get_territory_root_sets(exploration_data: dict, territory_neighbors: dict):
    """Generate a dict of root nodes within a territory and neighbors."""
    territory_root_sets = {t: set() for t in territory_neighbors}

    for node_key, node_data in exploration_data.items():
        if not node_data["is_base_town"]:
            continue

        root_territory = node_data["territory_key"]
        if root_territory not in territory_neighbors:
            continue

        for n in territory_neighbors[root_territory]:
            territory_root_sets[n].add(node_key)

    territory_root_sets = {k: list(sorted(v)) for k, v in territory_root_sets.items()}
    return territory_root_sets


def generate_territory_root_sets(exploration_data: dict):
    territory_neighbors = get_neighboring_territories(exploration_data)
    territory_root_sets = get_territory_root_sets(exploration_data, territory_neighbors)
    return territory_root_sets


def get_super_root(config: dict) -> dict:
    """Returns a base town exploration node with waypoint_key: 99999 and
    link_list consisting of all other exploration nodes with "is_base_town"
    attribute set. If valid_nodes is not empty super root's link_list will
    be filtered to the valid_nodes.

    This facilitates the connection of terminals through any potential root in the graph.

    NOTE: Adding super root to the graph breaks graph planarity!
    """
    data = get_clean_exploration_data(config)
    link_list = [k for k, v in data.items() if v["is_base_town"]]

    return {
        "waypoint_key": 99999,
        "region_key": 99999,
        "region_group_key": 99999,
        "territory_key": 99999,
        "character_key": 99999,
        "node_type": 1,
        "is_town": True,
        "is_base_town": True,
        "is_plantzone": False,
        "is_workerman_plantzone": False,
        "is_warehouse_town": False,
        "is_worker_npc_town": False,
        "need_exploration_point": 0,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "link_list": link_list,
        "worker_types": [],
        "region_houseinfo": {
            "has_rentable_lodging": False,
            "has_rentable_storage": False,
            "has_cashproduct_lodging": False,
            "has_cashproduct_storage": False,
        },
    }


# --------------------- Exploration graph betweenness centrality metrics ---------------------
def _load_or_compute_all_simple_paths(
    graph: rx.PyGraph | rx.PyDiGraph, cutoff: int = 20
) -> dict[tuple[int, int], list[list[int]]]:
    import ast
    import json

    filename = ds.path().joinpath("exploration_all_simple_paths.json")
    if filename.is_file():
        try:
            with filename.open("r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_pairs = set(ast.literal_eval(k) for k in data)

            # Simple validation
            active_pairs = {
                (t, r)
                for t in graph.attrs["terminal_indices"]
                for r in graph[t].get("prizes", {})
                if graph[t]["prizes"][r] > 0
            }
            if active_pairs.issuperset(loaded_pairs):
                logger.info(f"      loaded {len(data)} (t,r) pairs from {filename.name}")
                all_asp = {ast.literal_eval(k): [path[:] for path in v] for k, v in data.items()}
                return all_asp
            else:
                logger.warning("Cached paths incomplete; recomputing...")
        except (json.JSONDecodeError, ValueError, SyntaxError):
            logger.warning("Invalid cache; recomputing...")

    logger.info(
        f"      computing/saving ASP for {len(graph.attrs['terminal_indices'])} terminals (cutoff={cutoff})..."
    )
    all_asp = {}
    for t in graph.attrs["terminal_indices"]:
        for r_key in graph[t].get("prizes", {}):
            r = int(r_key)  # Assuming int keys
            if graph[t]["prizes"][r] <= 0:  # Filter active now; prizes static for paths
                continue
            paths = list(rx.all_simple_paths(graph, t, r, cutoff=cutoff))
            valid_paths = [path[:] for path in paths if len(path) >= 3]
            if valid_paths:
                all_asp[(t, r)] = valid_paths

    # Compact save: No indent, no ascii escape
    save_data = {str((k[0], k[1])): v for k, v in all_asp.items()}
    with filename.open("w", encoding="utf-8") as f:
        json.dump(save_data, f, separators=(",", ":"), ensure_ascii=False)
    logger.info(
        f"      saved {len(save_data)} pairs to {filename.name} (~{filename.stat().st_size / 1e6:.1f}MB)"
    )
    return all_asp


def compute_shortest_paths_subset(
    graph: rx.PyGraph | rx.PyDiGraph,
    sources: list[int],
    targets: list[int] | None = None,
) -> dict[tuple[int, int], list[int]]:
    if not has_edge_weights(graph):
        populate_edge_weights(graph)
    if targets is None:
        targets = list(sources)
    results = {}
    for s in sources:
        sp = rx.dijkstra_shortest_paths(graph, s, weight_fn=lambda e: e["need_exploration_point"])
        for t in targets:
            if t in sp:
                results[(s, t)] = list(sp[t])
    return results


def compute_path_overlap(paths: dict[tuple[int, int], list[int]]):
    node_count = Counter()
    edge_count = Counter()
    for path in paths.values():
        if path and len(path) >= 2:
            for v in path[1:-1]:
                node_count[v] += 1
            for u, v in zip(path, path[1:]):
                edge_count[(u, v)] += 1
    return node_count, edge_count


def compute_paths_using_node(paths: dict[tuple[int, int], list[int]]):
    usage = defaultdict(list)
    for (s, t), path in paths.items():
        if path and len(path) >= 3:
            for v in path[1:-1]:
                usage[v].append((s, t))
    return dict(usage)


def compute_restricted_betweenness(
    G: rx.PyGraph | rx.PyDiGraph, terminals: list[int], roots: list[int]
) -> dict[int, float]:
    bc = Counter()
    paths = compute_shortest_paths_subset(G, terminals, targets=roots)
    all_pairs_path_lengths = get_all_pairs_path_lengths(G)
    for (t, r), path in paths.items():
        prize = G[t].get("prizes", {}).get(r, 0)
        if prize <= 0:
            continue
        if len(path) < 3:
            continue
        path_cost = all_pairs_path_lengths[(t, r)]
        net_path_cost_value = prize / path_cost  # No epsilon
        for v in path[1:-1]:
            bc[v] += net_path_cost_value
    total = sum(bc.values())
    return {k: v / total if total > 0 else 0.0 for k, v in bc.items()}


def compute_restricted_betweenness_all_simple_paths(
    graph: rx.PyGraph | rx.PyDiGraph,
    terminals: list[int],
    roots: list[int],
    cutoff: int = 10,
) -> dict[int, float]:
    import time

    start_time = time.time()
    bc = Counter()
    all_asp = _load_or_compute_all_simple_paths(graph, cutoff)
    for (t, r), paths in all_asp.items():
        prize = graph[t]["prizes"][r]
        for path in paths:
            path_cost = sum(graph[v]["need_exploration_point"] for v in path[1:])
            if path_cost <= 0:
                continue
            for v in path[1:-1]:
                bc[v] += prize / path_cost
    total = sum(bc.values())
    result = {k: v / total if total > 0 else 0.0 for k, v in bc.items()}
    logger.info(f"      computing restricted betweenness took {time.time() - start_time:.2f} seconds")
    return result


def attach_metrics_to_graph(G: rx.PyGraph | rx.PyDiGraph, config: dict | None = None):
    """Attach betweenness centrality metrics to graph"""
    logger.info("    computing graph betweenness centrality metrics...")

    if not has_edge_weights(G):
        populate_edge_weights(G)

    terminal_indices = G.attrs["terminal_indices"]
    root_indices = G.attrs["root_indices"]
    S = sorted(set(terminal_indices) | set(root_indices))
    paths = compute_shortest_paths_subset(G, S)
    node_overlap, edge_overlap = compute_path_overlap(paths)
    paths_using_node = compute_paths_using_node(paths)
    bc_tr = compute_restricted_betweenness(G, terminal_indices, root_indices)
    bc_all = rx.betweenness_centrality(G)
    bc_tr_asp = None
    if config and config.get("use_asp_bc", False):
        bc_tr_asp = compute_restricted_betweenness_all_simple_paths(
            G,
            terminal_indices,
            root_indices,
            cutoff=config.get("asp_cutoff", 10),
        )
    metrics = {
        "S": S,
        "paths": paths,
        "node_overlap": node_overlap,
        "edge_overlap": edge_overlap,
        "paths_using_node": paths_using_node,
        "bc_sources": S,
        "bc_tr": bc_tr,
        "bc_all": bc_all,
    }
    if bc_tr_asp:
        metrics["bc_tr_asp"] = bc_tr_asp
    G.attrs["exploration_metrics"] = metrics
    return metrics
