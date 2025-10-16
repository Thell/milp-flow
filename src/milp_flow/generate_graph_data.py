# generate_graph.data.py

from typing import Any

from bidict import bidict
from loguru import logger
import rustworkx as rx

from api_exploration_graph import get_all_pairs_path_lengths

# DEBUG_ROOTS = [1623, 1785, 1834, 2001]
DEBUG_ROOTS = []


def prep_graph_nodes(solver_graph: rx.PyDiGraph, data: dict[str, Any]) -> int | None:
    """Prepare a copy of the exploration graph for the HiGHS model."""
    logger.info("Preparing graph nodes...")

    super_root_index = setup_super_terminals(solver_graph, data)

    node_key_by_index = bidict({i: solver_graph[i]["waypoint_key"] for i in solver_graph.node_indices()})
    solver_graph.attrs = {"node_key_by_index": node_key_by_index}

    root_indices = [
        node_key_by_index.inv[t]
        for t in data["affiliated_town_region"].values()
        if data["exploration"][t]["is_worker_npc_town"]
    ]
    solver_graph.attrs["root_indices"] = root_indices
    logger.debug(f"Found {len(root_indices)} root indices:")
    logger.debug(f"(index, waypoint): {[(i, solver_graph[i]['waypoint_key']) for i in root_indices]}")

    setup_terminals(solver_graph, data)
    setup_roots(solver_graph, data)
    setup_node_transit_bounds(solver_graph, data, super_root_index)
    return super_root_index


def setup_super_terminals(solver_graph: rx.PyDiGraph, data: dict[str, Any]) -> int | None:
    """Injects the superroot node into the graph"""
    # Super terminals require the super root for connection via any basetown.
    # Flow is _into_ super root with no flow out to prevent worker flow routing shortcuts
    super_root_index = None

    if data["force_active_node_ids"]:
        logger.info("  setting up super-terminals...")
        from api_rx_pydigraph import inject_super_root

        for node in solver_graph.nodes():
            node["is_super_terminal"] = (
                True if node["waypoint_key"] in data["force_active_node_ids"] else False
            )

        node_key_by_index = bidict({i: solver_graph[i]["waypoint_key"] for i in solver_graph.node_indices()})
        solver_graph.attrs = {"node_key_by_index": node_key_by_index}
        super_root_index = inject_super_root({}, solver_graph, flow_direction="inbound")
        solver_graph[super_root_index]["ub"] = len(data["force_active_node_ids"])

    return super_root_index


def setup_terminals(solver_graph: rx.PyDiGraph, data: dict[str, Any]) -> list[int]:
    """Prepares terminals with top_n root values."""
    # The top_n root values per terminal retain their value and all others are set at zero value.
    top_n = data["config"]["top_n"]
    logger.info(f"  setting up plant zones with {top_n} highest valued base towns per plant...")

    node_key_by_index = solver_graph.attrs["node_key_by_index"]
    terminal_indices = []

    for i in solver_graph.node_indices():
        node = solver_graph[i]
        if not node["is_workerman_plantzone"]:
            continue
        terminal_indices.append(i)

        prizes = data["plant_values"][node["waypoint_key"]]
        prizes = sorted(prizes.items(), key=lambda x: x[1]["value"], reverse=True)
        prizes = dict(prizes[:top_n])
        values = {}
        # NOTE: Prize keys are affiliated town regions not waypoint keys, so translate!
        for warehouse_key, prize_data in prizes.items():
            value = int(prize_data["value"])
            if value == 0:
                break
            root_key = data["affiliated_town_region"][warehouse_key]
            root_index = node_key_by_index.inv[root_key]
            values[root_index] = value
        node["prizes"] = values

    town_to_region_map = {int(town): region for region, town in data["affiliated_town_region"].items()}
    for terminal_index in sorted(terminal_indices):
        node = solver_graph[terminal_index]
        prizes = node["prizes"]
        for root, prize in prizes.items():
            town = solver_graph[root]["waypoint_key"]
            region = town_to_region_map[int(town)]
            logger.debug(f"{node['waypoint_key']:>5} {region:>5} {town:>5} {prize:>5}")

    solver_graph.attrs["terminals"] = terminal_indices
    return terminal_indices


def setup_roots(solver_graph: rx.PyDiGraph, data: dict[str, Any]):
    """Prepares step-wise tiered root lodging costs."""
    # The lodging costs of a worker town is a simple dict keyed by 0..capacity with cost values.
    # The capacity is the minimum of the waypoint_ub and the towns maximum lodging capacity.
    logger.info("  setting up root lodging costs...")
    root_indices = solver_graph.attrs["root_indices"]

    for i in root_indices:
        node = solver_graph[i]
        region_key = node["region_key"]
        lodgings = data["lodging_data"][region_key]
        num_free = lodgings["lodging_bonus"] + 1

        dominant_lodgings = []
        for n, lodging_data in lodgings.items():
            if not isinstance(n, int):
                break
            current_lodging_capacity = lodging_data[0]["lodging"]
            current_lodging_cost = lodging_data[0]["cost"]
            while len(dominant_lodgings) > 1 and current_lodging_cost < dominant_lodgings[-1]["cost"]:
                dominant_lodgings.pop()
            dominant_lodgings.append({"capacity": current_lodging_capacity, "cost": current_lodging_cost})

        # increase the capacity of each dominant lodging by lodging_bonus + 1
        for lodging in dominant_lodgings:
            lodging["capacity"] += num_free

        ub = min(lodgings["max_ub"] + num_free, data["config"]["waypoint_ub"] + 1)
        capacity_cost = [0] * (ub)
        current_index = num_free

        while current_index <= ub + 1 and len(dominant_lodgings) > 0:
            capacity_limit = min(ub + 1, dominant_lodgings[0]["capacity"] + num_free)
            capacity_cost[current_index:capacity_limit] = [dominant_lodgings[0]["cost"]] * (
                capacity_limit - current_index
            )
            current_index = capacity_limit
            dominant_lodgings.pop(0)

        node["ub"] = ub - 1
        node["capacity_cost"] = capacity_cost

    for i in root_indices:
        logger.debug(
            f"{solver_graph[i]['waypoint_key']:>5} region_key: {solver_graph[i]['region_key']:>5} ub: {solver_graph[i]['ub']} costs: {solver_graph[i]['capacity_cost']}"
        )


def setup_node_transit_bounds(solver_graph: rx.PyDiGraph, data: dict[str, Any], super_root_index: int | None):
    # The upper bound on transit for any worker town in the nearest_n towns is the minimum of
    # the waypoint_ub and the towns maximum lodging capacity and is set during lodging setup.
    # The upper bound for all other worker towns is zero.
    # In the end a non-root node has _n entries and a root node has _n+1 entries
    nearest_n = data["config"]["nearest_n"]
    logger.info(f"  setting up intermediate nodes for the nearest {nearest_n} towns...")

    all_pairs_path_lengths = get_all_pairs_path_lengths(solver_graph)
    root_indices = solver_graph.attrs["root_indices"]

    root_transit_ub = {i: solver_graph[i]["ub"] for i in root_indices}
    if super_root_index is not None:
        root_transit_ub[super_root_index] = len(data["force_active_node_ids"])

    for i in solver_graph.node_indices():
        node = solver_graph[i]
        if i == super_root_index:
            # Super root does not carry transit for any other root.
            node["transit_bounds"] = {super_root_index: len(data["force_active_node_ids"])}
            continue

        nearest_roots = []
        nearest_n_lim = nearest_n
        for j in root_indices:
            if i == j:
                nearest_roots.append((j, 0))
                nearest_n_lim += 1
                continue
            pair = (i, j) if i < j else (j, i)
            nearest_roots.append((j, all_pairs_path_lengths[pair]))

        nearest_roots = sorted(nearest_roots, key=lambda x: x[1])
        nearest_roots = [i for i, _ in nearest_roots][:nearest_n_lim]

        transit_ubs = {r: root_transit_ub[r] for r in nearest_roots}
        node["transit_bounds"] = transit_ubs
        # logger.debug(
        #     f"{solver_graph[i]['waypoint_key']:>5} { {solver_graph[i]['waypoint_key']: ub for i, ub in transit_ubs.items()} }"
        # )


def prune_NPD1(G: rx.PyDiGraph, quiet: bool = False) -> int:
    """In-Place removal of non-plant non-forced leaf nodes."""
    if not quiet:
        logger.info("Pruning NPD1...")

    num_removed = 0
    while removal_nodes := [
        v
        for v in G.node_indices()
        if G.out_degree(v) == 1
        and not G[v].get("is_super_terminal", False)
        and not G[v]["is_workerman_plantzone"]
    ]:
        G.remove_nodes_from(removal_nodes)
        num_removed += len(removal_nodes)
        removal_nodes = []

    if not quiet:
        logger.debug(f"  removed {num_removed} leaf nodes")
    return num_removed


def reduce_bounds(G: rx.PyDiGraph, super_root_index: int | None):
    """Limits transit upper bounds via intersection with neighbor bounds."""
    logger.info("Reducing bounds...")

    ubs_to_deactivate: list[set[int]] = []
    for i in G.node_indices():
        node = G[i]
        successors = G.successor_indices(i)

        # NOTE: This may be too much and could cut out potential paths because it would
        # cascade unless we store the changes and then apply them at the end of processing
        # which short-circuits the cascading... testing on our input graph shows this is OK.

        # Reduce root ub each (node, successor) pair to the minimum between them
        # and then the minimum of those results, since each ub is equal to the capacity
        # of the root this is a simple all or nothing per root...
        # This essentially cuts out the fringe of the potential transit flow network.
        active_ubs = set(k for k in node["transit_bounds"].keys() if node["transit_bounds"][k] > 0)
        for j in successors:
            neighbor = G[j]
            neighbor_ubs = set(
                k for k in neighbor["transit_bounds"].keys() if neighbor["transit_bounds"][k] > 0
            )
            active_ubs &= neighbor_ubs

        deactivate_ubs = set(k for k in node["transit_bounds"].keys() if k not in active_ubs)
        ubs_to_deactivate.append(deactivate_ubs)

    for i, G_i in enumerate(G.node_indices()):
        node = G[G_i]
        for ub in ubs_to_deactivate[i]:
            node["transit_bounds"].pop(ub, None)

    # Remove root from disconnected transit layer components
    for r in G.attrs["root_indices"]:
        # skip super root
        if r == super_root_index:
            continue
        # node_map maps from subgraph node index to global graph node index
        subG, node_map = G.subgraph_with_nodemap([i for i in G.node_indices() if r in G[i]["transit_bounds"]])
        node_map = bidict(node_map)
        subg_r = node_map.inv[r]
        cc = rx.weakly_connected_components(subG)
        for c in cc:
            if subg_r in c:
                continue
            for i in c:
                G[node_map[i]]["transit_bounds"].pop(r, None)

        subG, node_map = G.subgraph_with_nodemap([i for i in G.node_indices() if r in G[i]["transit_bounds"]])


def reduce_bounds_via_root_pruning(G: rx.PyDiGraph) -> None:
    """
    Limits transit upper bounds by performing Pruning on the root-specific subgraphs.
    This effectively eliminates dead-end branches in the potential flow network for each root.
    """
    logger.info("Reducing bounds via root-specific pruning (RSP)...")

    from api_common import SUPER_ROOT

    super_root_index = None
    super_terminal_indices = []
    for i in G.node_indices():
        if G[i].get("is_super_terminal", False):
            super_terminal_indices.append(i)
        if G[i]["waypoint_key"] == SUPER_ROOT:
            super_root_index = i

    flow_roots = G.attrs["root_indices"].copy()
    if super_root_index is not None:
        flow_roots.append(super_root_index)

    for r in flow_roots:
        transit_nodes = {
            i
            for i in G.node_indices()
            if r in G[i].get("transit_bounds", {}) and G[i]["transit_bounds"][r] > 0
        }
        if not transit_nodes:
            continue

        transit_subgraph, node_map = G.subgraph_with_nodemap(list(transit_nodes))
        removed_count = prune_NPD1(transit_subgraph, quiet=True)
        surviving_transit_nodes = {node_map[i] for i in transit_subgraph.node_indices()}
        removed_transit_nodes = set(transit_nodes) - surviving_transit_nodes
        if removed_transit_nodes:
            for i in removed_transit_nodes:
                G[i]["transit_bounds"].pop(r, None)

        if removed_count:
            logger.debug(f"  removed {removed_count} leaf nodes from root {r}")


def setup_basin_bottlenecks(
    G: rx.PyDiGraph, roots_indices: list[int], terminal_indices: list[int], super_root_index: int | None
):
    """Compute per-root basins via Stoer-Wagner min-cut on undirected transit subgraphs."""
    basins = {}
    for r in roots_indices:
        if super_root_index is not None and r == super_root_index:
            continue

        transit_nodes = [
            i
            for i in G.node_indices()
            if r in G[i].get("transit_bounds", {}) and G[i]["transit_bounds"][r] > 0
        ]
        if len(transit_nodes) < 2:
            continue

        subG, node_map = G.subgraph_with_nodemap(transit_nodes)
        node_map = bidict(node_map)  # local -> global, global -> local
        global_to_local = node_map.inv

        undir = subG.to_undirected()
        cut_value, cut_partition = rx.stoer_wagner_min_cut(undir)

        if cut_value == 0:
            logger.debug(f"  Skipping root {r}: disconnected subgraph")
            continue

        # Partitions in local indices
        partition1_local = set(cut_partition)
        partition2_local = set(undir.node_indices()) - partition1_local

        # Ensure r in partition1
        r_local = global_to_local[r]
        if r_local in partition2_local:
            partition1_local, partition2_local = partition2_local, partition1_local

        # Global nodes
        p1_global = {node_map[ln] for ln in partition1_local}
        p2_global = {node_map[ln] for ln in partition2_local}

        basin_terminals = [t for t in terminal_indices if t in p1_global]
        logger.debug(
            f"  basin for root {G[r]['waypoint_key']} (cut_value={cut_value}): {[G[t]['waypoint_key'] for t in basin_terminals]}"
        )

        # Cut nodes: boundary (neighbors in p2 from p1)
        cut_nodes = set()
        for ln in partition1_local:
            gn = node_map[ln]
            for neigh_local in undir.neighbors(ln):
                neigh_gn = node_map[neigh_local]
                if neigh_gn in p2_global:
                    cut_nodes.add(gn)
                    cut_nodes.add(neigh_gn)
        cut_nodes = list(cut_nodes)

        if not cut_nodes:
            logger.debug(f"  No boundary for root {r}; skipping")
            continue

        logger.debug(f"  cut nodes: {[G[n]['waypoint_key'] for n in cut_nodes]}")

        basins[r] = (basin_terminals, cut_value, cut_nodes)

    G.attrs["basins"] = basins
    logger.info(f"Computed basins for {len(basins)} roots")


def setup_ranked_basin_bottlenecks(G: rx.PyDiGraph, super_root_index: int | None) -> None:
    """Compute per-root basins via rank-1 core union and connectivity backfill."""
    basins: dict[int, tuple[list[int], int, list[int]]] = {}
    roots_indices = G.attrs["root_indices"]
    terminal_indices = G.attrs["terminals"]

    # Add edge costs to edges based on destination need_exploration_point
    for u, v in G.edge_list():
        G.update_edge(u, v, {"weight": G[v]["need_exploration_point"]})

    for r in roots_indices:
        # Step a: Root transit subgraph (omit super root)
        if super_root_index is not None and r == super_root_index:
            continue
        transit_nodes = {i for i in G.node_indices() if r in G[i].get("transit_bounds", {})}
        transit_subG, node_map = G.subgraph_with_nodemap(list(transit_nodes))
        node_map = bidict(node_map)

        do_debug = True if G[r]["waypoint_key"] in DEBUG_ROOTS else False
        if do_debug:
            logger.warning(f"  Computing rank-1 basins for root {G[r]['waypoint_key']}...")
            logger.warning(
                f"    transit_subG nodes: {sorted([transit_subG[n]['waypoint_key'] for n in transit_subG.node_indices()])}"
            )

        # Step b: Remove non-rank1 terminals from transit_subG
        transit_terminals = [t for t in terminal_indices if t in transit_nodes]

        # for rank 1 only
        rank1_ts = [t for t in transit_terminals if list(G[t]["prizes"].keys())[0] == r]

        # for all [:n] ranks
        # rank1_ts = [t for t in transit_terminals if r not in list(G[t]["prizes"].keys())[:1]]

        non_rank1_ts = set(transit_terminals) - set(rank1_ts)
        transit_subG_prime = transit_subG.copy()
        transit_subG_prime.remove_nodes_from([node_map.inv[t] for t in non_rank1_ts if t in node_map])
        prune_NPD1(transit_subG_prime, quiet=True)

        if do_debug:
            logger.warning(f"    rank1 terminals: {[G[n]['waypoint_key'] for n in rank1_ts]}")
            logger.warning(
                f"    rank1 transit_subG_prime nodes: {sorted([transit_subG_prime[n]['waypoint_key'] for n in transit_subG_prime.node_indices()])}"
            )

        # Step c: All-pairs shortest paths on remaining terminals + root in subG
        shortest_paths = rx.digraph_all_pairs_dijkstra_shortest_paths(
            transit_subG_prime, lambda payload: payload["weight"]
        )
        used_nodes = set()
        for source, dest_paths in shortest_paths.items():
            if node_map[source] != r and node_map[source] not in rank1_ts:
                continue
            for dest, path in dest_paths.items():
                if node_map[dest] not in rank1_ts:
                    # if do_debug:
                    #     logger.warning(
                    #         f"    skipping non-rank1 terminal {transit_subG_prime[dest]['waypoint_key']} from {transit_subG_prime[source]['waypoint_key']}"
                    #     )
                    continue
                used_nodes.update(path)
                if do_debug:
                    logger.warning(
                        f"    shortest path from {transit_subG_prime[source]['waypoint_key']} to {transit_subG_prime[dest]['waypoint_key']}: {sorted([transit_subG_prime[n]['waypoint_key'] for n in path])}"
                    )

        if do_debug:
            logger.warning(
                f"    used nodes: {sorted([transit_subG_prime[n]['waypoint_key'] for n in used_nodes])}"
            )

        # Step d: Keep only union nodes in transit_subG
        core_global = {node_map[ln] for ln in used_nodes}
        if len(core_global) < 2:
            logger.debug(f"  Skipping root {r}: insufficient core")
            continue

        if do_debug:
            logger.warning(f"    core nodes: {sorted([G[n]['waypoint_key'] for n in core_global])}")

        # Step e: New subG with global terminals + core
        basin_nodes = core_global | set(terminal_indices)
        basin_subG, basin_map = G.subgraph_with_nodemap(list(basin_nodes))
        basin_map = bidict(basin_map)

        # Step f: Remove isolates
        isolates_local = rx.isolates(basin_subG)
        basin_subG.remove_nodes_from(isolates_local)
        basin_global = {basin_map[ln] for ln in basin_subG.node_indices()}
        all_enclosed_ts = sorted([t for t in terminal_indices if t in basin_global])

        if do_debug:
            logger.warning(f"    basin nodes: {sorted([G[n]['waypoint_key'] for n in basin_global])}")
            logger.warning(
                f"    all enclosed terminals: {sorted([G[n]['waypoint_key'] for n in all_enclosed_ts])}"
            )

        # Step g: Full transit subG for cut
        full_transit_subG, _ = G.subgraph_with_nodemap(list(transit_nodes))

        # Step h: Cut nodes = transit nodes not in basin_global with a neighbor in basin_global
        cut_nodes = set()
        # for node in G.node_indices():
        # if node in basin_global:
        #     continue
        for node in sorted(list(transit_nodes - basin_global)):
            if any([nb in basin_global for nb in G.neighbors(node)]):
                cut_nodes.add(node)
        cut_value = len(cut_nodes)

        if not all_enclosed_ts or not cut_nodes:
            logger.debug(f"  Skipping root {r}: empty basin or cuts")
            continue

        basins[r] = (all_enclosed_ts, cut_value, cut_nodes)
        if do_debug:
            logger.warning(
                f"  basin for root {G[r]['waypoint_key']} (cut_value={cut_value}): "
                f"{sorted([G[t]['waypoint_key'] for t in all_enclosed_ts])}"
            )
            logger.warning(f"  cut nodes: {sorted([G[n]['waypoint_key'] for n in cut_nodes])}")

    G.attrs["basins"] = basins
    logger.info(f"Computed ranked basins for {len(basins)} roots")


def generate_graph_data(
    data: dict[str, Any], do_prune: bool, do_reduce: bool, basin_type: int
) -> dict[str, Any]:
    """Generate and return a GraphData Dict composing the LP empire data."""
    print("Generating graph data...")

    solver_graph = data["exploration_graph"].copy()

    super_root_index = prep_graph_nodes(solver_graph, data)
    if do_prune:
        # overall pruning of non terminal leaf nodes from graph
        prune_NPD1(solver_graph)
    if do_prune:
        # NPD1 pruning of transit region boundary flow per root
        reduce_bounds_via_root_pruning(solver_graph)
    if do_reduce:
        # transit region boundary flow reductions
        reduce_bounds(solver_graph, super_root_index)
        # clean up transit layer tree branches
        reduce_bounds_via_root_pruning(solver_graph)

    if basin_type > 0:
        if basin_type == 1:
            setup_basin_bottlenecks(
                solver_graph,
                solver_graph.attrs["root_indices"],
                solver_graph.attrs["terminals"],
                super_root_index,
            )
        else:
            setup_ranked_basin_bottlenecks(solver_graph, super_root_index)

    # Debugging transit layer subgraph nodes
    for r in solver_graph.attrs["root_indices"]:
        transit_nodes = [i for i in solver_graph.node_indices() if r in solver_graph[i]["transit_bounds"]]
        logger.debug(
            f"Root {solver_graph[r]['waypoint_key']}: transit layer: {[solver_graph[r]['waypoint_key']] + [solver_graph[i]['waypoint_key'] for i in transit_nodes]}"
        )

    data["solver_graph"] = solver_graph

    return data
