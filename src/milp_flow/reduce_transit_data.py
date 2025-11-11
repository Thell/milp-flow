# reduce_solver_graph.py

from typing import Any
from bidict import bidict
from loguru import logger
from rustworkx import (
    PyDiGraph,
    weakly_connected_components,
    stoer_wagner_min_cut,
    digraph_all_pairs_dijkstra_shortest_paths,
    isolates,
)

# DEBUG_ROOTS = [1623, 1785, 1834, 2001]
DEBUG_ROOTS = []


def prune_NPD1(solver_graph: PyDiGraph, quiet: bool = False) -> int:
    """In-Place removal of non-plant non-forced leaf nodes."""
    if not quiet:
        logger.info("      pruning NPD1...")

    num_removed = 0
    while removal_nodes := [
        v
        for v in solver_graph.node_indices()
        if solver_graph.out_degree(v) == 1
        and not solver_graph[v].get("is_super_terminal", False)
        and not solver_graph[v]["is_workerman_plantzone"]
        and not solver_graph[v]["is_warehouse_town"]
    ]:
        solver_graph.remove_nodes_from(removal_nodes)
        num_removed += len(removal_nodes)
        removal_nodes = []

    if not quiet:
        logger.debug(f"      removed {num_removed} leaf nodes")
    return num_removed


def reduce_bounds(solver_graph: PyDiGraph):
    """Limits transit upper bounds via intersection with neighbor bounds."""
    logger.info("      reducing upper bounds via intersections...")

    super_root_index = solver_graph.attrs["super_root_index"]
    ubs_to_deactivate: list[set[int]] = []
    for i in solver_graph.node_indices():
        node = solver_graph[i]
        successors = solver_graph.successor_indices(i)

        # NOTE: This may be too much and could cut out potential paths because it would
        # cascade unless we store the changes and then apply them at the end of processing
        # which short-circuits the cascading... testing on our input graph shows this is OK.

        # Reduce root ub each (node, successor) pair to the minimum between them
        # and then the minimum of those results, since each ub is equal to the capacity
        # of the root this is a simple all or nothing per root...
        # This essentially cuts out the fringe of the potential transit flow network.
        active_ubs = set(k for k in node["transit_bounds"].keys() if node["transit_bounds"][k] > 0)
        for j in successors:
            neighbor = solver_graph[j]
            neighbor_ubs = set(
                k for k in neighbor["transit_bounds"].keys() if neighbor["transit_bounds"][k] > 0
            )
            active_ubs &= neighbor_ubs

        deactivate_ubs = set(k for k in node["transit_bounds"].keys() if k not in active_ubs)
        ubs_to_deactivate.append(deactivate_ubs)

    for i, G_i in enumerate(solver_graph.node_indices()):
        node = solver_graph[G_i]
        for ub in ubs_to_deactivate[i]:
            node["transit_bounds"].pop(ub, None)

    # Remove root from disconnected transit layer components
    for r in solver_graph.attrs["root_indices"]:
        # skip super root
        if r == super_root_index:
            continue
        # node_map maps from subgraph node index to global graph node index
        subG, node_map = solver_graph.subgraph_with_nodemap([
            i for i in solver_graph.node_indices() if r in solver_graph[i]["transit_bounds"]
        ])
        node_map = bidict(node_map)
        subg_r = node_map.inv[r]
        cc = weakly_connected_components(subG)
        for c in cc:
            if subg_r in c:
                continue
            for i in c:
                solver_graph[node_map[i]]["transit_bounds"].pop(r, None)

        subG, node_map = solver_graph.subgraph_with_nodemap([
            i for i in solver_graph.node_indices() if r in solver_graph[i]["transit_bounds"]
        ])


def reduce_bounds_via_root_pruning(solver_graph: PyDiGraph):
    """
    Limits transit upper bounds by performing Pruning on the root-specific subgraphs.
    This effectively eliminates dead-end branches in the potential flow network for each root.
    """
    logger.info("      reducing bounds via root-specific pruning...")

    from api_common import SUPER_ROOT

    super_root_index = None
    super_terminal_indices = []
    for i in solver_graph.node_indices():
        if solver_graph[i].get("is_super_terminal", False):
            super_terminal_indices.append(i)
        if solver_graph[i]["waypoint_key"] == SUPER_ROOT:
            super_root_index = i

    flow_roots = solver_graph.attrs["root_indices"].copy()
    if super_root_index is not None:
        flow_roots.append(super_root_index)

    for r in flow_roots:
        transit_nodes = {
            i
            for i in solver_graph.node_indices()
            if r in solver_graph[i].get("transit_bounds", {}) and solver_graph[i]["transit_bounds"][r] > 0
        }
        if not transit_nodes:
            continue

        transit_subgraph, node_map = solver_graph.subgraph_with_nodemap(list(transit_nodes))
        removed_count = prune_NPD1(transit_subgraph, quiet=True)
        surviving_transit_nodes = {node_map[i] for i in transit_subgraph.node_indices()}
        removed_transit_nodes = set(transit_nodes) - surviving_transit_nodes
        if removed_transit_nodes:
            for i in removed_transit_nodes:
                solver_graph[i]["transit_bounds"].pop(r, None)

        if removed_count:
            logger.debug(f"      removed {removed_count} leaf nodes from root {r}")


def setup_basin_bottlenecks(solver_graph: PyDiGraph):
    """Compute per-root basins via Stoer-Wagner min-cut on undirected transit subgraphs."""
    root_indices = solver_graph.attrs["root_indices"]
    super_root_index = solver_graph.attrs["super_root_index"]
    terminal_indices = solver_graph.attrs["terminal_indices"]

    basins = {}
    for r in root_indices:
        if super_root_index is not None and r == super_root_index:
            continue

        transit_nodes = [
            i
            for i in solver_graph.node_indices()
            if r in solver_graph[i].get("transit_bounds", {}) and solver_graph[i]["transit_bounds"][r] > 0
        ]
        if len(transit_nodes) < 2:
            continue

        subG, node_map = solver_graph.subgraph_with_nodemap(transit_nodes)
        node_map = bidict(node_map)  # local -> global, global -> local
        global_to_local = node_map.inv

        undir = subG.to_undirected()
        result = stoer_wagner_min_cut(undir)
        assert result is not None
        cut_value, cut_partition = result

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
            f"  basin for root {solver_graph[r]['waypoint_key']} (cut_value={cut_value}): {[solver_graph[t]['waypoint_key'] for t in basin_terminals]}"
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

        logger.debug(f"  cut nodes: {[solver_graph[n]['waypoint_key'] for n in cut_nodes]}")

        basins[r] = (basin_terminals, cut_value, cut_nodes)

    solver_graph.attrs["basins"] = basins
    logger.info(f"      computed basins for {len(basins)} roots")


def setup_ranked_basin_bottlenecks(solver_graph: PyDiGraph):
    """Compute per-root basins via rank-1 core union and connectivity backfill."""
    basins: dict[int, tuple[list[int], int, list[int]]] = {}
    roots_indices = solver_graph.attrs["root_indices"]
    super_root_index = solver_graph.attrs["super_root_index"]
    terminal_indices = solver_graph.attrs["terminal_indices"]

    # Add edge costs to edges based on destination need_exploration_point
    for u, v in solver_graph.edge_list():
        solver_graph.update_edge(u, v, {"weight": solver_graph[v]["need_exploration_point"]})

    for r in roots_indices:
        # Step a: Root transit subgraph (omit super root)
        if super_root_index is not None and r == super_root_index:
            continue
        transit_nodes = {
            i for i in solver_graph.node_indices() if r in solver_graph[i].get("transit_bounds", {})
        }
        transit_subG, node_map = solver_graph.subgraph_with_nodemap(list(transit_nodes))
        node_map = bidict(node_map)

        do_debug = True if solver_graph[r]["waypoint_key"] in DEBUG_ROOTS else False
        if do_debug:
            logger.warning(f"Computing rank-1 basins for root {solver_graph[r]['waypoint_key']}...")
            logger.warning(
                f"transit_subG nodes: {sorted([transit_subG[n]['waypoint_key'] for n in transit_subG.node_indices()])}"
            )

        # Step b: Remove non-rank1 terminals from transit_subG
        transit_terminals = [t for t in terminal_indices if t in transit_nodes]

        # for rank 1 only
        rank1_ts = [t for t in transit_terminals if list(solver_graph[t]["prizes"].keys())[0] == r]

        # for all [:n] ranks
        # rank1_ts = [t for t in transit_terminals if r not in list(G[t]["prizes"].keys())[:1]]

        non_rank1_ts = set(transit_terminals) - set(rank1_ts)
        transit_subG_prime = transit_subG.copy()
        transit_subG_prime.remove_nodes_from([node_map.inv[t] for t in non_rank1_ts if t in node_map])
        prune_NPD1(transit_subG_prime, quiet=True)

        if do_debug:
            logger.warning(f"rank1 terminals: {[solver_graph[n]['waypoint_key'] for n in rank1_ts]}")
            logger.warning(
                f"rank1 transit_subG_prime nodes: {sorted([transit_subG_prime[n]['waypoint_key'] for n in transit_subG_prime.node_indices()])}"
            )

        # Step c: All-pairs shortest paths on remaining terminals + root in subG
        shortest_paths = digraph_all_pairs_dijkstra_shortest_paths(
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
                        f"shortest path from {transit_subG_prime[source]['waypoint_key']} to {transit_subG_prime[dest]['waypoint_key']}: {sorted([transit_subG_prime[n]['waypoint_key'] for n in path])}"
                    )

        if do_debug:
            logger.warning(
                f"used nodes: {sorted([transit_subG_prime[n]['waypoint_key'] for n in used_nodes])}"
            )

        # Step d: Keep only union nodes in transit_subG
        core_global = {node_map[ln] for ln in used_nodes}
        if len(core_global) < 2:
            logger.debug(f"Skipping root {r}: insufficient core")
            continue

        if do_debug:
            logger.warning(f"core nodes: {sorted([solver_graph[n]['waypoint_key'] for n in core_global])}")

        # Step e: New subG with global terminals + core
        basin_nodes = core_global | set(terminal_indices)
        basin_subG, basin_map = solver_graph.subgraph_with_nodemap(list(basin_nodes))
        basin_map = bidict(basin_map)

        # Step f: Remove isolates
        isolates_local = isolates(basin_subG)
        basin_subG.remove_nodes_from(isolates_local)
        basin_global = {basin_map[ln] for ln in basin_subG.node_indices()}
        all_enclosed_ts = sorted([t for t in terminal_indices if t in basin_global])

        if do_debug:
            logger.warning(
                f"    basin nodes: {sorted([solver_graph[n]['waypoint_key'] for n in basin_global])}"
            )
            logger.warning(
                f"    all enclosed terminals: {sorted([solver_graph[n]['waypoint_key'] for n in all_enclosed_ts])}"
            )

        # Step g: Full transit subG for cut
        full_transit_subG, _ = solver_graph.subgraph_with_nodemap(list(transit_nodes))

        # Step h: Cut nodes = transit nodes not in basin_global with a neighbor in basin_global
        cut_nodes = set()
        for node in sorted(list(transit_nodes - basin_global)):
            if any([nb in basin_global for nb in solver_graph.neighbors(node)]):
                cut_nodes.add(node)
        cut_value = len(cut_nodes)

        if not all_enclosed_ts or not cut_nodes:
            logger.debug(f"  Skipping root {r}: empty basin or cuts")
            continue

        basins[r] = (all_enclosed_ts, cut_value, list(cut_nodes))
        if do_debug:
            logger.warning(
                f"  basin for root {solver_graph[r]['waypoint_key']} (cut_value={cut_value}): "
                f"{sorted([solver_graph[t]['waypoint_key'] for t in all_enclosed_ts])}"
            )
            logger.warning(f"  cut nodes: {sorted([solver_graph[n]['waypoint_key'] for n in cut_nodes])}")

    solver_graph.attrs["basins"] = basins
    logger.info(f"      computed ranked basins for {len(basins)} roots")


def reduce_transit_data(data: dict[str, Any]):
    """Reduces per root transit layer data via pruning and reduction.

    SAFETY: The solver_graph is modified in-place!
    """
    logger.info("    reducing transit data...")

    solver_graph = data["solver_graph"]

    start_flow_var_count = sum(len(solver_graph[i]["transit_bounds"]) for i in solver_graph.node_indices())

    basin_type = data["config"]["transit_basin_type"]
    do_prune = data["config"]["transit_prune"]
    do_reduce = data["config"]["transit_reduce"]

    if do_prune:
        # overall pruning of non terminal leaf nodes from graph
        prune_NPD1(solver_graph)
    if do_prune:
        # NPD1 pruning of transit region boundary flow per root
        reduce_bounds_via_root_pruning(solver_graph)
    if do_reduce:
        # transit region boundary flow reductions
        reduce_bounds(solver_graph)
        # clean up transit layer tree branches
        reduce_bounds_via_root_pruning(solver_graph)

    if basin_type > 0:
        if basin_type == 1:
            # Type 1 is a simple Stoer-Wagner min-cut - just used for testing.
            setup_basin_bottlenecks(solver_graph)
        else:
            setup_ranked_basin_bottlenecks(solver_graph)

    # Remove nodes with no transit bounds
    no_transit_bounds_nodes = [
        i for i in solver_graph.node_indices() if not solver_graph[i].get("transit_bounds")
    ]
    solver_graph.remove_nodes_from(no_transit_bounds_nodes)
    if no_transit_bounds_nodes:
        logger.info(f"      removed {len(no_transit_bounds_nodes)} nodes with no transit bounds")

    # Debugging transit layer subgraph nodes
    flow_var_count = 0
    for r in solver_graph.attrs["root_indices"]:
        transit_nodes = [i for i in solver_graph.node_indices() if r in solver_graph[i]["transit_bounds"]]
        flow_var_count += len(transit_nodes)
        logger.trace(
            f"Root {solver_graph[r]['waypoint_key']}: transit layer: {[solver_graph[r]['waypoint_key']] + [solver_graph[i]['waypoint_key'] for i in transit_nodes]}"
        )
    logger.info(
        f"    final transit flow var count: {flow_var_count} / {start_flow_var_count} ({flow_var_count / start_flow_var_count * 100:.2f}%)"
    )
