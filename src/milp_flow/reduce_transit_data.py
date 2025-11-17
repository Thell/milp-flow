# reduce_transit_data.py

from typing import Any
from bidict import bidict
from loguru import logger
from rustworkx import (
    PyDiGraph,
    strongly_connected_components,
    weakly_connected_components,
    stoer_wagner_min_cut,
    digraph_all_pairs_dijkstra_shortest_paths,
    isolates,
)

from api_rx_pydigraph import subgraph_stable

# DEBUG_ROOTS = [1623, 1785, 1834, 2001]
DEBUG_ROOTS = [1750]


def prune_NTD1(graph: PyDiGraph, non_removables: set[int] | None = None, quiet: bool = False) -> int:
    """Recursive removal of transit-only leaf nodes."""
    if not quiet:
        logger.info("      pruning NTD1...")

    if non_removables is None:
        non_removables = set(graph.attrs.get("root_indices", []))
        non_removables.update(set(graph.attrs.get("terminal_indices", [])))
        if graph.attrs.get("super_root_index", None) is not None:
            non_removables.add(graph.attrs["super_root_index"])
            non_removables.update(set(graph.attrs.get("super_terminal_indices", [])))

    num_removed = 0
    while removal_nodes := [
        v for v in graph.node_indices() if graph.out_degree(v) == 1 and v not in non_removables
    ]:
        graph.remove_nodes_from(removal_nodes)
        num_removed += len(removal_nodes)
        removal_nodes = []

    if not quiet:
        logger.debug(f"      removed {num_removed} leaf nodes")
    return num_removed


def prune_transit_layer_NTD1(solver_graph: PyDiGraph):
    """Recursive removal of transit-only leaf nodes for each root-specific transit layer."""
    logger.info("      pruning root-specific transit layers NTD1...")

    flow_roots = solver_graph.attrs["root_indices"].copy()
    super_root_index = solver_graph.attrs["super_root_index"]
    if super_root_index is not None:
        flow_roots.append(super_root_index)

    cc_mismatch_found = False

    for r in flow_roots:
        transit_layer_nodes = {
            i for i in solver_graph.node_indices() if r in solver_graph[i].get("transit_bounds", {})
        }
        if not transit_layer_nodes:
            continue

        transit_layer_subgraph = subgraph_stable(solver_graph, transit_layer_nodes)
        assert isinstance(transit_layer_subgraph, PyDiGraph)

        pre_prune_cc = strongly_connected_components(transit_layer_subgraph)
        logger.debug(f"        transit layer {r} has {len(pre_prune_cc)} connected components")

        # Non_removables for this layer are the root and the terminals with prizes for this root
        non_removables = {r}
        non_removables.update({
            i for i in transit_layer_subgraph.node_indices() if i in solver_graph.attrs["terminal_indices"]
        })
        removed_count = prune_NTD1(transit_layer_subgraph, non_removables, quiet=True)

        post_prune_cc = strongly_connected_components(transit_layer_subgraph)
        if len(post_prune_cc) != len(pre_prune_cc):
            cc_mismatch_found = True
            logger.warning(
                f"        transit layer {r} has {len(post_prune_cc)} connected components after pruning"
            )
            logger.warning(f"        pre-prune CC: {list(pre_prune_cc)}")
            logger.warning(f"        post-prune CC: {list(post_prune_cc)}")
            breakpoint()

        surviving_transit_layer_nodes = set(transit_layer_subgraph.node_indices())
        removed_transit_layer_nodes = set(transit_layer_nodes) - surviving_transit_layer_nodes
        for i in removed_transit_layer_nodes:
            solver_graph[i]["transit_bounds"].pop(r)

        if removed_count:
            logger.debug(f"        removed {removed_count} leaf transit entries from root {r}")

        assert not cc_mismatch_found


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

    logger.info(
        f"      reduced upper bounds via intersections: {sum(len(x) for x in ubs_to_deactivate)} removed"
    )

    # Remove root from disconnected transit layer components
    root_transit_bounds_removed = 0
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
                root_transit_bounds_removed += 1
                solver_graph[node_map[i]]["transit_bounds"].pop(r, None)

        subG, node_map = solver_graph.subgraph_with_nodemap([
            i for i in solver_graph.node_indices() if r in solver_graph[i]["transit_bounds"]
        ])

    logger.info(f"      reduced upper bounds via intersections: {root_transit_bounds_removed} removed")


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
    do_debug = logger._core.min_level <= logger._core.levels_lookup["DEBUG"][2]

    # Add edge costs to edges based on destination need_exploration_point
    for u, v in solver_graph.edge_list():
        solver_graph.update_edge(u, v, {"weight": solver_graph[v]["need_exploration_point"]})

    for r in roots_indices:
        # Step a: Root transit subgraph (omit super root)
        if super_root_index is not None and r == super_root_index:
            continue
        transit_nodes = {i for i in solver_graph.node_indices() if r in solver_graph[i]["transit_bounds"]}
        transit_layer_subgraph = subgraph_stable(solver_graph, transit_nodes)
        assert isinstance(transit_layer_subgraph, PyDiGraph)
        assert len(strongly_connected_components(transit_layer_subgraph)) == 1

        # Non_removables for this layer are the root and the terminals with prizes for this root
        non_removables = {r}
        non_removables.update({
            i for i in transit_layer_subgraph.node_indices() if i in solver_graph.attrs["terminal_indices"]
        })
        _removed_count = prune_NTD1(transit_layer_subgraph, non_removables, quiet=True)

        if do_debug and solver_graph[r]["waypoint_key"] in DEBUG_ROOTS:
            logger.warning(f"Computing rank-1 basins for root {solver_graph[r]['waypoint_key']}...")
            logger.warning(
                f"transit_layer_subgraph nodes: {sorted([transit_layer_subgraph[n]['waypoint_key'] for n in transit_layer_subgraph.node_indices()])}"
            )

        # Step b: Remove non-rank1 terminals from transit_layer_subgraph
        transit_terminals = [t for t in terminal_indices if t in transit_nodes]

        # # for rank 1 only
        rank1_ts = [t for t in transit_terminals if list(solver_graph[t]["prizes"].keys())[0] == r]

        # # for all [:n] ranks
        # top_n = min(2, len(solver_graph[transit_terminals[0]]["prizes"]))
        # rank1_ts = [t for t in transit_terminals if r not in list(solver_graph[t]["prizes"].keys())[:top_n]]

        non_rank1_ts = set(transit_terminals) - set(rank1_ts)
        transit_layer_subgraph_prime = transit_layer_subgraph.copy()
        transit_layer_subgraph_prime.remove_nodes_from(list(non_rank1_ts))
        prune_NTD1(transit_layer_subgraph_prime, non_removables, quiet=True)

        if do_debug and solver_graph[r]["waypoint_key"] in DEBUG_ROOTS:
            logger.warning(f"rank1 terminals: {[solver_graph[n]['waypoint_key'] for n in rank1_ts]}")
            logger.warning(
                f"rank1 transit_layer_subgraph_prime nodes: {sorted([transit_layer_subgraph_prime[n]['waypoint_key'] for n in transit_layer_subgraph_prime.node_indices()])}"
            )

        # Step c: All-pairs shortest paths on remaining terminals + root in subG
        shortest_paths = digraph_all_pairs_dijkstra_shortest_paths(
            transit_layer_subgraph_prime, lambda payload: payload["weight"]
        )
        used_nodes = set()
        for source, dest_paths in shortest_paths.items():
            if source != r and source not in rank1_ts:
                continue
            for dest, path in dest_paths.items():
                if dest not in rank1_ts:
                    # if do_debug:
                    #     logger.warning(
                    #         f"    skipping non-rank1 terminal {transit_layer_subgraph_prime[dest]['waypoint_key']} from {transit_layer_subgraph_prime[source]['waypoint_key']}"
                    #     )
                    continue
                used_nodes.update(path)
                # if do_debug and solver_graph[r]["waypoint_key"] in DEBUG_ROOTS:
                #     logger.warning(
                #         f"shortest path from {transit_layer_subgraph_prime[source]['waypoint_key']} to {transit_layer_subgraph_prime[dest]['waypoint_key']}: {sorted([transit_layer_subgraph_prime[n]['waypoint_key'] for n in path])}"
                #     )

        if do_debug and solver_graph[r]["waypoint_key"] in DEBUG_ROOTS:
            logger.warning(
                f"used nodes: {sorted([transit_layer_subgraph_prime[n]['waypoint_key'] for n in used_nodes])}"
            )

        if len(used_nodes) < 2:
            logger.debug(f"Skipping root {r}: insufficient core")
            continue

        if do_debug and solver_graph[r]["waypoint_key"] in DEBUG_ROOTS:
            logger.warning(f"core nodes: {sorted([solver_graph[n]['waypoint_key'] for n in used_nodes])}")

        # Step e: New subG with global terminals + core
        basin_nodes = used_nodes | set(terminal_indices)
        basin_subG, basin_map = solver_graph.subgraph_with_nodemap(list(basin_nodes))
        basin_map = bidict(basin_map)

        # Step f: Remove isolates
        isolates_local = isolates(basin_subG)
        basin_subG.remove_nodes_from(isolates_local)
        basin_global = {basin_map[ln] for ln in basin_subG.node_indices()}
        all_enclosed_ts = sorted([t for t in terminal_indices if t in basin_global])

        if do_debug and solver_graph[r]["waypoint_key"] in DEBUG_ROOTS:
            logger.warning(
                f"    basin nodes: {sorted([solver_graph[n]['waypoint_key'] for n in basin_global])}"
            )
            logger.warning(
                f"    all enclosed terminals: {sorted([solver_graph[n]['waypoint_key'] for n in all_enclosed_ts])}"
            )

        # Step g: Full transit subG for cut
        _full_transit_layer_subgraph = subgraph_stable(solver_graph, transit_nodes)

        # Step h: Cut nodes = transit nodes not in basin_global with a neighbor in basin_global
        cut_nodes = set()
        for node in sorted(list(transit_nodes - basin_global)):
            if any([nb in basin_global for nb in solver_graph.neighbors(node)]):
                cut_nodes.add(node)
        cut_value = len(cut_nodes)

        if not all_enclosed_ts or not cut_nodes:
            logger.debug(f"  Skipping root {solver_graph[r]['waypoint_key']}: empty basin or cuts")
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


def prune_low_asp_nodes(solver_graph: PyDiGraph, data: dict[str, Any]) -> int:
    """DO NOT USE YET! Prune non-terminal nodes with low ASP betweenness (if enabled)."""
    logger.info("      pruning low-ASP nodes...")
    if not data["config"].get("transit_prune_low_asp", False):
        return 0

    metrics = solver_graph.attrs.get("exploration_metrics", {})
    if "bc_tr_asp" not in metrics:
        logger.warning("        no bc_tr_asp; skipping ASP prune")
        return 0

    untouchables = set(solver_graph.attrs["root_indices"]).union(solver_graph.attrs["terminal_indices"])

    bc_asp = metrics["bc_tr_asp"]
    max_score = max(bc_asp.values() or [1])
    threshold = 0.0000012283633212355
    low_asp = [n for n, score in bc_asp.items() if score < threshold and n not in untouchables]
    num_pruned = 0
    for n in low_asp:
        logger.warning(f"        pruning low-ASP node {solver_graph[n]['waypoint_key']}")
        solver_graph.remove_node(n)
        num_pruned += 1
    logger.warning(f"        pruned {num_pruned} low-ASP nodes (threshold={threshold:.4f})")
    return num_pruned


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
    do_asp_prune = data["config"]["transit_prune_low_asp"]

    if do_prune:
        prune_NTD1(solver_graph)
        prune_transit_layer_NTD1(solver_graph)
    if do_reduce:
        # transit layer boundary reductions
        reduce_bounds(solver_graph)
        prune_transit_layer_NTD1(solver_graph)
    # if do_asp_prune:
    #     # Second pass
    #     prune_low_asp_nodes(solver_graph, data)
    #     # overall pruning of non terminal leaf nodes from graph
    #     prune_NTD1(solver_graph)
    #     # transit layer NTD1 pruning of flows per root
    #     reduce_bounds_via_root_pruning(solver_graph)

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

    # # Just testing to see what happens if we do this _after_ the transit pruning...
    # if data["config"]["prune_prizes"]:
    #     import terminal_prize_utils as tpu
    #     from reduce_prize_data import apply_global_rank_prize_filtering

    #     prunable_entries = []
    #     prunable_entries = apply_global_rank_prize_filtering(solver_graph, data)
    #     tpu.nullify_prize_entries(solver_graph, data, prunable_entries)
    #     tpu.prune_null_prize_entries(solver_graph, prunable_entries)
    #     tpu.prune_prize_drained_terminals(solver_graph)

    #     # Validate all terminal indices are in graph, remove if not
    #     terminal_indices = solver_graph.attrs["terminal_indices"]
    #     for t in terminal_indices.copy():
    #         if not solver_graph.has_node(t):
    #             terminal_indices.remove(t)
    #     solver_graph.attrs["terminal_indices"] = terminal_indices
    # if data["config"]["prune_prizes"]:
    #     logger.warning("    pruning prizes again...")
    #     import terminal_prize_utils as tpu
    #     from reduce_prize_data import apply_global_rank_prize_filtering

    #     orig_config = data["config"].copy()
    #     tmp_config = data["config"].copy()
    #     tmp_config["prize_pruning_threshold_factors"] = {
    #         "min": {"only_child": 1, "dominant": 1, "protected": 1},
    #         "max": {"only_child": 1, "dominant": 1, "protected": 1},
    #     }
    #     data["config"] = tmp_config

    #     prunable_entries = []
    #     prunable_entries = apply_global_rank_prize_filtering(solver_graph, data)
    #     tpu.nullify_prize_entries(solver_graph, data, prunable_entries)
    #     tpu.prune_null_prize_entries(solver_graph, prunable_entries)
    #     tpu.prune_prize_drained_terminals(solver_graph)

    #     data["config"] = orig_config

    #     # Validate all terminal indices are in graph, remove if not
    #     terminal_indices = solver_graph.attrs["terminal_indices"]
    #     for t in terminal_indices.copy():
    #         if not solver_graph.has_node(t):
    #             terminal_indices.remove(t)
    #     solver_graph.attrs["terminal_indices"] = terminal_indices
