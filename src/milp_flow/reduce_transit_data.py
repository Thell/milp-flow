# reduce_transit_data.py

from typing import Any

from loguru import logger
from rustworkx import PyDiGraph, weakly_connected_components

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

        pre_prune_cc = weakly_connected_components(transit_layer_subgraph)
        logger.debug(f"        transit layer {r} has {len(pre_prune_cc)} connected components")

        # Non_removables for this layer are the root and the terminals with prizes for this root
        non_removables = {r}
        non_removables.update({
            i for i in transit_layer_subgraph.node_indices() if i in solver_graph.attrs["terminal_indices"]
        })
        removed_count = prune_NTD1(transit_layer_subgraph, non_removables, quiet=True)

        post_prune_cc = weakly_connected_components(transit_layer_subgraph)
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
        if r == super_root_index:
            continue
        # node_map maps from subgraph node index to global graph node index
        subG = subgraph_stable(
            solver_graph, [i for i in solver_graph.node_indices() if r in solver_graph[i]["transit_bounds"]]
        )
        assert isinstance(subG, PyDiGraph)

        cc = weakly_connected_components(subG)
        for c in cc:
            if r in c:
                continue
            for i in c:
                root_transit_bounds_removed += 1
                solver_graph[i]["transit_bounds"].pop(r, None)

    logger.info(f"      reduced upper bounds via intersections: {root_transit_bounds_removed} removed")


def reduce_transit_data(data: dict[str, Any]):
    """Reduces per root transit layer data via pruning and reduction.

    SAFETY: The solver_graph is modified in-place!
    """
    logger.info("    reducing transit data...")

    solver_graph = data["solver_graph"]

    start_flow_var_count = sum(len(solver_graph[i]["transit_bounds"]) for i in solver_graph.node_indices())

    do_prune = data["config"]["transit_prune"]
    do_reduce = data["config"]["transit_reduce"]

    if do_prune:
        prune_NTD1(solver_graph)
        prune_transit_layer_NTD1(solver_graph)
    if do_reduce:
        # transit layer boundary reductions
        reduce_bounds(solver_graph)
        prune_transit_layer_NTD1(solver_graph)

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

    #     # Validate all terminal indices are in graph, remove if not
    #     terminal_indices = solver_graph.attrs["terminal_indices"]
    #     for t in terminal_indices.copy():
    #         if not solver_graph.has_node(t):
    #             terminal_indices.remove(t)
    #     solver_graph.attrs["terminal_indices"] = terminal_indices
