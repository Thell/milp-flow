# generate_graph_data.py

from collections import Counter
from typing import Any

from bidict import bidict
from loguru import logger
from rustworkx import PyDiGraph, strongly_connected_components
from api_exploration_graph import attach_metrics_to_graph
from api_rx_pydigraph import subgraph_stable
import terminal_prize_utils as tpu

# DEBUG_ROOTS = [1623, 1785, 1834, 2001]
DEBUG_ROOTS = []


def setup_super_terminals(G: PyDiGraph, data: dict[str, Any]):
    """Injects the superroot node into the graph"""
    # Super terminals require the super root for connection via any basetown.
    # Flow is _into_ super root with no flow out to prevent worker flow routing shortcuts
    super_root_index = None
    super_terminal_indices = []
    if data["force_active_node_ids"]:
        logger.info("    setting up super-terminals...")
        from api_rx_pydigraph import inject_super_root

        for node in G.nodes():
            if node["waypoint_key"] in data["force_active_node_ids"]:
                super_terminal_indices.append(node["index"])
                node["is_super_terminal"] = True

        node_key_by_index = bidict({i: G[i]["waypoint_key"] for i in G.node_indices()})
        G.attrs = {"node_key_by_index": node_key_by_index}
        super_root_index = inject_super_root({}, G, flow_direction="inbound")
        G[super_root_index]["ub"] = len(data["force_active_node_ids"])
    else:
        for node in G.nodes():
            node["is_super_terminal"] = False

    G.attrs["super_root_index"] = super_root_index
    G.attrs["super_terminal_indices"] = super_terminal_indices


def setup_terminals(G: PyDiGraph, data: dict[str, Any]) -> list[int]:
    """Prepares terminals with full prize values."""
    logger.info("    setting up terminals (plant zones)...")

    node_key_by_index = G.attrs["node_key_by_index"]
    terminal_indices = []

    for i in G.node_indices():
        node = G[i]
        if not node["is_workerman_plantzone"]:
            node["is_terminal"] = False
            continue

        node["is_terminal"] = True
        prizes = data["plant_values"][node["waypoint_key"]]
        prizes = dict(sorted(prizes.items(), key=lambda x: x[1]["value"], reverse=True))
        values = {}
        # NOTE: Prize keys are affiliated town regions not waypoint keys, so translate!
        for warehouse_key, prize_data in prizes.items():
            value = prize_data["value"]
            # Not all terminals allow all worker species from all towns
            if int(value) == 0:
                continue
            root_key = data["affiliated_town_region"][warehouse_key]
            root_index = node_key_by_index.inv[root_key]
            values[root_index] = value
        if not values:
            continue
        terminal_indices.append(i)
        node["prizes"] = values

    logger.info(f"      initialized terminals: {len(terminal_indices)}")

    logger.trace("Base graph active (terminal, root) prize pairs:")
    town_to_region_map = {int(town): region for region, town in data["affiliated_town_region"].items()}
    tr_pair_count = 0
    for terminal_index in sorted(terminal_indices):
        node = G[terminal_index]
        prizes = node["prizes"]
        for r_idx, prize in prizes.items():
            town = G[r_idx]["waypoint_key"]
            region = town_to_region_map[int(town)]
            node_weight = node["need_exploration_point"]
            path_cost = data["all_pairs_path_lengths"].get((r_idx, terminal_index), 2**31)
            logger.trace(
                f"{node['waypoint_key']:>5} {region:>5} {town:>5} {prize:>5} {node_weight:>5} {path_cost:>5}"
            )
        tr_pair_count += len(prizes)
    logger.info(f"      initialized (terminal, root) prize pairs: {tr_pair_count}")

    G.attrs["terminal_indices"] = terminal_indices
    return terminal_indices


def setup_roots(G, data):
    """Populate each root node with ub and capacity_cost using precomputed bounds_costs."""
    root_indices = G.attrs["root_indices"]
    logger.info(f"    setting up root lodging costs for {len(root_indices)} roots...")

    for i in root_indices:
        node = G[i]
        region_key = node["region_key"]
        lodging = data["lodging_data"][region_key]

        # See generate_reference_data.get_region_lodging_bounds_costs for details
        bounds_costs = lodging["bounds_costs"]
        max_ub = lodging["max_ub"]

        # capacity_cost[cap] = cost for cap in 0..max_ub
        # We keep index 0 = 0 for the model's SOS1
        capacity_cost = [0] * (max_ub + 1)

        prev_cap = 0
        prev_cost = 0
        for cap, cost in bounds_costs:
            cap = min(cap, max_ub)
            # expand capacities to fill (prev_cap+1 .. cap) with this bound's cost
            for idx in range(prev_cap + 1, cap + 1):
                capacity_cost[idx] = cost
            prev_cap = cap
            prev_cost = cost

            if prev_cap >= max_ub:
                break

        # fill any remaining capacities up to max_ub with the last known cost
        for idx in range(prev_cap + 1, max_ub + 1):
            capacity_cost[idx] = prev_cost

        node["ub"] = max_ub
        node["capacity_cost"] = capacity_cost

        logger.trace(
            f"{node['waypoint_key']:>5} region_key: {region_key:>5} ub: {max_ub} costs: {capacity_cost}"
        )


def setup_node_transit_bounds(G: PyDiGraph, data: dict[str, Any]):
    # The upper bound on transit for any worker town in the nearest_n towns is the minimum of
    # the waypoint_ub and the towns maximum lodging capacity and is set during lodging setup.
    # The upper bound for all other worker towns is zero.
    # In the end a non-root node has _n entries and a root node has _n+1 entries
    nearest_n = data["config"]["nearest_n"]
    logger.info(f"    setting up intermediate nodes for the nearest {nearest_n} towns...")

    all_pairs_path_lengths = data["all_pairs_path_lengths"]
    root_indices = G.attrs["root_indices"]

    root_transit_ub = {i: G[i]["ub"] for i in root_indices}
    super_root_index = G.attrs.get("super_root_index")
    if super_root_index is not None:
        root_transit_ub[super_root_index] = len(data["force_active_node_ids"])

    for i in G.node_indices():
        node = G[i]
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
        logger.trace(
            f"{G[i]['waypoint_key']:>5} { {G[i]['waypoint_key']: ub for i, ub in transit_ubs.items()} }"
        )


def setup_transit_layers(G: PyDiGraph, data: dict[str, Any]):
    """Ensures a single connected component for each transit layer."""
    logger.info("    setting up transit layers...")

    root_indices = G.attrs["root_indices"]

    for root in root_indices:
        layer_nodes = [i for i in G.node_indices() if root in G[i]["transit_bounds"]]
        subG = subgraph_stable(G, layer_nodes)
        assert isinstance(subG, PyDiGraph)
        cc = strongly_connected_components(subG)
        if len(cc) == 1:
            continue

        root_cc = [i for i, c in enumerate(cc) if root in c][0]
        for i, c in enumerate(cc):
            if i == root_cc:
                continue
            for j in c:
                G[j]["transit_bounds"].pop(root)
                if subG[j]["is_terminal"]:
                    G[j]["prizes"].pop(root, None)

        # Rebuild the subgraph
        layer_nodes = [i for i in G.node_indices() if root in G[i]["transit_bounds"]]
        subG = subgraph_stable(G, layer_nodes)
        assert isinstance(subG, PyDiGraph)
        cc = strongly_connected_components(subG)
        assert len(cc) == 1


def setup_prize_rank_data(G: PyDiGraph, data: dict[str, Any]):
    logger.info("    setting up global terminal families...")
    families = tpu.get_terminal_families(G)
    logger.debug(
        f"Found {len(families)} families covering {sum(len(fs) for fs in families.values())} terminals\n"
        f"Family sizes and counts: {dict(Counter(len(ts) for ts in families.values()))}",
    )
    data["families"] = families

    logger.info("    setting up global terminal root prizes...")
    data["terminal_root_prizes"] = tpu.get_terminal_root_prizes(G, data["all_pairs_path_lengths"])

    logger.info("    setting up global prizes by terminal...")
    tr_data = data["terminal_root_prizes"]
    data["prizes_by_terminal_view"] = tpu.get_prizes_by_terminal_view(G, tr_data)

    logger.info("    setting up global prizes by root...")
    data["prizes_by_root_view"] = tpu.get_prizes_by_root_view(G, data["terminal_root_prizes"])

    logger.info("    computing global rank-based percentiles...")
    data["prize_ranks"] = tpu.get_full_prize_ranks(
        G,
        data["terminal_root_prizes"],
        data["prizes_by_root_view"],
        data["prizes_by_terminal_view"],
    )

    data["pct_thresholds_lookup"] = tpu.PCTThresholdLookup(
        tpu._THRESHOLDS_DATA,
        data["config"].get(
            "prize_pruning_threshold_factors",
            {
                "min": {"only_child": 1, "dominant": 1, "protected": 1},
                "max": {"only_child": 1, "dominant": 1, "protected": 1},
            },
        ),
    )

    logger.info("    identifying terminal categories...")
    tpu.assign_terminal_categories(G, families)


def generate_graph_data(data: dict[str, Any]):
    """Generate and return a GraphData Dict composing the LP empire data.

    SAFETY: data is modified in-place!
    """
    print("Generating global graph data...")

    G = data["exploration_graph"].copy()
    G.attrs = {}

    logger.info("  preparing graph nodes...")

    # Set up super terminals first to ensure stable node indices
    setup_super_terminals(G, data)

    node_key_by_index = bidict({i: G[i]["waypoint_key"] for i in G.node_indices()})
    G.attrs["node_key_by_index"] = node_key_by_index

    root_indices = [
        node_key_by_index.inv[t]
        for t in data["affiliated_town_region"].values()
        if data["exploration"][t]["is_worker_npc_town"]
    ]
    G.attrs["root_indices"] = root_indices
    logger.debug(f"Found {len(root_indices)} root indices:")
    logger.debug(f"(index, waypoint): {[(i, G[i]['waypoint_key']) for i in root_indices]}")

    setup_terminals(G, data)
    setup_roots(G, data)
    setup_node_transit_bounds(G, data)
    setup_prize_rank_data(G, data)
    setup_transit_layers(G, data)
    # attach_metrics_to_graph(G, config={"use_asp_bc": True, "asp_cutoff": 20})

    data["G"] = G
