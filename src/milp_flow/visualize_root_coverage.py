# visualize_root_coverage.py

"""
Visualizes root coverage rings on the BDO map using Folium.
Adapted for value-based ring coverage (cumulative hulls per bin of rings).
"""

import os
import tempfile
import webbrowser

import distinctipy
from loguru import logger
import numpy as np
import pandas as pd
import rustworkx as rx
from shapely import Point
from shapely.geometry import MultiPolygon, Polygon
import alphashape
import folium
from folium import FeatureGroup
from folium.plugins import FeatureGroupSubGroup, GroupedLayerControl, BeautifyIcon
import matplotlib.colors as mcolors

from api_common import get_clean_exploration_data, SUPER_ROOT
from api_exploration_graph import (
    get_exploration_graph,
    get_all_pairs_path_lengths,
    populate_edge_weights,
    has_edge_weights,
)
from api_rx_pydigraph import inject_super_root, set_graph_terminal_sets_attribute, subgraph_stable
import data_store as ds


# Copied from analyze_tr_coverage.py for ring generation
def get_cartesian(G, idx: int) -> tuple[float, float]:
    p = G[idx]["position"]
    return (float(p["x"]), float(p["z"]))


def get_hull(G, settlement: set[int], alpha=0.25):
    if len(settlement) < 3:
        return None
    coords = [get_cartesian(G, idx) for idx in settlement]
    try:
        hull = alphashape.alphashape(coords, alpha)
        return hull.buffer(0.5)
    except Exception as e:
        logger.warning(f"Hull failed: {e}")
        return None


def find_frontier_nodes(G, settlement: set[int]) -> set[int]:
    frontier = set()
    for v in settlement:
        frontier.update(G.neighbors(v))
    frontier.difference_update(settlement)
    return frontier


def expand_frontier_to_new_terminals(G, settlement: set[int], terminals: set[int]) -> set[int]:
    wild_frontier_terminals = terminals - settlement
    new_settlement = settlement.copy()
    new_frontier = find_frontier_nodes(G, new_settlement)
    while not new_frontier.intersection(wild_frontier_terminals):
        new_settlement.update(new_frontier)
        new_frontier = find_frontier_nodes(G, new_settlement)
    new_settlement.update(new_frontier)
    new_settlement.difference_update(settlement)
    return new_settlement


def ensure_all_inter_terminal_paths_are_covered(
    G, settlement: set[int], settled_terminals: set[int], newly_settled_terminals: set[int]
):
    new_settlement = settlement.copy()
    for terminal in newly_settled_terminals:
        for s_terminal in settled_terminals | newly_settled_terminals:
            if s_terminal == terminal:
                continue
            path = rx.dijkstra_shortest_paths(
                G, s_terminal, terminal, weight_fn=lambda e: e["need_exploration_point"]
            )
            new_settlement.update(path[terminal])
    if new_settlement != settlement:
        settlement.update(new_settlement)
        return True
    return False


def prune_NTD1(graph, non_removables=None):
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

    return num_removed


def get_root_connected_subset(graph, root: int, settlement: set[int]):
    new_settlement = settlement.copy()
    subG = subgraph_stable(graph, new_settlement)
    for v in settlement:
        if not rx.has_path(subG, root, v):
            new_settlement.remove(v)
    return new_settlement


def generate_root_coverage_ring_subgraphs(G, root: int, terminals: set[int], alpha=0.25):
    rings = []
    remaining_terminals = set(terminals)
    settlement = {root}
    settled_terminals = set()

    while remaining_terminals:
        newly_settled_nodes = expand_frontier_to_new_terminals(G, settlement, remaining_terminals)
        if not newly_settled_nodes:
            break

        settlement.update(newly_settled_nodes)
        newly_from_frontier = remaining_terminals.intersection(settlement)
        remaining_terminals.difference_update(newly_from_frontier)

        while True:
            changed = False

            hull = get_hull(G, settlement, alpha=alpha)
            if hull is not None:
                # Simplified; in full, use get_hull_covered_settlement
                pass  # Assume no change for viz

            if ensure_all_inter_terminal_paths_are_covered(
                G, settlement, settled_terminals, newly_from_frontier
            ):
                changed = True

            if newly_from_frontier:
                settled_terminals.update(newly_from_frontier)
                newly_from_frontier = set()

            if not changed:
                break

        rings.append(settlement.copy())
        settlement = get_root_connected_subset(G, root, settlement)

    ring_subgraphs = [subgraph_stable(G, r) for r in rings]
    for subgraph in ring_subgraphs:
        assert isinstance(subgraph, rx.PyDiGraph)
        prune_NTD1(subgraph)

    return ring_subgraphs


def analyze_ring(G, prev_ring, ring, r_idx: int, terminal_entries: list[dict]):
    nodes_in_ring = (
        set(ring.node_indices()) - set(prev_ring.node_indices())
        if prev_ring is not None
        else set(ring.node_indices())
    )

    terminals_in_ring = [i for i in nodes_in_ring if ring[i]["is_workerman_plantzone"]]
    terminal_value_in_ring = sum(round(G[i]["prizes"][r_idx] / 1e6, 3) for i in terminals_in_ring)

    terminals_in_hull = [i for i in ring.node_indices() if G[i]["is_workerman_plantzone"]]
    terminal_value_in_hull = sum(round(G[i]["prizes"][r_idx] / 1e6, 3) for i in terminals_in_hull)

    ttl_terminal_count = len(terminal_entries)
    ttl_terminal_value = sum(e["value"] for e in terminal_entries)

    return {
        "terminal_count": len(terminals_in_ring),
        "cum_terminal_count": len(terminals_in_hull),
        "terminal_count_pct": (len(terminals_in_ring) / ttl_terminal_count) if ttl_terminal_count else 0.0,
        "cum_terminal_count_pct": (len(terminals_in_hull) / ttl_terminal_count)
        if ttl_terminal_count
        else 0.0,
        "terminal_value": terminal_value_in_ring,
        "cum_terminal_value": terminal_value_in_hull,
        "terminal_value_pct": (terminal_value_in_ring / ttl_terminal_value) if ttl_terminal_value else 0.0,
        "cum_terminal_value_pct": (terminal_value_in_hull / ttl_terminal_value)
        if ttl_terminal_value
        else 0.0,
        "ring_nodes": sorted(nodes_in_ring),
        "hull_nodes": sorted(ring.node_indices()),
    }


def generate_root_df_analysis(ring_analysis: list[dict], bin_count: int = 5):
    total_rings = len(ring_analysis)
    if total_rings == 0:
        return pd.DataFrame()

    bin_count = min(bin_count, total_rings)
    rings_per_bin = total_rings // bin_count
    extra = total_rings % bin_count

    bins = []
    start = 0
    for b in range(bin_count):
        end = start + rings_per_bin + (1 if b < extra else 0)
        bin_rings = ring_analysis[start:end]

        if not bin_rings:
            continue

        bin_row = {
            "num_terms": sum(r["terminal_count"] for r in bin_rings),
            "cum_terms": bin_rings[-1]["cum_terminal_count"],
            "sum_prize": sum(r["terminal_value"] for r in bin_rings),
            "cum_prize": bin_rings[-1]["cum_terminal_value"],
            "hull_nodes": bin_rings[-1]["hull_nodes"],  # Cumulative for bin
        }
        total_prize = sum(r["terminal_value"] for r in ring_analysis)
        total_terms = sum(r["terminal_count"] for r in ring_analysis)
        bin_row["per_%_total_prize"] = (bin_row["sum_prize"] / total_prize * 100) if total_prize else 0.0
        bin_row["cum_%_total_prize"] = (bin_row["cum_prize"] / total_prize * 100) if total_prize else 0.0
        bin_row["value_range"] = (
            f"{bin_row['cum_%_total_prize'] - bin_row['per_%_total_prize']:.1f}-{bin_row['cum_%_total_prize']:.1f}"
        )
        bins.append(bin_row)
        start = end

    value_bins = pd.DataFrame(bins)
    full_row = {
        "num_terms": sum(r["terminal_count"] for r in ring_analysis),
        "cum_terms": ring_analysis[-1]["cum_terminal_count"],
        "sum_prize": sum(r["terminal_value"] for r in ring_analysis),
        "cum_prize": ring_analysis[-1]["cum_terminal_value"],
        "per_%_total_prize": 100.0,
        "cum_%_total_prize": 100.0,
        "value_range": "0.0-100.0",
        "hull_nodes": ring_analysis[-1]["hull_nodes"],
    }
    value_bins = pd.concat([value_bins, pd.DataFrame([full_row])], ignore_index=True)

    return value_bins


TILE_SCALE = 12800

ROOT_COLORS = distinctipy.get_colors(46, pastel_factor=0.7)
MARKER_COLORS = distinctipy.get_colors(46)

# Color Constants
MAIN_GRAPH_EDGE_COLOR = "darkturquoise"
SUBGRAPH_EDGE_COLOR = "yellow"
REGULAR_NODE_COLOR = "yellow"
SUPER_ROOT_COLOR = ROOT_COLORS[-1]

# Line Thicknesses
MAIN_GRAPH_EDGE_WEIGHT = 2
SUBGRAPH_EDGE_WEIGHT = 3


def add_edges_from_graph(
    fg: folium.FeatureGroup,
    graph: rx.PyGraph | rx.PyDiGraph,
    color: str | None = None,
    weight: float = 0.75,
):
    for u_index, v_index in graph.edge_list():
        if u_index > v_index and not graph.has_edge(v_index, u_index):
            continue

        u_node = graph[u_index]
        v_node = graph[v_index]
        if u_node["waypoint_key"] == 99999 or v_node["waypoint_key"] == 99999:
            continue

        start = u_node["position"]
        end = v_node["position"]

        start_lat = start["z"] / TILE_SCALE
        start_lng = start["x"] / TILE_SCALE
        end_lat = end["z"] / TILE_SCALE
        end_lng = end["x"] / TILE_SCALE

        folium.PolyLine(
            locations=[
                (start_lat, start_lng),
                (end_lat, end_lng),
            ],
            color=color,
            weight=weight,
            opacity=1,
            popup=f"Edge: {u_node['waypoint_key']} - {v_node['waypoint_key']}",
            tooltip=f"Edge: {u_node['waypoint_key']} - {v_node['waypoint_key']}",
            **{"interactive": True, "bubblingMouseEvents": True},
        ).add_to(fg)


def add_node_markers_from_graph(fg: folium.FeatureGroup, graph: rx.PyGraph | rx.PyDiGraph):
    base_towns = sorted(node["waypoint_key"] for node in graph.nodes() if node["is_town"])
    base_town_colors = {waypoint: ROOT_COLORS[i] for i, waypoint in enumerate(base_towns)}

    for node in graph.nodes():
        node_color = None
        node_key = node["waypoint_key"]
        cost = node["need_exploration_point"]

        lng = node["position"]["x"] / TILE_SCALE
        lat = node["position"]["z"] / TILE_SCALE

        popup_text = f"Node Key: {node_key}, Cost: {cost}"

        if node_key == 99999:
            popup_text += " (Super Root)"
            node_color = SUPER_ROOT_COLOR

        if node_color is None:
            if node["is_base_town"]:
                node_color = base_town_colors[node_key]
            else:
                node_color = REGULAR_NODE_COLOR

        folium.CircleMarker(
            location=(lat, lng),
            radius=1 if not node["is_base_town"] else 4,
            color=node_color,
            fill=True,
            fill_color=node_color,
            popup=popup_text,
            tooltip=popup_text,
        ).add_to(fg)


def add_root_coverage_overlays(
    m: folium.Map,
    G,
    bin_count: int = 5,
    alpha: float = 0.1,
    max_cum_pct: float = 100.0,
):
    node_key_by_index = G.attrs["node_key_by_index"]
    root_indices = G.attrs["root_indices"]
    terminal_indices = G.attrs["terminal_indices"]

    if not has_edge_weights(G):
        populate_edge_weights(G)

    # Unique hue per root
    base_towns = sorted(node["waypoint_key"] for node in G.nodes() if node["is_town"])
    root_colors = {waypoint: MARKER_COLORS[i] for i, waypoint in enumerate(base_towns)}

    def get_cartesian(idx) -> tuple[float, float]:
        p = G[idx]["position"]
        return (float(p["x"] / TILE_SCALE), float(p["z"] / TILE_SCALE))

    def gradient_for_root(rgb, n_bins):
        base = np.array(rgb)
        white = np.array([1, 1, 1])
        factors = np.linspace(0.85, 0.35, n_bins)
        return [mcolors.to_hex(base * f + white * (1 - f)) for f in factors]

    fg_root_coverage_master = FeatureGroup(name="All Root Coverage", show=True)
    fg_root_coverage_master.add_to(m)
    assert isinstance(fg_root_coverage_master, FeatureGroup)

    root_coverage_feature_groups: dict[int, FeatureGroupSubGroup] = {}

    for root_idx in root_indices:
        root_key = node_key_by_index[root_idx]
        terminal_entries = [
            {
                "t_idx": t,
                "r_idx": root_idx,
                "value": round(G[t]["prizes"][root_idx] / 1e6, 3),
                "dist": None,  # Unused for rings
            }
            for t in terminal_indices
            if G[t]["prizes"].get(root_idx) is not None
        ]
        if not terminal_entries:
            continue

        terminals: set[int] = {t["t_idx"] for t in terminal_entries}  # type: ignore
        ring_subgraphs = generate_root_coverage_ring_subgraphs(G, root_idx, terminals)

        ring_analysis = []
        for prev_ring, ring in zip([None] + ring_subgraphs[:-1], ring_subgraphs):
            ring_analysis.append(analyze_ring(G, prev_ring, ring, root_idx, terminal_entries))

        value_bins = generate_root_df_analysis(ring_analysis, bin_count)
        root_rows = value_bins[value_bins["cum_%_total_prize"] <= max_cum_pct].copy()
        if root_rows.empty:
            continue

        num_bins = len(root_rows) - 1  # Exclude total row
        if num_bins == 0:
            continue

        fg_root_coverage = FeatureGroupSubGroup(fg_root_coverage_master, name=f"Root {root_key}", show=False)
        fg_root_coverage.add_to(m)
        root_coverage_feature_groups[root_key] = fg_root_coverage

        bin_colors = gradient_for_root(root_colors[G[root_idx]["waypoint_key"]], num_bins)

        for bin_idx, (_, row) in enumerate(root_rows[:-1].iterrows()):  # Exclude total
            color = bin_colors[bin_idx]
            hull_nodes = row["hull_nodes"]  # Cumulative nodes for this bin

            if len(hull_nodes) < 3:
                logger.debug(f"Skipped bin for root {root_key}: {len(hull_nodes)} nodes < 3")
                continue

            subG = subgraph_stable(G, hull_nodes)
            assert isinstance(subG, rx.PyDiGraph)

            # Ensure connected to root
            connected_nodes = [idx for idx in subG.node_indices() if rx.has_path(subG, root_idx, idx)]
            if len(connected_nodes) < 3:
                logger.debug(f"Skipped bin for root {root_key}: {len(connected_nodes)} connected < 3")
                continue

            coords_cartesian = [get_cartesian(idx) for idx in connected_nodes]
            try:
                hull = alphashape.alphashape(coords_cartesian, alpha)
                hull = hull.buffer(0.5)
            except Exception as e:
                logger.warning(f"Alpha shape failed for root {root_key}, bin {row['value_range']}: {e}")
                continue

            if hull.is_empty:
                logger.debug(f"Empty hull for root {root_key}, bin {row['value_range']}")
                continue

            polygons = (
                [hull]
                if isinstance(hull, Polygon)
                else list(hull.geoms)
                if isinstance(hull, MultiPolygon)
                else []
            )
            for poly in polygons:
                ext = [(y, x) for x, y in poly.exterior.coords]
                folium.Polygon(
                    locations=ext,
                    color=color,
                    weight=2,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.25,
                    popup=(
                        f"<b>Root:</b> {root_key}<br>"
                        f"<b>Value bin:</b> {row['value_range']}<br>"
                        f"<b>Cum prize:</b> {row['cum_%_total_prize']:.1f}%"
                    ),
                ).add_to(fg_root_coverage)

    return fg_root_coverage_master, root_coverage_feature_groups


def add_terminal_sets_markers(
    m: folium.Map, graph: rx.PyGraph | rx.PyDiGraph
) -> tuple[folium.FeatureGroup | None, dict[str, FeatureGroupSubGroup]]:
    if "terminal_sets" not in graph.attrs:
        print("Warning: terminal_sets attribute not found in graph.")
        return None, {}

    terminal_sets = graph.attrs["terminal_sets"]
    base_towns = sorted(node["waypoint_key"] for node in graph.nodes() if node["is_town"])
    base_town_colors = {waypoint: MARKER_COLORS[i] for i, waypoint in enumerate(base_towns)}

    fg_terminal_sets_master = FeatureGroup(name="All Terminal Sets", show=True)
    fg_terminal_sets_master.add_to(m)
    assert isinstance(fg_terminal_sets_master, FeatureGroup)

    terminal_set_feature_groups: dict[str, FeatureGroupSubGroup] = {}

    for color_index, (root, terminal_set) in enumerate(terminal_sets.items()):
        layer_name = f"Terminal Set {graph[root]['waypoint_key']}"
        fg_terminal_set = FeatureGroupSubGroup(fg_terminal_sets_master, name=layer_name, show=False)
        fg_terminal_set.add_to(m)
        terminal_set_feature_groups[layer_name] = fg_terminal_set

        root_node = graph[root]
        root_lng = root_node["position"]["x"] / TILE_SCALE
        root_lat = root_node["position"]["z"] / TILE_SCALE
        terminal_set_color = base_town_colors[root_node["waypoint_key"]]

        folium.Marker(
            location=(root_lat, root_lng),
            icon=BeautifyIcon(
                icon="house",
                icon_shape="marker",
                background_color=distinctipy.get_hex(terminal_set_color),
                text_color="white",
                border_color="white",
            ),  # type: ignore
            popup=f"Root Node Key: {root_node['waypoint_key']}, Cost: {root_node['need_exploration_point']}",
            tooltip=f"Root Node Key: {root_node['waypoint_key']}, Cost: {root_node['need_exploration_point']}",
        ).add_to(fg_terminal_set)

        for terminal in terminal_set:
            terminal_node = graph[terminal]
            terminal_lng = terminal_node["position"]["x"] / TILE_SCALE
            terminal_lat = terminal_node["position"]["z"] / TILE_SCALE

            folium.Marker(
                location=(terminal_lat, terminal_lng),
                icon=BeautifyIcon(
                    icon="",
                    iconSize=[20, 20],
                    iconAnchor=[10, 20],
                    icon_shape="marker",
                    background_color=distinctipy.get_hex(terminal_set_color),
                    text_color="transparent",
                    border_color="transparent",
                ),  # type: ignore
                popup=f"Terminal Node Key: {terminal_node['waypoint_key']}, Cost: {terminal_node['need_exploration_point']}",
                tooltip=f"Terminal Node Key: {terminal_node['waypoint_key']}, Cost: {terminal_node['need_exploration_point']}",
            ).add_to(fg_terminal_set)

    return fg_terminal_sets_master, terminal_set_feature_groups


def visualize_solution_graph(
    main_graph: rx.PyGraph | rx.PyDiGraph,
    subgraph: rx.PyGraph | rx.PyDiGraph | None = None,
    bin_count: int = 5,
):
    m = folium.Map(
        crs="Simple",
        location=[32.5, 0],
        zoom_start=1.9,
        zoom_snap=0.25,
        tiles=None,
    )

    tile_pane = folium.map.CustomPane("tile_pane", z_index=1)
    m.add_child(tile_pane)

    tile_layer = folium.TileLayer(
        name="BDO Router",
        attr="Map Tiles @ BDO",
        min_zoom=1,
        max_zoom=7,
        no_wrap=True,
        pane="tile_pane",
        show=True,
        tiles=os.path.join(ds.path(), "maptiles", "{z}", "{x}_{y}.webp"),
    )
    tile_layer.add_to(m)

    fg_all_nodes = folium.FeatureGroup(name="All Nodes", show=True)
    fg_main_graph_edges = folium.FeatureGroup(name="Main Graph Edges", show=True)
    fg_subgraph_edges = folium.FeatureGroup(name="Subgraph Edges", show=True)

    add_node_markers_from_graph(fg_all_nodes, subgraph if subgraph else main_graph)

    add_edges_from_graph(
        fg_main_graph_edges,
        main_graph,
        color=MAIN_GRAPH_EDGE_COLOR,
        weight=MAIN_GRAPH_EDGE_WEIGHT,
    )

    if subgraph:
        add_edges_from_graph(
            fg_subgraph_edges,
            subgraph,
            color=SUBGRAPH_EDGE_COLOR,
            weight=SUBGRAPH_EDGE_WEIGHT,
        )

    fg_terminal_master, terminal_groups = add_terminal_sets_markers(m, main_graph)
    assert isinstance(fg_terminal_master, folium.FeatureGroup)

    fg_root_coverage_master, root_coverage_subgroups = add_root_coverage_overlays(m, main_graph, bin_count)
    assert isinstance(fg_root_coverage_master, folium.FeatureGroup)

    fg_all_nodes.add_to(m)
    fg_main_graph_edges.add_to(m)
    fg_subgraph_edges.add_to(m)

    m.keep_in_front(
        fg_main_graph_edges,
        fg_root_coverage_master,
        *root_coverage_subgroups.values(),
        fg_subgraph_edges,
        fg_all_nodes,
        fg_terminal_master,
        *terminal_groups.values(),
    )

    group_layer_control = GroupedLayerControl(
        groups={
            "Base": [fg_all_nodes, fg_main_graph_edges, fg_subgraph_edges],
        },
        collapsed=False,
        exclusive_groups=False,
        position="topright",
    )
    group_layer_control.add_to(m)

    group_layer_control = GroupedLayerControl(
        groups={
            "Terminal Sets": [fg_terminal_master, *terminal_groups.values()],
        },
        collapsed=False,
        exclusive_groups=False,
        position="bottomright",
    )
    group_layer_control.add_to(m)

    group_layer_control = GroupedLayerControl(
        groups={
            "Root Coverage": [
                fg_root_coverage_master,
                *root_coverage_subgroups.values(),
            ],
        },
        collapsed=False,
        exclusive_groups=False,
        position="topleft",
    )
    group_layer_control.add_to(m)

    tmp_file = os.path.join(tempfile.gettempdir(), "bdo_map.html")
    m.save(tmp_file)
    webbrowser.open("file://" + tmp_file)


if __name__ == "__main__":
    data = {}
    data["exploration"] = get_clean_exploration_data({})

    config = {}
    config["exploration_data"] = {"directed": True, "edge_weighted": False, "omit_great_ocean": True}
    G = get_exploration_graph(config)
    assert isinstance(G, rx.PyDiGraph)

    root_indices = [i for i in G.node_indices() if G[i]["is_worker_npc_town"]]
    root_waypoints = {G[i]["waypoint_key"] for i in root_indices}

    # fmt: off
    # MARK: Subgraph - same as old
    terminals = {
        131: 1, 132: 1, 135: 1, 136: 1, 144: 61, 160: 1, 171: 61, 172: 61, 175: 61, 176: 61,
        183: 1, 184: 61, 188: 61, 203: 61, 435: 301, 436: 301, 438: 301, 439: 301, 440: 301,
        443: 301, 455: 301, 464: 301, 476: 301, 480: 301, 488: 301, 840: 601, 841: 601,
        842: 601, 852: 601, 853: 601, 854: 601, 855: 601, 901: 601, 902: 601, 903: 601,
        905: 608, 907: 608, 908: 608, 910: 601, 912: 601, 913: 601, 914: 601, 951: 601,
        952: 602, 957: 601, 958: 601, 1201: 1141, 1203: 1101, 1204: 1101, 1206: 1141,
        1208: 1101, 1209: 1101, 1212: 301, 1213: 301, 1215: 1141, 1216: 301, 1219: 1141,
        1220: 1141, 1501: 1101, 1502: 1101, 1504: 1101, 1505: 1101, 1507: 1101, 1508: 1101,
        1510: 1319, 1513: 1319, 1514: 1314, 1515: 1314, 1516: 1314, 1517: 1314, 1520: 1319,
        1521: 1319, 1522: 1319, 1523: 1319, 1527: 1319, 1528: 1319, 1529: 1314, 1530: 1314,
        1531: 1314, 1534: 1301, 1535: 1301, 1536: 1301, 1537: 1301, 1538: 1301, 1554: 1301,
        1555: 1380, 1556: 1301, 1558: 1301, 1561: 1301, 1562: 1301, 1565: 1301, 1566: 1301,
        1636: 1623, 1637: 1623, 1642: 1623, 1645: 1623, 1679: 1649, 1681: 1649, 1682: 1649,
        1683: 302, 1684: 302, 1685: 301, 1686: 301, 1687: 1649, 1688: 1649, 1710: 1691,
        1711: 1691, 1713: 1691, 1716: 1691, 1769: 1750, 1770: 1750, 1771: 1750, 1772: 1750,
        1777: 1750, 1778: 1750, 1807: 1781, 1808: 1781, 1809: 1781, 1813: 1795, 1815: 1781,
        1819: 1795, 1820: 1795, 1821: 1781, 1822: 1781, 1823: 1785, 1826: 1795, 1827: 1795,
        1828: 1795, 1829: 1795, 1830: 1781, 1879: 1857, 1880: 1857, 1881: 1857, 1882: 1853,
        1883: 1853, 1884: 1853, 1885: 1853, 1886: 1857, 1887: 1853, 1888: 1853, 1889: 1857,
        1890: 1853, 1891: 1858, 1892: 1858, 1893: 1858, 1894: 1853, 1895: 1853, 1896: 1853,
        1897: 1853, 1899: 1834, 1902: 1843, 1903: 1843, 1904: 1834, 1905: 1843, 1906: 1843,
        1907: 1843, 1908: 1843, 1909: 1843, 1910: 1843, 1911: 602, 1912: 602, 1913: 602,
        1914: 301, 2037: 2001, 2038: 2001, 2039: 2001, 2040: 2001, 2044: 2001, 2045: 2001,
        2046: 2001, 2047: 2001, 2048: 2001, 2049: 2001, 2050: 2001
    }

    highlights = set([
        1,21,22,24,42,45,46,48,49,61,64,66,67,131,132,135,136,144,160,171,172,175,176,183,
        184,188,203,301,302,305,309,322,323,324,327,341,344,345,347,372,435,436,438,439,
        440,443,455,464,476,480,488,601,602,608,609,624,628,629,633,638,651,652,656,660,
        661,662,664,667,668,670,672,674,675,677,703,705,706,707,708,710,715,716,717,718,
        720,721,722,723,724,840,841,842,852,853,854,855,901,902,903,905,907,908,910,912,
        913,914,951,952,957,958,1101,1133,1134,1136,1137,1140,1141,1144,1149,1154,1155,1156,
        1159,1161,1162,1170,1201,1203,1204,1206,1208,1209,1212,1213,1215,1216,1219,1220,1301,
        1302,1303,1304,1305,1306,1307,1309,1310,1314,1315,1318,1319,1321,1325,1327,1328,1329,
        1330,1345,1346,1350,1351,1352,1354,1355,1379,1380,1385,1387,1388,1389,1390,1501,1502,
        1504,1505,1507,1508,1510,1513,1514,1515,1516,1517,1520,1521,1522,1523,1527,1528,1529,
        1530,1531,1534,1535,1536,1537,1538,1554,1555,1556,1558,1561,1562,1565,1566,1609,1619,
        1621,1622,1623,1625,1636,1637,1642,1645,1649,1652,1653,1654,1655,1656,1657,1658,1660,
        1663,1664,1665,1666,1679,1681,1682,1683,1684,1685,1686,1687,1688,1691,1695,1702,1703,
        1710,1711,1713,1716,1740,1741,1742,1743,1750,1755,1756,1757,1759,1762,1769,1770,1771,
        1772,1777,1778,1781,1785,1788,1789,1790,1792,1793,1795,1796,1797,1799,1807,1808,1809,
        1813,1815,1819,1820,1821,1822,1823,1826,1827,1828,1829,1830,1834,1837,1838,1840,1843,
        1845,1846,1847,1848,1849,1850,1851,1852,1853,1854,1855,1857,1858,1859,1860,1861,1863,
        1864,1868,1870,1874,1875,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887,1888,
        1889,1890,1891,1892,1893,1894,1895,1896,1897,1899,1902,1903,1904,1905,1906,1907,1908,
        1909,1910,1911,1912,1913,1914,2001,2002,2004,2006,2009,2014,2016,2019,2020,2022,2024,
        2025,2026,2027,2028,2029,2030,2034,2035,2037,2038,2039,2040,2044,2045,2046,2047,2048,
        2049,2050
    ])
    # fmt: on

    terminals = dict(sorted(terminals.items(), key=lambda item: item[1]))

    if SUPER_ROOT in terminals.values():
        inject_super_root(config, G)
    set_graph_terminal_sets_attribute(G, terminals)
    highlights.update(key for key in terminals.keys())
    highlights.update(key for key in terminals.values())
    waypoint_to_index = {node["waypoint_key"]: i for i, node in enumerate(G.nodes())}
    highlight_indices = [waypoint_to_index[key] for key in highlights]
    highlight_graph = G.subgraph(highlight_indices)
    cost = sum(node["need_exploration_point"] for node in highlight_graph.nodes())
    print(f"Cost of highlighted nodes: {cost}")

    G.attrs["root_indices"] = root_indices
    G.attrs["terminal_indices"] = [i for i in G.node_indices() if G[i]["is_workerman_plantzone"]]

    visualize_solution_graph(
        main_graph=G,
        subgraph=highlight_graph,
        bin_count=5,  # Adjust as needed
    )
