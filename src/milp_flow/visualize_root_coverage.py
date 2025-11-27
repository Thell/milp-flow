# visualize_root_coverage.py

"""
Visualizes root coverage rings on the BDO map using Folium.
Adapted for value-based ring coverage (cumulative hulls per bin of rings).
"""

import ast
import colorsys
from copy import deepcopy
import os
import tempfile
from typing import Any
import webbrowser
import re

from loguru import logger

import numpy as np
import pandas as pd
from rustworkx import PyDiGraph

from folium import CircleMarker, FeatureGroup, Map, Marker, PolyLine, TileLayer
from folium import Polygon as FoliumPolygon
from folium.map import CustomPane
from folium.plugins import FeatureGroupSubGroup, GroupedLayerControl, BeautifyIcon
import matplotlib.colors as mcolors
from shapely import LineString, Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import MultiLineString
from shapely.ops import unary_union

import data_store as ds
from api_common import scale_dict_values, set_logger, SUPER_ROOT, OQUILLAS_EYE_KEY
from api_exploration_graph import (
    GraphCoords,
    add_scaled_coords_to_graph,
    get_all_pairs_all_shortest_paths,
    get_all_pairs_shortest_paths,
)
from api_rx_pydigraph import set_graph_terminal_sets_attribute, subgraph_stable
from generate_graph_data import generate_graph_data
from generate_reference_data import generate_reference_data
from reduce_prize_data import reduce_prize_data
from reduce_transit_data import prune_NTD1


# Edge Constants
MAIN_GRAPH_EDGE_COLOR = "aqua"
SUBGRAPH_EDGE_COLOR = "yellow"

MAIN_GRAPH_EDGE_WEIGHT = 1
SUBGRAPH_EDGE_WEIGHT = 10

# Node Constants
MAIN_GRAPH_NODE_COLOR = "red"
SUB_GRAPH_NODE_COLOR = "yellow"
ROOT_NODE_COLOR = "white"
SUPER_ROOT_COLOR = "black"

NODE_CIRCLE_RADIUS = MAIN_GRAPH_EDGE_WEIGHT
ROOT_CIRCLE_RADIUS = MAIN_GRAPH_EDGE_WEIGHT * 2
SUB_GRAPH_CIRCLE_RADIUS = MAIN_GRAPH_EDGE_WEIGHT * 2


def get_prize_flow_edge_weights(G: PyDiGraph, data: dict[str, Any]) -> dict[tuple[int, int], float]:
    """Accumulates prize values for all terminal, root pairs for each family and distributes along
    the edges of the shortest path(s) from parent to root.
    """
    from collections import defaultdict

    all_shortest_paths = G.attrs["all_shortest_paths"]
    families = data["families"]

    # G can be either the main reference graph or a subset of the main graph.
    # When subG is passed in as G it is a subset and the attrs still exist from the original G
    # And when prize pruning is enabled some terminals and terminal prizes may be omitted completely.
    # So we need to ensure that all nodes for weighted edges are in G.
    # Roots have cost zero and may be an intermediate node in a different root, terminal path.

    weights = defaultdict(float)
    for root_idx in (r for r in G.attrs["root_indices"] if G.has_node(r)):
        for parent_idx in (p for p in families if G.has_node(p)):
            if root_idx == parent_idx:
                continue
            root_value = sum(
                data["solver_graph"][child]["prizes"].get(root_idx, 0.0) / G[child]["need_exploration_point"]
                for child in families[parent_idx]
                if G.has_node(child) and data["solver_graph"].has_node(child)
            )
            paths = all_shortest_paths.get((root_idx, parent_idx), [])
            for path in paths:
                if not all(G.has_node(u) for u in path):
                    continue
                for u, v in zip(path, path[1:]):
                    weights[u, v] += root_value / max(G[v].get("need_exploration_point"), 1)
                    if G.has_edge(v, u):
                        weights[v, u] += root_value / max(G[u].get("need_exploration_point"), 1)

    # The weights of edge_central_betweenness are interpreted as distances, so we need to invert them
    # such that higher weights indicate higher flow and higher flow indicates higher value
    global_max_value = max(weights.values())
    weights = {(u, v): global_max_value - value for (u, v), value in weights.items()}

    return weights


def get_scaled_edge_betweenness(
    G: PyDiGraph,
    data: dict[str, Any],
    feature_range: tuple[float, float] = (0.01, 1.0),
    omit_node_indices: set[int] = set(),
    weight_type: str | None = None,
) -> dict[tuple[int, int], float]:

    subG = G.copy()
    subG.remove_nodes_from(omit_node_indices)

    if weight_type is None:
        from rustworkx import edge_betweenness_centrality

        ebc = edge_betweenness_centrality(subG, normalized=False)  # unweighted
        ebc = {subG.get_edge_endpoints_by_index(e): v for e, v in ebc.items()}
    else:
        import networkx as nx

        if weight_type == "exploration":
            weights = {e: subG[e[1]]["need_exploration_point"] for e in subG.edge_list()}
        elif weight_type == "prize":
            weights = get_prize_flow_edge_weights(G, data)
        else:
            raise ValueError(f"Invalid weight_type: {weight_type}")

        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from(list(subG.edge_list()))
        nx.set_edge_attributes(nx_graph, weights, "weight")
        ebc = nx.edge_betweenness_centrality(nx_graph, normalized=False, weight="weight")

    X_scaled_dict = scale_dict_values(ebc, feature_range)  # type: ignore
    skipped_edges = set(G.edge_list()).difference(X_scaled_dict.keys())
    X_scaled_dict.update(dict.fromkeys(skipped_edges, feature_range[0]))

    return X_scaled_dict


def get_combined_betweenness_edge_weights(
    G: PyDiGraph, data: dict[str, Any], feature_range: tuple[float, float] = (0.01, 1.0)
) -> dict[tuple[int, int], float]:

    # OQUILLAS_EYE is a bottleneck node that workers cannot traverse
    OQUILLAS_EYE_IDX = {G.attrs["node_key_by_index"].inv[OQUILLAS_EYE_KEY]}

    # The PyDiGraph edge betweenness uses unweighted edges yielding a better
    # representation of overall flow between nodes.
    omitted_nodes = OQUILLAS_EYE_IDX
    rx_weights = get_scaled_edge_betweenness(G, data, feature_range, omitted_nodes, None)

    # The networkx edge betweenness uses weighted edges.
    # - exploration: yields a better visualization of the transit network's structural betweenness.
    # - prize: yields a better visualization of the prize flow network's structural betweenness.
    # Leaf nodes are omitted because when left in it skews visualization of the first hops.
    omitted_nodes = {i for i in G.node_indices() if G.out_degree(i) == 1} | OQUILLAS_EYE_IDX
    exploration_weights = get_scaled_edge_betweenness(G, data, feature_range, omitted_nodes, "exploration")
    prize_weights = get_scaled_edge_betweenness(G, data, feature_range, omitted_nodes, "prize")

    all_keys = exploration_weights.keys() | rx_weights.keys() | prize_weights.keys()
    min_weight = feature_range[0]
    avg_weights = {
        k: (
            rx_weights.get(k, min_weight)
            + exploration_weights.get(k, min_weight)
            + prize_weights.get(k, min_weight)
        )
        / 3
        for k in all_keys
    }

    return avg_weights


def add_edges_from_graph(fg: FeatureGroup, G: PyDiGraph, data: dict[str, Any], is_subgraph: bool):
    coords: GraphCoords = G.attrs["coords"]

    edge_weights = get_combined_betweenness_edge_weights(G, data, (1, 10))
    for u_idx, v_idx in G.edge_list():
        u_key = G[u_idx]["waypoint_key"]
        v_key = G[v_idx]["waypoint_key"]
        if v_idx < u_idx and SUPER_ROOT not in [u_key, v_key]:
            weight = edge_weights[(u_idx, v_idx)]
            opacity = (weight - 1) / 10
            if is_subgraph:
                opacity = 0.75 + 0.25 * opacity
            else:
                opacity = 0.5 + 0.5 * opacity

            PolyLine(
                locations=coords.as_geographics([u_idx, v_idx]),
                color=SUBGRAPH_EDGE_COLOR if is_subgraph else MAIN_GRAPH_EDGE_COLOR,
                weight=weight,
                opacity=opacity,
                popup=f"Edge: {u_key} - {v_key}",
                tooltip=f"Edge: {u_key} - {v_key}",
            ).add_to(fg)


def add_node_markers_from_graph(fg: FeatureGroup, G: PyDiGraph, data: dict[str, Any], is_subgraph: bool):
    coords = G.attrs["coords"]
    for node_idx in G.node_indices():
        node = G[node_idx]
        key = G[node_idx]["waypoint_key"]
        cost = G[node_idx]["need_exploration_point"]
        popup_text = f"Node Key: {key}, Cost: {cost}"

        if key == SUPER_ROOT:
            popup_text += " (Super Root)"
            node_color = SUPER_ROOT_COLOR
            circle_radius = ROOT_CIRCLE_RADIUS
        elif node["is_base_town"]:
            node_color = ROOT_NODE_COLOR
            circle_radius = ROOT_CIRCLE_RADIUS
        else:
            node_color = SUB_GRAPH_NODE_COLOR if is_subgraph else MAIN_GRAPH_NODE_COLOR
            circle_radius = SUB_GRAPH_CIRCLE_RADIUS if is_subgraph else NODE_CIRCLE_RADIUS

        CircleMarker(
            location=coords.as_geographic(node_idx),
            radius=circle_radius,
            color=node_color,
            fill_color=node_color,
            popup=popup_text,
            tooltip=popup_text,
        ).add_to(fg)


def read_rings_csv(csv_path: str = "tr_coverage_report.csv") -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        logger.debug(f"Loaded {len(df)} rows from {csv_path}")

        # Filter to per-ring rows only (numeric indices)
        ring_df = df[
            (pd.to_numeric(df["ring_index"], errors="coerce") >= 0)
            & (~df["ring_index"].astype(str).str.startswith("bin_"))
        ].copy()

        if ring_df.empty:
            logger.warning(f"No ring data found in {csv_path} — skipping overlays.")
            return None
    except FileNotFoundError:
        logger.warning(f"CSV not found: '{csv_path}' — skipping overlays.")
        return None

    return ring_df


def concave_hull_from_linestrings(G: PyDiGraph, node_indices: set[int]):
    coords = G.attrs["coords"]
    subG = subgraph_stable(G, node_indices)

    lines = [LineString(coords.as_cartesians(e)) for e in subG.edge_list() if e[0] < e[1]]
    if not lines:
        return None
    ml = MultiLineString(lines)

    buffer_dist = G.attrs.get("visual_buffer_dist", 2.0)
    buffered = ml.buffer(buffer_dist, cap_style="round", join_style="mitre")
    if buffered.is_empty:
        return None

    merged = unary_union(buffered)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda p: p.area)

    return merged


def get_binned_root_rings(
    root_key: int, ring_df: pd.DataFrame, bin_count: int = 5, max_cum_pct: float = 50.0
):
    rings = ring_df[ring_df["root"] == root_key].copy()
    if rings.empty:
        return []

    # Gather rings up to ring that _includes_ max_cum_pct
    starting_root_ring_count = len(rings)
    rings = rings.sort_values("Cum%Prize")
    cutoff_idx = rings["Cum%Prize"].searchsorted(max_cum_pct, side="right")
    rings = rings.iloc[: cutoff_idx + 1]
    if rings.empty:
        return []
    post_cutoff_root_ring_count = len(rings)

    # Re-aggregate rings to bin_count bin groups
    num_rings = len(rings)
    effective_bin_count = max(1, min(bin_count, num_rings))
    rings_per_bin = max(1, num_rings // effective_bin_count)
    extra = num_rings % effective_bin_count

    binned_rings = []
    start = 0
    for b in range(effective_bin_count):
        end = start + rings_per_bin + (1 if b < extra else 0)
        end = min(end, num_rings)
        if start < end:
            last_ring = rings.iloc[end - 1]
            try:
                hull = ast.literal_eval(last_ring["hull_nodes"])
            except (ValueError, SyntaxError) as e:
                logger.error(f"Bad hull_nodes in ring {end - 1}: {e} | row: {last_ring}")
                hull = []
            binned_rings.append({"hull_nodes": hull, "cum_pct": last_ring["Cum%Prize"]})
        start = end

    num_bins = len(binned_rings)
    logger.info(
        f"{root_key=}, {bin_count=}, {starting_root_ring_count=}, {cutoff_idx=}, {post_cutoff_root_ring_count=}, {effective_bin_count=}, {num_bins=}"
    )

    return binned_rings


def add_root_coverage_overlays(
    m: Map,
    G,
    csv_path: str = "tr_coverage_report.csv",
    bin_count: int = 5,
    max_cum_pct: float = 50.0,
):
    ring_df = read_rings_csv(csv_path)
    if ring_df is None:
        return None, {}

    fg_root_coverage_master = FeatureGroup(name="All Root Coverage", show=True)
    fg_root_coverage_master.add_to(m)
    assert isinstance(fg_root_coverage_master, FeatureGroup)
    root_coverage_feature_groups: dict[int, FeatureGroupSubGroup] = {}

    for root_idx in G.attrs["root_indices"]:
        key = G[root_idx]["waypoint_key"]
        binned_rings = get_binned_root_rings(key, ring_df, bin_count, max_cum_pct)
        if not binned_rings:
            continue

        fg_root_coverage = FeatureGroupSubGroup(fg_root_coverage_master, name=f"Root {key}", show=False)
        fg_root_coverage.add_to(m)
        root_coverage_feature_groups[key] = fg_root_coverage

        for bin_idx, bin_info in enumerate(reversed(binned_rings)):
            poly = concave_hull_from_linestrings(G, set(bin_info["hull_nodes"]))
            if poly is None or poly.is_empty:
                continue
            poly = poly.buffer(0.8).buffer(-0.4)

            polygons = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
            for p in polygons:
                FoliumPolygon(
                    locations=[(y, x) for x, y in p.exterior.coords],
                    color="white",
                    weight=2,
                    fillColor="white",
                    fillOpacity=0.3,
                    popup=(
                        f"<b>Root:</b> {key}<br>"
                        f"<b>Bin:</b> {bin_idx + 1}<br>"
                        f"<b>Prize coverage:</b> {bin_info['cum_pct']:.1f}%"
                    ),
                ).add_to(fg_root_coverage)

    return fg_root_coverage_master, root_coverage_feature_groups


def add_terminal_sets_markers(
    m: Map, G: PyDiGraph
) -> tuple[FeatureGroup | None, dict[str, FeatureGroupSubGroup]]:
    if "terminal_sets" not in G.attrs:
        print("Warning: terminal_sets attribute not found in graph.")
        return None, {}

    def add_marker(type: str, location: tuple[float, float], key: int, cost: int, parent_fg):
        Marker(
            location=location,
            icon=BeautifyIcon(
                icon="house" if type == "Root" else "",
                icon_shape="marker",
                iconSize=[30, 30] if type == "Root" else [20, 20],
                iconAnchor=[15, 15] if type == "Root" else [10, 20],
                background_color="white",
                text_color="white",
                border_color="white",
            ),  # type: ignore
            popup=f"{type} Node Key: {key}, Cost: {cost}",
            tooltip=f"{type} Node Key: {key}, Cost: {cost}",
        ).add_to(parent_fg)

    fg_terminal_sets_master = FeatureGroup(name="All Terminal Sets", show=True)
    fg_terminal_sets_master.add_to(m)
    assert isinstance(fg_terminal_sets_master, FeatureGroup)
    terminal_set_feature_groups: dict[str, FeatureGroupSubGroup] = {}

    coords = G.attrs["coords"]
    terminal_sets = G.attrs["terminal_sets"]

    for root, terminal_set in terminal_sets.items():
        layer_name = f"Terminal Set {G[root]['waypoint_key']}"
        fg_terminal_set = FeatureGroupSubGroup(fg_terminal_sets_master, name=layer_name, show=False)
        fg_terminal_set.add_to(m)
        terminal_set_feature_groups[layer_name] = fg_terminal_set

        cost = G[root]["need_exploration_point"]
        key = G[root]["waypoint_key"]
        location = coords.as_geographic(root)
        add_marker("Root", location, key, cost, fg_terminal_set)

        for terminal in terminal_set:
            cost = G[terminal]["need_exploration_point"]
            key = G[terminal]["waypoint_key"]
            location = coords.as_geographic(terminal)
            add_marker("Terminal", location, key, cost, fg_terminal_set)

    return fg_terminal_sets_master, terminal_set_feature_groups


def recolorize(m):
    """
    Recompute conflict-free colors for all root regions and their terminals,
    based on the final polygon geometries already added to the map.
    """
    ROOT_RE = re.compile(r"Root (\d+)")
    TERM_RE = re.compile(r"Terminal Set (\d+)")

    # ------------------------------------------------------------
    # 1. Extract polygons grouped by root_id
    # ------------------------------------------------------------
    root_polys = {}  # root_id -> [shapely_poly, ...]
    fg_lookup_coverage = {}  # root_id -> list of FGSubGroups for coverage
    fg_lookup_terminal = {}  # root_id -> list of FGSubGroups for terminals

    for _, fg in m._children.items():
        if not isinstance(fg, FeatureGroupSubGroup):
            continue
        name = getattr(fg, "layer_name", None)
        if not isinstance(name, str):
            continue

        # Root overlays
        m_root = ROOT_RE.fullmatch(name)
        if m_root:
            root = int(m_root.group(1))
            fg_lookup_coverage.setdefault(root, []).append(fg)
            for child in fg._children.values():
                if child._name == "Polygon":
                    poly = Polygon([(lng, lat) for lat, lng in child.locations])  # type: ignore
                    root_polys.setdefault(root, []).append(poly)

        # Terminal sets
        m_term = TERM_RE.fullmatch(name)
        if m_term:
            root = int(m_term.group(1))
            fg_lookup_terminal.setdefault(root, []).append(fg)

    if not root_polys:
        logger.warning("No root polygons found in map... skipping recoloring.")
        return

    # ------------------------------------------------------------
    # 2. Unified polygon per root (for conflict detection)
    # ------------------------------------------------------------
    unified = {root: unary_union(polys) for root, polys in root_polys.items()}

    # ------------------------------------------------------------
    # 3. Build conflict graph
    # ------------------------------------------------------------
    roots = list(unified.keys())
    conflicts = {r: set() for r in roots}
    for i in range(len(roots)):
        r1, p1 = roots[i], unified[roots[i]]
        for j in range(i + 1, len(roots)):
            r2, p2 = roots[j], unified[roots[j]]
            if p1.intersects(p2):
                conflicts[r1].add(r2)
                conflicts[r2].add(r1)

    # ------------------------------------------------------------
    # 4. DSATUR coloring
    # ------------------------------------------------------------
    color_of = {}
    saturation = {r: 0 for r in roots}

    def choose_next():
        return max((r for r in roots if r not in color_of), key=lambda r: (saturation[r], len(conflicts[r])))

    while len(color_of) < len(roots):
        r = choose_next()
        neighbor_colors = {color_of[n] for n in conflicts[r] if n in color_of}
        c = 0
        while c in neighbor_colors:
            c += 1
        color_of[r] = c
        for n in conflicts[r]:
            if n not in color_of:
                neighbor_c = {color_of[x] for x in conflicts[n] if x in color_of}
                saturation[n] = len(neighbor_c)

    # ------------------------------------------------------------
    # 5. Generate root colors
    # ------------------------------------------------------------
    # fmt: off
    PALETTE = [
        "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#42D4F4",
        "#F032E6", "#BFEF45", "#FABED4", "#469990", "#DCBEFF", "#9A6324", "#800000",
        "#AAFFC3", "#808000", "#FFD8B1", "#000075", "#FF4500", "#00FA9A", "#FF1493",
        "#00CED1", "#9400D3", "#FF8C00", "#32CD32", "#8A2BE2", "#B22222", "#228B22",
        "#9932CC", "#FF6347", "#40E0D0", "#EE82EE", "#D2691E", "#CD5C5C", "#00FF7F",
        "#DC143C", "#ADFF2F", "#FF00FF", "#1E90FF", "#FF69B4", "#00FF00", "#FF0000",
        "#0000FF", "#FFFF00",
    ]
    # fmt: on

    max_c_idx = max(color_of.values())
    n_needed = max_c_idx + 1
    if n_needed <= len(PALETTE):
        root_colors = PALETTE[:n_needed]
    else:
        # Fallback: cycle with slight HSV shift (very rare)
        base = PALETTE
        root_colors = []
        for i in range(n_needed):
            idx = i % len(base)
            r, g, b = mcolors.to_rgb(base[idx])
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h = (h + 0.17 * (i // len(base))) % 1.0
            s = min(1.0, s * 1.1)
            v = min(1.0, v * 1.1)
            root_colors.append(mcolors.to_hex(colorsys.hsv_to_rgb(h, s, v)))
    logger.info(f"Root colors: {root_colors}")

    # ------------------------------------------------------------
    # 6. Gradient generator for bins (vibrant, noticeable)
    # ------------------------------------------------------------
    def gradient_for_root(base_hex: str, n_bins: int) -> list[str]:
        """
        - Bin 0 (innermost): 90% of base color (dark, rich)
        - Bin N (outermost): 50% base + 50% white → still recognizable, never gray/pale
        """
        if n_bins <= 0:
            return []
        if n_bins == 1:
            return [base_hex]

        base_rgb = np.array(mcolors.to_rgb(base_hex))

        # Inner: slightly darkened version
        dark = base_rgb * 0.9
        dark = np.clip(dark, 0, 1)

        # Outer: blend toward white, cap at 50% to avoid washout
        light = base_rgb * 0.6 + np.array([0.4, 0.4, 0.4])
        colors = np.linspace(dark, light, n_bins)

        return [mcolors.to_hex(c.clip(0, 1)) for c in colors]

    # ------------------------------------------------------------
    # 7. Apply colors to all coverage and terminals
    # ------------------------------------------------------------
    for root in fg_lookup_coverage:
        c_idx = color_of[root]
        base_color = root_colors[c_idx]
        logger.debug(f"Applying color {base_color} (index {c_idx}) to root {root}")

        # Apply gradient to coverage bins
        for fg in fg_lookup_coverage[root]:
            n_bins = len(fg._children)
            colors = gradient_for_root(base_color, n_bins)
            for child, color in zip(fg._children.values(), colors):
                if hasattr(child, "locations") and isinstance(child.locations, list):
                    child.options["color"] = color
                    child.options["fillColor"] = color
                    child.options["fillOpacity"] = 0.35

        # Terminals: use base color (same as root)
        for fg in fg_lookup_terminal.get(root, []):
            for child in fg._children.values():
                if hasattr(child, "icon"):
                    child.icon.options["background_color"] = base_color


def visualize_solution_graph(
    main_graph: PyDiGraph,
    data: dict[str, Any],
    sub_graph: PyDiGraph | None = None,
    bin_count: int = 5,
    max_cum_pct: float = 50.0,
):
    m = Map(
        crs="Simple",
        location=[32.5, 0],
        zoom_start=1.9,  # type: ignore
        zoom_snap=0.25,
        tiles=None,
    )

    tile_pane = CustomPane("tile_pane", z_index=1)  # type: ignore
    m.add_child(tile_pane)

    tile_layer = TileLayer(
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

    fg_main_graph_nodes = FeatureGroup(name="Main Graph Nodes", show=True)
    fg_sub_graph_nodes = FeatureGroup(name="Subgraph Nodes", show=True)
    fg_main_graph_edges = FeatureGroup(name="Main Graph Edges", show=True)
    fg_sub_graph_edges = FeatureGroup(name="Subgraph Edges", show=True)

    logger.info("    setting up main graph...")
    add_node_markers_from_graph(fg_main_graph_nodes, main_graph, data, is_subgraph=False)
    add_edges_from_graph(fg_main_graph_edges, main_graph, data, is_subgraph=False)

    if sub_graph is not None:
        logger.info("    setting up subgraph...")
        add_node_markers_from_graph(fg_sub_graph_nodes, sub_graph, data, is_subgraph=True)
        add_edges_from_graph(fg_sub_graph_edges, sub_graph, data, is_subgraph=True)

    fg_terminal_master, terminal_groups = add_terminal_sets_markers(m, main_graph)
    assert isinstance(fg_terminal_master, FeatureGroup)

    fg_root_coverage_master, root_coverage_subgroups = add_root_coverage_overlays(
        m,
        main_graph,
        bin_count=bin_count,
        max_cum_pct=max_cum_pct,
    )
    assert isinstance(fg_root_coverage_master, FeatureGroup)

    fg_main_graph_nodes.add_to(m)
    fg_sub_graph_nodes.add_to(m)
    fg_main_graph_edges.add_to(m)
    fg_sub_graph_edges.add_to(m)
    recolorize(m)

    m.keep_in_front(
        fg_main_graph_edges,
        fg_root_coverage_master,
        *root_coverage_subgroups.values(),
        fg_sub_graph_edges,
        fg_main_graph_nodes,
        fg_sub_graph_nodes,
        fg_terminal_master,
        *terminal_groups.values(),
    )

    group_layer_control = GroupedLayerControl(
        groups={
            "Base": [fg_main_graph_nodes, fg_sub_graph_nodes, fg_main_graph_edges, fg_sub_graph_edges],
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


def main(config: dict[str, Any], kwargs: dict[str, Any]):
    lodging = kwargs["min_lodging"]
    prices = kwargs["prices"]
    modifiers = kwargs["modifiers"]
    grindTakenList = kwargs["grind_taken_list"]
    bin_count = kwargs["bin_count"]
    max_cum_pct = kwargs["max_cum_pct"]
    solution_terminals = kwargs["solution_terminals"]
    solution_nodes = kwargs["solution_nodes"]

    data = generate_reference_data(config, prices, modifiers, lodging, grindTakenList)
    generate_graph_data(data)
    solver_graph = deepcopy(data["G"].copy())
    solver_graph.attrs = deepcopy(data["G"].attrs)
    data["solver_graph"] = solver_graph

    reduce_prize_data(data)

    G = data["solver_graph"]
    assert isinstance(G, PyDiGraph)

    prune_NTD1(G)
    G.attrs["shortest_paths"] = get_all_pairs_shortest_paths(G)
    G.attrs["all_shortest_paths"] = get_all_pairs_all_shortest_paths(G)
    add_scaled_coords_to_graph(G)

    # Ensure all solution terminals are in the solution nodes.
    solution_nodes.update(key for key in solution_terminals.keys())
    solution_nodes.update(key for key in solution_terminals.values())

    node_key_by_index = G.attrs["node_key_by_index"]
    solution_indices = [node_key_by_index.inv[key] for key in solution_nodes]

    subG = subgraph_stable(G, solution_indices)
    assert isinstance(subG, PyDiGraph)
    subG.attrs = G.attrs
    if SUPER_ROOT in solution_terminals.values():
        from api_rx_pydigraph import inject_super_root

        inject_super_root(config, G)
    set_graph_terminal_sets_attribute(subG, solution_terminals)

    cost = sum(node["need_exploration_point"] for node in subG.nodes())
    print(f"Cost of highlighted nodes: {cost}")

    visualize_solution_graph(
        main_graph=G,
        data=data,
        sub_graph=subG,
        bin_count=bin_count,
        max_cum_pct=max_cum_pct,
    )


if __name__ == "__main__":
    # Common config
    budget = 500

    config = {}
    config["name"] = "Empire"
    config["budget"] = budget  # N/A
    config["top_n"] = 24  # all prizes
    config["nearest_n"] = 24  # all roots
    config["max_waypoint_ub"] = 30  # N/A

    config["prune_prizes"] = False
    prize_t = 1 - (budget * 0.0005)
    config["prize_pruning_threshold_factors"] = {
        "min": {"only_child": prize_t, "dominant": prize_t, "protected": prize_t},
        "max": {"only_child": 1, "dominant": 1, "protected": 1},
    }
    config["capacity_mode"] = "min"
    config["solver"] = {}  # N/A

    config["logger"] = {"level": "INFO", "format": "<level>{message}</level>"}
    set_logger(config)

    # Common data
    kwargs = {
        "prices": ds.read_json("en_lta_prices.json")["effectivePrices"],
        "modifiers": {},
        "grind_taken_list": [],
        "min_lodging": ds.read_json("lodging_specifications.json")["min"],
        "max_lodging": ds.read_json("lodging_specifications.json")["max"],
        "bin_count": 4,
        "max_cum_pct": 35.0,
    }

    # fmt: off
    # MARK: Subgraph
    solution_terminals = {
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
    solution_nodes = set([
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

    solution_terminals = dict(sorted(solution_terminals.items(), key=lambda item: item[1]))

    kwargs["solution_terminals"] = solution_terminals
    kwargs["solution_nodes"] = solution_nodes

    main(config, kwargs)
