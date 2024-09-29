# from enum import IntEnum, auto
# import itertools
# import json
import locale
# import os
# from typing import Dict, Set

# from natsort import natsorted
# import networkx as nx
# from numpy import average

# from milp_flow.generate_empire_data (
#     Node,
#     NodeType,
#     generate_empire_data,
#     get_reference_data,
#     get_link_node_type,
#     GraphData,
# )

locale.setlocale(locale.LC_ALL, "")

config = {
    "budget": 501,
    "lodging_bonus": 0,
    "top_n": 4,
    "nearest_n": 5,
    "waypoint_ub": 25,
    "solve_log_level": "INFO",
    "data_log_level": "INFO",
    "solver": {
        "file_prefix": "",
        "file_suffix": "",
        "mips_tol": 1e-6,
        "mips_gap": "auto",
        "time_limit": "inf",
    },
}

# ref_data = get_reference_data(config)
# ref_data["waypoint_capacity"] = min(config["waypoint_ub"], config["budget"])
# GD: GraphData = generate_empire_data(config)


# plant_values_by_group = {}
# for plant_id, group_data in ref_data["plant_values"].items():
#     for group_id, details in group_data.items():
#         if group_id not in plant_values_by_group:
#             plant_values_by_group[group_id] = {}
#         plant_values_by_group[group_id][plant_id] = details

# for group_id in plant_values_by_group:
#     plant_values_by_group[group_id] = dict(
#         sorted(
#             plant_values_by_group[group_id].items(), key=lambda item: item[1]["value"], reverse=True
#         )
#     )

# top_n_per_group = set()
# for group_id, details in plant_values_by_group.items():
#     for i, (plant_id, entry) in enumerate(details.items()):
#         if i > 20:
#             break
#         top_n_per_group.add(plant_id)
#         print(group_id, plant_id, entry["value"])
# print(natsorted(top_n_per_group))
# print(len(top_n_per_group))
# all_plants = set(n.id for n in GD["P"].values())
# worst_nodes = all_plants.symmetric_difference(top_n_per_group)
# print(natsorted(worst_nodes))
# print(len(worst_nodes))

# placements = {}

# for pz, pz_values in origin_values.items():
#     for position, key in enumerate(pz_values.keys()):
#         if key not in placements.keys():
#             placements[key] = []
#         placements[key].append(position)

# for k, v in placements.items():
#     print(k, sorted(Counter(v).most_common()))


# def get_explore_node_type(node):
#     if node in towns:
#         return NodeType.TOWN
#     if node in all_plantzones:
#         if node not in origins:
#             return NodeType.INVALID
#         return NodeType.ORIGIN
#     return NodeType.WAYPOINT


# link_count = 0
# type_pairs = set()


# G = nx.Graph()
# dG = nx.DiGraph()
# for link in ref_data["waypoint_links"]:
#     u_type = get_link_node_type(str(link[0]), ref_data)
#     v_type = get_link_node_type(str(link[1]), ref_data)
#     if u_type < v_type:
#         u, v = link[0], link[1]
#     else:
#         u_type, v_type = v_type, u_type
#         u, v = link[1], link[0]

#     cost = (
#         ref_data["waypoint_data"][str(u)]["CP"],
#         ref_data["waypoint_data"][str(v)]["CP"],
#     )

#     G.add_edge(u, v, weight=sum(cost))
#     G.nodes[u]["weight"] = cost[0]
#     G.nodes[v]["weight"] = cost[1]
#     G.nodes[u]["type"] = u_type
#     G.nodes[v]["type"] = v_type
#     dG.add_edge(u, v, weight=cost[1])
#     dG.add_edge(v, u, weight=cost[0])

# print(G)
# print(dG)

# plants = [int(p) for p in ref_data["plants"]]
# towns = [int(t) for t in ref_data["towns"]]
# town = towns[0]

# G_distances = {
#     # Edge costs are sum(cost)/2. The final node's cost is 1/2 actual cost, so add the other 1/2.
#     p: int((nx.shortest_path_length(G, town, p, weight="weight") + G.nodes[p]["weight"]) / 2)
#     for p in plants
# }
# dG_distances = {p: nx.shortest_path_length(dG, town, p, weight="weight") for p in plants}

# G_paths = {p: nx.shortest_path(G, town, p, weight="weight") for p in plants}
# G_paths = dict(sorted(G_paths.items(), key=lambda x: x[0]))
# dG_paths = {p: nx.shortest_path(dG, town, p, weight="weight") for p in plants}
# dG_paths = dict(sorted(dG_paths.items(), key=lambda x: x[0]))

# for k, G_path, dG_path in zip(G_paths.keys(), G_paths.values(), dG_paths.values()):
#     if G_path != dG_path:
#         num_G_towns = sum([v in towns for v in G_path])
#         num_dG_towns = sum([v in towns for v in dG_path])
#         print(" G_path", G_path, f"({G_distances[k]})", f"[{num_G_towns}]")
#         print("dG_path", dG_path, f"({dG_distances[k]})", f"[{num_dG_towns}]")
#         assert {G_distances[k]} == {dG_distances[k]}
#         print()


# parents = set()
# for node in G:
#     for neighbor in nx.neighbors(G, node):
#         if G.nodes[neighbor]["type"] is NodeType.plant:
#             parents.add(node)
# print("Num plant parent nodes:", len(parents))

# for warehouse, plantzones_data in origin_values.items():
#     for plantzone, plantzone_data in plantzones_data.items():
#         if plantzone_data["value"] == 0:
#             continue
#         # Link source -> demand -> origin
#         link_count = link_count + 2

# for warehouse, warehouse_lodging_data in lodging_per_warehouse.items():
#     # Link town -> warehouse
#     link_count = link_count + 1

#     for lodging_data in warehouse_lodging_data:
#         # Link warehouse -> lodging -> sink
#         link_count = link_count + 2

# print("Link count", link_count)
# for type_pair in sorted(list(type_pairs)):
#     print(type_pair)

# outputs = []
# for pz, values_data in origin_values.items():
#     v_w = list(values_data.keys())
#     w_a = v_w[0]
#     w_b = v_w[-1]
#     values = list([d["value"] for d in values_data.values()])
#     v_a = values[0]
#     v_b = values[-1]
#     loss_percent = round((v_a - v_b) / v_a, 4)
#     outputs.append((pz, (w_a, v_a), (w_b, v_b), loss_percent))
# outputs = sorted(outputs, key=lambda x: x[3], reverse=True)
# for output in outputs:
#     print(output)

# with open("/home/thell/milp-flow/data/user/empire_solution_11.json", "r") as file:
#     data = json.load(file)

#     for k, v in data["solution_vars"].items():
#         if v["value"] > 0:
#             print(k, v)

# G = nx.DiGraph()
# for node_id, node in graph_data["nodes"].items():
#     G.add_node(node.id, type=node.type)

# for arc_id, arc in graph_data["arcs"].items():
#     weight = arc.destination.cost
#     if "1727" in arc.name():
#         weight = 999999
#     G.add_arc(arc.source.id, arc.destination.id, weight=weight)

# all_pairs_shortest_paths = dict(nx.all_pairs_bellman_ford_path_length(G, weight="weight"))

# nearest_n = 5
# nearest_warehouses = {}
# nearest_warehouse_nodes = {}

# for node_id, node in graph_data["nodes"].items():
#     if node.type in {NodeType.waypoint, NodeType.town}:
#         distances = []
#         for warehouse_key, warehouse in graph_data["warehouse_nodes"].items():
#             town_id = ref_data["warehouse_to_town"][warehouse.id]
#             distances.append((warehouse_key, all_pairs_shortest_paths[node.id][town_id]))
#         nearest_warehouses[node_id] = sorted(distances, key=lambda x: x[1])[:nearest_n]

# print(nearest_warehouses["waypoint_42"])
# print(nearest_warehouses["waypoint_326"])

# find all nodes not part of __a__ shortest path in the paths from supply to nearest_n towns


# def read_workerman_json(filepath):
#     """Read and return json as dict."""
#     with open(os.path.join(os.path.dirname(__file__), filepath), "r") as file:
#         data = json.load(file)
#     return data


# def get_rooted_tree(town, terminals, ref_data):
#     link_nodes: Dict[str, Node] = {}
#     nodes, graph, terminals = get_links_graph(link_nodes, ref_data)
#     tree = process_graph(graph, terminals, town)
#     for terminal in terminals:
#         short_path = nx.shortest_path(tree, town, terminal)
#         for u, v in zip(short_path, short_path[1:]):
#             tree.edges[(u, v)]["weight"] = tree.nodes[v]["weight"]
#     return nodes, tree


# def order_discounted_terminals(tree, town, terminals, ref_data):
#     remaining_terminals = terminals.copy()
#     warehouse = ref_data["town_to_root"][town]

#     ordered_terminals = []
#     while remaining_terminals:
#         best = {"terminal": 0, "value": 0, "cost": 9999}
#         i = 0
#         for i, terminal in enumerate(remaining_terminals):
#             value = ref_data["terminal_values"][terminal][warehouse]["value"]
#             if value == 0:
#                 remaining_terminals.pop(i)
#                 continue
#             cost = nx.shortest_path_length(tree, town, terminal, weight="weight")
#             if cost <= best["cost"] and value >= best["value"]:
#                 best = {"terminal": terminal, "value": round(value / cost), "cost": cost}
#         ordered_terminals.append(best)
#         remaining_terminals.pop(remaining_terminals.index(best["terminal"]))

#         short_path = nx.shortest_path(tree, town, best["terminal"])
#         path_cost = 0
#         for u, v in zip(short_path, short_path[1:]):
#             path_cost += tree.edges[(u, v)]["weight"]
#             if tree.edges[(u, v)]["weight"] >= 1:
#                 tree.edges[(u, v)]["weight"] = 0

#     return ordered_terminals


# def get_all_used_nodes(ref_data):
#     used_nodes: Set[int] = set()
#     for town in natsorted(ref_data["towns"]):
#         terminals = list(ref_data["terminals"])
#         nodes, tree = get_rooted_tree(town, terminals, ref_data)
#         used_nodes = used_nodes.union(set(list(tree)))
#     return used_nodes


# def get_unused_nodes(ref_data):
#     nodes: Dict[str, Node] = {}
#     graph, terminals = get_links_graph(nodes, ref_data)

#     towns = ref_data["towns"]
#     for u, v in itertools.combinations(towns, 2):
#         graph.add_edge(u, v)
#     print(graph)
#     tree = process_graph(graph, terminals)
#     print(tree)
#     unused_nodes = set(list(n.id for n in nodes.values())).difference(set(list(tree)))
#     return unused_nodes


# unused_nodes = get_unused_nodes(ref_data)
# print(natsorted(unused_nodes))
# print(len(unused_nodes))

# all_used_nodes = natsort.natsorted(get_all_used_nodes(ref_data))
# print(len(all_used_nodes))


# for town in natsort.natsorted(ref_data["towns"]):
#     # if town != "1":
#     #     continue
#     terminals = list(ref_data["terminals"])
#     nodes, tree = get_rooted_tree(town, terminals, ref_data)
#     discounted_terminals = order_discounted_terminals(tree.copy(), town, terminals, ref_data)

#     root = ref_data["town_to_root"][town]
#     values: List[Dict[str, Any]] = []

#     for terminal in ref_data["terminals"]:
#         value = round(ref_data["terminal_values"][terminal][root]["value"])
#         if value > 0:
#             rank = list(ref_data["terminal_values"][terminal].keys()).index(root) + 1
#             t_values = ref_data["terminal_values"][terminal]
#             data = {"town": town, "terminal": terminal, "tt_value": value, "tt_rank": rank}
#             values.append(data)

#     values = sorted(values, key=lambda x: x["tt_value"], reverse=True)
#     for i, v in enumerate(values, start=1):
#         v["tv_rank"] = i

#     for data in values:
#         for i, terminal in enumerate(discounted_terminals, start=1):
#             if terminal["terminal"] == data["terminal"]:
#                 data["dt_value"] = round(terminal["value"])
#                 data["dt_rank"] = i

#     taken_terminals = [
#         str(w["job"]["pzk"]) for w in sol_data["userWorkers"] if w["tnk"] == int(town)
#     ]
#     for data in values:
#         data["taken"] = data["terminal"] in taken_terminals

#     for data in values:
#         data["rank_score"] = round(average([data[k] for k in data.keys() if "rank" in k]))

#     values = sorted(values, key=lambda x: x["rank_score"])

#     # values_to_print = [v for v in values if v["taken"] or v["tt_rank"] <= 4]
#     # for value in sorted(values_to_print, key=lambda x: x["dt_rank"]):
#     values_to_print = values
#     top_4s = 0
#     for i, value in enumerate(values_to_print):
#         if value["tt_rank"] <= 4:
#             top_4s += 1
#         if value["taken"]:
#             print(i, value)
#     print("len top 4s:", top_4s)

#     print()

# seed = 42  # For reproducibility
# pos = nx.planar_layout(tree, 5)
# pos = nx.spring_layout(tree, pos=pos, seed=seed)
# pos = nx.kamada_kawai_layout(tree, pos=pos)
# plt.figure()
# plt.title(f"Tree rooted at {town}")
# nx.draw(tree, pos=pos, with_labels=True, alpha=0.5)
# plt.show()


# def parse_model(filename):
#     filepath = os.path.join(os.path.dirname(__file__), filename)
#     with open(filepath, "r") as sol:
#         outputs = []
#         token0 = ""
#         token1 = ""
#         for line in sol:
#             if line.startswith("End"):
#                 break
#             if line.startswith(" "):
#                 continue

#             tokens = line.split("_")
#             if ":" in tokens[0]:
#                 token0 = tokens[0].split(":")[0]
#                 token1 = ""
#                 outputs.append((token0, token1))
#             elif "_" in line:
#                 if token0 == tokens[0] and token1 == tokens[1]:
#                     continue
#                 else:
#                     token0 = tokens[0]
#                     token1 = tokens[1]
#                     outputs.append((token0, token1))
#             else:
#                 outputs.append(line)
#         return outputs


# outputs = parse_model("highs_output/models/unicode_cleanup_mc5_lb0_tn4_nn5_wc25_gdefault.lp")
# for output in outputs:
#     print(output)

# lodgingP2W = {
#     "lodgingP2W": {
#         "5": 5,
#         "32": 5,
#         "52": 5,
#         "77": 5,
#         "88": 5,
#         "107": 0,
#         "120": 0,
#         "126": 0,
#         "182": 0,
#         "202": 0,
#         "218": 0,
#         "221": 0,
#         "229": 0,
#         "601": 0,
#         "605": 0,
#         "619": 0,
#         "693": 0,
#         "706": 0,
#         "735": 0,
#         "873": 0,
#         "955": 0,
#         "1000": 0,
#         "1124": 0,
#         "1210": 0,
#         "1219": 0,
#         "1246": 0,
#         "1375": 0,
#         "1382": 0,
#         "1420": 0,
#         "1424": 0,
#         "1444": 0,
#     }
# }

# lodging_bonuses = {}
# for k, v in lodgingP2W["lodgingP2W"].items():
#     town = ref_data["group_to_townname"].get(k, None)
#     if town == None:
#         print(k)
#     else:
#         lodging_bonuses[town] = v
# print(lodging_bonuses)

# purchased_lodging = {
#     "Velia": 0,
#     "Heidel": 0,
#     "Glish": 0,
#     "Calpheon City": 0,
#     "Olvia": 0,
#     "Keplan": 0,
#     "Port Epheria": 0,
#     "Trent": 0,
#     "Iliya Island": 0,
#     "Altinova": 0,
#     "Tarif": 0,
#     "Valencia City": 0,
#     "Shakatu": 0,
#     "Sand Grain Bazaar": 0,
#     "Ancado Inner Harbor": 0,
#     "Arehaza": 0,
#     "Old Wisdom Tree": 0,
#     "Grána": 0,
#     "Duvencrune": 0,
#     "O'draxxia": 0,
#     "Eilton": 0,
#     "Dalbeol Village": 0,
#     "Nampo's Moodle Village": 0,
#     "Nopsae's Byeot County": 0,
#     "Muzgar": 0,
#     "Yukjo Street": 0,
#     "Godu Village": 0,
#     "Bukpo": 0,
# }
