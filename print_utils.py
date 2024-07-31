from operator import itemgetter
import natsort
import pulp

from generate_empire_data import NodeType


def print_solver_output(prob, data, arc_vars, doFull=False):
    if doFull:
        # Write model for debugging
        prob.writeLP("sample_LP.lp")

        # Print all decision variables with non-zero values
        print("Decision Variables with values:")
        for group in [
            "flow_source",
            "flow_prod",
            "flow_city_",
            "flow_active_prod",
            "flow_transit",
            "flow_warehouse",
            "load_transit",
            "has_load",
        ]:
            for v in prob.variables():
                if v.name.startswith(group):
                    if v.varValue is not None and v.varValue > 0:
                        print(v.name, "=", v.varValue)

        # Print the activated arcs with their details
        print("\nActivated arcs:")
        for group in [
            "source",
            "prod",
            "city_",
            "active_prod",
            "transit",
            "warehouse",
        ]:
            for arc in data["arcs"]:
                if arc["source"].startswith(group):
                    var = arc_vars[(arc["source"], arc["destination"])]
                    if var.varValue is not None and var.varValue > 0:
                        print(
                            f"arc from {arc['source']} to {arc['destination']} with flow {var.varValue} has value {arc['value']} and cost {arc['cost']}"
                        )

    # Output results
    total_value = 0
    total_cost = 0
    counted_destinations = set()
    city_active_prods = {city: [] for city in data["cities"]}
    active_transit_nodes = []

    for arc in data["arcs"]:
        arc_var = arc_vars[(arc["source"], arc["destination"])]
        flow_value = pulp.value(arc_var)
        if flow_value is None and doFull:
            print(f"arc_var: {arc_var} is NONE")
            continue
        elif flow_value > 0:
            total_value += arc["value"] * flow_value
            destination_node = arc["destination"]
            if destination_node not in counted_destinations:
                total_cost += arc["cost"]
                counted_destinations.add(destination_node)
            if arc["destination"].startswith("transit_"):
                active_transit_nodes.append(arc["destination"])
            if arc["destination"].startswith("active_prod") and not arc["source"].startswith(
                "transit_"
            ):
                city_id = int(arc["source"].split("_")[1])
                city_active_prods[city_id].append(arc["destination"])
            if doFull:
                print(
                    f"arc from {arc['source']} to {arc['destination']} with flow {flow_value} has value {arc['value']} and cost {arc['cost']}"
                )

    # Summary output
    print("\nSummary:")
    print(f"Total Value Achieved: {total_value}")
    print(f"Total Cost: {total_cost} with a max of {data["max_cost"]}")
    for city, active_prods in city_active_prods.items():
        print(
            f"City {city} Active Production Nodes ({len(active_prods)}): {', '.join([prod.replace('active_prod_', '') for prod in active_prods])}"
        )
    transit_nodes = sorted(
        [int(node.replace("transit_", "")) for node in list(set(active_transit_nodes))]
    )
    print(f"Transit nodes ({len(transit_nodes)}):\n{transit_nodes}")


def print_empire_solver_output(prob, graph_data, ref_data, max_cost, top_n, detailed=False):
    solution_vars = {}
    gt0lt1_vars = set()

    for v in prob.variables():
        if v.varValue is not None and v.varValue > 0:
            if v.varValue < 1:
                gt0lt1_vars.add(v.name)
            rounded_value = round(v.varValue)
            if rounded_value >= 1:
                solution_vars[v.name] = {
                    "value": rounded_value,
                    "lowBound": v.lowBound,
                    "upBound": v.upBound,
                }

    calculated_cost = 0
    outputs = []
    arc_loads = []
    waypoint_loads = []
    for k, v in solution_vars.items():
        if k.startswith("Load_"):
            kname = k.replace("Load_", "")
            if "_to_" in k:
                # An arc
                source, destination = kname.split("_to_")
                arc_key = (source, destination)
                arc = graph_data["arcs"].get(arc_key, None)
                outputs.append(f"{arc}, {v}")
                if arc.source.type is NodeType.waypoint or arc.destination.type is NodeType.waypoint:
                    arc_loads.append(v["value"])
            else:
                # A node
                node = graph_data["nodes"].get(kname, None)
                outputs.append(f"{node}, {v}")
                calculated_cost = calculated_cost + node.cost
                if node.type is NodeType.waypoint:
                    waypoint_loads.append((node, v["value"]))
    outputs = natsort.natsorted(outputs)

    solver_cost = 0
    if "cost" in solution_vars.keys():
        solver_cost = int(solution_vars["cost"]["value"])

    solver_value = round(solution_vars["value"]["value"])

    if detailed:
        print("\nDetailed Solution:")
        for output in outputs:
            print(output)

    print()
    print("         Load_source:", solution_vars["Load_source"]["value"])
    print("           Load_sink:", solution_vars["Load_sink"]["value"])
    print()
    print("     Calculated cost:", calculated_cost)
    print("         Solver cost:", calculated_cost)
    print("            Max cost:", max_cost)
    print("               Value:", round(solver_value))
    print("          Value/Cost:", round(solver_value / max(1, solver_cost, calculated_cost)))
    print()
    print("    Num origin nodes:", len([x for x in outputs if x.startswith("Node(name: origin_")]))
    print("   Max waypoint load:", max(waypoint_loads, key=itemgetter(1)))
    print("       Max arc load:", max(arc_loads))
    print()
    if gt0lt1_vars:
        print("WARNING: 0 < x < 1 vars count:", len(gt0lt1_vars))
        print()

    data = {"max_cost": max_cost, "solution_vars": solution_vars}
    return data
