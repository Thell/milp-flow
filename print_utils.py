import pulp


def print_solver_output(prob, data, edge_vars, doFull=False):
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

        # Print the activated edges with their details
        print("\nActivated Edges:")
        for group in [
            "source",
            "prod",
            "city_",
            "active_prod",
            "transit",
            "warehouse",
        ]:
            for edge in data["edges"]:
                if edge["source"].startswith(group):
                    var = edge_vars[(edge["source"], edge["destination"])]
                    if var.varValue is not None and var.varValue > 0:
                        print(
                            f"Edge from {edge['source']} to {edge['destination']} with flow {var.varValue} has value {edge['value']} and cost {edge['cost']}"
                        )

    # Output results
    total_value = 0
    total_cost = 0
    counted_destinations = set()
    city_active_prods = {city: [] for city in data["cities"]}
    active_transit_nodes = []

    for edge in data["edges"]:
        edge_var = edge_vars[(edge["source"], edge["destination"])]
        flow_value = pulp.value(edge_var)
        if flow_value is None and doFull:
            print(f"edge_var: {edge_var} is NONE")
            continue
        elif flow_value > 0:
            total_value += edge["value"] * flow_value
            destination_node = edge["destination"]
            if destination_node not in counted_destinations:
                total_cost += edge["cost"]
                counted_destinations.add(destination_node)
            if edge["destination"].startswith("transit_"):
                active_transit_nodes.append(edge["destination"])
            if edge["destination"].startswith("active_prod") and not edge["source"].startswith(
                "transit_"
            ):
                city_id = int(edge["source"].split("_")[1])
                city_active_prods[city_id].append(edge["destination"])
            if doFull:
                print(
                    f"Edge from {edge['source']} to {edge['destination']} with flow {flow_value} has value {edge['value']} and cost {edge['cost']}"
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
