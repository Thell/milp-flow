import logging
import pulp

from generate_empire_data import NodeType

logger = logging.getLogger(__name__)


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
