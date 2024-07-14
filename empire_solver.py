"""MILP Worker Node Empire Problem
(See `generate_empire_data.py` for node and edge generation details.)

Goal: Maximize demand value - cost

* The model is based on flow conservation where the source outbound load == sink inbound load.

* All nodes except source have the same "base" inbound constraints.
* All nodes except sink have the same "base" outbound constraints.

* All edges share the same "base" inbound and outbound constraints.
* All edges have a reverse edge.

* All edges and nodes have a capacity which the calculated load must not exceed.

ref_data dict structure is:
{
  "nodes": [{"id": str, "capacity": int, "cost": int, "value": float},...],
  "edges": [{"source": str, "destination": str, "capacity": int}, ...],
  "warehouses": [str, ...],
  "plantzones": [str, ...]
}

where:
  - all numeric values are >= 0
  - all node "id" values except "source" and "sink" are prefixed with
    demand_, origin_, waypoint_, town_, warehouse_ or lodging_
  - edge "source" and "destination" are node "id"s.
  - warehouses and plantzones are str id numbers for the respective nodes.

Solver edge_var names should signify "{source}_to_{destination}"
"""

import hashlib
import json
from typing import Dict, Any

import pulp

from file_utils import read_empire_json
from print_utils import print_empire_solver_output

M = 1e6


def create_load_vars(ref_data: Dict[str, Any]):
    """Integer variables to represent the dynamic load for each node and edge."""
    load_vars = {}
    for node in ref_data["nodes"]:
        key = node["id"]
        load_vars[key] = pulp.LpVariable(f"load_{key}", lowBound=0, cat="Integer")

    for edge in ref_data["edges"]:
        key = f"{edge['source']}_to_{edge['destination']}"
        load_vars[key] = pulp.LpVariable(f"load_{key}", lowBound=0, cat="Integer")

    return load_vars


def create_has_load_vars(ref_data: Dict[str, Any], load_vars):
    """Binary variables to represent the load condition for each node and edge."""
    has_load_vars = {}
    for node in ref_data["nodes"]:
        key = node["id"]
        has_load_vars[key] = pulp.LpVariable(f"has_load_{key}", cat=pulp.LpBinary)

    for edge in ref_data["edges"]:
        key = f"{edge['source']}_to_{edge['destination']}"
        has_load_vars[key] = pulp.LpVariable(f"has_load_{key}", cat=pulp.LpBinary)

    return has_load_vars


def create_warehouse_demand_vars(ref_data: Dict[str, Any], load_vars):
    """Sum variable of demand load at each warehouse."""
    warehouse_demand_vars = {}
    for warehouse in ref_data["warehouses"]:
        plantzones_for_warehouse = []
        for plantzone in ref_data["plantzones"]:
            plantzones_for_warehouse.append(f"demand_plantzone_{plantzone}_at_warehouse_{warehouse}")

        warehouse_demand_vars[warehouse] = pulp.lpSum(
            load_vars[plantzone_node] for plantzone_node in plantzones_for_warehouse
        )

    return warehouse_demand_vars


def create_demand_value_var(prob, ref_data: Dict[str, Any], has_load_vars):
    """Sum variable of value at demand nodes with load."""
    demand_value_var = pulp.LpVariable("demand_value", lowBound=0)

    prob += (
        demand_value_var
        == pulp.lpSum(
            node["value"] * has_load_vars[node["id"]]
            for node in ref_data["nodes"]
            if node["id"].startswith("demand") and node["value"] > 0
        ),
        "Total_demand_value_constraint",
    )

    return demand_value_var


def create_cost_var(prob, ref_data: Dict[str, Any], has_load_vars):
    """Sum variable of cost at nodes with load."""
    cost_var = pulp.LpVariable("cost", lowBound=0)  # Create a PuLP variable

    prob += (
        cost_var
        == pulp.lpSum(
            node["cost"] * has_load_vars[node["id"]]
            for node in ref_data["nodes"]
            if node["cost"] > 0
        ),
        "Total_cost_constraint",
    )

    return cost_var


def add_objective_function(prob, demand_value_var, cost_var):
    """Maximize demand value and minimize cost."""
    prob += demand_value_var - cost_var, "ObjectiveFunction"


def add_cost_constraint(prob, cost_var, max_cost):
    """Ensure total cost <= max_cost."""
    prob += cost_var <= max_cost, "Max_cost_constraint"


def add_demand_warehouse_balance_constraints(prob, ref_data, warehouse_demand_vars, load_vars):
    """Ensure demand is equal to load for each warehouse."""
    for warehouse_id in ref_data["warehouses"]:
        warehouse = f"warehouse_{warehouse_id}"
        prob += (
            warehouse_demand_vars[warehouse_id] == load_vars[warehouse],
            f"Demand_load_balance_{warehouse}",
        )


def add_base_constraints(prob, ref_data, load_var, has_load_var):
    """Add load, capacity and activation constraints to nodes and edges."""

    def constrain(prob, key, capacity):
        # Ensure load does not exceed capacity.
        prob += load_var[key] <= capacity, f"Capacity_{key}"
        # Ensure load activates has_load.
        prob += load_var[key] <= has_load_var[key] * M, f"Load_activation_{key}"
        # Ensure if activated, it must have a load.
        prob += has_load_var[key] <= load_var[key], f"Load_has_load_link_{key}"

    # Constrain all nodes.
    for node in ref_data["nodes"]:
        key = node["id"]
        capacity = node["capacity"]
        constrain(prob, key, capacity)

    # Constrain all edges.
    for edge in ref_data["edges"]:
        key = f"{edge['source']}_to_{edge['destination']}"
        capacity = edge["capacity"]
        constrain(prob, key, capacity)


def add_flow_constraints(prob, ref_data, load_vars):
    """Ensure flow conservation for all nodes except source and sink.
    Link the inbound edges load sum to the node load to the outbound edges load sum.
    """
    for node in ref_data["nodes"]:
        node_key = node["id"]
        if node_key not in ["source", "sink"]:
            in_edge_keys = [
                f"{edge['source']}_to_{edge['destination']}"
                for edge in ref_data["edges"]
                if edge["destination"] == node_key
            ]
            out_edge_keys = [
                f"{edge['source']}_to_{edge['destination']}"
                for edge in ref_data["edges"]
                if edge["source"] == node_key
            ]

            # Ensure the node load is equal to the sum of the inbound edge loads
            prob += (
                load_vars[node_key] == pulp.lpSum(load_vars[edge_key] for edge_key in in_edge_keys),
                f"Inbound_flow_to_load_{node_key}",
            )

            # Ensure the node load is equal to the sum of the outbound edge loads
            prob += (
                load_vars[node_key] == pulp.lpSum(load_vars[edge_key] for edge_key in out_edge_keys),
                f"Outbound_flow_from_load_{node_key}",
            )


def add_source_to_sink_constraint(prob, ref_data, load_vars):
    """Ensure source outbound load equals sink inbound load."""
    source_outbound_edges = [edge for edge in ref_data["edges"] if edge["source"] == "source"]
    sink_inbound_edges = [edge for edge in ref_data["edges"] if edge["destination"] == "sink"]

    prob += (
        pulp.lpSum(
            load_vars[f"{edge['source']}_to_{edge['destination']}"] for edge in source_outbound_edges
        )
        == pulp.lpSum(
            load_vars[f"{edge['source']}_to_{edge['destination']}"] for edge in sink_inbound_edges
        ),
        "Source_to_sink_flow_balance",
    )


def create_problem(ref_data, max_cost):
    """Create the problem and add the varaibles and constraints."""
    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    load_vars = create_load_vars(ref_data)
    has_load_vars = create_has_load_vars(ref_data, load_vars)
    warehouse_demand_vars = create_warehouse_demand_vars(ref_data, has_load_vars)
    total_demand_value_var = create_demand_value_var(prob, ref_data, has_load_vars)
    total_cost_var = create_cost_var(prob, ref_data, has_load_vars)

    add_objective_function(prob, total_demand_value_var, total_cost_var)

    add_cost_constraint(prob, total_cost_var, max_cost)
    add_demand_warehouse_balance_constraints(prob, ref_data, warehouse_demand_vars, load_vars)
    add_base_constraints(prob, ref_data, load_vars, has_load_vars)
    add_flow_constraints(prob, ref_data, load_vars)
    add_source_to_sink_constraint(prob, ref_data, load_vars)

    return prob


def compute_data_hash(ref_data):
    """Get the hash for the full ref data dict."""
    ref_data_str = json.dumps(ref_data, sort_keys=True)
    hash_object = hashlib.sha256(ref_data_str.encode())
    return hash_object.hexdigest()


def save_hash_to_file(hash_value, filename="ref_data_hash.txt"):
    with open(filename, "w") as f:
        f.write(hash_value)


def load_hash_from_file(filename="ref_data_hash.txt"):
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except:  # noqa: E722
        return 0


def main(max_cost=40):
    ref_data = read_empire_json("full_empire.json")
    current_data_hash = compute_data_hash(ref_data)
    stored_data_hash = load_hash_from_file()

    # Modify cost constraint when re-using existing ref_data.
    if current_data_hash == stored_data_hash:
        prob_vars, prob = pulp.LpProblem.from_json("MaximizeEmpireValue.json")
        cost_var = prob_vars["cost"]
        prob.constraints["Max_cost_constraint"] = cost_var <= max_cost
    else:
        prob = create_problem(ref_data, max_cost)
        prob.to_json("MaximizeEmpireValue.json")
        new_data_hash = compute_data_hash(ref_data)
        save_hash_to_file(new_data_hash)

    prob.solve()
    print_empire_solver_output(prob, ref_data, max_cost)


if __name__ == "__main__":
    main()
