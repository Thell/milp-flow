# optimize_poi.py

from importlib.resources import as_file, files

import pyoptinterface as poi
from pyoptinterface import highs

import milp_flow.data_store as ds
from milp_flow.generate_graph_data import Arc, GraphData, Node, NodeType as NT

import re

from milp_flow.optimize_par_poi import solve_par


def ensure_highs():
    try:
        for file in files("highspy").iterdir():
            if file.name.startswith("_core"):
                with as_file(file) as lib:
                    highs.load_library(lib.as_posix())
    except ModuleNotFoundError:
        print("Failed to import 'highspy' package. Ensure it's installed.")
        exit(1)


def filter_arcs(v: Node, groupflow: str, arcs: list[Arc]) -> list[Arc]:
    return [
        var
        for arc in arcs
        for key, var in arc.vars.items()
        if key.startswith("groupflow_") and (key == groupflow or v.isLodging)
    ]


def link_in_out_by_group(
    model: highs.Model, v: Node, in_arcs: list[Arc], out_arcs: list[Arc]
) -> None:
    all_inflows = []
    f = v.vars["f"]
    for group in v.groups:
        groupflow_key = f"groupflow_{group.id}"
        inflows = filter_arcs(v, groupflow_key, in_arcs)
        outflows = filter_arcs(v, groupflow_key, out_arcs)
        model.add_linear_constraint(
            poi.quicksum(inflows) - poi.quicksum(outflows),
            poi.Eq,
            0,
            f"balance_{groupflow_key}_at_{v.name()}",
        )
        all_inflows.extend(inflows)
    model.add_linear_constraint(f - poi.quicksum(all_inflows), poi.Eq, 0, f"flow_{v.name()}")
    model.add_linear_constraint(f - v.ub * v.vars["x"], poi.Leq, 0, f"x_{v.name()}")


def create_problem(config: dict, G: GraphData) -> tuple[highs.Model, dict]:
    """Create the problem and add the variables and constraints."""
    sol_vars = {}
    model = highs.Model()

    # Variables
    for v in G["V"].values():
        x = model.add_variable(lb=0, ub=1, domain=poi.VariableDomain.Binary, name=f"x_{v.name()}")
        v.vars["x"] = x
        sol_vars[f"x_{v.name()}"] = x

        f = model.add_variable(
            lb=0, ub=v.ub, domain=poi.VariableDomain.Integer, name=f"flow_{v.name()}"
        )
        v.vars["f"] = f
        sol_vars[f"flow_{v.name()}"] = f

    for arc in G["E"].values():
        for group in set(arc.source.groups).intersection(set(arc.destination.groups)):
            key = f"groupflow_{group.id}"
            ub = arc.ub if arc.source.type in [NT.group, NT.𝓢, NT.𝓣, NT.lodging] else group.ub
            domain = (
                poi.VariableDomain.Binary
                if arc.source.type in [NT.𝓢, NT.plant]
                else poi.VariableDomain.Integer
            )
            a = model.add_variable(lb=0, ub=ub, domain=domain, name=f"{key}_on_{arc.name()}")
            arc.vars[key] = a
            sol_vars[f"{key}_on_{arc.name()}"] = a

    # Objective
    model.set_objective(
        poi.quicksum(
            round(plant.group_prizes[group.id]["value"], 2) * arc.vars[f"groupflow_{group.id}"]
            for plant in G["P"].values()
            for group in plant.groups
            for arc in plant.inbound_arcs
        ),
        poi.ObjectiveSense.Maximize,
    )

    # Constraints
    model.add_linear_constraint(
        poi.quicksum(v.cost * v.vars["x"] for v in G["V"].values()),
        poi.Leq,
        config["budget"],
        name="cost",
    )

    for group in G["G"].values():
        vars = [lodge.vars["x"] for lodge in G["L"].values() if lodge.groups[0] == group]
        model.add_linear_constraint(poi.quicksum(vars), poi.Leq, 1, f"lodging_{group.id}")

    for v in G["V"].values():
        if v.type not in [NT.𝓢, NT.𝓣]:
            link_in_out_by_group(model, v, v.inbound_arcs, v.outbound_arcs)

    link_in_out_by_group(model, G["V"]["𝓣"], G["V"]["𝓣"].inbound_arcs, G["V"]["𝓢"].outbound_arcs)

    model.add_linear_constraint(G["V"]["𝓢"].vars["x"], poi.Eq, 1, "x_source")

    for node in G["V"].values():
        if node.type in [NT.S, NT.T]:
            continue

        in_neighbors = [arc.source.vars["x"] for arc in node.inbound_arcs]
        out_neighbors = [arc.destination.vars["x"] for arc in node.outbound_arcs]
        if node.isWaypoint:
            model.add_linear_constraint(poi.quicksum(in_neighbors) - 2 * node.vars["x"], poi.Geq, 0)
        else:
            model.add_linear_constraint(
                poi.quicksum(in_neighbors) + poi.quicksum(out_neighbors) - 2 * node.vars["x"],
                poi.Geq,
                0,
            )

        model.add_linear_constraint(poi.quicksum(out_neighbors) - node.vars["x"], poi.Geq, 0)

    return model, sol_vars


def load_primals(data: dict, model: highs.Model, model_vars: dict):
    primals = ds.read_primal(f"{data['config']['primal_budget']}_primal.json")

    def extract_capacity(s):
        return int(m.group(1)) if (m := re.search("_for_(\\d+)", s)) else None

    for primal_key, primal_value in primals.items():
        if (model_var := model_vars.get(primal_key, None)) is None and "lodging" in primal_key:
            if (prior_capacity := extract_capacity(primal_key)) is None:
                continue
            prefix = primal_key.split("_for_")[0] + "_for_"
            for candidate in (cand for cand in model_vars.keys() if cand.startswith(prefix)):
                candidate_capacity = extract_capacity(candidate)
                if candidate_capacity and candidate_capacity > prior_capacity:
                    model_var = model_vars[candidate]
                    break
        if model_var is not None:
            model.set_variable_attribute(model_var, poi.VariableAttribute.PrimalStart, primal_value)


def optimize_poi(data: dict, graph_data: GraphData) -> highs.Model:
    num_processes = data["config"]["solver"]["num_processes"]
    options = {k: v for k, v in data["config"]["solver"].items() if k != "num_processes"}

    print(
        f"\nSolving:  graph with {len(graph_data['V'])} nodes and {len(graph_data['E'])} arcs"
        f"\n  Using:  budget of {data["config"]["budget"]}"
        f"\n   With:  {num_processes} processes."
    )

    print("Creating mip problem...")
    model, model_vars = create_problem(data["config"], graph_data)

    if data["config"]["load_primal"]:
        print("Loading previous solution primal values...")
        load_primals(data, model, model_vars)

    print("Solving mip problem...")
    if num_processes == 1:
        print(f"Single process starting using {options}")
        if options["log_file"]:
            options["log_file"] = options["log_file"].as_posix()
        for primal_key, v in options.items():
            model.set_raw_parameter(primal_key, v)
        model.optimize()
    else:
        model = solve_par(model, options, num_processes)

    primal_vars = {}
    if data["config"]["save_primal"]:
        for primal_key, i in model_vars.copy().items():
            primal_value = round(model.get_value(i))
            if primal_value > 0:
                primal_vars[primal_key] = primal_value
        ds.write_primal(f"{data["config"]["budget"]}_primal.json", primal_vars)

    return model


def testing():
    ensure_highs()
    print(highs.is_library_loaded())


if __name__ == "__main__":
    testing()
