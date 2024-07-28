"""MILP Worker Node Empire Problem
(See `generate_empire_data.py` for node and edge generation details.)

Goal: Maximize supply value - cost

The model is based on flow where source outbound load == sink inbound load per supply destination.

* All edges and nodes have a load per supply warehouse equal to the sum of the inbound/source loads.
* All edges and nodes loads must be <= source supply. This is more robust for result verification
  but requires more time. Using the max warehouse load as the load limit works as well.

Nodes are subject to:
  - inbound == outbound
  - single outbound edge per load
Edges are subject to:
  - edge and reverse edge load mutual exclusivity

WARNING: Solver integer tolerance is extremely important in this problem. Solvers _will_ use
the tolerance to eliminate node costs which are calculated using the binary load indicators.
With HiGHS the default of 1e-6 is not tight enough, using `mip_feasibility_tolerance=1e-9`
results in proper binary assignments but takes much longer to solve.
"""

from typing import List

import pulp

from print_utils import print_empire_solver_output
from generate_empire_data import (
    NodeType,
    Node,
    Edge,
    generate_empire_data,
    get_reference_data,
    GraphData,
)


def create_load_hasload_vars(prob: pulp.LpProblem, obj: Node | Edge, load_vars: List[Node]):
    assert load_vars, "Each node must have at least one warehouse load variable."

    load_var = pulp.LpVariable(f"Load_{obj.name()}", lowBound=0, upBound=obj.capacity, cat="Integer")
    obj.pulp_vars["Load"] = load_var
    prob += load_var == pulp.lpSum(load_vars), f"TotalLoad_{obj.name()}"

    has_load_var = pulp.LpVariable(f"HasLoad_{obj.name()}", cat="Binary")
    obj.pulp_vars["HasLoad"] = has_load_var

    # obj_is_node = isinstance(obj, Node)
    # test_type = obj.type if obj_is_node else obj.destination.type
    # if test_type in [NodeType.waypoint, NodeType.town]:
    #     # Linking constraint to ensure TotalLoad is either 0 or within [1, capacity]
    #     prob += load_var >= has_load_var, f"ClampLoadMin_{obj.name()}"
    # elif test_type is NodeType.lodging:
    #     # Linking constraint to ensure TotalLoad is either 0 or within [min_capacity, capacity]
    #     min_capacity = obj.min_capacity if obj_is_node else obj.destination.min_capacity
    #     prob += load_var >= has_load_var * min_capacity, f"ClampLoadMin_{obj.name()}"

    # Ensure has_load_var is 1 if load_var is non-zero and 0 if load_var is zero

    # Linking constraint to ensure TotalLoad is less than capacity.
    # prob += load_var <= obj.capacity * has_load_var, f"ClampLoadMax_{obj.name()}"

    # This works fine with primal_feasibility_tolerance=1e-9
    # prob += load_var <= 1e-5 + (obj.capacity * has_load_var), f"ZeroOrPositiveLoad_{obj.name()}"

    # Testing to see if 1e-6 on this _and_ the tolerance setting will work...
    # I still got fractional values doing this, the cost was correct on 30
    # prob += load_var <= 1e-6 + (obj.capacity * has_load_var), f"ZeroOrPositiveLoad_{obj.name()}"
    # prob += load_var - (obj.capacity * has_load_var) <= 0, f"ForceZeroLoad_{obj.name()}"

    # Ensure consistency between TotalLoad and HasLoad
    # prob += has_load_var <= load_var, f"LinkLoadHasLoad_{obj.name()}"

    # This (with eps=0.0001) eliminated fractionals on max_cost=30.
    # perhaps tie this to mip_feasability_tol if max_cost=100 reintroduces fractionals?
    # 1e-5-> no fractionals.  50: 8m12s, 100: 37m both with phantom cycles but optimum results.
    eps = 1e-5
    M = obj.capacity + eps
    prob += load_var >= 0 + eps - M * (1 - has_load_var), f"ZeroOrPositiveLoad_{obj.name()}"
    prob += load_var <= 0 + M * has_load_var, f"ForceZeroLoad_{obj.name()}"


def create_node_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """has_load and load vars for nodes.
    `has_load`: binary indicator of a load.
    `LoadForWarehouse_{warehouse}`: integer value of a destination bound load.
    """
    for node in graph_data["nodes"].values():
        warehouse_load_vars = []
        for LoadForWarehouse in node.LoadForWarehouse:
            load_key = f"LoadForWarehouse_{LoadForWarehouse.id}"
            load_var = pulp.LpVariable(
                f"{load_key}_at_{node.name()}",
                lowBound=0,
                upBound=node.capacity
                if node.type in [NodeType.supply, NodeType.origin, NodeType.lodging]
                else LoadForWarehouse.capacity,
                cat="Integer",
            )
            node.pulp_vars[load_key] = load_var
            warehouse_load_vars.append(load_var)

        create_load_hasload_vars(prob, node, warehouse_load_vars)


def create_edge_vars(prob: pulp.LpProblem, graph_data: GraphData, max_cost: int):
    """has_load and load vars for edges.
    `has_load` is a binary indicator of a load.
    `LoadForWarehouse_{warehouse}`: integer value of a destination bound load.

    Edges can not transport a load the source doesn't send or a load the destination can't recieve.
    LoadForWarehouse is an intersection of the source and destination LoadForWarehouse lists.
    """

    for edge in graph_data["edges"].values():
        warehouse_load_vars = []
        for LoadForWarehouse in set(edge.source.LoadForWarehouse).intersection(
            set(edge.destination.LoadForWarehouse)
        ):
            load_key = f"LoadForWarehouse_{LoadForWarehouse.id}"
            load_var = pulp.LpVariable(
                f"{load_key}_on_{edge.name()}",
                lowBound=0,
                upBound=edge.capacity,
                cat="Integer",
            )
            edge.pulp_vars[load_key] = load_var
            warehouse_load_vars.append(load_var)

        create_load_hasload_vars(prob, edge, warehouse_load_vars)


def link_node_and_edge_vars(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure load per warehouse flow conservation for all nodes.
    Link the inbound edges load sum to the node load to the outbound edges load sum.
    Special handling for source and sink nodes.
    """

    def get_edge_vars(node: Node, node_var_key: str, edges: List[Edge]) -> List[pulp.LpVariable]:
        """Get the relevant edge variables for a given node and key."""
        is_source = node.type is NodeType.source
        edge_vars = []
        for edge in edges:
            for key, var in edge.pulp_vars.items():
                if key.startswith("LoadForWarehouse_") and (
                    key == node_var_key or node.type is NodeType.lodging
                ):
                    edge_vars.append(edge.pulp_vars["HasLoad"] if is_source else var)
        return edge_vars

    def link_node_to_edges(prob: pulp.LpProblem, node: Node, direction: str, edges: List[Edge]):
        """Link a node's load variable to the sum of its edge variables."""
        for key, var in node.pulp_vars.items():
            if key.startswith("LoadForWarehouse_"):
                edge_vars = get_edge_vars(node, key, edges)
                prob += var == pulp.lpSum(edge_vars), f"Link{direction}_{key}_at_{node.name()}"

    for node in graph_data["nodes"].values():
        if node.type is not NodeType.source:
            link_node_to_edges(prob, node, "Inbound", node.inbound_edges)
        if node.type is not NodeType.sink:
            link_node_to_edges(prob, node, "Outbound", node.outbound_edges)


def create_value_var(prob: pulp.LpProblem, graph_data: GraphData):
    """Sum variable of value at supply nodes with load."""
    value_var = pulp.LpVariable("value", lowBound=0)
    prob += (
        value_var
        == pulp.lpSum(
            node.value * node.pulp_vars["HasLoad"]
            for node in graph_data["nodes"].values()
            if node.type is NodeType.supply
        ),
        "TotalValue",
    )
    return value_var


def create_cost_var(prob: pulp.LpProblem, graph_data: GraphData, max_cost: int):
    """Sum variable of cost at nodes with load."""
    cost_var = pulp.LpVariable("cost", lowBound=0, upBound=max_cost)
    prob += (
        cost_var
        == pulp.lpSum(
            node.cost * node.pulp_vars["HasLoad"]
            for node in graph_data["nodes"].values()
            if node.type is not NodeType.source and node.type is not NodeType.sink
        ),
        "TotalCost",
    )
    return cost_var


def add_source_to_warehouse_constraints(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure each warehouse receives its total origin supply."""
    for warehouse in graph_data["warehouse_nodes"].values():
        source_load = graph_data["nodes"]["source"].pulp_vars[f"LoadForWarehouse_{warehouse.id}"]
        prob += (
            warehouse.pulp_vars["Load"] == source_load,
            f"BalancesupplyLoadForWarehouse_{warehouse.id}",
        )


def add_waypoint_clamp_by_source_constraints(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure each waypoint/town node's and edge's LoadForWarehouse is <= its' source load."""

    for node in graph_data["nodes"].values():
        if node.type in [NodeType.waypoint, NodeType.town]:
            for var_key, var in node.pulp_vars.items():
                if var_key.startswith("LoadForWarehouse_"):
                    source_load = graph_data["nodes"]["source"].pulp_vars[var_key]
                    var_name = f"ClampBySource{var_key}_at_{node.name()}"
                    prob += var <= source_load, var_name

    for edge in graph_data["edges"].values():
        if edge.destination.type in [NodeType.waypoint, NodeType.town]:
            for var_key, var in edge.pulp_vars.items():
                if var_key.startswith("LoadForWarehouse_"):
                    source_load = graph_data["nodes"]["source"].pulp_vars[var_key]
                    var_name = (f"ClampBySource{var_key}_on_{edge.name()}",)
                    prob += var <= source_load, var_name


def add_warehouse_to_lodging_constraints(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure a single warehouse -> lodging -> sink path for each warehouse."""
    for warehouse in graph_data["warehouse_nodes"].values():
        prob += pulp.lpSum(edge.pulp_vars["HasLoad"] for edge in warehouse.outbound_edges) <= 1


def add_source_to_sink_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure source outbound load equals sink inbound load."""
    prob += (
        graph_data["nodes"]["source"].pulp_vars["Load"]
        == graph_data["nodes"]["sink"].pulp_vars["Load"],
        "BalanceSourceSink",
    )


def add_reverse_edge_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure reverse edges on waypoint/town nodes don't return loads."""

    # Only waypoint/town edges have reverse edges.
    def is_transit_edge(edge):
        return (edge.source.type in [NodeType.waypoint, NodeType.town]) and (
            edge.destination.type in [NodeType.waypoint, NodeType.town]
        )

    seen_edges = []
    for edge in graph_data["edges"].values():
        if not is_transit_edge(edge):
            continue

        reverse_edge = graph_data["edges"][(edge.destination.name(), edge.source.name())]
        if edge.key in seen_edges or reverse_edge.key in seen_edges:
            continue
        seen_edges.append(edge.key)
        seen_edges.append(reverse_edge.key)

        shared_keys = list(set(edge.pulp_vars.keys()).intersection(reverse_edge.pulp_vars.keys()))
        for var_key in shared_keys:
            if var_key.startswith("LoadForWarehouse"):
                # Forward load indicator.
                b_forward = pulp.LpVariable(f"b{var_key}_on_{edge.name()}", cat="Binary")
                edge.pulp_vars[f"b{var_key}"] = b_forward
                prob += (
                    edge.pulp_vars[var_key] <= edge.destination.capacity * b_forward,
                    f"Linkb{var_key}_on_{edge.name()}",
                )

                # Reverse load indicator.
                b_reverse = pulp.LpVariable(f"b{var_key}_on_{reverse_edge.name()}", cat="Binary")
                reverse_edge.pulp_vars[f"b{var_key}"] = b_reverse
                prob += (
                    reverse_edge.pulp_vars[var_key] <= edge.source.capacity * b_reverse,
                    f"Linkb{var_key}_on_{reverse_edge.name()}",
                )

                # Mutual exclusion.
                prob += (
                    b_forward + b_reverse <= 1,
                    f"MutualExclusion{var_key}_{edge.name()}_{reverse_edge.name()}",
                )


def add_waypoint_single_outbound_constraint(prob: pulp.LpProblem, graph_data: GraphData):
    """Ensure transit nodes do not split a load to multiple outbound edges.
    NOTE: do not call this prior to calling add_reverse_edge_constraint() because
    transit edges have the binary indicators added in add_reverse_edge_constraint().
    The binary indicators are in the edge's pulp_vars keyed with `bLoadForWarehouse_{}`
    TODO: add binary indicators during edge creation?
    """

    def is_transit_edge(edge):
        return (edge.source.type in [NodeType.waypoint, NodeType.town]) and (
            edge.destination.type in [NodeType.waypoint, NodeType.town]
        )

    for node in graph_data["nodes"].values():
        if node.type not in [NodeType.waypoint, NodeType.town]:
            continue

        outbound_indicator_vars = {}
        for edge in node.outbound_edges:
            if is_transit_edge(edge):
                for key, var in edge.pulp_vars.items():
                    if key.startswith("bLoadForWarehouse_"):
                        if key not in outbound_indicator_vars:
                            outbound_indicator_vars[key] = []
                        outbound_indicator_vars[key].append(var)

        for vars in outbound_indicator_vars.values():
            if len(vars) > 1:
                prob += pulp.lpSum(vars) <= 1


def create_problem(graph_data, max_cost):
    """Create the problem and add the varaibles and constraints."""
    prob = pulp.LpProblem("MaximizeEmpireValue", pulp.LpMaximize)

    create_node_vars(prob, graph_data)
    create_edge_vars(prob, graph_data, max_cost)
    link_node_and_edge_vars(prob, graph_data)

    total_value_var = create_value_var(prob, graph_data)
    total_cost_var = create_cost_var(prob, graph_data, max_cost)

    # prob += total_value_var, "ObjectiveFunction"
    prob += total_value_var - total_cost_var, "ObjectiveFunction"
    prob += total_cost_var <= max_cost, "MaxCost"

    add_source_to_warehouse_constraints(prob, graph_data)
    add_warehouse_to_lodging_constraints(prob, graph_data)
    add_source_to_sink_constraint(prob, graph_data)

    # NOTE: Test runtimes with these constraints...
    # Right now the solver is coming up with correct solutions but with phantom loads (unnecessary
    # loads that don't contribute to the optimal solution) being added in the transit nodes where
    # they are allowed because there isn't a constraint to stop inter-waypoint cycles.
    # While these constraints would avoid direct A <-> B cycles and make the outbound routes more
    # specific, they still wouldn't stop A -> B -> C -> A cycles. The only thing that would stop all
    # of these types of cycles is linking the source load as the load capacity on waypoint nodes,
    # which is noted below.
    # When run to completion (aka low gap), these cycles only seem to exist on nodes that already
    # have a load incurring a node cost, so the only reason to use these would seem to be if they
    # improve performance.

    # Results with default "mip_feasibility_tolerance". (Solver 'd')
    # Without both: 10 => 7s, 30 => <6m, 50 => >11m (fractionals on 30 & 50, wrong cost on 50)
    #    With both: 10 => 11s, 30 => 3m, 50 => 18m (fractionals and wrong cost on 30 & 50)

    # Results with "mip_feasibility_tolerance=1e-9" to stop fractional result values. (Solver 'c')
    # Without both: 10 => 48s, 30 => <6m, 50 => >25m, 100 => 52m23s (phantom loads on 30, 50, 100)
    #    With both: 10 => 89s, 30 => <9m, 50 => >18m, 100 => 1h13m (no phantom loads)

    # Uncomment the following lines to add constraints aimed at preventing inter-waypoint cycles:
    # add_reverse_edge_constraint(prob, graph_data)  # stop phantom loads
    # add_waypoint_single_outbound_constraint(prob, graph_data)  # stop phantom loads

    # Don't use this as it adds complexity the solver doesn't like.
    # The source load is calculated as the sum of its outbound loads per warehouse.
    # Logically, it makes sense to use that as an upper bound constraint for waypoint
    # LoadForWarehouse_ values, but then every time a supply node is enabled/disabled,
    # all of the constraints for waypoint nodes that can carry a load for that warehouse
    # change as well. I believe that makes the problem more stochastic than linear.

    # NOTE: *** Tests with same as above along with waypoint clamp below. ***
    # Results with default "mip_feasibility_tolerance".
    # Without both: skipped testing this
    #    With both: 10 => 9s, 30 => >8m, 50 => <24m (massive fractionals and wrong cost on 50 of 88)

    # Results with "mip_feasibility_tolerance=1e-9" to stop fractional result values. (Solver 'c')
    # Without both: skipped testing this
    #    With both: 10 => <6s, 30 => <5m, 50 => 30m, 100 => 1h13m (no phantom loads)

    # For the 200 max_cost the comparative result between with and without clamp by source @ 6h27m.
    #           Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work
    #        Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time
    #   w   106594    3679     42977  75.01%   225901133.5852  222071941.9115     1.72%     3140   1045  10168    50610k 23215.3s
    #   w/o  83054    8794     30344  78.49%   229797003.0633  222142404.6765     3.45%     4611   1678   9882    27612k 23215.4s
    #
    # The w/o option stopped at 11h10m hours:
    #       148626   15771     57904  81.81%   229059269.8077  222142404.6765     3.11%     5592   1613   9639    48973k 40236.1s

    # Uncomment the following line to add constraints clamping waypoint loads by their source load:
    # add_waypoint_clamp_by_source_constraints(prob, graph_data)

    return prob


def main(max_cost=100, lodging_bonus=1, top_n=2, nearest_n=4, waypoint_capacity=15):
    """
    top_n: count of supply nodes per warehouse by value
    nearest_n: count of nearest warehouses available on waypoint nodes
    waypoint_capacity: max loads on a waypoint
    """
    graph_data: GraphData = generate_empire_data(lodging_bonus, top_n, nearest_n, waypoint_capacity)
    prob = create_problem(graph_data, max_cost)
    prob.writeLP(f"last_{max_cost}.lp")

    mips_tol = 1e-6
    solver = pulp.HiGHS_CMD(
        path="/home/thell/.local/bin/highs",
        keepFiles=True,
        threads=16,
        logPath=f"/home/thell/milp-flow/highs_log_{max_cost}.txt",
        options=[
            "parallel=on",
            "threads=16",
            # "mip_heuristic_effort=1",
            # testing the node/edge load indicator tolerance adjustment... put back to 1e-9 if needed
            f"mip_feasibility_tolerance={mips_tol}",
            f"primal_feasibility_tolerance={mips_tol}",
            "mip_improving_solution_save=on",
            f"mip_improving_solution_file=highs_improved_solution_{max_cost}_c.sol",
            # f"solution_file=highs_solution_{max_cost}_c.sol",
        ],
    )

    print()
    print(
        f"=== Solving: max_cost: {max_cost}, Lodging_bonus: {lodging_bonus} top_n: {top_n}, nearest_n: {nearest_n}, capacity={waypoint_capacity}"
    )
    print(f"Using threads=16 and mip_feasibility_tolerance={mips_tol}")
    print()

    solver.solve(prob)
    ref_data = get_reference_data(lodging_bonus, top_n)
    print_empire_solver_output(prob, graph_data, ref_data, max_cost, top_n, detailed=True)

    # solver = pulp.HiGHS_CMD(
    #     path="/home/thell/.local/bin/highs",
    #     keepFiles=True,
    #     threads=16,
    #     logPath=f"/home/thell/milp-flow/highs_log_{max_cost}.txt",
    #     options=[
    #         "parallel=on",
    #         "threads=16",
    #         # "mip_heuristic_effort=1",
    #         "mip_feasibility_tolerance=1e-9",
    #         "mip_improving_solution_save=on",
    #         f"mip_improving_solution_file=highs_improved_solution_{max_cost}_c.sol",
    #     ],
    # )
    # solver.solve(prob)
    # ref_data = get_reference_data(lodging_bonus, top_n)
    # print_empire_solver_output(prob, graph_data, ref_data, max_cost, top_n, detailed=True)

    # # This solution resulted in taking 11 minutes with a cost of 44 vs 40 max limit.
    # solver = pulp.HiGHS_CMD(
    #     path="/home/thell/.local/bin/highs",
    #     keepFiles=True,
    #     threads=16,
    #     logPath=f"/home/thell/milp-flow/highs_log_{max_cost}.txt",
    #     options=[
    #         "parallel=on",
    #         "threads=16",
    #         "mip_heuristic_effort=1",
    #         "mip_feasibility_tolerance=1e-5",
    #         "mip_improving_solution_save=on",
    #         f"mip_improving_solution_file=highs_improved_solution_{max_cost}_a.sol",
    #     ],
    # )
    # solver.solve(prob)
    # ref_data = get_reference_data(lodging_bonus, top_n)
    # print_empire_solver_output(prob, graph_data, ref_data, max_cost, top_n, detailed=True)

    # Interestingly this setup resulted in the solver taking 37 minutes and having a massive amount
    # of fractional values to eliminate cost. 88 actual cost vs 40 max limit.
    # solver = pulp.HiGHS_CMD(
    #     path="/home/thell/.local/bin/highs",
    #     keepFiles=True,
    #     threads=16,
    #     logPath=f"/home/thell/milp-flow/highs_log_{max_cost}.txt",
    #     options=[
    #         "parallel=on",
    #         "threads=16",
    #         "mip_heuristic_effort=1",
    #         "mip_feasibility_tolerance=1e-7",
    #         "mip_improving_solution_save=on",
    #         f"mip_improving_solution_file=highs_improved_solution_{max_cost}_b.sol",
    #     ],
    # )
    # solver.solve(prob)
    # ref_data = get_reference_data(lodging_bonus, top_n)
    # print_empire_solver_output(prob, graph_data, ref_data, max_cost, top_n, detailed=True)

    # prob.solve() # use cbc instead of HiGHS
    # ref_data = get_reference_data(lodging_bonus, top_n)
    # print_empire_solver_output(prob, graph_data, ref_data, max_cost, top_n, detailed=True)

    assert prob.variablesDict()["Load_source"].value() == prob.variablesDict()["Load_sink"].value()


if __name__ == "__main__":
    main()
