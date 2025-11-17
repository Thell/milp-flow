# Notes

11-16-2025

With the current state of the code the 'best' setup is most likely to use

top_n: 10, nearest_n: 10, no basin, and the threshold with budget based decay.

The next step will be implmenting a new setup approach for generating the graph
data. What I'm thinking is:

- Calculate the max used (t,r) path length when _all_ nodes are assigned for
both min capacity and max capacity modes (perhaps per root) after solving with
the solver and a high enough budget. I estimate a budget ~933 will be what's
needed, but that should be able to be confirmed using the max terminals
experiment (all prizes=1) to find the right budget and then do a full solve with
that budget. Then analyze the solution for the length metric.

- Use the length as a cutoff to create the initial transit layer per root,
instead of the nearest_n, then prune NTD1 for each layer.

- For each layer: add that layer's root prize to the payload for each terminal
and that layer's root capacity to the transit bounds of each node in the layer.

- This will create the reference graph with terminals populated for prizes that
can possibly be used instead of the top_n and transit routes that would possibly
be utilized instead of nearest_n.

The reductions and prunings would run against that data["G"] -> solver_graph.


# Older Notes
The purpose of this project is to figure out a way to optimize a Node Empire in
Black Desert Online.

The full Node Network in BDO consists of

* 24 Warehouse/Town Nodes (overlaps with Waypoints) for
  * Commodity Storage (count varies by town - max 192)
  * Worker Lodging (count varies by town)
* 917 Waypoint Nodes
* 274 Production Nodes

The sample Node Network consists of

* 2 Warehouse/Town Nodes (overlaps with Waypoints) for
  * Commodity Storage
  * Worker Lodging (5 loging in the first town and 4 in the second)
* 25 Waypoint Nodes
* 48 Production Nodes

A Node Empire typically works by

* Player earns 'contribution points' through in-game activities.
* Player spends the points to 'unlock' storage, lodging and nodes.
* Player can only unlock things in chains.
  * Cities are unlocked by default.
  * Cities have 1 worker lodging and some storage slots unlocked by default.
  * Player pays enough points to unlock something  attached to the town.
  * Player pays enough points to unlock something attached to the town or the
    previously unlocked thing.
* When a player has unlocked a production node they assign a worker from a
  connected town.
* The worker gets materials from the production node and puts them in storage.
* The town to node distance determines how much a worker gets (value).

We can think of it in quite a few ways. Team Orienteering Problem, Prize
Collecting Problem, Multi-Depot Vehicle Routing Problem and so on.


## Representation

If we flip the concept of the node empire on its head and think of the problem
as a flow problem it becomes feasible to obtain maximum value solutions while
still following the rules of a Node Empire.

By 'flipping' I mean that it would be typical to consider the towns as roots and
the worker nodes (plants) as terminals but if the plants are the roots and the
towns workers are the terminals it works like this...

* _Source_ node.
* _Roots_: Plants - nodes representing each production node for each town.
* _Connections_: Waypoints - nodes representing the connection network.
* _Connections_: Towns - part of the connection nodes but with arcs
  leading out of the connection network to Lodging Nodes.
* _Terminals_: Lodging - (the demand) representing the workers of each town.
* _Sink_ node.

In essence
1. Plant nodes are enabled.
2. Plant nodes are assigned to specific towns based on maximizing value.
4. Waypoints are enabled connecting the plant to the specific town.
5. Commodity arrives at Town node.
6. A worker collects the commodity.
7. Cost is contained to a budget.

## A Better Representation

Given an undirected node-weighted graph G transform the graph into a directed
edge weighted graph, then solve with

### Variables:

• cost ∈ ℕ₀ such that 0 ≤ cost ≤ budget

for all nodes v ∈ G[V]:  
• xᵥ ∈ {0, 1} (indicates if the node is included in the solution)  
• fᵥ ∈ ℕ₀ such that 0 ≤ fᵥ ≤ ubᵥ (total flow load at the node)

for all arcs e ∈ G[E], and for all groups g ∈ (source ∩ destination) groups:  
• fₑ,g ∈ {0, 1} if the source type is 𝓢 or plant.  
• fₑ,g ∈ ℕ₀ such that 0 ≤ fₑ,g ≤ ubₑ,g if the source type is not 𝓢 or plant.

### Objective: Maximize the total prize value:

• maximize ∑ₚ ∑ₑ vₚ,g * fₑ,g for all groups g

where:
vₚ,g is the value associated with group g in plant p
fₑ,g represents the binary flow of group g on arc e inbound to plant p

### Subject to:

Cost Constraint:

• cost = ∑ᵥ cᵥxᵥ

where cᵥ is the cost associated with node v, and xᵥ is the binary decision variable.

Lodging Exclusivity:

For each group g:

• ∑ᵥ xᵥ,g ≤ 1

where xᵥ is the binary variable indicating if the lodging node is selected,
and the sum is over all lodgings in group g.

Flow Conservation:

For all nodes v ∈ G[V]:

• ∑ₑ fₑ⁻,g = ∑ₑ fₑ⁺,g

where fₑ⁻,g represents the flow variable for group g on inbound arcs e to node v,
and fₑ⁺,g represents the flow on outbound arcs e from node v.
(Includes sink 𝓣 inbound to source 𝓢 outbound.)

## Future Direction

Setup problem as a Budgeted Prize Collecting Steiner Forest Problem with dynamic
capacity constraints and figure out how to do that while retaining the planar
graph. I believe this problem can be solved with minimal MIP using
preprocessing, approximation, primal dual and branch and cut algos for Steiner Problems.

I just have to learn how...