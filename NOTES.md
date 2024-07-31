# Notes

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
  * Player pays enough points to unlock something attached to the town.
  * Player pays enough points to unlock something attached to the town or the
    previously unlocked thing.
* When a player has unlocked a production node they assign a worker from a town.
* The worker gets materials from the production node and puts them in storage.
* The town to node distance determines how much a worker gets (value).

We can think of it in quite a few ways. Team Orienteering Problem, Prize
Collecting Problem, Multi-Depot Vehicle Routing Problem and so on. The problem
constraints get difficult to describe as a MILP problem and are timely to
calculate when calculated and validated iteratively.


## Alternative Representation

If we flip the concept of the node empire on its head and think of the problem
as a maximum flow problem I think it might be feasible to obtain maximum value
solutions while still following the rules of a Node Empire.

By 'flipping' I mean to consider the Node Empire like this...

* Source node - with a supply equal to the number of possible empire workers.
* Supply nodes - nodes representing each production node.
* Supply Valued nodes - nodes representing each production node to each town.
* Chosen Supply nodes - nodes representing each production node.
* Unused Supply nodes - representing unassigned production nodes.
* Network nodes (waypoints) - nodes representing the connection network.
* Demand Aggregation nodes (Town Nodes) - nodes that are part of the Network nodes and have arcs
  leading out of the connection network to Demand Nodes.
* Demand nodes - representing the workers of each town.
* Unused Demand nodes - representing unassigned production nodes.
* Sink node.

In essence
1. Production nodes are enabled.
2. Production node's commodities are assigned to a town or unused.
3. Production node is enabled in the connection network.
4. Waypoints are enabled.
5. Commodities arrive at network exit nodes (Town nodes).
6. A worker collects the commodity.

I believe that by doing this we can create constraints that will properly model
the Node Empire by following the connection rules and maximizing the value.

Constraints:

- Supply nodes: Impose a single capatown limit for production nodes.
- Supply Valued nodes: Impose a single town assignment for the commodities of a
production node.
- Chosen Supply nodes: Impose that the count of enabled production nodes
is <= number of town workers for each town and the total number of enabled
production nodes + unused nodes == total production node count. (For debugging,
this ensures our flow is correct up to the entry into the Network Nodes.)
- Demand Nodes: Impose that the count of enabled Demand Nodes (town workers) ==
the count of enabled Supply Valued nodes for the supplier (Town) and the total
enabled workers from all cities + unused demand == total production node count.
(For debugging, this ensures our flow is correct up to the Network Nodes exit.)
- Sink: Impose the sink count == the supply count at source.


## Data Prep

In order to facilitate any representation of a Node Empire we'll need some basic
data.


### Optimized Node Values Per Town

- Towns that can assign workers: use the keys from distances_tk2pzk.json.
- Production nodes: use the keys in plantzone_drops.json.
- Production node drops: plantzone_drops.json is the count of drops per cycle.
* Production node workload: use default workloads from plantzone_drops.json.
- Distance: distances_tk2pzk.json is the worker travel distance from each town
to each production node.
- Workers: worker_static.json and skills.json provide the worker stats.
- Item pricing: market.json is a recent static snapshot of item values.

To calculate an optimum node value per town each production node needs a value
and since each node can potentially supply one or more different commodities the
node values must be calculated using the node's item values and the worker stats.
The optimum worker stats are determined by the combination of stats that yield
the highest value for the node from a given town.

