# CONSTRUCTION ZONE

Current testing workflow:

- edit `market.json`
- run `generate_optimized_node_values.py`
- run `empire_solver.py`
- run `generate_workerman_json.py` against new output in `highs_output/solution-vars`
- import the newly generated output from `workerman_output` into workerman


----

## Goal: Solve Minimum Cost Multi-Commodity Flow Problem

Attempting to come up with a reasonably efficient solution.

## Python -> pulp -> (coin) cbc

Doesn't scale.

Using a max cost limit of 40 takes just over 10 minutes.


## WSL -> Python -> pulp -> HiGHS

Still shouldn't scale.

Parallel doesn't kick in for the problem even with parallel and threads flags.

Using a max cost limit of 40 takes just under 4 minutes but bumping the cost
up to 50 requires almost 16 minutes.

## TODO: Try hybrid LP

Should scale better.

More complex.