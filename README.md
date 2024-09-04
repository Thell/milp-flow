# CONSTRUCTION ZONE

Current testing workflow:

- edit `market.json`
- run `generate_optimized_node_values.py`
- run `empire_solver.py`
- run `generate_workerman_json.py` against new output in `highs_output/solution-vars`
- import the newly generated output from `workerman_output` into workerman


----

## Goal: Solve Multi-Commodity Flow Problem

Attempting to come up with a reasonably efficient solution.

## Python -> pulp -> (coin) cbc

Doesn't scale.

Using a max cost limit of 40 takes just over 10 minutes.


## WSL -> Python -> pulp -> HiGHS

Still doesn't scale. Perhaps when the parallel solver makes it into HiGHS?

Using a max cost limit of 40 takes just under 4 minutes but bumping the cost
up to 50 requires about 15 minutes.

## TODO: Try hybrid approach as Steiner Forest

Solving as a Steiner Forest on instances with a small budget can be done in
under a 1s using `scip-jack` but that is because the small budget naturally
constrains the solution to a single tree whereas larger budgets need the forests
and need those forest to be able to be combined.

Definitely more complex but should "solve" any possible instance of the
parameters in seconds.
