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

With the most recent model and budget of 30 optimal is found in 1039 seconds and
proven in 1104 seconds.

```sh
Problem MODEL has 5761 rows, 9907 columns and 27863 elements
Coin0008I MODEL read with 0 errors
Option for timeMode changed from cpu to elapsed
Continuous objective value is 1.1202e+08 - 0.03 seconds
Cgl0002I 212 variables fixed
Cgl0003I 0 fixed, 0 tightened bounds, 2 strengthened rows, 0 substitutions
Cgl0004I processed model has 2615 rows, 6036 columns (6036 integer (1880 of which binary)) and 19323 elements
```
```sh
Cbc0001I Search completed - best objective -58540725, took 3849896 iterations and 32956 nodes (1104.65 seconds)
Cbc0032I Strong branching done 24150 times (196926 iterations), fathomed 326 nodes and fixed 439 variables
Cbc0035I Maximum depth 98, 3407148 variables fixed on reduced cost
Cuts at root node changed objective from -1.11312e+08 to -7.16507e+07
Probing was tried 37760 times and created 483273 cuts of which 214 were active after adding rounds of cuts (32.829 seconds)
Gomory was tried 37501 times and created 6438 cuts of which 0 were active after adding rounds of cuts (56.102 seconds)
Knapsack was tried 37503 times and created 42785 cuts of which 0 were active after adding rounds of cuts (65.913 seconds)
Clique was tried 35 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.003 seconds)
MixedIntegerRounding2 was tried 37503 times and created 45064 cuts of which 0 were active after adding rounds of cuts (25.980 seconds)
FlowCover was tried 35 times and created 22 cuts of which 0 were active after adding rounds of cuts (0.051 seconds)
TwoMirCuts was tried 37501 times and created 5374 cuts of which 0 were active after adding rounds of cuts (17.161 seconds)
ZeroHalf was tried 37503 times and created 41669 cuts of which 0 were active after adding rounds of cuts (185.402 seconds)
ImplicationCuts was tried 937 times and created 2684 cuts of which 0 were active after adding rounds of cuts (0.102 seconds)
```


## WSL -> Python -> pulp -> HiGHS

Still doesn't scale. Substantially faster than cbc.
Perhaps when the parallel solver makes it into HiGHS it would be 'fast enough'.

With the most recent model and budget of 30 optimal is found in 84 seconds and
proven in 304 seconds.

```sh
Number of BV entries in BOUNDS section is 3458
MIP  MaximizeEmpireValue-pulp has 5761 rows; 9907 cols; 27863 nonzeros; 9907 integer variables
Coefficient ranges:
  Matrix [1e+00, 3e+02]
  Cost   [1e+05, 9e+06]
  Bound  [1e+00, 3e+02]
  RHS    [1e+00, 1e+00]
Presolving model
3667 rows, 7757 cols, 22319 nonzeros  0s
2692 rows, 6839 cols, 20069 nonzeros  0s
2528 rows, 5849 cols, 17947 nonzeros  0s

Solving MIP model with:
   2528 rows
   5849 cols (1838 binary, 4011 integer, 0 implied int., 0 continuous)
   17947 nonzeros
```
```sh
  Nodes             2459
  LP iterations     1047798 (total)
                    309584 (strong br.)
                    49888 (separation)
                    312332 (heuristics)
```


## TODO: Try hybrid approach as Steiner Forest

Solving as a Steiner Forest on instances with a small budget can be done in
under a 1s using `scip-jack` but that is because the small budget naturally
constrains the solution to a single tree whereas larger budgets need the forests
and need those forest to be able to be combined.

Definitely more complex but should "solve" any possible instance of the
parameters in seconds.
