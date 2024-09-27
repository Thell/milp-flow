# Pruning input graph notes:

Node weighted, undirected input graph.

## Not pruned.

```
INFO:root:Starting link_graph: Graph with 820 nodes and 924 edges
INFO:root:Reduced link_graph: Graph with 820 nodes and 924 edges
Num 1-degree nodes: 401
Num 2-degree nodes: 122
INFO:root:
Solving:    graph with 915 nodes and 2010 arcs
  Using:    budget: 10, lodging_bonus: 0, top_n: 4, nearest_n: 5, capacity: 25
   With:    threads: 16, mip_feasibility_tolerance: 1e-06, gap: default
Running HiGHS 1.7.2 (git hash: 00e812dab): Copyright (c) 2024 HiGHS under MIT licence terms
Number of BV entries in BOUNDS section is 3372
MIP  MaximizeEmpireValue-pulp has 5503 rows; 9563 cols; 26917 nonzeros; 9563 integer variables
Coefficient ranges:
  Matrix [1e+00, 3e+02]
  Cost   [1e+05, 9e+06]
  Bound  [1e+00, 3e+02]
  RHS    [1e+00, 1e+00]
Presolving model
3572 rows, 7567 cols, 21654 nonzeros  0s
2584 rows, 6636 cols, 19373 nonzeros  0s
2379 rows, 5575 cols, 17155 nonzeros  0s

Solving MIP model with:
   2379 rows
   5575 cols (1743 binary, 3832 integer, 0 implied int., 0 continuous)
   17155 nonzeros

        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work      
     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   538775957.22    -inf                 inf        0      0      0         0     0.1s
         0       0         0   0.00%   45306302.07458  -inf                 inf        0      0      4       908     0.1s
 C       0       0         0   0.00%   30532456.8388   -0                 Large      278     33     61      1382     0.3s
 L       0       0         0   0.00%   25014394.85418  24954483.03        0.24%     2046    151     61      2667     2.0s
         1       0         1 100.00%   24955211.42028  24954483.03        0.00%     1903    151     61      4294     2.0s

Solving report
  Status            Optimal
  Primal bound      24954483.03
  Dual bound        24955211.4203
  Gap               0.00292% (tolerance: 0.01%)
  Solution status   feasible
                    24954483.03 (objective)
                    0 (bound viol.)
                    1.73194791842e-14 (int. viol.)
                    0 (row viol.)
  Timing            1.98 (total)
                    0.08 (presolve)
                    0.00 (postsolve)
  Nodes             1
  LP iterations     4294 (total)
                    0 (strong br.)
                    1759 (separation)
                    1602 (heuristics)
Writing the solution to MaximizeEmpireValue-pulp.sol
```

## Simple non-Plant leaf removal.

```
INFO:root:Starting link_graph: Graph with 820 nodes and 924 edges
INFO:root:Reduced link_graph: Graph with 693 nodes and 797 edges
Num 1-degree nodes: 281
Num 2-degree nodes: 136
INFO:root:
Solving:    graph with 855 nodes and 1890 arcs
  Using:    budget: 10, lodging_bonus: 0, top_n: 4, nearest_n: 5, capacity: 25
   With:    threads: 16, mip_feasibility_tolerance: 1e-06, gap: default
Running HiGHS 1.7.2 (git hash: 00e812dab): Copyright (c) 2024 HiGHS under MIT licence terms
Number of BV entries in BOUNDS section is 3325
MIP  MaximizeEmpireValue-pulp has 5083 rows; 8870 cols; 24959 nonzeros; 8870 integer variables
Coefficient ranges:
  Matrix [1e+00, 3e+02]
  Cost   [1e+05, 9e+06]
  Bound  [1e+00, 3e+02]
  RHS    [1e+00, 1e+00]
Presolving model
3423 rows, 7146 cols, 20784 nonzeros  0s
2570 rows, 6347 cols, 18927 nonzeros  0s
2414 rows, 5661 cols, 17453 nonzeros  0s

Solving MIP model with:
   2414 rows
   5661 cols (1781 binary, 3880 integer, 0 implied int., 0 continuous)
   17453 nonzeros

        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work      
     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   535936750.66    -inf                 inf        0      0      0         0     0.1s
         0       0         0   0.00%   45306302.07458  -inf                 inf        0      0      4       934     0.1s
 C       0       0         0   0.00%   30233602.30697  -0                 Large      244     33    103      1354     0.3s
 L       0       0         0   0.00%   25442209.06891  24954483.03        1.95%     1563    133    103      3129     2.2s

95.1% inactive integer columns, restarting
Model after restart has 37 rows, 61 cols (31 bin., 30 int., 0 impl., 0 cont.), and 139 nonzeros

         0       0         0   0.00%   25045139.75419  24954483.03        0.36%       16      0      0      5083     2.4s
         1       0         1 100.00%   24954483.03     24954483.03        0.00%       16      9      0      5110     2.4s

Solving report
  Status            Optimal
  Primal bound      24954483.03
  Dual bound        24954483.03
  Gap               0% (tolerance: 0.01%)
  Solution status   feasible
                    24954483.03 (objective)
                    0 (bound viol.)
                    0 (int. viol.)
                    0 (row viol.)
  Timing            2.38 (total)
                    0.08 (presolve)
                    0.00 (postsolve)
  Nodes             1
  LP iterations     5110 (total)
                    0 (strong br.)
                    2313 (separation)
                    1796 (heuristics)
Writing the solution to MaximizeEmpireValue-pulp.sol
```

## Repeated non-Plant leaf removal.

```
INFO:root:Starting link_graph: Graph with 820 nodes and 924 edges
INFO:root:Reduced link_graph: Graph with 685 nodes and 789 edges
Num 1-degree nodes: 274
Num 2-degree nodes: 136
INFO:root:
Solving:    graph with 847 nodes and 1874 arcs
  Using:    budget: 10, lodging_bonus: 0, top_n: 4, nearest_n: 5, capacity: 25
   With:    threads: 16, mip_feasibility_tolerance: 1e-06, gap: default
Running HiGHS 1.7.2 (git hash: 00e812dab): Copyright (c) 2024 HiGHS under MIT licence terms
Number of BV entries in BOUNDS section is 3311
MIP  MaximizeEmpireValue-pulp has 5027 rows; 8774 cols; 24687 nonzeros; 8774 integer variables
Coefficient ranges:
  Matrix [1e+00, 3e+02]
  Cost   [1e+05, 9e+06]
  Bound  [1e+00, 3e+02]
  RHS    [1e+00, 1e+00]
Presolving model
3401 rows, 7084 cols, 20648 nonzeros  0s
2562 rows, 6299 cols, 18831 nonzeros  0s
2412 rows, 5657 cols, 17449 nonzeros  0s

Solving MIP model with:
   2412 rows
   5657 cols (1779 binary, 3878 integer, 0 implied int., 0 continuous)
   17449 nonzeros

        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work      
     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   535936750.66    -inf                 inf        0      0      0         0     0.1s
         0       0         0   0.00%   45306302.07458  -inf                 inf        0      0      2       901     0.1s
 C       0       0         0   0.00%   30603281.02295  -0                 Large      327     44     86      1583     0.3s
 L       0       0         0   0.00%   25838900.50407  24954483.03        3.54%     1235    166     86      3709     2.4s

79.8% inactive integer columns, restarting
Model after restart has 362 rows, 788 cols (257 bin., 531 int., 0 impl., 0 cont.), and 2214 nonzeros

         0       0         0   0.00%   25813972.06102  24954483.03        3.44%       86      0      0      6787     2.9s
         0       0         0   0.00%   25795169.03876  24954483.03        3.37%       86     71      2      7152     2.9s

8.2% inactive integer columns, restarting
Model after restart has 340 rows, 713 cols (220 bin., 493 int., 0 impl., 0 cont.), and 2001 nonzeros

         0       0         0   0.00%   25767892.44455  24954483.03        3.26%       71      0      0      8465     3.0s
         0       0         0   0.00%   25767892.44455  24954483.03        3.26%       71     68      2      8711     3.1s

6.2% inactive integer columns, restarting
Model after restart has 299 rows, 636 cols (207 bin., 429 int., 0 impl., 0 cont.), and 1769 nonzeros

         0       0         0   0.00%   25728313.20228  24954483.03        3.10%       66      0      0      9569     3.2s

10.8% inactive integer columns, restarting
Model after restart has 262 rows, 563 cols (189 bin., 374 int., 0 impl., 0 cont.), and 1555 nonzeros

         0       0         0   0.00%   25728313.20228  24954483.03        3.10%       51      0      0      9785     3.2s
         0       0         0   0.00%   25728313.20228  24954483.03        3.10%       51     50      4      9969     3.2s

7.3% inactive integer columns, restarting
Model after restart has 230 rows, 487 cols (161 bin., 326 int., 0 impl., 0 cont.), and 1332 nonzeros

         0       0         0   0.00%   25717500.77774  24954483.03        3.06%       50      0      0     10577     3.3s
         0       0         0   0.00%   25717500.77774  24954483.03        3.06%       50     49      6     10740     3.3s

8.0% inactive integer columns, restarting
Model after restart has 165 rows, 366 cols (114 bin., 252 int., 0 impl., 0 cont.), and 950 nonzeros

         0       0         0   0.00%   25688939.35623  24954483.03        2.94%       44      0      0     11753     3.4s
         0       0         0   0.00%   25688939.35623  24954483.03        2.94%       44     39      6     11873     3.4s

19.1% inactive integer columns, restarting
Model after restart has 136 rows, 274 cols (96 bin., 178 int., 0 impl., 0 cont.), and 753 nonzeros

         0       0         0   0.00%   25454104.05712  24954483.03        2.00%       25      0      0     11928     3.4s

46.0% inactive integer columns, restarting
Model after restart has 69 rows, 143 cols (60 bin., 83 int., 0 impl., 0 cont.), and 403 nonzeros

         0       0         0   0.00%   25291454.01017  24954483.03        1.35%        8      0      0     11981     3.4s
         0       0         0   0.00%   25291454.01017  24954483.03        1.35%        8      8      6     12019     3.4s
         1       0         1 100.00%   24954483.03     24954483.03        0.00%       93     11     50     12046     3.4s

Solving report
  Status            Optimal
  Primal bound      24954483.03
  Dual bound        24954483.03
  Gap               0% (tolerance: 0.01%)
  Solution status   feasible
                    24954483.03 (objective)
                    0 (bound viol.)
                    0 (int. viol.)
                    0 (row viol.)
  Timing            3.41 (total)
                    0.13 (presolve)
                    0.00 (postsolve)
  Nodes             1
  LP iterations     12046 (total)
                    0 (strong br.)
                    3120 (separation)
                    6595 (heuristics)
Writing the solution to MaximizeEmpireValue-pulp.sol
```

## Remove simple 2-degree nodes with neighbors linked.

```
INFO:root:Starting link_graph: Graph with 820 nodes and 924 edges
INFO:root:Reduced link_graph: Graph with 682 nodes and 783 edges
Num 1-degree nodes: 274
Num 2-degree nodes: 135
INFO:root:
Solving:    graph with 844 nodes and 1862 arcs
  Using:    budget: 10, lodging_bonus: 0, top_n: 4, nearest_n: 5, capacity: 25
   With:    threads: 16, mip_feasibility_tolerance: 1e-06, gap: default
Running HiGHS 1.7.2 (git hash: 00e812dab): Copyright (c) 2024 HiGHS under MIT licence terms
Number of BV entries in BOUNDS section is 3308
MIP  MaximizeEmpireValue-pulp has 5006 rows; 8708 cols; 24495 nonzeros; 8708 integer variables
Coefficient ranges:
  Matrix [1e+00, 3e+02]
  Cost   [1e+05, 9e+06]
  Bound  [1e+00, 3e+02]
  RHS    [1e+00, 1e+00]
Presolving model
3380 rows, 7018 cols, 20456 nonzeros  0s
2542 rows, 6234 cols, 18637 nonzeros  0s
2391 rows, 5590 cols, 17255 nonzeros  0s

Solving MIP model with:
   2391 rows
   5590 cols (1776 binary, 3814 integer, 0 implied int., 0 continuous)
   17255 nonzeros

        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work      
     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   535936750.66    -inf                 inf        0      0      0         0     0.1s
         0       0         0   0.00%   45306302.07458  -inf                 inf        0      0      2       917     0.1s
 C       0       0         0   0.00%   30913489.36497  614933.1        4927.13%      215     29     57      1295     0.2s
 L       0       0         0   0.00%   26288199.42009  24954483.03        5.34%     1488    107     57      2437     2.9s

75.4% inactive integer columns, restarting
Model after restart has 456 rows, 982 cols (289 bin., 693 int., 0 impl., 0 cont.), and 2800 nonzeros

         0       0         0   0.00%   26252621.89821  24954483.03        5.20%       72      0      0      6042     3.3s

16.1% inactive integer columns, restarting
Model after restart has 360 rows, 754 cols (217 bin., 537 int., 0 impl., 0 cont.), and 2136 nonzeros

         0       0         0   0.00%   26249341.40835  24954483.03        5.19%       60      0      0      6443     3.3s

11.8% inactive integer columns, restarting
Model after restart has 284 rows, 601 cols (180 bin., 421 int., 0 impl., 0 cont.), and 1678 nonzeros

         0       0         0   0.00%   26249341.40835  24954483.03        5.19%       56      0      0      6700     3.3s
         0       0         0   0.00%   26242361.99399  24954483.03        5.16%       56     53      2      6899     3.3s

77.4% inactive integer columns, restarting
Model after restart has 38 rows, 74 cols (46 bin., 28 int., 0 impl., 0 cont.), and 167 nonzeros

         0       0         0   0.00%   25402744.95538  24954483.03        1.80%       16      0      0      7798     3.4s

29.7% inactive integer columns, restarting
Model after restart has 29 rows, 51 cols (34 bin., 17 int., 0 impl., 0 cont.), and 124 nonzeros

         0       0         0   0.00%   25402744.95538  24954483.03        1.80%       11      0      0      7820     3.4s
         0       0         0   0.00%   25402744.95538  24954483.03        1.80%       11     11      4      7834     3.4s
         1       0         1 100.00%   24954483.03     24954483.03        0.00%       51     12     20      7868     3.4s

Solving report
  Status            Optimal
  Primal bound      24954483.03
  Dual bound        24954483.03
  Gap               0% (tolerance: 0.01%)
  Solution status   feasible
                    24954483.03 (objective)
                    0 (bound viol.)
                    1.99840144433e-15 (int. viol.)
                    0 (row viol.)
  Timing            3.44 (total)
                    0.10 (presolve)
                    0.00 (postsolve)
  Nodes             1
  LP iterations     7868 (total)
                    0 (strong br.)
                    1616 (separation)
                    4392 (heuristics)
Writing the solution to MaximizeEmpireValue-pulp.sol
```

## Merge 2-degree waypoint neighbors.

```
INFO:root:Starting link_graph: Graph with 820 nodes and 924 edges
INFO:root:Reduced link_graph: Graph with 654 nodes and 755 edges
Num 1-degree nodes: 274
Num 2-degree nodes: 107
INFO:root:
Solving:    graph with 816 nodes and 1825 arcs
  Using:    budget: 10, lodging_bonus: 0, top_n: 4, nearest_n: 5, capacity: 25
   With:    threads: 16, mip_feasibility_tolerance: 1e-06, gap: default
Running HiGHS 1.7.2 (git hash: 00e812dab): Copyright (c) 2024 HiGHS under MIT licence terms
Number of BV entries in BOUNDS section is 3274
MIP  MaximizeEmpireValue-pulp has 4810 rows; 8467 cols; 23640 nonzeros; 8467 integer variables
Coefficient ranges:
  Matrix [1e+00, 3e+02]
  Cost   [1e+05, 9e+06]
  Bound  [1e+00, 3e+02]
  RHS    [1e+00, 1e+00]
Presolving model
3202 rows, 6797 cols, 19675 nonzeros  0s
2421 rows, 6068 cols, 18026 nonzeros  0s
2233 rows, 5322 cols, 16549 nonzeros  0s

Solving MIP model with:
   2233 rows
   5322 cols (1743 binary, 3579 integer, 0 implied int., 0 continuous)
   16549 nonzeros

        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work      
     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   535936750.66    -inf                 inf        0      0      0         0     0.1s
         0       0         0   0.00%   45306302.07458  -inf                 inf        0      0      3       907     0.1s
 C       0       0         0   0.00%   30913489.36497  383764.89       7955.32%      214     29     79      1366     0.3s
 L       0       0         0   0.00%   26051934.3691   24954483.03        4.40%     1086    149     79      3192     2.0s

74.5% inactive integer columns, restarting
Model after restart has 475 rows, 1055 cols (296 bin., 759 int., 0 impl., 0 cont.), and 3019 nonzeros

         0       0         0   0.00%   26032298.09922  24954483.03        4.32%       85      0      0      7164     2.6s

12.4% inactive integer columns, restarting
Model after restart has 415 rows, 914 cols (237 bin., 677 int., 0 impl., 0 cont.), and 2624 nonzeros

         0       0         0   0.00%   26032291.337    24954483.03        4.32%       75      0      0      7675     2.7s
         0       0         0   0.00%   26032287.48781  24954483.03        4.32%       75     74      4      8082     2.7s

22.6% inactive integer columns, restarting
Model after restart has 295 rows, 631 cols (175 bin., 456 int., 0 impl., 0 cont.), and 1765 nonzeros

         0       0         0   0.00%   25990134.61881  24954483.03        4.15%       72      0      0      9467     2.9s
         0       0         0   0.00%   25988124.73231  24954483.03        4.14%       72     64      2      9709     2.9s

24.1% inactive integer columns, restarting
Model after restart has 213 rows, 452 cols (128 bin., 324 int., 0 impl., 0 cont.), and 1255 nonzeros

         0       0         0   0.00%   25913504.00298  24954483.03        3.84%       51      0      0     10444     3.0s
         0       0         0   0.00%   25913504.00298  24954483.03        3.84%       51     46      2     10620     3.0s

29.6% inactive integer columns, restarting
Model after restart has 136 rows, 285 cols (104 bin., 181 int., 0 impl., 0 cont.), and 777 nonzeros

         0       0         0   0.00%   25599495.80359  24954483.03        2.58%       40      0      0     11573     3.1s
         0       0         0   0.00%   25566664.66361  24954483.03        2.45%       40     37      5     11680     3.2s
         1       0         1 100.00%   24954483.36601  24954483.03        0.00%       52     38     44     11710     3.2s

Solving report
  Status            Optimal
  Primal bound      24954483.03
  Dual bound        24954483.366
  Gap               1e-06% (tolerance: 0.01%)
  Solution status   feasible
                    24954483.03 (objective)
                    0 (bound viol.)
                    5.53532632775e-14 (int. viol.)
                    0 (row viol.)
  Timing            3.15 (total)
                    0.12 (presolve)
                    0.00 (postsolve)
  Nodes             1
  LP iterations     11710 (total)
                    0 (strong br.)
                    2739 (separation)
                    6476 (heuristics)
Writing the solution to MaximizeEmpireValue-pulp.sol
```


## Remove dead-zone nodes (no transit in or through nodes).

```
INFO:root:Starting link_graph: Graph with 820 nodes and 924 edges
INFO:root:Reduced link_graph: Graph with 647 nodes and 742 edges
Num 1-degree nodes: 274
Num 2-degree nodes: 102
INFO:root:
Solving:    graph with 806 nodes and 1796 arcs
  Using:    budget: 10, lodging_bonus: 0, top_n: 4, nearest_n: 5, capacity: 25
   With:    threads: 16, mip_feasibility_tolerance: 1e-06, gap: default
Running HiGHS 1.7.2 (git hash: 00e812dab): Copyright (c) 2024 HiGHS under MIT licence terms
Number of BV entries in BOUNDS section is 3264
MIP  MaximizeEmpireValue-pulp has 4740 rows; 8302 cols; 23197 nonzeros; 8302 integer variables
Coefficient ranges:
  Matrix [1e+00, 3e+02]
  Cost   [1e+05, 9e+06]
  Bound  [1e+00, 3e+02]
  RHS    [1e+00, 1e+00]
Presolving model
3130 rows, 6632 cols, 19226 nonzeros  0s
2359 rows, 5911 cols, 17595 nonzeros  0s
2174 rows, 5184 cols, 16142 nonzeros  0s

Solving MIP model with:
   2174 rows
   5184 cols (1735 binary, 3449 integer, 0 implied int., 0 continuous)
   16142 nonzeros

        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work      
     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   536433392.26    -inf                 inf        0      0      0         0     0.1s
         0       0         0   0.00%   45306302.07458  -inf                 inf        0      0      2       861     0.1s
 C       0       0         0   0.00%   30637795.72836  3850033.89       695.78%      263     38    104      1437     0.3s
 L       0       0         0   0.00%   26169783.31875  24954483.03        4.87%      972    126    104      3237     2.6s

71.6% inactive integer columns, restarting
Model after restart has 545 rows, 1201 cols (338 bin., 863 int., 0 impl., 0 cont.), and 3454 nonzeros

         0       0         0   0.00%   26102854.67511  24954483.03        4.60%       83      0      0      7662     3.1s

15.7% inactive integer columns, restarting
Model after restart has 429 rows, 942 cols (273 bin., 669 int., 0 impl., 0 cont.), and 2681 nonzeros

         0       0         0   0.00%   26102854.67511  24954483.03        4.60%       72      0      0      8087     3.2s
         0       0         0   0.00%   26102854.67511  24954483.03        4.60%       72     69      6      8514     3.2s

14.1% inactive integer columns, restarting
Model after restart has 341 rows, 737 cols (231 bin., 506 int., 0 impl., 0 cont.), and 2074 nonzeros

         0       0         0   0.00%   26023821.54712  24954483.03        4.29%       68      0      0      9715     3.4s
         0       0         0   0.00%   26023801.52584  24954483.03        4.29%       68     65      8     10090     3.4s

Symmetry detection completed in 0.0s
Found 2 generator(s) and 1 full orbitope(s) acting on 2 columns

 L       0       0         0   0.00%   25917837.27768  24954483.03        3.86%      197     67    145     10411     3.7s
 T       0       0         0   0.00%   25917837.27768  24954483.03        3.86%      208     67    208     12040     3.7s
         1       0         1 100.00%   24954483.03     24954483.03        0.00%      209     67    208     12608     3.7s

Solving report
  Status            Optimal
  Primal bound      24954483.03
  Dual bound        24954483.03
  Gap               0% (tolerance: 0.01%)
  Solution status   feasible
                    24954483.03 (objective)
                    0 (bound viol.)
                    5.56356337029e-14 (int. viol.)
                    0 (row viol.)
  Timing            3.75 (total)
                    0.13 (presolve)
                    0.00 (postsolve)
  Nodes             1
  LP iterations     12608 (total)
                    440 (strong br.)
                    2791 (separation)
                    7160 (heuristics)
Writing the solution to MaximizeEmpireValue-pulp.sol
```