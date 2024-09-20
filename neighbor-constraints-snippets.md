See:

https://docs.google.com/spreadsheets/d/1-B9mb-WuFuFsYunEBpC-JL7KmdKtqMuXVRAN6T6J988/edit?gid=1829888294#gid=1829888294

```python
# I think either 1 or 6 based upon a second test of #2 that had real bad Parx8 results (~6h40m)
# One more test and which ever is more stable between #1 and #6 will get it.
# One wins... barely, I guess I could say it is a toss up...
# After doing a test with b5_lb5_tn6_nn7_wc35_gdefault #1 wins.

# 1 - Works fine, good times. I select this as the winner.
if node.isWaypoint:
    prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
else:
    prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0
prob += lpSum(out_neighbors) >= node.vars["x"]

# 2 - Works fine, good times. Same as Variant 1 but different order.
prob += lpSum(out_neighbors) >= node.vars["x"]
if node.isWaypoint:
    prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
else:
    prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0

# 4 - Works fine, good times.
if node.isWaypoint or node.isPlant:
    prob += lpSum(in_neighbors) >= lpSum(out_neighbors)
if node.isWaypoint:
    prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
else:
    prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0
prob += lpSum(out_neighbors) >= node.vars["x"]

# 6 - Works fine, good times, I'd say this is the best of the three overall, but still testing.
if node.type in [NT.plant, NT.waypoint, NT.group]:
    prob += lpSum(in_neighbors) >= lpSum(out_neighbors)
if node.isWaypoint:
    prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
else:
    prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0
prob += lpSum(out_neighbors) >= node.vars["x"]


# 6b -  Same as Variant 6 but different order.
prob += lpSum(out_neighbors) >= node.vars["x"]
if node.type in [NT.plant, NT.waypoint, NT.group]:
    prob += lpSum(in_neighbors) >= lpSum(out_neighbors)
if node.isWaypoint:
    prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
else:
    prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0


# 7 - Works fine, bad times.
if node.isWaypoint:
    prob += lpSum(in_neighbors) - 2 * node.vars["x"] >= 0
else:
    prob += lpSum(in_neighbors) + lpSum(out_neighbors) - 2 * node.vars["x"] >= 0

# 7 - Works fine, bad times.
prob += lpSum(out_neighbors) >= node.vars["x"]
```