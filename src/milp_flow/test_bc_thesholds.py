import pandas as pd
import numpy as np
from bisect import bisect_left

df = pd.read_csv("raw_used_centrality.csv")

# Group by budget/capacity for per-group analysis
groups = df.groupby(["budget", "capacity"])

thresholds = []
min_rows = []
global_suggested = groups["bc_tr_asp"].quantile(0.05).median()  # Median 5th %ile as global start
print(f"Global suggested threshold: {global_suggested:.8f}")

for (budget, capacity), group in groups:
    used_bc = sorted(group["bc_tr_asp"].unique())  # Sorted unique used bc for bisect
    if len(used_bc) == 0:
        continue

    # Start with global
    thresh = global_suggested
    leaks = group[group["bc_tr_asp"] < thresh]

    # Halve until no leaks
    while len(leaks) > 0 and thresh > 0:
        thresh /= 2
        leaks = group[group["bc_tr_asp"] < thresh]

    # Bisect up: Find highest thresh where no leaks (just below min used)
    low, high = thresh, used_bc[0]
    while high - low > 1e-8:
        mid = (low + high) / 2
        leaks_mid = group[group["bc_tr_asp"] < mid]
        if len(leaks_mid) == 0:
            low = mid
        else:
            high = mid

    safe_thresh = low
    thresholds.append({
        "budget": budget,
        "capacity": capacity,
        "safe_threshold": safe_thresh,
        "min_used": used_bc[0],
    })
    print(f"Budget {budget} {capacity}: Safe thresh={safe_thresh:.8f} (min used={used_bc[0]:.8f})")

    # Add min row for this group
    min_idx = group["bc_tr_asp"].idxmin()
    min_row = df.loc[min_idx].copy()
    min_row["budget"] = budget
    min_row["capacity"] = capacity
    min_rows.append(min_row)

# Save thresholds
pd.DataFrame(thresholds).to_csv("per_budget_safe_thresholds.csv", index=False)
print("Saved per_budget_safe_thresholds.csv")

# Save min rows CSV
min_df = pd.DataFrame(min_rows)
min_df.to_csv("min_used_bc_asp.csv", index=False)
print("Saved min_used_bc_asp.csv")
