# initialize.py

# import json
import milp_flow.data_store as ds


workerman_data_filenames = [
    # Used for node value generation
    "distances_tk2pzk.json",
    "plantzone_drops.json",
    "skills.json",
    "worker_static.json",
    # Used for empire data generation
    # "all_lodging_storage.json", # replacing
    # "deck_links.json", # replacing
    # "exploration.json", # replacing
]

local_data_filenames = [
    # Used for empire data generation
    # "townnames.json",
    # "warehouse_to_townname.json",
]


def initialize_workerman_data(last_sha: str) -> None:
    for filename in workerman_data_filenames:
        print(f"Getting `{filename}`...", end="")
        ds.download_json(filename)
        print("complete.")

    ds.path().joinpath("git_commit.txt").write_text(last_sha)


def initialize_data() -> None:
    # Any error in initialization is fatal and is raised via urllib.
    print("Checking data files...")
    last_sha = ds.download_sha()
    if not ds.initialized(last_sha, workerman_data_filenames + local_data_filenames):
        initialize_workerman_data(last_sha)
    print("Initialized...")
