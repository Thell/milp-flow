import json
from os import makedirs, path
from urllib import error, request


def download_json_file(filename, outpath):
    outpath = path.join(outpath, f"{filename}")
    if filename in ["plantzone_drops.json", "skills.json"]:
        filename = f"manual/{filename}"
    raw_url = f"https://raw.githubusercontent.com/shrddr/workermanjs/refs/heads/main/data/{filename}"

    try:
        request.urlretrieve(raw_url, outpath)
    except error.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")


def extract_tk2tnk_from_js(outpath):
    import json
    import re

    url = "https://raw.githubusercontent.com/shrddr/workermanjs/refs/heads/main/src/stores/game.js"
    try:
        with request.urlopen(url) as response:
            js_content = response.read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching the file: {e}")
        return None

    # re to match the _tk2tnk dictionary
    dict_pattern = re.compile(r"this\._tk2tnk\s*=\s*{([^}]*)}", re.DOTALL)
    match = dict_pattern.search(js_content)
    # Since shrddr's comments in the source indicating reading this from the bss in the future...
    if not match:
        print("Dictionary not found, check the source file!")
        return None

    dict_content = match.group(1).strip()
    pair_pattern = re.compile(r"(\d+)\s*:\s*(\d+)")
    tk2tnk_dict = {}
    for line in dict_content.splitlines():
        pair_match = pair_pattern.search(line)
        if pair_match:
            key, value = int(pair_match.group(1)), int(pair_match.group(2))
            tk2tnk_dict[key] = str(value)
    tnk2tk_dict = {v: str(k) for k, v in tk2tnk_dict.items()}

    out_dict = {"tk2tnk": tk2tnk_dict, "tnk2tk": tnk2tk_dict}
    outpath = path.join(outpath, "town_node_translate.json")
    with open(outpath, "w") as json_file:
        json.dump(out_dict, json_file, indent=4)


def extract_town_names(outpath):
    url = "https://raw.githubusercontent.com/shrddr/workermanjs/refs/heads/main/data/loc.json"
    try:
        with request.urlopen(url) as response:
            content = response.read().decode("utf-8")
            json_data = json.loads(content)
    except Exception as e:
        print(f"Error fetching the file: {e}")
        return None

    json_data = json_data["en"]["town"]
    outpath = path.join(outpath, "warehouse_to_townname.json")
    with open(outpath, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    return json_data


def generate_warehouse_to_town_names(town_names, outpath):
    with open(f"{outpath}/town_node_translate.json", "r") as f:
        translator = json.load(f)

    out_dict = {}
    for k, v in town_names.items():
        out_dict[translator["tk2tnk"][k]] = v

    outpath = path.join(outpath, "townnames.json")
    with open(outpath, "w") as json_file:
        json.dump(out_dict, json_file, indent=4)


def main():
    filepath = path.join(path.dirname(__file__), "data", "workerman")
    makedirs(path.dirname(filepath), exist_ok=True)

    filelist = [
        # Used for node value generation
        "distances_tk2pzk.json",
        "plantzone.json",
        "plantzone_drops.json",
        "skills.json",
        "worker_static.json",
        # Used for empire data generation
        "all_lodging_storage.json",
        "deck_links.json",
        "exploration.json",
        "plantzone.json",
    ]
    for filename in filelist:
        print(f"Getting `{filename}`...", end="")
        download_json_file(filename, filepath)
        print("complete.")

    print("Generating `town_node_translate.json`...", end="")
    extract_tk2tnk_from_js(filepath)
    print("complete.")

    print("Extracting town name list...", end="")
    town_names = extract_town_names(filepath)
    print("complete.")

    print("Generating warehouse to town name list...", end="")
    generate_warehouse_to_town_names(town_names, filepath)
    print("complete.")


if __name__ == "__main__":
    main()
