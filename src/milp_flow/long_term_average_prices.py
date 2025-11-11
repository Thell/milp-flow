import json
import requests
import time
import os

# Try hardcoded path first (forward slashes work on Windows too)
input_file = "C:/Users/thell/Downloads/custom_prices (22).json"

# Verify path exists
if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    # Optional: Prompt for path
    input_file = input("Enter full path to JSON file: ").strip().strip("\"'")
    if not os.path.exists(input_file):
        print("Still not found. Check spelling/spaces/parentheses.")
        exit(1)

print(f"Loading: {input_file}")

# Load input JSON
with open(input_file, "r") as f:
    data = json.load(f)

# Rest of your script unchanged...
effective_prices = data.get("effectivePrices", {})
if not effective_prices:
    print("No 'effective_prices' found in input file.")
    exit(1)

now_ms = int(time.time() * 1000)
start_ts = int((time.time() - 365 * 24 * 3600) * 1000)
finish_ts = now_ms

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

for id_str, price in list(effective_prices.items()):
    try:
        id_int = int(id_str)
        url = f"https://apiv2.bdolytics.com/market/analytics/{id_int}?start_date={start_ts}&end_date={finish_ts}&region=NA&enhancement_level=0"
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        j = r.json()
        if j.get("status"):
            raise ValueError(j["status"])

        valid_data = [d for d in j.get("data", []) if d[1] is not None]
        if valid_data:
            avg_price = round(sum(d[1] for d in valid_data) / len(valid_data))
            effective_prices[id_str] = float(avg_price)
            print(f"Updated {id_str}: {price} -> {avg_price}")
        else:
            print(f"No data for {id_str}, keeping {price}")
    except Exception as e:
        print(f"Error for {id_str}: {e}, keeping {price}")

data["effective_prices"] = effective_prices
output_file = "en_lta_prices.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"Output written to {output_file}")
