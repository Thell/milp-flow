### Optimized Worker Production Node Valuations Per Town

import json
import os
import math


def read_data_json(file):
    filepath = os.path.join(os.path.dirname(__file__), "data", file)
    with open(filepath, "r") as file:
        return json.load(file)


DISTANCES_TK2PZK = read_data_json("distances_tk2pzk.json")

# A fix for ordering since workerman sorts by nearest node to town.
# for town in DISTANCES_TK2PZK:
#     DISTANCES_TK2PZK[town] = sorted(DISTANCES_TK2PZK[town], key=lambda x: x[0])
# filepath = os.path.join(os.path.dirname(__file__), "data", "distances_tk2pzk.json")
# with open(filepath, "w") as file:
#     json.dump(DISTANCES_TK2PZK, file, sort_keys=False, indent=4, ensure_ascii=True)

MARKET_VALUE = read_data_json("market.json")
PLANTZONE = read_data_json("plantzone.json")
PLANTZONE_DROPS = read_data_json("plantzone_drops.json")
WORKER_SKILLS = read_data_json("skills.json")
WORKER_STATIC = read_data_json("worker_static.json")


def isGiant(charkey):
    return WORKER_STATIC[str(charkey)]["species"] in [2, 4, 8]


def skill_bonus(skill_set):
    bonus = {"wspd": 0, "mspd": 0, "luck": 0}
    for sk in skill_set:
        skill_bonuses = WORKER_SKILLS.get(sk, {})
        bonus["wspd"] += skill_bonuses.get("wspd", 0)
        bonus["wspd"] += skill_bonuses.get("wspd_farm", 0)
        bonus["mspd"] += skill_bonuses.get("mspd", 0)
        bonus["luck"] += skill_bonuses.get("luck", 0)
    return bonus


def worker_stats(worker, skill_set):
    bonus = skill_bonus(skill_set)
    wspd = worker["wspd"] + bonus["wspd"]
    mspd_base = WORKER_STATIC[str(worker["charkey"])]["mspd"] / 100
    mspd = mspd_base * ((worker["mspd"] / mspd_base) + bonus["mspd"] / 100)
    luck = worker["luck"] + bonus["luck"]
    return {"wspd": wspd, "mspd": mspd, "luck": luck}


def calcCyclesDaily(baseWorkload, wspd, dist, mspd):
    moveMinutes = 2 * dist / mspd / 60
    activeWorkload = baseWorkload * 2
    workMinutes = math.ceil(activeWorkload / wspd)
    cycleMinutes = 10 * workMinutes + moveMinutes
    return 24 * 60 / cycleMinutes


def price_bunch(bunch):
    return sum(MARKET_VALUE[k] * q for k, q in bunch.items())


def price_pzd(pzd, luck):
    unlucky_price = price_bunch(pzd.get("unlucky", {}))
    if "lucky" in pzd:
        lucky_price = price_bunch(pzd["lucky"])
        return (luck / 100) * lucky_price + (1 - luck / 100) * unlucky_price
    return unlucky_price


def price_lerp(lucky_price, unlucky_price, luck):
    if lucky_price is None:
        return unlucky_price
    return (luck / 100) * lucky_price + (1 - luck / 100) * unlucky_price


def profitPzTownStats(pzk, tnk, dist, wspd, mspd, luck, is_giant):
    if dist == 9999999:
        return 0

    drop = PLANTZONE_DROPS[str(pzk)]
    luckyPart = price_bunch(drop["lucky"])
    unluckyValue = price_bunch(drop["unlucky"])
    luckyValue = unluckyValue + luckyPart
    unluckyValue_gi = price_bunch(drop["unlucky_gi"])
    luckyValue_gi = unluckyValue_gi + luckyPart

    cycleValue = (
        price_lerp(luckyValue_gi, unluckyValue_gi, luck)
        if is_giant
        else price_lerp(luckyValue, unluckyValue, luck)
    )
    cyclesDaily = calcCyclesDaily(drop["workload"], wspd, dist, mspd)
    priceDaily = cyclesDaily * cycleValue / 1000000
    return priceDaily


def profit(town, plantzone, dist, worker, skill_set):
    stats = worker_stats(worker, skill_set)
    priceDaily = profitPzTownStats(
        plantzone,
        town,
        dist,
        stats["wspd"],
        stats["mspd"],
        stats["luck"],
        isGiant(worker["charkey"]),
    )
    return priceDaily


def makeMedianChar(charkey):
    ret = {}
    stat = WORKER_STATIC[str(charkey)]
    pa_wspd = stat["wspd"]
    pa_mspdBonus = 0
    pa_luck = stat["luck"]

    for i in range(2, 41):
        pa_wspd += (stat["wspd_lo"] + stat["wspd_hi"]) / 2
        pa_mspdBonus += (stat["mspd_lo"] + stat["mspd_hi"]) / 2
        pa_luck += (stat["luck_lo"] + stat["luck_hi"]) / 2

    pa_mspd = stat["mspd"] * (1 + pa_mspdBonus / 1e6)

    ret["wspd"] = round(pa_wspd / 1e6 * 100) / 100
    ret["mspd"] = round(pa_mspd) / 100
    ret["luck"] = round(pa_luck / 1e4 * 100) / 100
    ret["charkey"] = charkey
    ret["isGiant"] = isGiant(charkey)

    return ret


def medianGoblin(tnk):
    if tnk == 1623:
        return makeMedianChar(8003)  # grana
    if tnk == 1604:
        return makeMedianChar(8003)  # owt
    if tnk == 1691:
        return makeMedianChar(8023)  # oddy
    if tnk == 1750:
        return makeMedianChar(8035)  # eilton
    if tnk == 1781:
        return makeMedianChar(8050)  # lotml
    if tnk == 1785:
        return makeMedianChar(8050)  # lotml
    if tnk == 1795:
        return makeMedianChar(8050)  # lotml
    return makeMedianChar(7572)


def medianGiant(tnk):
    if tnk == 1623:
        return makeMedianChar(8006)  # grana
    if tnk == 1604:
        return makeMedianChar(8006)  # owt
    if tnk == 1691:
        return makeMedianChar(8027)  # oddy
    if tnk == 1750:
        return makeMedianChar(8039)  # eilton
    if tnk == 1781:
        return makeMedianChar(8058)  # lotml
    if tnk == 1785:
        return makeMedianChar(8058)  # lotml
    if tnk == 1795:
        return makeMedianChar(8058)  # lotml
    return makeMedianChar(7571)


def medianHuman(tnk):
    if tnk == 1623:
        return makeMedianChar(8009)  # grana
    if tnk == 1604:
        return makeMedianChar(8009)  # owt
    if tnk == 1691:
        return makeMedianChar(8031)  # oddy
    if tnk == 1750:
        return makeMedianChar(8043)  # eilton
    if tnk == 1781:
        return makeMedianChar(8054)  # lotml
    if tnk == 1785:
        return makeMedianChar(8054)  # lotml
    if tnk == 1795:
        return makeMedianChar(8054)  # lotml
    return makeMedianChar(7573)


def optimize_skills(town, plantzone, dist, worker):
    max_skills = 9
    w_bonuses = {0: {"skills": [], "profit": 0}}
    w_actions = ["wspd"]
    w_actions.append("wspd_farm")

    w_skills = []
    for key, skill in WORKER_SKILLS.items():
        if any(act in skill for act in w_actions):
            w_skills.append(
                {
                    "key": key,
                    "amount": skill.get("wspd", 0) + skill.get("wspd_farm", 0),
                    "mspd": skill.get("mspd", 0),
                }
            )

    w_skills.sort(key=lambda x: (x["amount"], x["mspd"]), reverse=True)

    for i in range(1, max_skills + 1):
        temp_skills = [w["key"] for w in w_skills[:i]]
        new_profit = profit(town, plantzone, dist, worker, temp_skills)
        w_bonuses[i] = {"skills": temp_skills, "profit": new_profit}

        if all(not WORKER_SKILLS[sk].get("mspd", 0) for sk in temp_skills):
            mod_skills = temp_skills.copy()
            wm_skills = [ss for ss in w_skills if ss["mspd"] > 0]
            if wm_skills:
                mod_skills[-1] = wm_skills[0]["key"]
                mod_profit = profit(town, plantzone, dist, worker, mod_skills)
                if mod_profit > new_profit:
                    w_bonuses[i] = {"skills": mod_skills, "profit": mod_profit}

    ml_actions = ["mspd", "luck"]
    ml_skills = {
        key for key, skill in WORKER_SKILLS.items() if any(act in skill for act in ml_actions)
    }

    step_results = [w_bonuses[max_skills]]
    ml_best_skills = []
    for i in range(1, max_skills + 1):
        step_base_skills = w_bonuses[max_skills - i]["skills"] + ml_best_skills
        step_candidates = []

        for sk in ml_skills:
            if sk in w_bonuses[max_skills - i]["skills"]:
                continue
            temp_skills = step_base_skills + [sk]
            new_profit = profit(town, plantzone, dist, worker, temp_skills)
            step_candidates.append({"sk": sk, "profit": new_profit})

        if step_candidates:
            step_candidates.sort(key=lambda x: x["profit"], reverse=True)
            step_best_skill = step_candidates[0]["sk"]
            step_skills = step_base_skills + [step_best_skill]
            step_results.append({"skills": step_skills, "profit": step_candidates[0]["profit"]})
            ml_best_skills.append(step_best_skill)
            ml_skills.remove(step_best_skill)
        else:
            ml_best_skills.append(0)

    step_results.sort(key=lambda x: x["profit"], reverse=True)
    return step_results[0]


SAMPLE_FILTER = read_data_json("sample_filter_nodes.json")
output = {}
for town in DISTANCES_TK2PZK.keys():
    if town not in SAMPLE_FILTER["towns"]:
        continue

    if str(town) not in output:
        output[str(town)] = {}

    median_giant = medianGiant(town)
    median_goblin = medianGoblin(town)
    median_human = medianHuman(town)

    for data in DISTANCES_TK2PZK[town]:
        plantzone, dist = data
        if plantzone not in SAMPLE_FILTER["production_nodes"]:
            continue
        if not PLANTZONE[str(plantzone)]["node"]["is_plantzone"]:
            continue
        if PLANTZONE[str(plantzone)]["node"]["kind"] in [12, 13]:
            continue

        optimized_workers = {
            "giant": optimize_skills(town, plantzone, dist, median_giant),
            "goblin": optimize_skills(town, plantzone, dist, median_goblin),
            "human": optimize_skills(town, plantzone, dist, median_human),
        }
        optimized_worker = max(optimized_workers.items(), key=lambda item: item[1]["profit"])

        output[str(town)][str(plantzone)] = {}
        output[str(town)][str(plantzone)]["worker"] = optimized_worker[0]
        output[str(town)][str(plantzone)]["skills"] = optimized_worker[1]["skills"]
        output[str(town)][str(plantzone)]["value"] = optimized_worker[1]["profit"]

filepath = os.path.join(os.path.dirname(__file__), "data", "sample_node_values_per_city.json")
with open(filepath, "w") as outfile:
    json.dump(output, outfile, indent=4)
