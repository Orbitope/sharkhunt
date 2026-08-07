"""The numbers the article quotes. Run this, then write prose against the output.

Everything here is cheap - no MCMC. The population fits live in
``sharkhunt/fit_experiments.py``, which takes minutes rather than seconds.

    python -m sharkhunt.experiments --out analysis/results.json
"""

import argparse
import json
import pathlib
import statistics as st

from sharkhunt.detectors import (
    DEFAULT_DELTA,
    HierarchicalJointLLR,
    MixtureJointLLR,
    joint_cell_table,
)
from sharkhunt.elo import pinning_tank_rate, wald_thresholds
from sharkhunt.engine import (
    detection_rates,
    false_positive_by_archetype,
    honest_baseline_profit,
    shark_economics,
    simulate_career,
    simulate_field,
)
from sharkhunt.players import HonestPlayer, Shark, Tilter, Whale
from sharkhunt.rng import Rng
from sharkhunt.wagers import RAKE, STAKES, TIER_NAMES, population_profile

LOWER, UPPER = wald_thresholds()
DETECTORS = ["outcome", "weighted", "joint", "hierarchical", "mixture"]

CASES = {
    "honest-cautious": lambda: HonestPlayer(profile="cautious"),
    "honest-typical": lambda: HonestPlayer(profile="typical"),
    "honest-aggressive": lambda: HonestPlayer(profile="aggressive"),
    "whale": Whale,
    "tilter": Tilter,
    "shark-blatant": lambda: Shark(hidden_delta=400, tank_rate=0.0),
    "shark-pinned": lambda: Shark(hidden_delta=400, bet_correlation=1.0),
    "shark-partial": lambda: Shark(hidden_delta=400, bet_correlation=0.5),
    "shark-hidden": lambda: Shark(hidden_delta=400, bet_correlation=0.0),
}


def _median(xs):
    return st.median(xs) if xs else None


def archetype_table(trials=300, matches=400):
    """How every detector scores every kind of player. The article's main table."""
    out = {}
    for case, make in CASES.items():
        acc = {d: {"flag": 0, "clear": 0, "at": [], "final": []} for d in DETECTORS}
        win_rates, profits, drifts = [], [], []
        for seed in range(trials):
            career, dets = simulate_career(make(), matches, Rng(seed * 7919 + 13))
            win_rates.append(career.win_rate())
            profits.append(career.profit_per_match())
            drifts.append(career.rating_drift())
            for name in DETECTORS:
                d = dets[name]
                acc[name]["flag"] += d.flagged
                acc[name]["clear"] += min(d.history) <= LOWER
                acc[name]["final"].append(d.score)
                if d.flagged_at:
                    acc[name]["at"].append(d.flagged_at)
        out[case] = {
            "trials": trials,
            "matches": matches,
            "win_rate": st.mean(win_rates),
            "profit_per_match": st.mean(profits),
            "rating_drift": st.mean(drifts),
            "detectors": {
                name: {
                    "accused_rate": a["flag"] / trials,
                    "cleared_rate": a["clear"] / trials,
                    "median_matches_to_flag": _median(a["at"]),
                    "final_score_mean": st.mean(a["final"]),
                    "final_score_sd": st.pstdev(a["final"]),
                }
                for name, a in acc.items()
            },
        }
    return out


def calibration_by_betting_habit(trials=300, matches=400):
    """Whether a fixed threshold means the same thing to different bettors.

    This is the concrete cost of the wager-weighted score not being a likelihood
    ratio: how far the running total travels depends on stake size, so the same
    thresholds resolve a whale's case quickly and a cautious player's never.
    """
    out = {}
    for case in ["honest-cautious", "honest-typical", "honest-aggressive", "whale"]:
        row = {}
        for name in DETECTORS:
            resolved = 0
            for seed in range(trials):
                _, dets = simulate_career(CASES[case](), matches, Rng(seed + 500))
                d = dets[name]
                resolved += (min(d.history) <= LOWER) or d.flagged
            row[name] = resolved / trials
        out[case] = row
    return out


def evidence_table(expected=0.5):
    """Per-cell log-likelihood ratio for all six (bet tier, outcome) pairs."""
    tau = pinning_tank_rate(expected, DEFAULT_DELTA)
    joint = joint_cell_table(expected, tau=tau)
    weighted_scale = [STAKES[t] / max(STAKES) for t in range(len(TIER_NAMES))]
    return {
        "expected": expected,
        "pinning_tank_rate": tau,
        "tiers": list(TIER_NAMES),
        "joint": {TIER_NAMES[t]: {"loss": joint[t][0], "win": joint[t][1]}
                  for t in range(len(TIER_NAMES))},
        "weighted_multiplier": {TIER_NAMES[t]: weighted_scale[t]
                                for t in range(len(TIER_NAMES))},
    }


def pinning_curve():
    """The tank rate that hides a given amount of hidden skill, across matchups."""
    return [
        {
            "expected": e,
            "delta": d,
            "tank_rate": pinning_tank_rate(e, d),
        }
        for d in (200, 400, 600, 800)
        for e in (0.3, 0.4, 0.5, 0.6, 0.7)
    ]


def prior_strength_sweep(trials=120, matches=300):
    """The Dirichlet dial: shark detection against whale false alarms."""
    out = []
    for strength in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 400):
        shark = whale = honest = 0
        for seed in range(trials):
            _, d = simulate_career(
                Shark(hidden_delta=400, bet_correlation=1.0), matches, Rng(seed),
                detectors={"h": HierarchicalJointLLR(strength=strength)})
            shark += d["h"].flagged
            _, d = simulate_career(
                Whale(), matches, Rng(seed + 700),
                detectors={"h": HierarchicalJointLLR(strength=strength)})
            whale += d["h"].flagged
            _, d = simulate_career(
                HonestPlayer(), matches, Rng(seed + 1300),
                detectors={"h": HierarchicalJointLLR(strength=strength)})
            honest += d["h"].flagged
        out.append({
            "strength": strength,
            "shark_caught": shark / trials,
            "whale_accused": whale / trials,
            "honest_accused": honest / trials,
        })
    return out


def economics_curve(trials=40, matches=600):
    """Profit against detection, as the shark decorrelates their bets."""
    baseline = st.mean(honest_baseline_profit(Rng(17 + s)) for s in range(20))
    rows = []
    for corr in (1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0):
        profits, caught, at = [], 0, []
        for seed in range(trials):
            r = shark_economics(400, corr, matches, Rng(seed * 31 + 7))
            profits.append(r["profit_per_match"])
            flagged = r["flagged_at"]["mixture"]
            caught += flagged is not None
            if flagged:
                at.append(flagged)
        rows.append({
            "bet_correlation": corr,
            "profit_per_match": st.mean(profits),
            "profit_sd": st.pstdev(profits),
            "caught_rate": caught / trials,
            "median_matches_to_flag": _median(at),
        })
    return {"honest_baseline_profit_per_match": baseline, "rake": RAKE, "curve": rows}


def hidden_skill_sweep(trials=60, matches=400):
    """How much hidden skill you need before the wager evidence bites."""
    rows = []
    for delta in (100, 200, 300, 400, 600, 800):
        caught, at, profit = 0, [], []
        for seed in range(trials):
            career, dets = simulate_career(
                Shark(hidden_delta=delta, bet_correlation=1.0), matches, Rng(seed * 13 + 5))
            d = dets["mixture"]
            caught += d.flagged
            if d.flagged_at:
                at.append(d.flagged_at)
            profit.append(career.profit_per_match())
        rows.append({
            "hidden_delta": delta,
            "pinning_tank_rate": pinning_tank_rate(0.5, delta),
            "caught_rate": caught / trials,
            "median_matches_to_flag": _median(at),
            "profit_per_match": st.mean(profit),
        })
    return rows


def field_study(n_players=2000, matches=300, shark_fraction=0.05):
    """Operational rates at population scale, for each detector."""
    rows = simulate_field(
        n_players, matches, Rng(5),
        shark_fraction=shark_fraction,
        shark_kw={"hidden_delta": 400, "bet_correlation": 1.0},
    )
    return {
        "n_players": n_players,
        "matches": matches,
        "shark_fraction": shark_fraction,
        "rates": {d: detection_rates(rows, d) for d in DETECTORS},
        "false_positives_by_archetype": {
            d: false_positive_by_archetype(rows, d) for d in DETECTORS
        },
    }


def run_all(quick=False):
    scale = 0.25 if quick else 1.0
    n = lambda x: max(20, int(x * scale))  # noqa: E731
    return {
        "config": {
            "wald_lower": LOWER,
            "wald_upper": UPPER,
            "stakes": list(STAKES),
            "rake": RAKE,
            "population_profile": list(population_profile()),
            "default_delta": DEFAULT_DELTA,
            "quick": quick,
        },
        "evidence_table": evidence_table(),
        "pinning_curve": pinning_curve(),
        "archetype_table": archetype_table(trials=n(300)),
        "calibration_by_betting_habit": calibration_by_betting_habit(trials=n(300)),
        "prior_strength_sweep": prior_strength_sweep(trials=n(120)),
        "economics_curve": economics_curve(trials=n(40)),
        "hidden_skill_sweep": hidden_skill_sweep(trials=n(60)),
        "field_study": field_study(n_players=n(2000)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/results.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    results = run_all(quick=args.quick)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")

    tbl = results["archetype_table"]
    print(f"\n{'case':18} " + " ".join(f"{d:>13}" for d in DETECTORS))
    for case, row in tbl.items():
        cells = " ".join(f"{row['detectors'][d]['accused_rate']:12.1%} " for d in DETECTORS)
        print(f"{case:18} {cells}")
    econ = results["economics_curve"]
    print(f"\nhonest baseline profit/match: {econ['honest_baseline_profit_per_match']:+.4f}")
    for r in econ["curve"]:
        print(f"  corr {r['bet_correlation']:.2f}  profit {r['profit_per_match']:+.4f}  "
              f"caught {r['caught_rate']:.0%}")


if __name__ == "__main__":
    main()
