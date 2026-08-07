"""Population fits: parameter recovery, and carrying a prior between games.

Slow - this is the MCMC half of the analysis. Produces the numbers behind the
article's claim that the shark hypothesis can be learned from unlabelled play
rather than asserted, and that what it learns is portable.

    python -m sharkhunt.fit_experiments --out analysis/fits.json
"""

import argparse
import json
import pathlib
import time

import numpy as np

from sharkhunt.detectors import MixtureJointLLR
from sharkhunt.engine import simulate_career
from sharkhunt.hierarchical import (
    as_transfer_prior,
    build_grid,
    fit_population,
    player_shark_logodds,
)
from sharkhunt.players import CAUTIOUS_MIX, HONEST_MIX, HonestPlayer, Shark, Whale, make_honest
from sharkhunt.rng import Rng
from sharkhunt.wagers import SHARK_PLAY_PROFILE, SHARK_TANK_PROFILE

# The established title we learn from, and the new one we transfer into. They
# differ in the things a real pair of games differ in - how the crowd bets, how
# common cheating is, how much skill the cheats are hiding - while the *shape* of
# what a shark does is assumed to carry across.
GAME_A = {
    "mix": HONEST_MIX,
    "prevalence": 0.08,
    "shark": {"hidden_delta": 400, "bet_correlation": 1.0},
    "label": "established title, mixed betting culture",
}
#: The new title is deliberately a hard case, because an easy one proves nothing.
#: With 120 matches per player and blatant sharks, a from-scratch fit already
#: separates the two groups perfectly and there is no room for a prior to help.
#: Three weeks of history is more like 40 matches each, and the sharks worth
#: catching are the subtle ones - here only partly correlating their bets, the
#: regime where the detector sat at 50-60% rather than 100%.
GAME_B = {
    "mix": CAUTIOUS_MIX,
    "prevalence": 0.06,
    "shark": {"hidden_delta": 300, "bet_correlation": 0.6},
    "label": "new title, cautious betting culture, subtle sharks",
}


def make_population(spec, n_players, matches, seed):
    """Simulate one game's population.

    The number of sharks is set to ``round(prevalence * n)`` rather than drawn,
    with a floor of two. At the small end of the transfer sweep - 25 players at
    6% prevalence - a Bernoulli draw regularly yields zero or one shark, and a
    separation score computed on zero positives is undefined rather than bad.
    Fixing the count keeps the comparison across sizes about the *amount of
    data*, which is the thing under study, instead of about how many cheaters
    happened to be dealt.
    """
    rng = Rng(seed)
    n_sharks = max(2, int(round(spec["prevalence"] * n_players)))
    flags = [i < n_sharks for i in range(n_players)]
    # Shuffle so shark indices are not all at the front.
    for i in range(n_players - 1, 0, -1):
        j = rng.randint(0, i + 1)
        flags[i], flags[j] = flags[j], flags[i]

    careers, labels = [], []
    for i in range(n_players):
        sub = rng.spawn(i + 1)
        is_shark = flags[i]
        player = Shark(**spec["shark"]) if is_shark else make_honest(sub, mix=spec["mix"])
        career, _ = simulate_career(player, matches, sub, detectors={})
        careers.append(career.observations)
        labels.append(is_shark)
    return careers, np.array(labels)


def auc(scores, labels):
    """Rank-based AUC. 0.5 is coin-flipping, 1.0 is a perfect separation."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def profile_error(fit):
    """Total absolute error in the recovered shark betting signature."""
    play = np.abs(np.array(fit["play_profile"]) - np.array(SHARK_PLAY_PROFILE)).sum()
    tank = np.abs(np.array(fit["tank_profile"]) - np.array(SHARK_TANK_PROFILE)).sum()
    return {"play_l1": float(play), "tank_l1": float(tank), "total_l1": float(play + tank)}


def detector_scorecard(fit, trials=60, matches=300, seed0=0):
    """Turn a fitted posterior into a detector and see how it does."""

    def rate(make, s0):
        hits = 0
        for seed in range(trials):
            _, d = simulate_career(
                make(), matches, Rng(seed + s0),
                detectors={"m": MixtureJointLLR.from_posterior(fit)},
            )
            hits += d["m"].flagged
        return hits / trials

    return {
        "shark_caught": rate(lambda: Shark(hidden_delta=400, bet_correlation=1.0), seed0),
        "honest_accused": rate(HonestPlayer, seed0 + 5000),
        "whale_accused": rate(Whale, seed0 + 9000),
    }


def summarise(fit, careers, labels, elapsed):
    logodds = player_shark_logodds(fit)
    return {
        "prevalence": fit["prevalence"],
        "prior_strength": fit["prior_strength"],
        "population_profile": list(fit["population_profile"]),
        "play_profile": list(fit["play_profile"]),
        "tank_profile": list(fit["tank_profile"]),
        "weights": fit["weights"],
        "profile_error": profile_error(fit),
        "auc": auc(logodds, labels),
        "n_players": len(careers),
        "n_sharks": int(labels.sum()),
        "matches": len(careers[0]),
        "seconds": round(elapsed, 1),
    }


def run(args):
    grid = build_grid()
    out = {
        "grid": grid,
        "truth": {
            "play_profile": list(SHARK_PLAY_PROFILE),
            "tank_profile": list(SHARK_TANK_PROFILE),
            "game_a": {k: v for k, v in GAME_A.items() if k != "mix"},
            "game_b": {k: v for k, v in GAME_B.items() if k != "mix"},
        },
    }

    careers_a, labels_a = make_population(GAME_A, args.a_players, args.a_matches, seed=99)
    cache = pathlib.Path(args.a_cache) if args.a_cache else None
    if cache and cache.exists():
        print(f"reusing cached game A fit from {cache}", flush=True)
        cached = json.loads(cache.read_text())
        fit_a = cached["fit"]
        out["game_a"] = cached["summary"]
    else:
        print(f"fitting game A: {args.a_players} players x {args.a_matches} matches", flush=True)
        t = time.time()
        fitted = fit_population(
            careers_a, grid=grid, draws=args.draws, tune=args.tune, chains=args.chains, seed=1
        )
        out["game_a"] = summarise(fitted, careers_a, labels_a, time.time() - t)
        out["game_a"]["detector"] = detector_scorecard(fitted)
        fit_a = {k: v for k, v in fitted.items() if k != "idata"}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"fit": fit_a, "summary": out["game_a"]}, indent=2))
    print(f"  play {np.round(fit_a['play_profile'], 3)} "
          f"tank {np.round(fit_a['tank_profile'], 3)} "
          f"auc {out['game_a']['auc']} in {out['game_a']['seconds']}s", flush=True)

    prior = as_transfer_prior(fit_a, confidence=args.confidence)
    out["transfer"] = []
    for n_players in args.b_players:
        careers_b, labels_b = make_population(GAME_B, n_players, args.b_matches, seed=4242)
        row = {"n_players": n_players, "matches": args.b_matches,
               "n_sharks": int(labels_b.sum())}
        for condition, use_prior in (("cold", None), ("warm", prior)):
            t = time.time()
            fit_b = fit_population(
                careers_b, grid=grid, prior=use_prior,
                draws=args.draws, tune=args.tune, chains=args.chains, seed=7,
            )
            row[condition] = summarise(fit_b, careers_b, labels_b, time.time() - t)
            row[condition]["detector"] = detector_scorecard(fit_b, trials=40)
            got = row[condition]["auc"]
            print(f"  B n={n_players:4} {condition:5} "
                  f"auc {'n/a' if got is None else format(got, '.3f')} "
                  f"profile_err {row[condition]['profile_error']['total_l1']:.3f} "
                  f"caught {row[condition]['detector']['shark_caught']:.0%} "
                  f"({row[condition]['seconds']}s)", flush=True)
        out["transfer"].append(row)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/fits.json")
    ap.add_argument("--a-cache", default="analysis/fit_a.json",
                    help="reuse the established title's posterior instead of refitting it")
    ap.add_argument("--a-players", type=int, default=400)
    ap.add_argument("--a-matches", type=int, default=200)
    ap.add_argument("--b-players", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--b-matches", type=int, default=40)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--confidence", type=float, default=1.0)
    args = ap.parse_args()

    results = run(args)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
