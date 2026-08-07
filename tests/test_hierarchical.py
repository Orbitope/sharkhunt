"""The population model: correctness first, then whether it recovers the truth.

The recovery test is the load-bearing one. Simulate a population whose shark
parameters we chose, hide the labels, fit, and check the posterior finds them.
If that fails, the article's claim that the hypothesis can be learned rather
than asserted is not true.
"""

import numpy as np
import pytest

from sharkhunt.detectors import MixtureJointLLR
from sharkhunt.engine import simulate_career
from sharkhunt.hierarchical import (
    as_transfer_prior,
    build_grid,
    build_model,
    fit_population,
    honest_outcome_loglik,
    numpy_honest_tier_loglik,
    numpy_player_loglik,
    numpy_shark_loglik,
    pack,
    player_shark_logodds,
)
from sharkhunt.players import HonestPlayer, Shark, Whale, make_honest
from sharkhunt.rng import Rng
from sharkhunt.wagers import SHARK_PLAY_PROFILE, SHARK_TANK_PROFILE

TRUE_PREVALENCE = 0.10


def make_population(n_players, matches, seed, prevalence=TRUE_PREVALENCE, **shark_kw):
    rng = Rng(seed)
    careers, labels = [], []
    shark_kw.setdefault("hidden_delta", 400)
    shark_kw.setdefault("bet_correlation", 1.0)
    for i in range(n_players):
        sub = rng.spawn(i + 1)
        is_shark = sub.chance(prevalence)
        player = Shark(**shark_kw) if is_shark else make_honest(sub)
        career, _ = simulate_career(player, matches, sub, detectors={})
        careers.append(career.observations)
        labels.append(is_shark)
    return careers, np.array(labels)


# --- correctness ------------------------------------------------------------


def test_pack_preserves_the_record():
    careers, _ = make_population(12, 40, seed=3)
    data = pack(careers)
    assert data["n_players"] == 12
    assert data["mask"].sum() == 12 * 40
    assert np.allclose(data["counts"].sum(axis=1), 40)
    for p, obs_list in enumerate(careers):
        assert data["tier"][p, 7] == obs_list[7].tier
        assert bool(data["won"][p, 7]) == obs_list[7].won
        assert data["expected"][p, 7] == pytest.approx(obs_list[7].expected)


def test_ragged_careers_are_masked_not_counted():
    careers, _ = make_population(6, 40, seed=4)
    careers[2] = careers[2][:11]
    data = pack(careers)
    assert data["n"][2] == 11
    assert data["counts"][2].sum() == 11
    assert data["mask"][2, 11:].sum() == 0
    # Padded slots must not contribute to any likelihood term.
    grid = build_grid()
    ll = numpy_shark_loglik(data, grid, SHARK_PLAY_PROFILE, SHARK_TANK_PROFILE)
    data2 = pack([careers[2]])
    ll2 = numpy_shark_loglik(data2, grid, SHARK_PLAY_PROFILE, SHARK_TANK_PROFILE)
    assert np.allclose(ll[2], ll2[0])


def test_pytensor_likelihood_matches_the_numpy_reference():
    """The model graph and the hand-written reference must agree to machine precision."""
    careers, _ = make_population(40, 60, seed=5, prevalence=0.2)
    data = pack(careers)
    grid = build_grid()
    model, grid = build_model(data, grid)

    params = {
        "prevalence": 0.1,
        "prior_strength": 8.0,
        "population_profile": (0.35, 0.38, 0.27),
        "play_profile": SHARK_PLAY_PROFILE,
        "tank_profile": SHARK_TANK_PROFILE,
        "weights": [1.0 / len(grid)] * len(grid),
    }
    reference = float(numpy_player_loglik(data, grid, params).sum())
    point = {model[k]: np.asarray(v, dtype="float64") for k, v in params.items()}
    graph = float(model.potentials[0].eval(point))
    assert graph == pytest.approx(reference, rel=1e-12)


def test_honest_branch_terms_are_finite_and_ordered():
    careers, _ = make_population(30, 80, seed=6)
    data = pack(careers)
    outcome = honest_outcome_loglik(data)
    tiers = numpy_honest_tier_loglik(data, 8.0, (0.35, 0.38, 0.27))
    assert np.all(np.isfinite(outcome)) and np.all(outcome < 0)
    assert np.all(np.isfinite(tiers)) and np.all(tiers < 0)

    # A stronger prior should fit an idiosyncratic bettor worse, not better.
    whale, _ = simulate_career(Whale(), 80, Rng(1), detectors={})
    wdata = pack([whale.observations])
    weak = numpy_honest_tier_loglik(wdata, 0.5, (0.35, 0.38, 0.27))[0]
    strong = numpy_honest_tier_loglik(wdata, 200.0, (0.35, 0.38, 0.27))[0]
    assert weak > strong


# --- does it actually learn? ------------------------------------------------


@pytest.mark.slow
def test_fit_recovers_the_shark_betting_signature_without_labels():
    """The claim the article rests on.

    Nobody tells the model what a shark looks like. It sees an unlabelled pile of
    match records and has to discover that one subpopulation stakes heavily on
    the games it means to win and minimally on the ones it means to throw.
    """
    careers, labels = make_population(200, 150, seed=99, prevalence=0.08)
    fit = fit_population(careers, draws=300, tune=300, chains=1, seed=1)

    for i, (got, want) in enumerate(zip(fit["play_profile"], SHARK_PLAY_PROFILE)):
        assert got == pytest.approx(want, abs=0.08), f"play profile tier {i}"
    for i, (got, want) in enumerate(zip(fit["tank_profile"], SHARK_TANK_PROFILE)):
        assert got == pytest.approx(want, abs=0.08), f"tank profile tier {i}"
    assert 0.01 < fit["prevalence"] < 0.25, "prevalence should land in the right order"

    # And the per-player posterior should separate the two groups it never saw.
    logodds = player_shark_logodds(fit)
    assert np.median(logodds[labels]) > np.median(logodds[~labels]) + 10


@pytest.mark.slow
def test_fitted_posterior_builds_a_working_detector():
    """The learned hypothesis has to survive being turned back into a test."""
    careers, _ = make_population(150, 150, seed=77, prevalence=0.1)
    fit = fit_population(careers, draws=250, tune=250, chains=1, seed=2)

    def catch_rate(make, trials=40, seed0=0):
        hits = 0
        for seed in range(trials):
            _, d = simulate_career(
                make(), 300, Rng(seed + seed0),
                detectors={"m": MixtureJointLLR.from_posterior(fit)},
            )
            hits += d["m"].flagged
        return hits / trials

    assert catch_rate(lambda: Shark(hidden_delta=400, bet_correlation=1.0)) > 0.9
    assert catch_rate(HonestPlayer, seed0=500) <= 0.05
    assert catch_rate(Whale, seed0=500) <= 0.05


def test_transfer_prior_shape():
    """A transfer prior must be a usable set of hyperparameters, and confidence 0
    must fall all the way back to flat."""
    fake = {
        "prevalence": 0.08,
        "prior_strength": 6.0,
        "population_profile": (0.35, 0.38, 0.27),
        "play_profile": SHARK_PLAY_PROFILE,
        "tank_profile": SHARK_TANK_PROFILE,
        "weights": [1.0 / 15] * 15,
    }
    assert as_transfer_prior(fake, confidence=0.0) == {}

    prior = as_transfer_prior(fake, confidence=1.0)
    assert prior["prevalence_alpha"] < prior["prevalence_beta"], "should still favour rarity"
    assert np.all(prior["play_a"] > 0) and np.all(prior["tank_a"] > 0)
    assert prior["play_a"].argmax() == 2, "prior should remember sharks bet big to win"
    assert prior["tank_a"].argmax() == 0, "and small to lose"

    careers, _ = make_population(20, 40, seed=8)
    model, _ = build_model(pack(careers), build_grid(), prior)
    assert "weights" in model.named_vars, "the model should build with a transfer prior"

    # Higher confidence pins the new game closer to the old one's posterior.
    tighter = as_transfer_prior(fake, confidence=4.0)
    assert tighter["strength_sigma"] < prior["strength_sigma"]
    assert tighter["play_a"].sum() > prior["play_a"].sum()
