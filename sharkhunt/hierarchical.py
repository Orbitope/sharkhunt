"""Fit the shark hypothesis to a population instead of asserting it.

Every detector in detectors.py takes the alternative hypothesis as given: this
much hidden skill, this tank rate, these betting habits, this much trust in the
population prior. Those are five numbers somebody made up. This module learns
them from the population, with a hierarchical Bayesian model whose latent
per-player class - honest or shark - is marginalised out, so no labels are
needed.

The model, per player p with matches i:

    z_p ~ Bernoulli(prevalence)                        who they are
    honest: tier_i ~ Categorical(pi_p),  pi_p ~ Dirichlet(strength * pop)
            won_i  ~ Bernoulli(e_i)
    shark:  (delta, tau) ~ Categorical(w) over a fixed grid
            each match is a two-component mixture of "play" and "tank",
            exactly as in detectors.JointLLR

Two things are marginalised analytically so NUTS only ever sees a smooth
posterior. The per-player betting habits pi_p integrate out to a
Dirichlet-multinomial over that player's tier counts, which is also what makes
``strength`` - the article's prior-strength dial - an estimable parameter rather
than a knob. And z_p integrates out to a two-term logsumexp per player.

What comes back is a posterior over the population, which
``MixtureJointLLR.from_posterior`` turns straight into a serving-time detector.
Fitting is offline and occasional; scoring a live player stays a closed-form
running sum. PyMC does the learning, arithmetic does the serving.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import gammaln, logsumexp

from sharkhunt.detectors import DEFAULT_DELTA_GRID, DEFAULT_TAU_GRID, TANK_WIN_PROB
from sharkhunt.wagers import N_TIERS

_EPS = 1e-12


def build_grid(deltas=DEFAULT_DELTA_GRID, taus=DEFAULT_TAU_GRID):
    """The (hidden skill, tank rate) cells the shark component mixes over."""
    return [(float(d), float(t)) for d in deltas for t in taus]


def pack(careers):
    """Turn simulated careers into the padded arrays the model consumes.

    ``careers`` is a sequence of Observation lists, one per player. Players may
    have played different numbers of matches; short ones are padded and masked.
    """
    n_players = len(careers)
    n_max = max(len(c) for c in careers)
    expected = np.zeros((n_players, n_max))
    won = np.zeros((n_players, n_max))
    tier = np.zeros((n_players, n_max), dtype=np.int64)
    mask = np.zeros((n_players, n_max))
    for p, obs_list in enumerate(careers):
        for i, o in enumerate(obs_list):
            expected[p, i] = o.expected
            won[p, i] = 1.0 if o.won else 0.0
            tier[p, i] = o.tier
            mask[p, i] = 1.0
    # Guard the padding so log() never sees a zero.
    expected = np.where(mask > 0, expected, 0.5)
    onehot = np.eye(N_TIERS)[tier] * mask[..., None]
    counts = onehot.sum(axis=1)
    return {
        "expected": expected,
        "won": won,
        "tier": tier,
        "mask": mask,
        "onehot": onehot,
        "counts": counts,
        "n": mask.sum(axis=1),
        "n_players": n_players,
    }


def honest_outcome_loglik(data):
    """Sum of log Bernoulli(won ; expected). Free of parameters, so precompute."""
    e = np.clip(data["expected"], _EPS, 1 - _EPS)
    per_match = data["won"] * np.log(e) + (1 - data["won"]) * np.log(1 - e)
    return (per_match * data["mask"]).sum(axis=1)


def _dirichlet_multinomial_logp(counts, n, alpha):
    """Log probability of a player's *ordered* tier sequence with pi integrated out.

    The multinomial coefficient is deliberately omitted: we are scoring the
    sequence the player actually produced, not the set of counts, and the shark
    branch scores the same sequence. Dropping it consistently keeps the two
    branches comparable.
    """
    return (
        pt.gammaln(pt.sum(alpha, axis=-1))
        - pt.gammaln(n + pt.sum(alpha, axis=-1))
        + pt.sum(pt.gammaln(counts + alpha) - pt.gammaln(alpha), axis=-1)
    )


def _shark_loglik(data, grid, play_profile, tank_profile, tau_vec, delta_vec):
    """[P, K] log-likelihood of each player's record under each grid cell."""
    e = pt.as_tensor_variable(np.clip(data["expected"], _EPS, 1 - _EPS))[:, :, None]
    won = pt.as_tensor_variable(data["won"])[:, :, None]
    mask = pt.as_tensor_variable(data["mask"])[:, :, None]
    onehot = pt.as_tensor_variable(data["onehot"])

    # phi[t] for the tier actually played, per player per match.
    play_t = pt.dot(onehot, play_profile)[:, :, None]
    tank_t = pt.dot(onehot, tank_profile)[:, :, None]

    g = 10.0 ** (-delta_vec[None, None, :] / 400.0)
    e_delta = e / (e + (1.0 - e) * g)
    play_out = won * e_delta + (1.0 - won) * (1.0 - e_delta)
    tank_out = won * TANK_WIN_PROB + (1.0 - won) * (1.0 - TANK_WIN_PROB)

    tau = tau_vec[None, None, :]
    p = tau * tank_t * tank_out + (1.0 - tau) * play_t * play_out
    return pt.sum(pt.log(pt.maximum(p, _EPS)) * mask, axis=1)


def build_model(data, grid=None, prior=None):
    """The PyMC model. ``prior`` carries a previous game's posterior, if any."""
    grid = grid or build_grid()
    tau_vec = np.array([t for _, t in grid])
    delta_vec = np.array([d for d, _ in grid])
    prior = prior or {}

    honest_out = honest_outcome_loglik(data)
    counts = pt.as_tensor_variable(data["counts"])
    n = pt.as_tensor_variable(data["n"])

    coords = {"tier": list(range(N_TIERS)), "cell": list(range(len(grid)))}
    with pm.Model(coords=coords) as model:
        prevalence = pm.Beta(
            "prevalence",
            alpha=prior.get("prevalence_alpha", 1.0),
            beta=prior.get("prevalence_beta", 19.0),
        )
        # Concentration of the per-player betting prior: the article's
        # "prior strength" dial, here estimated rather than chosen.
        prior_strength = pm.LogNormal(
            "prior_strength",
            mu=prior.get("strength_mu", np.log(8.0)),
            sigma=prior.get("strength_sigma", 1.0),
        )
        population_profile = pm.Dirichlet(
            "population_profile", a=prior.get("population_a", np.ones(N_TIERS)), dims="tier"
        )
        play_profile = pm.Dirichlet(
            "play_profile", a=prior.get("play_a", np.ones(N_TIERS)), dims="tier"
        )
        tank_profile = pm.Dirichlet(
            "tank_profile", a=prior.get("tank_a", np.ones(N_TIERS)), dims="tier"
        )
        weights = pm.Dirichlet(
            "weights", a=prior.get("weights_a", np.ones(len(grid))), dims="cell"
        )

        honest_tiers = _dirichlet_multinomial_logp(
            counts, n, prior_strength * population_profile[None, :]
        )
        ll_honest = honest_tiers + pt.as_tensor_variable(honest_out)

        shark_cells = _shark_loglik(
            data, grid, play_profile, tank_profile, tau_vec, delta_vec
        )
        ll_shark = pt.logsumexp(pt.log(weights)[None, :] + shark_cells, axis=1)

        per_player = pt.logsumexp(
            pt.stack(
                [pt.log1p(-prevalence) + ll_honest, pt.log(prevalence) + ll_shark], axis=1
            ),
            axis=1,
        )
        pm.Potential("population", pt.sum(per_player))

        # Reported for inspection: the posterior probability each player is a shark.
        pm.Deterministic(
            "shark_logodds", pt.log(prevalence) + ll_shark - pt.log1p(-prevalence) - ll_honest
        )
    return model, grid


def fit_population(
    careers,
    grid=None,
    prior=None,
    draws=1000,
    tune=1000,
    chains=4,
    cores=1,
    seed=0,
    progressbar=False,
):
    """Fit the population model and return everything the detector needs.

    ``careers`` is one Observation list per player, labels not required and not
    used. The returned dict plugs straight into
    ``MixtureJointLLR.from_posterior``.

    Chains run sequentially by default. The per-gradient working set is a
    [players, matches, grid cells] tensor, and forking several copies of it has
    been enough to take the sampler's worker processes out on a laptop. Fitting
    happens offline and rarely, so the wall clock is not worth the fragility.
    """
    data = pack(careers)
    model, grid = build_model(data, grid, prior)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=True,
        )
    post = idata.posterior
    mean = {k: post[k].mean(dim=("chain", "draw")).values for k in
            ["prevalence", "prior_strength", "population_profile", "play_profile",
             "tank_profile", "weights"]}
    return {
        "components": grid,
        "weights": [float(w) for w in mean["weights"]],
        "prior_strength": float(mean["prior_strength"]),
        "prevalence": float(mean["prevalence"]),
        "population_profile": tuple(float(x) for x in mean["population_profile"]),
        "play_profile": tuple(float(x) for x in mean["play_profile"]),
        "tank_profile": tuple(float(x) for x in mean["tank_profile"]),
        "idata": idata,
    }


def posterior_summary(fit, var_names=None):
    """Compact posterior table: mean, sd, 94% HDI, r_hat."""
    import arviz as az

    var_names = var_names or [
        "prevalence", "prior_strength", "population_profile", "play_profile", "tank_profile"
    ]
    return az.summary(fit["idata"], var_names=var_names, round_to=4)


def as_transfer_prior(fit, confidence=1.0):
    """Turn a fitted posterior into the prior for another game.

    This is the cold-start move. A new title has a handful of players and no
    labelled cheaters, which is nowhere near enough to identify a mixture model
    on its own. But what a shark *looks like* - bet heavily on the matches you
    mean to win, minimally on the ones you mean to throw - is not especially
    game-specific, even though the ratings, the stakes and the betting culture
    all are. So carry the shape across and let the new game's own data move it.

    ``confidence`` scales how strongly the previous game's posterior is imposed:
    0 falls back to flat priors, larger values pin the new fit closer to the old.
    """
    if confidence <= 0:
        return {}
    k = 20.0 * confidence
    prev = fit["prevalence"]
    return {
        "prevalence_alpha": max(prev * k, 1e-2),
        "prevalence_beta": max((1.0 - prev) * k, 1e-2),
        "strength_mu": float(np.log(max(fit["prior_strength"], 1e-3))),
        "strength_sigma": max(1.0 / confidence, 0.05),
        "population_a": np.maximum(np.array(fit["population_profile"]) * k, 1e-2),
        "play_a": np.maximum(np.array(fit["play_profile"]) * k, 1e-2),
        "tank_a": np.maximum(np.array(fit["tank_profile"]) * k, 1e-2),
        "weights_a": np.maximum(np.array(fit["weights"]) * k, 1e-2),
    }


def player_shark_logodds(fit):
    """Posterior mean log-odds that each player is a shark."""
    return fit["idata"].posterior["shark_logodds"].mean(dim=("chain", "draw")).values


def numpy_shark_loglik(data, grid, play_profile, tank_profile):
    """Reference implementation of the shark branch, for testing the tensor one."""
    out = np.zeros((data["n_players"], len(grid)))
    e = np.clip(data["expected"], _EPS, 1 - _EPS)
    for k, (delta, tau) in enumerate(grid):
        g = 10.0 ** (-delta / 400.0)
        e_delta = e / (e + (1 - e) * g)
        play_out = np.where(data["won"] > 0, e_delta, 1 - e_delta)
        tank_out = np.where(data["won"] > 0, TANK_WIN_PROB, 1 - TANK_WIN_PROB)
        play_t = np.array(play_profile)[data["tier"]]
        tank_t = np.array(tank_profile)[data["tier"]]
        p = tau * tank_t * tank_out + (1 - tau) * play_t * play_out
        out[:, k] = (np.log(np.maximum(p, _EPS)) * data["mask"]).sum(axis=1)
    return out


def numpy_honest_tier_loglik(data, strength, population):
    """Reference implementation of the Dirichlet-multinomial branch."""
    alpha = strength * np.array(population)
    return (
        gammaln(alpha.sum())
        - gammaln(data["n"] + alpha.sum())
        + (gammaln(data["counts"] + alpha) - gammaln(alpha)).sum(axis=1)
    )


def numpy_player_loglik(data, grid, params):
    """Full marginal log-likelihood per player, in numpy. Used by the tests."""
    ll_h = numpy_honest_tier_loglik(
        data, params["prior_strength"], params["population_profile"]
    ) + honest_outcome_loglik(data)
    cells = numpy_shark_loglik(data, grid, params["play_profile"], params["tank_profile"])
    ll_s = logsumexp(np.log(np.array(params["weights"]))[None, :] + cells, axis=1)
    prev = params["prevalence"]
    return logsumexp(
        np.stack([np.log1p(-prev) + ll_h, np.log(prev) + ll_s], axis=1), axis=1
    )
