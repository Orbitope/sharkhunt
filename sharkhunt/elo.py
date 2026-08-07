"""Elo expectations, plus the tank rate that makes a rating hold still.

Pure Python on purpose: docs/sharkhunt.js mirrors this file line for line so the
article's widgets and the offline experiments compute the same numbers.
"""

import math

DEFAULT_K = 24.0


def expected_score(rating, opponent_rating):
    """Probability the first player wins, per the standard Elo curve."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def shift_expectation(expected, delta):
    """Re-price an expectation for a player who is ``delta`` Elo stronger.

    Rearranging the Elo curve, an expectation ``e`` implies a rating gap of
    ``400 log10(e / (1 - e))``. Adding ``delta`` to that gap and folding it back
    gives a closed form that never needs the ratings themselves:

        e_delta = e / (e + (1 - e) * 10^(-delta/400))
    """
    g = 10.0 ** (-delta / 400.0)
    return expected / (expected + (1.0 - expected) * g)


def update_rating(rating, expected, score, k=DEFAULT_K):
    return rating + k * (score - expected)


def pinning_tank_rate(expected, delta, tank_win_prob=0.02):
    """The tank rate that makes a hidden ``delta`` of skill statistically invisible.

    A player who is ``delta`` stronger than their rating wins with probability
    ``e_delta``. If they deliberately throw a fraction ``tau`` of matches (winning
    those only with probability ``tank_win_prob``), their observed win rate is

        (1 - tau) * e_delta + tau * tank_win_prob

    Setting that equal to ``expected`` - the win rate their *shown* rating
    predicts - and solving gives

        tau* = (e_delta - expected) / (e_delta - tank_win_prob)

    At tau* the outcome distribution under "shark" is *identical* to the outcome
    distribution under "honest". No test that looks only at wins and losses can
    tell them apart, at any sample size. This is the hole the wager evidence is
    there to fill.
    """
    e_delta = shift_expectation(expected, delta)
    if e_delta <= tank_win_prob:
        return 0.0
    tau = (e_delta - expected) / (e_delta - tank_win_prob)
    return min(max(tau, 0.0), 1.0)


def wald_thresholds(alpha=0.01, beta=0.01):
    """(lower, upper) log-likelihood-ratio bounds for a sequential test.

    ``alpha`` is the false-accusation rate we are willing to run, ``beta`` the
    rate of clearing a real shark. Wald's approximation puts the accept-H1
    boundary at log((1-beta)/alpha) and the accept-H0 boundary at
    log(beta/(1-alpha)).
    """
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)
