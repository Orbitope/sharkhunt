"""Wager tiers and the population prior over betting habits.

Bets are discretised into three tiers. That is a modelling choice, not a
limitation of the method - it keeps every distribution a three-cell categorical,
which makes the likelihood ratio in detectors.py readable and makes the
hierarchical prior a plain Dirichlet with a closed-form posterior.
"""

TIER_NAMES = ("min", "mid", "max")
N_TIERS = 3

# Stake per tier as a fraction of the table maximum.
STAKES = (0.10, 0.40, 1.00)

# House cut of the pot. A player with no edge loses money at this rate, which is
# the honest-player baseline the shark's profit gets compared against.
RAKE = 0.05


def payoff(tier, won):
    """Net change in bankroll for one match, in units of the table maximum.

    Both players post the same stake. The winner takes the pot less the rake, so
    a win nets ``s * (1 - 2 * RAKE)`` and a loss nets ``-s``.
    """
    s = STAKES[tier]
    return s * (1.0 - 2.0 * RAKE) if won else -s


# Betting habits, as probabilities over (min, mid, max).
PROFILES = {
    "cautious": (0.70, 0.25, 0.05),
    "typical": (0.20, 0.55, 0.25),
    "aggressive": (0.10, 0.30, 0.60),
    "whale": (0.00, 0.00, 1.00),
}

# What a shark's bets look like conditioned on private intent: heavy when they
# mean to win, minimal when they mean to lose.
SHARK_PLAY_PROFILE = (0.05, 0.25, 0.70)
SHARK_TANK_PROFILE = (0.75, 0.20, 0.05)


def population_profile(weights=None):
    """The field's average betting habits - the mean of the Dirichlet prior."""
    weights = weights or {"cautious": 0.35, "typical": 0.45, "aggressive": 0.15, "whale": 0.05}
    out = [0.0] * N_TIERS
    total = sum(weights.values())
    for name, w in weights.items():
        for t in range(N_TIERS):
            out[t] += (w / total) * PROFILES[name][t]
    return tuple(out)


def blend(a, b, weight):
    """``weight`` of ``a`` mixed with the rest of ``b``, kept normalised."""
    return tuple(weight * a[t] + (1.0 - weight) * b[t] for t in range(N_TIERS))


class FixedProfile:
    """A betting profile the detector already knows. Used for ablations."""

    def __init__(self, probs):
        self.probs = tuple(probs)

    def predict(self):
        return self.probs

    def observe(self, tier):
        pass


class DirichletProfile:
    """Each player's betting habits, learned from their own history.

    A player with four matches has no usable profile of their own, and a shark
    could otherwise manufacture a convenient one. So each player's tier
    probabilities carry a Dirichlet prior centred on the population average with
    concentration ``strength``, and the detector scores match i against the
    posterior predictive built from matches 1..i-1 only:

        P(tier = t) = (count_t + strength * pop_t) / (n + strength)

    Scoring prequentially matters - if a match were allowed into the profile that
    judges it, every player would look unsurprising to themselves.

    ``strength`` is the knob the article makes interactive. Near zero, players
    are judged only against their own history, and a patient shark can normalise
    any habit they like. Very large, everyone is judged against the field, and
    honest players with unusual habits start tripping the alarm.
    """

    def __init__(self, population, strength=8.0):
        self.population = tuple(population)
        self.strength = float(strength)
        self.counts = [0.0] * N_TIERS
        self.n = 0.0

    def predict(self):
        denom = self.n + self.strength
        return tuple(
            (self.counts[t] + self.strength * self.population[t]) / denom
            for t in range(N_TIERS)
        )

    def observe(self, tier):
        self.counts[tier] += 1.0
        self.n += 1.0
