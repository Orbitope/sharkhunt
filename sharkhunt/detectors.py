"""Four sequential detectors, in the order the article builds them up.

Every one of them accumulates a running score over a player's matches and
compares it to a pair of thresholds. They differ in what they are willing to
look at:

    OutcomeSPRT     wins and losses only
    WeightedLLR     the same, with each term scaled by the size of the bet
    JointLLR        bet size and outcome modelled together
    Hierarchical    the same, with the player's betting habits learned under a
                    population prior

Only the last two are genuine likelihood ratios. That is the point of the
sequence: the middle one looks like it works and is the one worth breaking on
purpose.
"""

import math

from sharkhunt.elo import shift_expectation, wald_thresholds
from sharkhunt.wagers import (
    N_TIERS,
    SHARK_PLAY_PROFILE,
    SHARK_TANK_PROFILE,
    STAKES,
    DirichletProfile,
    FixedProfile,
    population_profile,
)

# Default alternative hypothesis: the player is 400 Elo stronger than they show
# and throws matches often enough to sit still at that rating.
DEFAULT_DELTA = 400.0
DEFAULT_TAU = 0.40
TANK_WIN_PROB = 0.02

_EPS = 1e-12


class Observation:
    """One match, as the detector sees it.

    ``expected`` is what the two *shown* ratings predict, ``won`` is the result,
    ``tier`` is which of the three wager tiers the player under test posted.
    Nothing here is private to the player - a detector only ever sees what the
    platform already records.
    """

    __slots__ = ("expected", "won", "tier")

    def __init__(self, expected, won, tier):
        self.expected = expected
        self.won = bool(won)
        self.tier = int(tier)

    def as_dict(self):
        return {"expected": self.expected, "won": self.won, "tier": self.tier}


class Detector:
    """Running score plus Wald thresholds. Subclasses supply ``increment``."""

    name = "detector"
    calibrated = True  # whether the thresholds mean what they claim

    def __init__(self, alpha=0.01, beta=0.01):
        self.lower, self.upper = wald_thresholds(alpha, beta)
        self.score = 0.0
        self.history = [0.0]
        self.flagged_at = None
        self.n = 0

    def increment(self, obs):
        raise NotImplementedError

    def update(self, obs):
        inc = self.increment(obs)
        self.score += inc
        self.n += 1
        self.history.append(self.score)
        if self.flagged_at is None and self.score >= self.upper:
            self.flagged_at = self.n
        return inc

    def run(self, observations):
        for obs in observations:
            self.update(obs)
        return self

    @property
    def flagged(self):
        return self.flagged_at is not None


class OutcomeSPRT(Detector):
    """Wald's sequential test on match results alone.

    H0: the player wins with the probability their shown rating predicts.
    H1: they are ``delta`` stronger and throw a fraction ``tau`` of matches, so
        they win with probability ``(1 - tau) * e_delta + tau * TANK_WIN_PROB``.

    Each match contributes ``log P(result | H1) - log P(result | H0)``.

    This is a correct test and it is genuinely useful against a player who simply
    plays their real strength. It has one structural blind spot, and it is not a
    matter of needing more data: whenever the shark picks the tau that makes
    those two win probabilities equal, every increment is exactly zero. See
    ``elo.pinning_tank_rate``.
    """

    name = "outcome"

    def __init__(self, delta=DEFAULT_DELTA, tau=DEFAULT_TAU, **kw):
        super().__init__(**kw)
        self.delta = delta
        self.tau = tau

    def win_prob_h1(self, expected):
        e_delta = shift_expectation(expected, self.delta)
        return (1.0 - self.tau) * e_delta + self.tau * TANK_WIN_PROB

    def increment(self, obs):
        p0 = min(max(obs.expected, _EPS), 1 - _EPS)
        p1 = min(max(self.win_prob_h1(obs.expected), _EPS), 1 - _EPS)
        return math.log(p1 / p0) if obs.won else math.log((1 - p1) / (1 - p0))


class WeightedLLR(OutcomeSPRT):
    """The intuitive first fix: scale each outcome term by the size of the bet.

    A win that the player backed heavily counts for more than one they backed
    with pocket change; a match they staked almost nothing on barely counts at
    all. Against the obvious shark - big bets on the matches they intend to win,
    minimum bets on the ones they intend to throw - this works, and it works
    fast.

    It is not a likelihood ratio. Multiplying log-likelihood terms by an
    arbitrary weight produces a quantity that is no longer the log of any
    probability ratio, so Wald's thresholds no longer carry the error rates they
    were derived for, and ``calibrated`` is False to say so. The practical
    consequences the article demonstrates:

      - A player who always bets the maximum has every term scaled up, so their
        score random-walks further and crosses the accusation line far more often
        than the nominal alpha.
      - A minimum-bet loss contributes almost exactly zero. Under a model where
        sharks throw matches cheaply, that event is *evidence*, and this detector
        throws it away.
    """

    name = "weighted"
    calibrated = False

    def increment(self, obs):
        return super().increment(obs) * STAKES[obs.tier] / max(STAKES)


class JointLLR(Detector):
    """Model the bet and the result together. The correlation is the signal.

    Under H0 there is nothing to know: the player's skill is their rating, so
    what they stake carries no information about how the match will go. Bet tier
    and outcome are independent given the ratings:

        P(t, y | H0) = pi_t * Bernoulli(y ; e)

    Under H1 the player privately decides, before each match, whether to play or
    to throw it - and that decision drives both the bet and the result:

        P(t, y | H1) = tau     * phi_tank[t] * Bernoulli(y ; TANK_WIN_PROB)
                     + (1-tau) * phi_play[t] * Bernoulli(y ; e_delta)

    Both sides sum to one over the six (tier, outcome) cells, so the ratio is a
    real likelihood ratio and the thresholds mean what they say.

    Three things fall out that the weighted score cannot do:

      - A minimum-bet loss is now positive evidence, because throwing matches
        cheaply is exactly what H1 predicts. The weighted score scored that same
        event as zero.
      - A player who always bets the same amount contributes no wager evidence
        either way: their tier term appears in numerator and denominator and
        mostly cancels, leaving the outcome test. Whales stop being suspects.
      - A shark can still hide, but only by decorrelating their bets from their
        intent - which means staking full size on the matches they are about to
        throw. That is not free, and the price is what makes this worth doing.
    """

    name = "joint"

    def __init__(
        self,
        delta=DEFAULT_DELTA,
        tau=DEFAULT_TAU,
        play_profile=SHARK_PLAY_PROFILE,
        tank_profile=SHARK_TANK_PROFILE,
        profile=None,
        **kw,
    ):
        super().__init__(**kw)
        self.delta = delta
        self.tau = tau
        self.play_profile = tuple(play_profile)
        self.tank_profile = tuple(tank_profile)
        self.profile = profile or FixedProfile(population_profile())

    def _p_h0(self, obs, pi):
        p = pi[obs.tier] * (obs.expected if obs.won else 1.0 - obs.expected)
        return max(p, _EPS)

    def _p_h1(self, obs):
        e_delta = shift_expectation(obs.expected, self.delta)
        play_out = e_delta if obs.won else 1.0 - e_delta
        tank_out = TANK_WIN_PROB if obs.won else 1.0 - TANK_WIN_PROB
        p = (
            self.tau * self.tank_profile[obs.tier] * tank_out
            + (1.0 - self.tau) * self.play_profile[obs.tier] * play_out
        )
        return max(p, _EPS)

    def increment(self, obs):
        pi = self.profile.predict()
        inc = math.log(self._p_h1(obs) / self._p_h0(obs, pi))
        self.profile.observe(obs.tier)
        return inc


class HierarchicalJointLLR(JointLLR):
    """JointLLR with the player's betting habits learned under a population prior.

    JointLLR as written above assumes we already know pi, the player's own
    distribution over bet tiers. We don't. Estimating it from the player's own
    history alone is unstable early and manipulable later; assuming everyone bets
    like the average player punishes anyone who doesn't. The Dirichlet prior in
    wagers.py interpolates between those two failure modes, and ``strength`` is
    the dial.
    """

    name = "hierarchical"

    def __init__(self, strength=8.0, population=None, **kw):
        kw.pop("profile", None)
        super().__init__(**kw)
        self.profile = DirichletProfile(population or population_profile(), strength)


#: Tank rates and hidden-skill gaps the mixture detector averages over when no
#: fitted posterior is supplied. Deliberately coarse: the point of the grid is to
#: stop the test assuming one particular kind of shark, not to be exact.
DEFAULT_TAU_GRID = (0.0, 0.15, 0.30, 0.45, 0.60)
DEFAULT_DELTA_GRID = (200.0, 400.0, 700.0)


class MixtureJointLLR(HierarchicalJointLLR):
    """Joint detection that does not assume one kind of shark.

    Every detector above fixes a single (delta, tau) pair, and that turns out to
    matter. A test tuned for a shark who throws 40% of their matches scores a
    shark who throws *none* of them as unremarkable, because their bets show no
    trace of the tanking the hypothesis expects. Outcome-only testing catches
    that player instantly, so the two detectors have complementary blind spots -
    an awkward place to leave a system.

    Averaging fixes it. H1 becomes a mixture over a grid of shark strategies:

        P(t, y | H1) = sum_k w_k * P(t, y | H1 ; delta_k, tau_k)

    A mixture of proper distributions is a proper distribution, so this remains a
    real likelihood ratio and the thresholds still mean what they say. The
    weights are the interesting part: left alone they are uniform, but
    ``sharkhunt.hierarchical`` fits them from the population, so the test ends up
    tuned to the sharks a given game actually has rather than the ones we
    guessed at.
    """

    name = "mixture"

    def __init__(self, components=None, weights=None, **kw):
        super().__init__(**kw)
        if components is None:
            components = [(d, t) for d in DEFAULT_DELTA_GRID for t in DEFAULT_TAU_GRID]
        self.components = [(float(d), float(t)) for d, t in components]
        if weights is None:
            weights = [1.0 / len(self.components)] * len(self.components)
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def _p_h1(self, obs):
        total = 0.0
        base_delta, base_tau = self.delta, self.tau
        try:
            for (delta, tau), w in zip(self.components, self.weights):
                self.delta, self.tau = delta, tau
                total += w * super()._p_h1(obs)
        finally:
            self.delta, self.tau = base_delta, base_tau
        return max(total, _EPS)

    @classmethod
    def from_posterior(cls, fit, **kw):
        """Build the serving-time detector from a fitted population posterior.

        ``fit`` is the dict ``sharkhunt.hierarchical.fit_population`` returns.
        Each posterior draw becomes one mixture component, which makes the served
        likelihood ratio a posterior-predictive one rather than a plug-in
        estimate at the posterior mean.
        """
        kw.setdefault("strength", fit["prior_strength"])
        kw.setdefault("population", fit["population_profile"])
        kw.setdefault("play_profile", fit["play_profile"])
        kw.setdefault("tank_profile", fit["tank_profile"])
        return cls(components=fit["components"], weights=fit["weights"], **kw)


def detector_suite(**kw):
    """One fresh instance of each detector, for scoring the same match stream."""
    return {
        "outcome": OutcomeSPRT(**kw),
        "weighted": WeightedLLR(**kw),
        "joint": JointLLR(**kw),
        "hierarchical": HierarchicalJointLLR(**kw),
        "mixture": MixtureJointLLR(**kw),
    }


def joint_cell_table(expected, delta=DEFAULT_DELTA, tau=DEFAULT_TAU, pi=None):
    """The per-cell log-likelihood ratio for all six (tier, outcome) pairs.

    Handy for the article: it shows at a glance which observations incriminate,
    which exonerate, and by how much - including the sign flip on a cheap loss
    that the weighted score gets wrong.
    """
    pi = pi or population_profile()
    det = JointLLR(delta=delta, tau=tau, profile=FixedProfile(pi))
    table = []
    for tier in range(N_TIERS):
        row = []
        for won in (False, True):
            obs = Observation(expected, won, tier)
            row.append(math.log(det._p_h1(obs) / det._p_h0(obs, pi)))
        table.append(row)
    return table
