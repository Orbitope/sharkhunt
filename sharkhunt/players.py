"""Player archetypes: who is in the lobby, how they bet, how they play.

Each archetype exposes the same tiny interface - given the Elo expectation for
the current match, decide a bet tier and a win probability - so the engine never
needs to know which kind of player it is holding, and the detectors never see
anything except the resulting (tier, outcome) pair.

The honest archetypes exist to be *hard*, not easy. A cautious player, a whale
who always bets the table maximum, and a tilter who chases losses all have odd
wager patterns, and a detector that flags them is worse than no detector.
"""

from sharkhunt.elo import expected_score, pinning_tank_rate, shift_expectation, update_rating
from sharkhunt.wagers import (
    PROFILES,
    SHARK_PLAY_PROFILE,
    SHARK_TANK_PROFILE,
    blend,
    payoff,
)

TANK_WIN_PROB = 0.02


class Player:
    """Base class. ``rating`` is public, ``hidden_delta`` is not."""

    archetype = "player"
    hidden_delta = 0.0

    def __init__(self, rating=1200.0, name=None):
        self.rating = float(rating)
        self.name = name or self.archetype
        self.bankroll = 0.0
        self.matches = 0
        self.last_won = None

    # -- decisions ---------------------------------------------------------

    def bet_profile(self, expected):
        raise NotImplementedError

    def decide(self, expected, rng):
        """Return ``(tier, win_probability, intent)`` for one match.

        ``intent`` is the player's private plan - "play" for everyone honest, and
        "play" or "tank" for a shark. It is recorded for analysis only; no
        detector is ever allowed to see it.
        """
        tier = rng.choice_p(self.bet_profile(expected))
        return tier, expected, "play"

    # -- bookkeeping -------------------------------------------------------

    def settle(self, expected, tier, won, k):
        self.bankroll += payoff(tier, won)
        self.rating = update_rating(self.rating, expected, 1.0 if won else 0.0, k)
        self.matches += 1
        self.last_won = won


class HonestPlayer(Player):
    """Skill equals rating. Bets are drawn from a fixed habit, never from the
    result, because they have no idea what the result will be."""

    archetype = "honest"

    def __init__(self, profile="typical", **kw):
        super().__init__(**kw)
        self.profile_name = profile
        self.profile = PROFILES[profile]

    def bet_profile(self, expected):
        return self.profile


class Whale(HonestPlayer):
    """Honest, and always bets the table maximum.

    The archetype that breaks the wager-weighted score. Every one of their
    matches gets the largest possible multiplier, so their running total takes
    the biggest possible steps in both directions and wanders past the
    accusation threshold on nothing but variance.
    """

    archetype = "whale"

    def __init__(self, **kw):
        kw.setdefault("profile", "whale")
        super().__init__(**kw)


class Tilter(HonestPlayer):
    """Honest, but chases: bets bigger after a loss.

    A deliberate near-miss. Their bet size genuinely is correlated with an
    outcome - the *previous* one. If the joint detector cannot tell that apart
    from correlation with the *current* outcome, it will flag people for being
    bad at gambling rather than for cheating.
    """

    archetype = "tilter"

    def __init__(self, chase=0.75, **kw):
        kw.setdefault("profile", "typical")
        super().__init__(**kw)
        self.chase = chase

    def bet_profile(self, expected):
        if self.last_won is False:
            return blend(PROFILES["aggressive"], self.profile, self.chase)
        return self.profile


class Shark(Player):
    """Stronger than their rating shows, and throwing matches to keep it that way.

    Three dials:

    ``hidden_delta``  how much stronger than their rating they really are.
    ``tank_rate``     the fraction of matches they deliberately throw. Left at
                      None it is solved for exactly, per match, so their observed
                      win rate matches what their shown rating predicts - the
                      setting that makes an outcome-only test blind forever.
    ``bet_correlation``  how tightly they let bet size track private intent.
                      At 1.0 they bet big when they mean to win and minimum when
                      they mean to lose. At 0.0 both intents draw from the same
                      distribution, so the bets say nothing - but they are then
                      staking full size on matches they are about to throw.

    The marginal bet distribution is deliberately held fixed as
    ``bet_correlation`` varies, so sweeping that dial isolates the correlation
    itself rather than confounding it with a change in betting habits.
    """

    archetype = "shark"

    def __init__(
        self,
        hidden_delta=400.0,
        tank_rate=None,
        bet_correlation=1.0,
        play_profile=SHARK_PLAY_PROFILE,
        tank_profile=SHARK_TANK_PROFILE,
        **kw,
    ):
        super().__init__(**kw)
        self.hidden_delta = float(hidden_delta)
        self.tank_rate = tank_rate
        self.bet_correlation = float(bet_correlation)
        self.play_profile = tuple(play_profile)
        self.tank_profile = tuple(tank_profile)

    def true_rating(self):
        return self.rating + self.hidden_delta

    def effective_tank_rate(self, expected):
        if self.tank_rate is not None:
            return self.tank_rate
        return pinning_tank_rate(expected, self.hidden_delta, TANK_WIN_PROB)

    def marginal_profile(self, tau):
        """Bet habits with intent averaged out. Invariant in bet_correlation."""
        return tuple(
            tau * self.tank_profile[t] + (1.0 - tau) * self.play_profile[t]
            for t in range(3)
        )

    def bet_profile(self, expected):
        return self.marginal_profile(self.effective_tank_rate(expected))

    def decide(self, expected, rng):
        tau = self.effective_tank_rate(expected)
        base = self.marginal_profile(tau)
        tanking = rng.chance(tau)
        intent_profile = self.tank_profile if tanking else self.play_profile
        tier = rng.choice_p(blend(intent_profile, base, self.bet_correlation))
        if tanking:
            return tier, TANK_WIN_PROB, "tank"
        return tier, shift_expectation(expected, self.hidden_delta), "play"


ARCHETYPES = {
    "noob": lambda **kw: HonestPlayer(profile="cautious", name="noob", **kw),
    "average": lambda **kw: HonestPlayer(profile="typical", name="average", **kw),
    "aggressive": lambda **kw: HonestPlayer(profile="aggressive", name="aggressive", **kw),
    "whale": lambda **kw: Whale(name="whale", **kw),
    "tilter": lambda **kw: Tilter(name="tilter", **kw),
    "shark": lambda **kw: Shark(name="shark", **kw),
}

# How the simulated lobby is composed. Sharks are added separately at whatever
# prevalence an experiment asks for.
HONEST_MIX = {"noob": 0.30, "average": 0.40, "aggressive": 0.18, "whale": 0.05, "tilter": 0.07}


#: A second game's crowd: far more cautious, almost no whales. Used to show what
#: transfers between titles and what does not - the shark's betting signature
#: carries over, the local betting culture emphatically does not.
CAUTIOUS_MIX = {"noob": 0.62, "average": 0.28, "aggressive": 0.05, "whale": 0.01, "tilter": 0.04}


def make_honest(rng, rating=1200.0, mix=None):
    """Draw one honest player from a lobby mix."""
    mix = mix or HONEST_MIX
    names = list(mix)
    probs = [mix[n] for n in names]
    total = sum(probs)
    return ARCHETYPES[names[rng.choice_p([p / total for p in probs])]](rating=rating)


def expected_for(player, opponent):
    return expected_score(player.rating, opponent.rating)
