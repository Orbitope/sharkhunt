"""The match loop: pair players, resolve wagers, move ratings, score detectors.

Two entry points. ``simulate_career`` follows one player under scrutiny through a
stream of opponents, which is what every widget in the article shows. ``simulate_field``
runs a whole lobby, which is what the false-positive numbers come from.
"""

from sharkhunt.detectors import Observation, detector_suite
from sharkhunt.elo import DEFAULT_K, expected_score
from sharkhunt.players import Shark, make_honest
from sharkhunt.wagers import payoff


class Career:
    """Everything one player's match history produced."""

    def __init__(self, player):
        self.player = player
        self.observations = []
        self.intents = []
        self.ratings = [player.rating]
        self.bankroll = [0.0]

    @property
    def n(self):
        return len(self.observations)

    def win_rate(self):
        if not self.observations:
            return 0.0
        return sum(o.won for o in self.observations) / self.n

    def profit_per_match(self):
        return self.bankroll[-1] / self.n if self.n else 0.0

    def rating_drift(self):
        return self.ratings[-1] - self.ratings[0]

    def summary(self):
        return {
            "archetype": self.player.archetype,
            "matches": self.n,
            "win_rate": self.win_rate(),
            "rating_start": self.ratings[0],
            "rating_end": self.ratings[-1],
            "rating_drift": self.rating_drift(),
            "profit_per_match": self.profit_per_match(),
        }


def opponent_rating(player_rating, rng, spread=80.0):
    """Matchmaking: an opponent near the player's shown rating."""
    return player_rating + spread * rng.normal()


def play_match(player, opp_rating, rng, k=DEFAULT_K):
    """One wagered match. Returns ``(Observation, intent)``."""
    expected = expected_score(player.rating, opp_rating)
    tier, win_prob, intent = player.decide(expected, rng)
    won = rng.chance(win_prob)
    player.settle(expected, tier, won, k)
    return Observation(expected, won, tier), intent


def simulate_career(
    player,
    matches,
    rng,
    detectors=None,
    k=DEFAULT_K,
    spread=80.0,
    **detector_kw,
):
    """Run one player through ``matches`` wagered games, scoring every detector.

    Returns ``(career, detectors)``. The detectors see only the Observation
    stream - expectation, result, bet tier - never the player's archetype, their
    hidden skill, or their per-match intent.
    """
    detectors = detectors if detectors is not None else detector_suite(**detector_kw)
    career = Career(player)
    for _ in range(matches):
        obs, intent = play_match(player, opponent_rating(player.rating, rng, spread), rng, k)
        career.observations.append(obs)
        career.intents.append(intent)
        career.ratings.append(player.rating)
        career.bankroll.append(player.bankroll)
        for det in detectors.values():
            det.update(obs)
    return career, detectors


def simulate_field(
    n_players,
    matches,
    rng,
    shark_fraction=0.0,
    shark_kw=None,
    rating=1200.0,
    **detector_kw,
):
    """Run a whole lobby and report how each detector scored every player.

    Used for the numbers that actually matter operationally: how many honest
    players get accused, and how many sharks get caught, at a fixed threshold.
    """
    shark_kw = shark_kw or {}
    rows = []
    for i in range(n_players):
        sub = rng.spawn(i + 1)
        is_shark = sub.chance(shark_fraction)
        player = Shark(rating=rating, **shark_kw) if is_shark else make_honest(sub, rating)
        career, dets = simulate_career(player, matches, sub, **detector_kw)
        rows.append(
            {
                "index": i,
                "is_shark": is_shark,
                "archetype": player.archetype,
                "profile": getattr(player, "profile_name", None),
                "summary": career.summary(),
                "scores": {name: d.score for name, d in dets.items()},
                "flagged_at": {name: d.flagged_at for name, d in dets.items()},
            }
        )
    return rows


def detection_rates(rows, detector):
    """False-positive and true-positive rates for one detector over a field."""
    honest = [r for r in rows if not r["is_shark"]]
    sharks = [r for r in rows if r["is_shark"]]
    fp = sum(r["flagged_at"][detector] is not None for r in honest)
    tp = sum(r["flagged_at"][detector] is not None for r in sharks)
    caught = [r["flagged_at"][detector] for r in sharks if r["flagged_at"][detector]]
    caught.sort()
    return {
        "detector": detector,
        "n_honest": len(honest),
        "n_sharks": len(sharks),
        "false_positive_rate": fp / len(honest) if honest else 0.0,
        "true_positive_rate": tp / len(sharks) if sharks else 0.0,
        "median_matches_to_catch": caught[len(caught) // 2] if caught else None,
    }


def false_positive_by_archetype(rows, detector):
    """Which kinds of honest player a detector accuses. The whale check."""
    out = {}
    for row in rows:
        if row["is_shark"]:
            continue
        key = row["archetype"]
        hit, total = out.get(key, (0, 0))
        out[key] = (hit + (row["flagged_at"][detector] is not None), total + 1)
    return {k: {"flagged": h, "n": n, "rate": h / n} for k, (h, n) in out.items()}


def shark_economics(hidden_delta, bet_correlation, matches, rng, tank_rate=None, k=DEFAULT_K):
    """Profit and time-to-detection for a shark at one setting of the dials.

    This is the trade the article closes on: correlating your bets with your
    intent is what makes the edge pay, and it is also the thing that gets you
    caught.
    """
    shark = Shark(
        hidden_delta=hidden_delta,
        bet_correlation=bet_correlation,
        tank_rate=tank_rate,
        rating=1200.0,
    )
    career, dets = simulate_career(shark, matches, rng)
    return {
        "bet_correlation": bet_correlation,
        "profit_per_match": career.profit_per_match(),
        "win_rate": career.win_rate(),
        "rating_drift": career.rating_drift(),
        "flagged_at": {name: d.flagged_at for name, d in dets.items()},
        "final_scores": {name: d.score for name, d in dets.items()},
    }


def honest_baseline_profit(rng, matches=2000, profile="typical"):
    """What an honest player at their true rating makes per match: the rake, negative."""
    from sharkhunt.players import HonestPlayer

    p = HonestPlayer(profile=profile, rating=1200.0)
    total = 0.0
    for _ in range(matches):
        expected = expected_score(p.rating, opponent_rating(p.rating, rng))
        tier = rng.choice_p(p.bet_profile(expected))
        won = rng.chance(expected)
        total += payoff(tier, won)
    return total / matches
