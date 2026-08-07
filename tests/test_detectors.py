"""Every claim the article makes about the detectors, as an assertion.

If a test in here fails, a sentence in docs/index.html is wrong and has to
change. That is the point of the file.
"""

import math

import pytest

from sharkhunt.detectors import (
    DEFAULT_DELTA,
    TANK_WIN_PROB,
    HierarchicalJointLLR,
    JointLLR,
    Observation,
    OutcomeSPRT,
    WeightedLLR,
    detector_suite,
    joint_cell_table,
)
from sharkhunt.elo import expected_score, pinning_tank_rate, shift_expectation, wald_thresholds
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
from sharkhunt.wagers import N_TIERS, STAKES, FixedProfile, population_profile

LOWER, UPPER = wald_thresholds()


# --- the maths the article states -----------------------------------------


def test_shift_expectation_matches_rating_arithmetic():
    """The closed form must agree with actually moving the rating."""
    for ra, rb, delta in [(1200, 1200, 400), (1000, 1350, 250), (1600, 1450, -120)]:
        direct = expected_score(ra + delta, rb)
        via_closed_form = shift_expectation(expected_score(ra, rb), delta)
        assert via_closed_form == pytest.approx(direct, rel=1e-12)


@pytest.mark.parametrize("expected", [0.3, 0.5, 0.65, 0.8])
def test_pinning_tank_rate_makes_outcome_evidence_exactly_zero(expected):
    """The blind spot is exact, not approximate.

    At the pinning tank rate the shark's win probability under H1 equals the win
    probability under H0, so every single outcome-only increment is zero - for a
    win and for a loss alike. No amount of data helps.
    """
    tau = pinning_tank_rate(expected, DEFAULT_DELTA)
    det = OutcomeSPRT(delta=DEFAULT_DELTA, tau=tau)
    assert det.win_prob_h1(expected) == pytest.approx(expected, abs=1e-12)
    for won in (True, False):
        assert det.increment(Observation(expected, won, 1)) == pytest.approx(0.0, abs=1e-12)


def test_joint_model_is_a_real_likelihood_ratio():
    """Both hypotheses must be proper distributions over the six cells.

    This is what separates JointLLR from WeightedLLR: because numerator and
    denominator each sum to one, the ratio is a true likelihood ratio and Wald's
    thresholds carry their advertised error rates.
    """
    pi = population_profile()
    det = JointLLR(profile=FixedProfile(pi))
    for expected in (0.35, 0.5, 0.72):
        h0 = h1 = 0.0
        for tier in range(N_TIERS):
            for won in (False, True):
                obs = Observation(expected, won, tier)
                h0 += det._p_h0(obs, pi)
                h1 += det._p_h1(obs)
        assert h0 == pytest.approx(1.0, abs=1e-12)
        assert h1 == pytest.approx(1.0, abs=1e-12)


def test_cheap_loss_incriminates_under_joint_and_is_ignored_when_weighted():
    """The single cell where the two detectors most disagree.

    A minimum-bet loss is exactly what a shark throwing a match cheaply looks
    like, so the joint model scores it as positive evidence. The wager-weighted
    score multiplies its outcome term by the smallest stake, driving it to
    roughly nothing - it throws the tell away.
    """
    expected = 0.5
    tau = pinning_tank_rate(expected, DEFAULT_DELTA)
    table = joint_cell_table(expected, tau=tau)
    min_loss, min_win = table[0]
    max_loss, max_win = table[2]

    assert min_loss > 0.5, "a cheap loss should incriminate"
    assert max_win > 0.5, "an expensive win should incriminate"
    assert min_win < 0, "sharks do not win the matches they bet nothing on"
    assert max_loss < 0, "sharks do not lose at full stake"

    weighted = WeightedLLR(delta=DEFAULT_DELTA, tau=tau)
    cheap_loss = weighted.increment(Observation(expected, False, 0))
    assert abs(cheap_loss) < 1e-9, "at the pinning rate the weighted score sees nothing"

    # And the weighting itself is what discards it: the stake multiplier on the
    # cheapest tier is a tenth of the one on the most expensive.
    assert STAKES[0] / STAKES[2] == pytest.approx(0.1)


# --- behaviour over simulated careers --------------------------------------


def run(player, matches=400, seed=1, **kw):
    return simulate_career(player, matches, Rng(seed), **kw)


def test_honest_player_score_stays_between_the_thresholds():
    """Across many seeds an honest player should almost never be accused."""
    flagged = 0
    for seed in range(60):
        _, dets = run(HonestPlayer(profile="typical"), matches=400, seed=seed)
        if dets["hierarchical"].flagged:
            flagged += 1
    assert flagged <= 3, f"{flagged}/60 honest players accused - alpha is not holding"


def test_a_shark_who_does_not_tank_is_caught_by_outcomes_alone():
    """The easy case. Outcome-only testing is not useless; it is incomplete."""
    caught = 0
    for seed in range(20):
        _, dets = run(Shark(hidden_delta=400, tank_rate=0.0), matches=200, seed=seed)
        caught += dets["outcome"].flagged
    assert caught >= 18, "an unconcealed shark should be trivially detectable"


def test_pinned_shark_is_not_merely_missed_but_actively_cleared():
    """The article's central comparison, and it is worse than "the test is slow".

    A shark tanking at the pinning rate wins exactly as often as their shown
    rating predicts. An outcome-only test therefore never accumulates evidence
    against them - and because they slightly *under*perform the specific
    alternative it is testing, the score drifts the wrong way and the test
    terminates by declaring them clean. Their bets still give them away.
    """
    outcome_caught = outcome_cleared = joint_caught = 0
    trials = 60
    for seed in range(trials):
        _, dets = run(Shark(hidden_delta=400, bet_correlation=1.0), matches=400, seed=seed)
        outcome_caught += dets["outcome"].flagged
        outcome_cleared += min(dets["outcome"].history) <= LOWER
        joint_caught += dets["hierarchical"].flagged

    assert outcome_caught <= 2, f"outcome test flagged {outcome_caught}/{trials} - should be blind"
    assert outcome_cleared > trials * 0.7, (
        f"only {outcome_cleared}/{trials} pinned sharks were actively exonerated"
    )
    assert joint_caught >= trials - 3, f"joint detector only caught {joint_caught}/{trials}"


def test_pinned_shark_rating_really_does_hold_still():
    """The premise: tanking at tau* keeps them parked in a soft bracket."""
    drifts = []
    for seed in range(20):
        career, _ = run(Shark(hidden_delta=400, bet_correlation=1.0), matches=400, seed=seed)
        drifts.append(career.rating_drift())
    assert abs(sum(drifts) / len(drifts)) < 60, "the shark's rating should stay put"


# --- the ways the naive fix breaks -----------------------------------------


def test_weighted_score_threshold_means_different_things_to_different_bettors():
    """Why "not a likelihood ratio" is a practical problem, not a purist's one.

    Scaling each term by the stake rescales the whole random walk, so how far a
    player's score travels depends on how they bet rather than on how they play.
    At one fixed pair of thresholds a cautious bettor's test essentially never
    terminates, while a whale's almost always does - on identical, honest
    behaviour. There is no single threshold that is right for both, which means
    there is no error rate to quote.
    """
    def resolution_rate(make):
        resolved = 0
        for seed in range(120):
            _, dets = run(make(), matches=400, seed=seed + 500)
            d = dets["weighted"]
            resolved += (min(d.history) <= LOWER) or d.flagged
        return resolved / 120

    cautious = resolution_rate(lambda: HonestPlayer(profile="cautious"))
    whale = resolution_rate(Whale)
    assert cautious < 0.15, f"cautious bettors should stall unresolved, got {cautious:.1%}"
    assert whale > 0.70, f"whales should resolve fast, got {whale:.1%}"
    assert whale > cautious * 4, "the gap should be dramatic, not marginal"


def test_joint_detector_on_population_habits_accuses_every_whale():
    """The failure that forces the hierarchical layer.

    JointLLR compares a player's bets against the *field's* betting habits. A
    whale never makes a small bet, which the field does constantly, so match
    after match they look mildly more like the shark hypothesis than the honest
    one - and a mild bias, accumulated sequentially, convicts everybody.
    Giving each player their own learned profile removes it entirely.
    """
    trials = 100
    population_flags = hierarchical_flags = 0
    for seed in range(trials):
        _, dets = run(Whale(), matches=400, seed=seed + 500)
        population_flags += dets["joint"].flagged
        hierarchical_flags += dets["hierarchical"].flagged

    assert population_flags / trials > 0.9, "the population-profile detector should convict whales"
    assert hierarchical_flags / trials <= 0.02, (
        f"the hierarchical detector still flags {hierarchical_flags}/{trials} whales"
    )


def test_mixture_over_shark_strategies_covers_both_styles():
    """A single assumed tank rate leaves a hole; averaging over tank rates closes it.

    The detector tuned for a concealed shark scores a blatant one - somebody who
    never throws a match at all - as unremarkable, because their bets carry none
    of the tanking the hypothesis expects. Marginalising over a grid of tank
    rates catches both without giving up the false-positive rate.
    """
    def catch_rate(make, name, trials=60, seed0=0):
        hits = 0
        for seed in range(trials):
            _, dets = run(make(), matches=400, seed=seed + seed0)
            hits += dets[name].flagged
        return hits / trials

    blatant = lambda: Shark(hidden_delta=400, tank_rate=0.0)  # noqa: E731
    pinned = lambda: Shark(hidden_delta=400, bet_correlation=1.0)  # noqa: E731

    assert catch_rate(blatant, "hierarchical") < 0.95, "the gap this test is about"
    assert catch_rate(blatant, "mixture") > 0.97
    assert catch_rate(pinned, "mixture") > 0.97
    assert catch_rate(Whale, "mixture", trials=100, seed0=500) <= 0.03
    assert catch_rate(HonestPlayer, "mixture", trials=100, seed0=500) <= 0.03


def test_tilter_is_not_mistaken_for_a_shark():
    """A confound with real correlation in it, just the wrong correlation.

    A tilter's bet size tracks the *previous* match's result. A shark's tracks the
    one about to happen. If the detector cannot tell those apart it is flagging
    people for chasing losses.
    """
    flagged = 0
    for seed in range(80):
        _, dets = run(Tilter(), matches=400, seed=seed + 900)
        flagged += dets["hierarchical"].flagged
    assert flagged / 80 <= 0.05, f"tilters flagged at {flagged / 80:.1%}"


def test_weighted_thresholds_are_not_calibrated_and_say_so():
    assert WeightedLLR.calibrated is False
    assert JointLLR.calibrated is True
    assert OutcomeSPRT.calibrated is True


# --- the economics ---------------------------------------------------------


def test_decorrelating_bets_evades_detection_and_destroys_the_edge():
    """The closing argument.

    A shark can beat the joint detector by refusing to let bet size track intent
    - but that means posting full stake on the matches they intend to throw. The
    detector does not have to catch them. It has to make hiding cost more than
    cheating pays.
    """
    correlated = shark_economics(400, 1.0, 400, Rng(3))
    decorrelated = shark_economics(400, 0.0, 400, Rng(3))

    assert correlated["flagged_at"]["hierarchical"] is not None
    assert decorrelated["flagged_at"]["hierarchical"] is None
    assert correlated["profit_per_match"] > 0, "cheating should pay when unconstrained"
    assert decorrelated["profit_per_match"] < correlated["profit_per_match"] / 3, (
        f"evasion cost too little: {decorrelated['profit_per_match']:.4f} "
        f"vs {correlated['profit_per_match']:.4f}"
    )


def test_evasion_drags_profit_down_to_roughly_the_honest_baseline():
    """Honest players lose the rake. A fully hidden shark should end up near it."""
    baseline = honest_baseline_profit(Rng(17))
    decorrelated = shark_economics(400, 0.0, 1500, Rng(23))["profit_per_match"]
    assert baseline < 0, "the house edge should make honest play a slow loss"
    assert decorrelated < abs(baseline) * 2, "hiding should leave almost no edge"


def test_profit_falls_monotonically_as_bets_decorrelate():
    xs = [1.0, 0.75, 0.5, 0.25, 0.0]
    profits = [shark_economics(400, x, 1200, Rng(41))["profit_per_match"] for x in xs]
    assert profits == sorted(profits, reverse=True), f"not monotone: {profits}"


# --- the population layer ---------------------------------------------------


def test_prior_strength_trades_shark_camouflage_against_whale_false_alarms():
    """Both ends of the Dirichlet dial are bad, which is why it is a dial.

    Weak prior: each player is judged against their own history, so a shark who
    establishes an unusual betting habit is scored as normal for them. Strong
    prior: everyone is judged against the field, and honest players with unusual
    habits get accused.
    """
    def rates(strength):
        shark_hits = whale_hits = 0
        for seed in range(40):
            _, d = simulate_career(
                Shark(hidden_delta=400, bet_correlation=1.0), 300, Rng(seed),
                detectors={"h": HierarchicalJointLLR(strength=strength)},
            )
            shark_hits += d["h"].flagged
            _, d = simulate_career(
                Whale(), 300, Rng(seed + 700),
                detectors={"h": HierarchicalJointLLR(strength=strength)},
            )
            whale_hits += d["h"].flagged
        return shark_hits / 40, whale_hits / 40

    weak_tpr, weak_fpr = rates(0.5)
    huge_tpr, huge_fpr = rates(400.0)
    assert huge_fpr > weak_fpr, "a very strong prior should start accusing whales"
    assert weak_tpr > 0.5, "even a weak prior should still catch a blatant shark"


def test_field_simulation_reports_sane_rates():
    rows = simulate_field(300, 250, Rng(5), shark_fraction=0.1,
                          shark_kw={"hidden_delta": 400, "bet_correlation": 1.0})
    joint = detection_rates(rows, "hierarchical")
    outcome = detection_rates(rows, "outcome")

    assert joint["n_sharks"] > 10, "need enough sharks for the rates to mean anything"
    assert joint["true_positive_rate"] > 0.85
    assert joint["false_positive_rate"] < 0.05
    assert outcome["true_positive_rate"] < joint["true_positive_rate"]

    by_type = false_positive_by_archetype(rows, "joint")
    assert by_type["whale"]["rate"] > by_type["honest"]["rate"], (
        "the population-profile detector's false positives should land on whales"
    )
    hier_by_type = false_positive_by_archetype(rows, "hierarchical")
    assert hier_by_type["whale"]["rate"] <= 0.05, "the hierarchical layer should clear them"


def test_detector_suite_scores_one_stream_consistently():
    """All four detectors must see the identical match stream."""
    career, dets = run(HonestPlayer(), matches=50, seed=2)
    assert len(career.observations) == 50
    for det in dets.values():
        assert det.n == 50
        assert len(det.history) == 51
    assert math.isfinite(sum(d.score for d in dets.values()))
