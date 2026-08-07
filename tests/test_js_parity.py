"""docs/sharkhunt.js must reproduce the Python package exactly.

The article's widgets re-run these simulations live rather than replaying stored
results, so a divergence here would mean the figures a reader plays with and the
numbers the prose quotes come from two different models. Whole careers are
compared match by match, not just summary statistics.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

from sharkhunt.detectors import detector_suite, joint_cell_table
from sharkhunt.elo import pinning_tank_rate, shift_expectation
from sharkhunt.engine import simulate_career
from sharkhunt.players import HonestPlayer, Shark, Tilter, Whale
from sharkhunt.rng import Rng

ROOT = pathlib.Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
DETECTORS = ["outcome", "weighted", "joint", "hierarchical", "mixture"]

CASES = [
    ("honest-typical", {"kind": "honest", "profile": "typical"}),
    ("honest-cautious", {"kind": "honest", "profile": "cautious"}),
    ("whale", {"kind": "whale"}),
    ("tilter", {"kind": "tilter"}),
    ("shark-pinned", {"kind": "shark", "hiddenDelta": 400, "betCorrelation": 1.0}),
    ("shark-hidden", {"kind": "shark", "hiddenDelta": 400, "betCorrelation": 0.0}),
    ("shark-blatant", {"kind": "shark", "hiddenDelta": 400, "tankRate": 0.0}),
]

DRIVER = """
require(%(rng)s);
require(%(sh)s);
var SH = globalThis.SharkHunt;
var cases = %(cases)s;
var out = {};
function build(spec) {
  if (spec.kind === 'honest') return new SH.HonestPlayer(spec.profile, 1200);
  if (spec.kind === 'whale') return new SH.Whale(1200);
  if (spec.kind === 'tilter') return new SH.Tilter(1200);
  return new SH.Shark({
    rating: 1200,
    hiddenDelta: spec.hiddenDelta,
    betCorrelation: spec.betCorrelation,
    tankRate: spec.tankRate === undefined ? null : spec.tankRate
  });
}
for (var i = 0; i < cases.length; i++) {
  var name = cases[i][0], spec = cases[i][1];
  var res = SH.simulateCareer(build(spec), %(matches)d, new SHRng(%(seed)d));
  var obs = [];
  for (var j = 0; j < res.career.observations.length; j++) {
    var o = res.career.observations[j];
    obs.push([o.expected, o.won ? 1 : 0, o.tier]);
  }
  var scores = {}, flags = {}, hist = {};
  for (var k in res.detectors) {
    if (!res.detectors.hasOwnProperty(k)) continue;
    scores[k] = res.detectors[k].score;
    flags[k] = res.detectors[k].flaggedAt;
    hist[k] = res.detectors[k].history;
  }
  out[name] = {
    observations: obs,
    ratings: res.career.ratings,
    bankroll: res.career.bankroll,
    intents: res.career.intents,
    scores: scores, flags: flags, history: hist
  };
}
out.__scalars = {
  pinning: SH.pinningTankRate(0.5, 400),
  shifted: SH.shiftExpectation(0.5, 400),
  cellTable: SH.jointCellTable(0.5, 400, SH.pinningTankRate(0.5, 400)),
  population: SH.populationProfile(),
  thresholds: SH.waldThresholds()
};
console.log(JSON.stringify(out));
"""


def _build_player(spec):
    if spec["kind"] == "honest":
        return HonestPlayer(profile=spec["profile"], rating=1200.0)
    if spec["kind"] == "whale":
        return Whale(rating=1200.0)
    if spec["kind"] == "tilter":
        return Tilter(rating=1200.0)
    return Shark(
        rating=1200.0,
        hidden_delta=spec["hiddenDelta"],
        bet_correlation=spec.get("betCorrelation", 1.0),
        tank_rate=spec.get("tankRate"),
    )


@pytest.fixture(scope="module")
def js_results():
    if NODE is None:
        pytest.skip("node not installed")
    driver = DRIVER % {
        "rng": json.dumps(str(ROOT / "docs" / "rng.js")),
        "sh": json.dumps(str(ROOT / "docs" / "sharkhunt.js")),
        "cases": json.dumps(CASES),
        "matches": 250,
        "seed": 12345,
    }
    proc = subprocess.run([NODE, "-e", driver], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"node driver failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


@pytest.mark.parametrize("name,spec", CASES)
def test_career_matches_match_for_match(js_results, name, spec):
    expected = js_results[name]
    career, dets = simulate_career(_build_player(spec), 250, Rng(12345))

    assert len(career.observations) == len(expected["observations"])
    for i, (obs, js) in enumerate(zip(career.observations, expected["observations"])):
        assert obs.expected == pytest.approx(js[0], rel=1e-12), f"{name} expectation at {i}"
        assert int(obs.won) == js[1], f"{name} outcome at match {i}"
        assert obs.tier == js[2], f"{name} bet tier at match {i}"

    assert career.intents == expected["intents"], f"{name} private intents diverged"
    for i, (r, js) in enumerate(zip(career.ratings, expected["ratings"])):
        assert r == pytest.approx(js, rel=1e-12), f"{name} rating at {i}"
    for i, (b, js) in enumerate(zip(career.bankroll, expected["bankroll"])):
        assert b == pytest.approx(js, rel=1e-12, abs=1e-12), f"{name} bankroll at {i}"


@pytest.mark.parametrize("name,spec", CASES)
def test_every_detector_traces_identically(js_results, name, spec):
    expected = js_results[name]
    _, dets = simulate_career(_build_player(spec), 250, Rng(12345))

    for det_name in DETECTORS:
        det = dets[det_name]
        assert det.score == pytest.approx(expected["scores"][det_name], rel=1e-11), (
            f"{name}/{det_name} final score"
        )
        assert det.flagged_at == expected["flags"][det_name], (
            f"{name}/{det_name} flagged at a different match"
        )
        for i, (a, b) in enumerate(zip(det.history, expected["history"][det_name])):
            assert a == pytest.approx(b, rel=1e-11, abs=1e-12), (
                f"{name}/{det_name} running score at match {i}"
            )


def test_scalar_helpers_agree(js_results):
    s = js_results["__scalars"]
    assert pinning_tank_rate(0.5, 400) == pytest.approx(s["pinning"], rel=1e-12)
    assert shift_expectation(0.5, 400) == pytest.approx(s["shifted"], rel=1e-12)

    table = joint_cell_table(0.5, tau=pinning_tank_rate(0.5, 400))
    for t, (row, js_row) in enumerate(zip(table, s["cellTable"])):
        for o, (a, b) in enumerate(zip(row, js_row)):
            assert a == pytest.approx(b, rel=1e-11), f"cell table tier {t} outcome {o}"


def test_detector_suites_have_the_same_members(js_results):
    assert set(js_results["honest-typical"]["scores"]) == set(detector_suite())
