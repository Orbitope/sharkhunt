"""sharkhunt/rng.py and docs/rng.js must produce identical streams.

Skipped when node isn't installed, but on a machine that has it this is the
guard that stops the article's in-browser simulations from quietly diverging
from the numbers the Python experiments report.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

from sharkhunt.rng import Rng

ROOT = pathlib.Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

DRIVER = """
require(%s);
const seeds = [1, 42, 1234567, -9, 2147483647];
const out = {};
for (const s of seeds) {
  const r = new SHRng(s);
  const u = [], n = [], c = [], k = [];
  for (let i = 0; i < 12; i++) u.push(r.random());
  for (let i = 0; i < 6; i++) n.push(r.normal());
  for (let i = 0; i < 8; i++) c.push(r.choiceP([0.2, 0.3, 0.5]));
  for (let i = 0; i < 4; i++) k.push(r.spawn(i).random());
  out[s] = { u, n, c, k };
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_streams_match():
    driver = DRIVER % json.dumps(str(ROOT / "docs" / "rng.js"))
    proc = subprocess.run(
        [NODE, "-e", driver], capture_output=True, text=True, check=True
    )
    js = json.loads(proc.stdout)

    for seed_str, expected in js.items():
        seed = int(seed_str)
        r = Rng(seed)
        got_u = [r.random() for _ in range(12)]
        got_n = [r.normal() for _ in range(6)]
        got_c = [r.choice_p([0.2, 0.3, 0.5]) for _ in range(8)]
        got_k = [r.spawn(i).random() for i in range(4)]

        for a, b in zip(got_u, expected["u"]):
            assert a == pytest.approx(b, abs=0, rel=0), f"uniform drift at seed {seed}"
        for a, b in zip(got_n, expected["n"]):
            assert a == pytest.approx(b, rel=1e-12), f"normal drift at seed {seed}"
        assert got_c == expected["c"], f"categorical drift at seed {seed}"
        for a, b in zip(got_k, expected["k"]):
            assert a == pytest.approx(b, abs=0, rel=0), f"spawn drift at seed {seed}"


def test_uniform_and_categorical_are_well_behaved():
    r = Rng(7)
    draws = [r.random() for _ in range(20000)]
    assert all(0.0 <= d < 1.0 for d in draws)
    assert 0.49 < sum(draws) / len(draws) < 0.51

    r = Rng(11)
    probs = [0.2, 0.3, 0.5]
    counts = [0, 0, 0]
    for _ in range(20000):
        counts[r.choice_p(probs)] += 1
    for got, want in zip(counts, probs):
        assert abs(got / 20000 - want) < 0.02


def test_spawned_streams_are_distinct():
    parent = Rng(3)
    heads = [parent.spawn(i).random() for i in range(16)]
    assert len(set(heads)) == len(heads)
