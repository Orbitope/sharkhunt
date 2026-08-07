# Hunting sharks

Detecting rating manipulators in wagered matchmaking, and the article that
explains it: [orbitope.github.io/sharkhunt](https://orbitope.github.io/sharkhunt/).

A player who is stronger than their rating and throws matches to keep it that
way is **provably invisible** to any test that looks only at wins and losses.
There is a tank rate at which their outcome distribution is identical to an
honest player's, so the evidence per match is exactly zero at any sample size.
What they wager gives them away instead.

## What's here

```
sharkhunt/          the detectors and the population model
  elo.py            Elo expectations; the tank rate that pins a rating
  wagers.py         bet tiers, betting habits, the Dirichlet prior
  players.py        archetypes - honest, whale, tilter, shark
  detectors.py      five sequential detectors, in the order the article builds them
  engine.py         match loop, ratings, careers, field studies
  hierarchical.py   the PyMC model that learns the shark hypothesis, unlabelled
  experiments.py    the cheap numbers            -> analysis/results.json
  fit_experiments.py  the MCMC ones              -> analysis/fits.json
  export.py         posterior means              -> docs/data.js

docs/               the article: index.html + rng.js + sharkhunt.js + widgets.js
```

`docs/sharkhunt.js` is a port of the Python package so the article's widgets can
re-run the simulations live rather than replay stored results.
`tests/test_js_parity.py` drives both from the same seeds and compares whole
careers match by match.

## The five detectors

| | evidence used | catches hiding sharks | accuses honest players |
|---|---|---|---|
| `OutcomeSPRT` | wins and losses | 0.0% | 0.84% |
| `WeightedLLR` | scaled by stake | 54.9% | 0.10% |
| `JointLLR` | bet and outcome jointly, population habits | 100% | 7.18% |
| `HierarchicalJointLLR` | ... with per-player habits | 100% | 0.52% |
| `MixtureJointLLR` | ... averaged over shark strategies | 100% | 0.63% |

2,000 players, 300 matches each, 5% sharks. Only the last three are genuine
likelihood ratios; `WeightedLLR` is included because it is the intuitive fix and
breaking it is instructive.

## Running it

```bash
uv venv && uv pip install -e ".[dev]"
```

Tests, including the differential test against PettingZoo and the Python/JS
parity check (needs `node`):

```bash
pytest
```

The MCMC fits are marked slow and skipped by default:

```bash
pytest -m slow
```

Reproduce the article's numbers. The detector experiments take under a minute;
the population fits take tens of minutes.

```bash
python -m sharkhunt.experiments --out analysis/results.json
python -m sharkhunt.fit_experiments --out analysis/fits.json
python -m sharkhunt.export --out docs/data.js
```

Then serve the article from any static file server:

```bash
python -m http.server -d docs 8899
```

## Caveats

The players are simulated and the adversary doesn't adapt. The article's
[closing section](https://orbitope.github.io/sharkhunt/#caveats) says what
that does and doesn't license you to conclude.

Every detector here needs a dozen-plus matches, so none of it places a brand new
account. That problem — reading skill from behaviour rather than results, using
RL checkpoints as the labelled training data — is its own project:
[coldopen](https://github.com/orbitope/coldopen).
