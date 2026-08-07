/* Port of the sharkhunt Python package, for the widgets on this page.

   The figures below are not pictures of results computed somewhere else - they
   re-run the simulation in your browser as you drag the controls. That only
   means anything if this file agrees with the package the offline experiments
   used, so tests/test_js_parity.py drives both from the same seeds and compares
   whole careers, match by match.

   Kept in ES5 style with the same function names and the same order of random
   draws as the Python, because "same order of random draws" is the entire
   contract. */
(function (global) {
  'use strict';

  var TIER_NAMES = ['min', 'mid', 'max'];
  var N_TIERS = 3;
  var STAKES = [0.10, 0.40, 1.00];
  var RAKE = 0.05;
  var TANK_WIN_PROB = 0.02;
  var DEFAULT_DELTA = 400.0;
  var DEFAULT_TAU = 0.40;
  var DEFAULT_K = 24.0;
  var EPS = 1e-12;

  var PROFILES = {
    cautious: [0.70, 0.25, 0.05],
    typical: [0.20, 0.55, 0.25],
    aggressive: [0.10, 0.30, 0.60],
    whale: [0.00, 0.00, 1.00]
  };
  var SHARK_PLAY_PROFILE = [0.05, 0.25, 0.70];
  var SHARK_TANK_PROFILE = [0.75, 0.20, 0.05];
  var POP_WEIGHTS = { cautious: 0.35, typical: 0.45, aggressive: 0.15, whale: 0.05 };
  var HONEST_MIX = { noob: 0.30, average: 0.40, aggressive: 0.18, whale: 0.05, tilter: 0.07 };

  function populationProfile() {
    var out = [0, 0, 0], total = 0, k;
    for (k in POP_WEIGHTS) if (POP_WEIGHTS.hasOwnProperty(k)) total += POP_WEIGHTS[k];
    for (k in POP_WEIGHTS) {
      if (!POP_WEIGHTS.hasOwnProperty(k)) continue;
      for (var t = 0; t < N_TIERS; t++) out[t] += (POP_WEIGHTS[k] / total) * PROFILES[k][t];
    }
    return out;
  }

  function blend(a, b, weight) {
    var out = [];
    for (var t = 0; t < N_TIERS; t++) out.push(weight * a[t] + (1 - weight) * b[t]);
    return out;
  }

  function payoff(tier, won) {
    var s = STAKES[tier];
    return won ? s * (1 - 2 * RAKE) : -s;
  }

  /* ---- Elo ---- */

  function expectedScore(rating, opponentRating) {
    return 1 / (1 + Math.pow(10, (opponentRating - rating) / 400));
  }

  function shiftExpectation(expected, delta) {
    var g = Math.pow(10, -delta / 400);
    return expected / (expected + (1 - expected) * g);
  }

  function updateRating(rating, expected, score, k) {
    return rating + (k === undefined ? DEFAULT_K : k) * (score - expected);
  }

  function pinningTankRate(expected, delta, tankWinProb) {
    var eps = tankWinProb === undefined ? TANK_WIN_PROB : tankWinProb;
    var eDelta = shiftExpectation(expected, delta);
    if (eDelta <= eps) return 0;
    return Math.min(Math.max((eDelta - expected) / (eDelta - eps), 0), 1);
  }

  function waldThresholds(alpha, beta) {
    alpha = alpha === undefined ? 0.01 : alpha;
    beta = beta === undefined ? 0.01 : beta;
    return [Math.log(beta / (1 - alpha)), Math.log((1 - beta) / alpha)];
  }

  /* ---- betting profiles ---- */

  function FixedProfile(probs) { this.probs = probs.slice(); }
  FixedProfile.prototype.predict = function () { return this.probs; };
  FixedProfile.prototype.observe = function () {};

  function DirichletProfile(population, strength) {
    this.population = population.slice();
    this.strength = strength === undefined ? 8.0 : strength;
    this.counts = [0, 0, 0];
    this.nObs = 0;
  }
  DirichletProfile.prototype.predict = function () {
    var denom = this.nObs + this.strength, out = [];
    for (var t = 0; t < N_TIERS; t++) {
      out.push((this.counts[t] + this.strength * this.population[t]) / denom);
    }
    return out;
  };
  DirichletProfile.prototype.observe = function (tier) {
    this.counts[tier] += 1;
    this.nObs += 1;
  };

  /* ---- detectors ---- */

  function Observation(expected, won, tier) {
    this.expected = expected;
    this.won = !!won;
    this.tier = tier | 0;
  }

  function Detector(opts) {
    opts = opts || {};
    var th = waldThresholds(opts.alpha, opts.beta);
    this.lower = th[0];
    this.upper = th[1];
    this.score = 0;
    this.history = [0];
    this.flaggedAt = null;
    this.n = 0;
    this.delta = opts.delta === undefined ? DEFAULT_DELTA : opts.delta;
    this.tau = opts.tau === undefined ? DEFAULT_TAU : opts.tau;
  }
  Detector.prototype.update = function (obs) {
    var inc = this.increment(obs);
    this.score += inc;
    this.n += 1;
    this.history.push(this.score);
    if (this.flaggedAt === null && this.score >= this.upper) this.flaggedAt = this.n;
    return inc;
  };
  Detector.prototype.flagged = function () { return this.flaggedAt !== null; };
  Detector.prototype.cleared = function () {
    for (var i = 0; i < this.history.length; i++) if (this.history[i] <= this.lower) return true;
    return false;
  };

  function inherit(Child, Parent) {
    Child.prototype = Object.create(Parent.prototype);
    Child.prototype.constructor = Child;
  }

  function OutcomeSPRT(opts) { Detector.call(this, opts); }
  inherit(OutcomeSPRT, Detector);
  OutcomeSPRT.prototype.name = 'outcome';
  OutcomeSPRT.prototype.calibrated = true;
  OutcomeSPRT.prototype.winProbH1 = function (expected) {
    var eDelta = shiftExpectation(expected, this.delta);
    return (1 - this.tau) * eDelta + this.tau * TANK_WIN_PROB;
  };
  OutcomeSPRT.prototype.increment = function (obs) {
    var p0 = Math.min(Math.max(obs.expected, EPS), 1 - EPS);
    var p1 = Math.min(Math.max(this.winProbH1(obs.expected), EPS), 1 - EPS);
    return obs.won ? Math.log(p1 / p0) : Math.log((1 - p1) / (1 - p0));
  };

  function WeightedLLR(opts) { OutcomeSPRT.call(this, opts); }
  inherit(WeightedLLR, OutcomeSPRT);
  WeightedLLR.prototype.name = 'weighted';
  WeightedLLR.prototype.calibrated = false;
  WeightedLLR.prototype.increment = function (obs) {
    return OutcomeSPRT.prototype.increment.call(this, obs) * STAKES[obs.tier] / STAKES[2];
  };

  function JointLLR(opts) {
    opts = opts || {};
    Detector.call(this, opts);
    this.playProfile = (opts.playProfile || SHARK_PLAY_PROFILE).slice();
    this.tankProfile = (opts.tankProfile || SHARK_TANK_PROFILE).slice();
    this.profile = opts.profile || new FixedProfile(populationProfile());
  }
  inherit(JointLLR, Detector);
  JointLLR.prototype.name = 'joint';
  JointLLR.prototype.calibrated = true;
  JointLLR.prototype.pH0 = function (obs, pi) {
    return Math.max(pi[obs.tier] * (obs.won ? obs.expected : 1 - obs.expected), EPS);
  };
  JointLLR.prototype.pH1One = function (obs, delta, tau) {
    var eDelta = shiftExpectation(obs.expected, delta);
    var playOut = obs.won ? eDelta : 1 - eDelta;
    var tankOut = obs.won ? TANK_WIN_PROB : 1 - TANK_WIN_PROB;
    return tau * this.tankProfile[obs.tier] * tankOut
      + (1 - tau) * this.playProfile[obs.tier] * playOut;
  };
  JointLLR.prototype.pH1 = function (obs) {
    return Math.max(this.pH1One(obs, this.delta, this.tau), EPS);
  };
  JointLLR.prototype.increment = function (obs) {
    var pi = this.profile.predict();
    var inc = Math.log(this.pH1(obs) / this.pH0(obs, pi));
    this.profile.observe(obs.tier);
    return inc;
  };

  function HierarchicalJointLLR(opts) {
    opts = opts || {};
    JointLLR.call(this, opts);
    this.profile = new DirichletProfile(
      opts.population || populationProfile(),
      opts.strength === undefined ? 8.0 : opts.strength
    );
  }
  inherit(HierarchicalJointLLR, JointLLR);
  HierarchicalJointLLR.prototype.name = 'hierarchical';

  var DEFAULT_TAU_GRID = [0.0, 0.15, 0.30, 0.45, 0.60];
  var DEFAULT_DELTA_GRID = [200.0, 400.0, 700.0];

  function defaultGrid() {
    var out = [];
    for (var i = 0; i < DEFAULT_DELTA_GRID.length; i++) {
      for (var j = 0; j < DEFAULT_TAU_GRID.length; j++) {
        out.push([DEFAULT_DELTA_GRID[i], DEFAULT_TAU_GRID[j]]);
      }
    }
    return out;
  }

  function MixtureJointLLR(opts) {
    opts = opts || {};
    HierarchicalJointLLR.call(this, opts);
    this.components = opts.components || defaultGrid();
    var w = opts.weights, total = 0, i;
    if (!w) {
      w = [];
      for (i = 0; i < this.components.length; i++) w.push(1 / this.components.length);
    }
    for (i = 0; i < w.length; i++) total += w[i];
    this.weights = [];
    for (i = 0; i < w.length; i++) this.weights.push(w[i] / total);
  }
  inherit(MixtureJointLLR, HierarchicalJointLLR);
  MixtureJointLLR.prototype.name = 'mixture';
  MixtureJointLLR.prototype.pH1 = function (obs) {
    var total = 0;
    for (var k = 0; k < this.components.length; k++) {
      total += this.weights[k] * this.pH1One(obs, this.components[k][0], this.components[k][1]);
    }
    return Math.max(total, EPS);
  };

  function detectorSuite(opts) {
    return {
      outcome: new OutcomeSPRT(opts),
      weighted: new WeightedLLR(opts),
      joint: new JointLLR(opts),
      hierarchical: new HierarchicalJointLLR(opts),
      mixture: new MixtureJointLLR(opts)
    };
  }

  function jointCellTable(expected, delta, tau, pi) {
    pi = pi || populationProfile();
    var det = new JointLLR({ delta: delta, tau: tau, profile: new FixedProfile(pi) });
    var table = [];
    for (var tier = 0; tier < N_TIERS; tier++) {
      var row = [];
      var outcomes = [false, true];
      for (var i = 0; i < 2; i++) {
        var obs = new Observation(expected, outcomes[i], tier);
        row.push(Math.log(det.pH1(obs) / det.pH0(obs, pi)));
      }
      table.push(row);
    }
    return table;
  }

  /* ---- players ---- */

  function Player(rating, name) {
    this.rating = rating === undefined ? 1200 : rating;
    this.name = name || 'player';
    this.bankroll = 0;
    this.matches = 0;
    this.lastWon = null;
    this.hiddenDelta = 0;
  }
  Player.prototype.archetype = 'player';
  Player.prototype.settle = function (expected, tier, won, k) {
    this.bankroll += payoff(tier, won);
    this.rating = updateRating(this.rating, expected, won ? 1 : 0, k);
    this.matches += 1;
    this.lastWon = won;
  };

  function HonestPlayer(profile, rating, name) {
    Player.call(this, rating, name || 'honest');
    this.profileName = profile || 'typical';
    this.profile = PROFILES[this.profileName];
  }
  inherit(HonestPlayer, Player);
  HonestPlayer.prototype.archetype = 'honest';
  HonestPlayer.prototype.betProfile = function () { return this.profile; };
  HonestPlayer.prototype.decide = function (expected, rng) {
    return { tier: rng.choiceP(this.betProfile(expected)), winProb: expected, intent: 'play' };
  };

  function Whale(rating) { HonestPlayer.call(this, 'whale', rating, 'whale'); }
  inherit(Whale, HonestPlayer);
  Whale.prototype.archetype = 'whale';

  function Tilter(rating, chase) {
    HonestPlayer.call(this, 'typical', rating, 'tilter');
    this.chase = chase === undefined ? 0.75 : chase;
  }
  inherit(Tilter, HonestPlayer);
  Tilter.prototype.archetype = 'tilter';
  Tilter.prototype.betProfile = function () {
    if (this.lastWon === false) return blend(PROFILES.aggressive, this.profile, this.chase);
    return this.profile;
  };

  function Shark(opts) {
    opts = opts || {};
    Player.call(this, opts.rating, 'shark');
    this.hiddenDelta = opts.hiddenDelta === undefined ? 400 : opts.hiddenDelta;
    this.tankRate = opts.tankRate === undefined ? null : opts.tankRate;
    this.betCorrelation = opts.betCorrelation === undefined ? 1.0 : opts.betCorrelation;
    this.playProfile = (opts.playProfile || SHARK_PLAY_PROFILE).slice();
    this.tankProfile = (opts.tankProfile || SHARK_TANK_PROFILE).slice();
  }
  inherit(Shark, Player);
  Shark.prototype.archetype = 'shark';
  Shark.prototype.effectiveTankRate = function (expected) {
    if (this.tankRate !== null) return this.tankRate;
    return pinningTankRate(expected, this.hiddenDelta, TANK_WIN_PROB);
  };
  Shark.prototype.marginalProfile = function (tau) {
    var out = [];
    for (var t = 0; t < N_TIERS; t++) {
      out.push(tau * this.tankProfile[t] + (1 - tau) * this.playProfile[t]);
    }
    return out;
  };
  Shark.prototype.betProfile = function (expected) {
    return this.marginalProfile(this.effectiveTankRate(expected));
  };
  Shark.prototype.decide = function (expected, rng) {
    var tau = this.effectiveTankRate(expected);
    var base = this.marginalProfile(tau);
    var tanking = rng.chance(tau);
    var intentProfile = tanking ? this.tankProfile : this.playProfile;
    var tier = rng.choiceP(blend(intentProfile, base, this.betCorrelation));
    if (tanking) return { tier: tier, winProb: TANK_WIN_PROB, intent: 'tank' };
    return { tier: tier, winProb: shiftExpectation(expected, this.hiddenDelta), intent: 'play' };
  };

  var ARCHETYPES = {
    noob: function () { return new HonestPlayer('cautious', 1200, 'noob'); },
    average: function () { return new HonestPlayer('typical', 1200, 'average'); },
    aggressive: function () { return new HonestPlayer('aggressive', 1200, 'aggressive'); },
    whale: function () { return new Whale(1200); },
    tilter: function () { return new Tilter(1200); },
    shark: function () { return new Shark({}); }
  };

  /* ---- engine ---- */

  function opponentRating(playerRating, rng, spread) {
    return playerRating + (spread === undefined ? 80 : spread) * rng.normal();
  }

  /* One wagered match. The order of random draws here is load-bearing: opponent
     rating first (two uniforms via normal), then the bet decision, then the
     result. Python's engine.play_match does the same, in the same order. */
  function playMatch(player, oppRating, rng, k) {
    var expected = expectedScore(player.rating, oppRating);
    var decision = player.decide(expected, rng);
    var won = rng.chance(decision.winProb);
    player.settle(expected, decision.tier, won, k);
    return { obs: new Observation(expected, won, decision.tier), intent: decision.intent };
  }

  function simulateCareer(player, matches, rng, opts) {
    opts = opts || {};
    var detectors = opts.detectors === undefined ? detectorSuite(opts.detectorOpts) : opts.detectors;
    var k = opts.k === undefined ? DEFAULT_K : opts.k;
    var spread = opts.spread === undefined ? 80 : opts.spread;
    var career = {
      player: player,
      observations: [],
      intents: [],
      ratings: [player.rating],
      bankroll: [0]
    };
    for (var i = 0; i < matches; i++) {
      var res = playMatch(player, opponentRating(player.rating, rng, spread), rng, k);
      career.observations.push(res.obs);
      career.intents.push(res.intent);
      career.ratings.push(player.rating);
      career.bankroll.push(player.bankroll);
      for (var name in detectors) {
        if (detectors.hasOwnProperty(name)) detectors[name].update(res.obs);
      }
    }
    career.winRate = function () {
      if (!this.observations.length) return 0;
      var w = 0;
      for (var j = 0; j < this.observations.length; j++) if (this.observations[j].won) w++;
      return w / this.observations.length;
    };
    career.profitPerMatch = function () {
      return this.observations.length
        ? this.bankroll[this.bankroll.length - 1] / this.observations.length : 0;
    };
    career.ratingDrift = function () {
      return this.ratings[this.ratings.length - 1] - this.ratings[0];
    };
    return { career: career, detectors: detectors };
  }

  global.SharkHunt = {
    TIER_NAMES: TIER_NAMES, N_TIERS: N_TIERS, STAKES: STAKES, RAKE: RAKE,
    TANK_WIN_PROB: TANK_WIN_PROB, DEFAULT_DELTA: DEFAULT_DELTA, DEFAULT_TAU: DEFAULT_TAU,
    PROFILES: PROFILES, HONEST_MIX: HONEST_MIX,
    SHARK_PLAY_PROFILE: SHARK_PLAY_PROFILE, SHARK_TANK_PROFILE: SHARK_TANK_PROFILE,
    populationProfile: populationProfile, blend: blend, payoff: payoff,
    expectedScore: expectedScore, shiftExpectation: shiftExpectation,
    updateRating: updateRating, pinningTankRate: pinningTankRate,
    waldThresholds: waldThresholds,
    FixedProfile: FixedProfile, DirichletProfile: DirichletProfile,
    Observation: Observation, OutcomeSPRT: OutcomeSPRT, WeightedLLR: WeightedLLR,
    JointLLR: JointLLR, HierarchicalJointLLR: HierarchicalJointLLR,
    MixtureJointLLR: MixtureJointLLR, defaultGrid: defaultGrid,
    detectorSuite: detectorSuite, jointCellTable: jointCellTable,
    Player: Player, HonestPlayer: HonestPlayer, Whale: Whale, Tilter: Tilter, Shark: Shark,
    ARCHETYPES: ARCHETYPES,
    opponentRating: opponentRating, playMatch: playMatch, simulateCareer: simulateCareer
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
