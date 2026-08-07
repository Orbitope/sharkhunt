/* mulberry32, mirrored exactly by sharkhunt/rng.py.
   The widgets on this page re-run the same simulations the Python package runs,
   so both sides have to produce identical streams from identical seeds.
   tests/test_rng_parity.py checks that claim. */
(function (global) {
  'use strict';

  function Rng(seed) {
    this.state = seed | 0;
  }

  Rng.prototype.random = function () {
    this.state = (this.state + 0x6D2B79F5) | 0;
    var a = this.state;
    var t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (((t + Math.imul(t ^ (t >>> 7), 61 | t)) | 0) ^ t) | 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  Rng.prototype.uniform = function (lo, hi) {
    return lo + (hi - lo) * this.random();
  };

  Rng.prototype.randint = function (lo, hi) {
    return lo + Math.floor(this.random() * (hi - lo));
  };

  Rng.prototype.chance = function (p) {
    return this.random() < p;
  };

  Rng.prototype.choiceP = function (probs) {
    var u = this.random(), acc = 0;
    for (var i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (u < acc) return i;
    }
    return probs.length - 1;
  };

  Rng.prototype.normal = function () {
    var u1 = Math.max(this.random(), 1e-12);
    var u2 = this.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  Rng.prototype.spawn = function (salt) {
    return new Rng(Math.imul(this.state ^ (salt | 0), 0x9E3779B1) | 0);
  };

  global.SHRng = Rng;
})(typeof globalThis !== 'undefined' ? globalThis : this);
