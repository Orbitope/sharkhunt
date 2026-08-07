/* Widgets for the shark-hunting article.

   Every detector widget re-runs the simulation live via sharkhunt.js, a port of
   the Python package that produced the offline numbers - tests/test_js_parity.py
   holds the two to match career by career. Only the population-fit widgets read
   precomputed results from data.js, because a browser cannot sample a posterior.

   Plain ES5, one IIFE per widget, each bailing out if its element is absent. */
(function () {
"use strict";

var SH = window.SharkHunt, Rng = window.SHRng, D = window.SHData || {};
var NS = 'http://www.w3.org/2000/svg';

/* ---- known-answer check ---------------------------------------------------
   If this page's simulation ever drifts from the package the prose quotes, say
   so rather than showing plausible-looking numbers. These constants come from
   the Python side. */
var SIM_OK = (function () {
  try {
    var r = new Rng(12345);
    var res = SH.simulateCareer(new SH.Shark({ hiddenDelta: 400, betCorrelation: 1.0 }), 60, r);
    var ok = Math.abs(SH.pinningTankRate(0.5, 400) - 0.4601226993865031) < 1e-12
          && Math.abs(SH.shiftExpectation(0.5, 400) - 0.9090909090909091) < 1e-12;
    var cells = SH.jointCellTable(0.5, 400, SH.pinningTankRate(0.5, 400));
    ok = ok && Math.abs(cells[0][0] - 0.6660524740323518) < 1e-9
            && Math.abs(cells[2][1] - 0.9354203801432172) < 1e-9;
    return ok && res.detectors.hierarchical.n === 60;
  } catch (e) { return false; }
})();

/* ---- tiny SVG helpers ---------------------------------------------------- */

function el(tag, attrs) {
  var e = document.createElementNS(NS, tag);
  for (var k in attrs) if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]);
  return e;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function txt(svg, x, y, s, fill, anchor, size, weight) {
  var t = el('text', {
    x: x, y: y, fill: fill || 'var(--text-muted)',
    'text-anchor': anchor || 'start', 'font-size': size || 10,
    'font-family': 'JetBrains Mono, monospace'
  });
  if (weight) t.setAttribute('font-weight', weight);
  t.textContent = s;
  svg.appendChild(t);
  return t;
}
function fmtPct(x, dp) { return (100 * x).toFixed(dp === undefined ? 0 : dp) + '%'; }
function fmtSigned(x, dp) { return (x >= 0 ? '+' : '') + x.toFixed(dp === undefined ? 3 : dp); }

/* A minimal linear-axis chart frame shared by every line plot on the page.
   Returns scale functions plus the group to draw into, so each widget only has
   to say what its series are. */
function frame(svg, opts) {
  clear(svg);
  var vb = svg.getAttribute('viewBox').split(/\s+/).map(Number);
  var W = vb[2], H = vb[3];
  var m = opts.margin || {};
  var L = m.left === undefined ? 52 : m.left;
  var R = m.right === undefined ? 14 : m.right;
  var T = m.top === undefined ? 16 : m.top;
  var B = m.bottom === undefined ? 34 : m.bottom;
  var x0 = opts.x0, x1 = opts.x1, y0 = opts.y0, y1 = opts.y1;
  var logx = !!opts.logx;

  function fx(v) {
    var t = logx
      ? (Math.log(Math.max(v, 1e-9)) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))
      : (v - x0) / (x1 - x0);
    return L + t * (W - L - R);
  }
  function fy(v) {
    var t = (v - y0) / (y1 - y0);
    return H - B - t * (H - B - T);
  }

  (opts.yTicks || []).forEach(function (tick) {
    var obj = typeof tick === 'object';
    var v = obj ? tick.v : tick;
    var lab = obj ? tick.label : String(tick);
    var y = fy(v);
    svg.appendChild(el('line', {
      x1: L, x2: W - R, y1: y, y2: y,
      stroke: (obj && tick.strong) ? 'var(--border)' : 'rgba(42,40,32,.55)',
      'stroke-width': 1,
      'stroke-dasharray': (obj && tick.dash) ? '3 4' : ''
    }));
    // A word-length label in the axis gutter collides with the rotated y-axis
    // title. `inline` puts it on the line itself instead, at the right edge.
    if (obj && tick.inline) {
      txt(svg, W - R - 4, y - 5, lab, tick.color || 'var(--text-muted)', 'end', 9.5, 600);
    } else {
      txt(svg, L - 8, y + 3.5, lab, (obj && tick.color) || 'var(--text-muted)', 'end', 9.5);
    }
  });
  (opts.xTicks || []).forEach(function (tick) {
    var v = typeof tick === 'object' ? tick.v : tick;
    var lab = typeof tick === 'object' ? tick.label : String(tick);
    // A long label centred on the first or last tick hangs off the side of the
    // viewBox and gets clipped, so tuck the outermost ones inward.
    var px = fx(v), anchor = 'middle';
    if (px <= L + 1) anchor = 'start';
    else if (px >= W - R - 1) anchor = 'end';
    txt(svg, px, H - B + 15, lab, 'var(--text-muted)', anchor, 9.5);
  });
  if (opts.xLabel) txt(svg, (L + W - R) / 2, H - 3, opts.xLabel, 'var(--text-muted)', 'middle', 9.5);
  if (opts.yLabel) {
    var t = txt(svg, 0, 0, opts.yLabel, 'var(--text-muted)', 'middle', 9.5);
    t.setAttribute('transform', 'translate(11,' + ((T + H - B) / 2) + ') rotate(-90)');
  }
  return { svg: svg, fx: fx, fy: fy, W: W, H: H, L: L, R: R, T: T, B: B };
}

function line(f, pts, stroke, width, opacity, dash) {
  if (!pts.length) return null;
  var d = '';
  for (var i = 0; i < pts.length; i++) {
    d += (i ? 'L' : 'M') + f.fx(pts[i][0]).toFixed(2) + ' ' + f.fy(pts[i][1]).toFixed(2);
  }
  var p = el('path', {
    d: d, fill: 'none', stroke: stroke,
    'stroke-width': width || 1.6, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'
  });
  if (opacity !== undefined) p.setAttribute('opacity', opacity);
  if (dash) p.setAttribute('stroke-dasharray', dash);
  f.svg.appendChild(p);
  return p;
}

function seriesFromHistory(history, stride) {
  stride = stride || 1;
  var pts = [];
  for (var i = 0; i < history.length; i += stride) pts.push([i, history[i]]);
  if ((history.length - 1) % stride !== 0) pts.push([history.length - 1, history[history.length - 1]]);
  return pts;
}

var TH = SH.waldThresholds();
var LOWER = TH[0], UPPER = TH[1];

/* ====== page chrome ====== */
(function () {
  var prog = document.getElementById('progress');
  if (prog) {
    addEventListener('scroll', function () {
      var h = document.documentElement;
      prog.style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100) + '%';
    }, { passive: true });
  }
  var reveals = [].slice.call(document.querySelectorAll('.reveal'));
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px' });
    reveals.forEach(function (r) { io.observe(r); });
  } else {
    reveals.forEach(function (r) { r.classList.add('in'); });
  }
  if (!SIM_OK) {
    var tag = document.getElementById('pin_tag');
    if (tag) { tag.className = 'tag no'; tag.textContent = 'SIM CHECK FAILED'; }
  }
})();

/* ====== 1. RATING PINNING ====== */
(function () {
  var svg = document.getElementById('pin_svg');
  if (!svg) return;
  var dIn = document.getElementById('pin_delta'), dOut = document.getElementById('pin_delta_v');
  var tIn = document.getElementById('pin_tau'), tOut = document.getElementById('pin_tau_v');
  var solve = document.getElementById('pin_solve');
  var cap = document.getElementById('pin_cap'), tag = document.getElementById('pin_tag');
  var MATCHES = 400;

  function render() {
    var delta = +dIn.value, tau = +tIn.value / 100;
    dOut.textContent = '+' + delta;
    tOut.textContent = fmtPct(tau);

    var runs = [];
    for (var s = 0; s < 6; s++) {
      var shark = new SH.Shark({ hiddenDelta: delta, tankRate: tau, betCorrelation: 1.0 });
      runs.push(SH.simulateCareer(shark, MATCHES, new Rng(4242 + s * 977), { detectors: {} }).career);
    }
    var lo = 1200, hi = 1200;
    runs.forEach(function (c) {
      c.ratings.forEach(function (r) { lo = Math.min(lo, r); hi = Math.max(hi, r); });
    });
    var pad = Math.max(60, (hi - lo) * 0.12);
    lo = Math.floor((lo - pad) / 50) * 50; hi = Math.ceil((hi + pad) / 50) * 50;

    var ticks = [];
    var step = Math.max(50, Math.round((hi - lo) / 5 / 50) * 50);
    for (var v = lo; v <= hi + 1; v += step) ticks.push({ v: v, label: String(v) });

    var f = frame(svg, {
      x0: 0, x1: MATCHES, y0: lo, y1: hi,
      yTicks: ticks,
      xTicks: [0, 100, 200, 300, 400],
      xLabel: 'matches played', yLabel: 'shown rating'
    });
    line(f, [[0, 1200], [MATCHES, 1200]], 'var(--steel)', 1, 0.5, '4 4');
    runs.forEach(function (c, i) {
      line(f, seriesFromHistory(c.ratings, 4), 'var(--amber-bright)', i === 0 ? 2 : 1.1,
        i === 0 ? 1 : 0.35);
    });

    var main = runs[0];
    var drift = main.ratingDrift();
    document.getElementById('pin_wr').textContent = fmtPct(main.winRate(), 1);
    var dEl = document.getElementById('pin_drift');
    dEl.textContent = fmtSigned(drift, 0);
    dEl.className = 'big ' + (Math.abs(drift) < 60 ? 'sage' : 'coral');
    document.getElementById('pin_profit').textContent = fmtSigned(main.profitPerMatch(), 3);

    var star = SH.pinningTankRate(0.5, delta);
    var pinned = Math.abs(drift) < 60;
    tag.className = 'tag ' + (pinned ? 'ok' : 'idle');
    tag.textContent = pinned ? 'rating pinned' : 'drifting ' + fmtSigned(drift, 0);
    cap.innerHTML = 'Six sharks hiding <b>' + delta + '</b> Elo, throwing <b>' + fmtPct(tau)
      + '</b> of matches. The invisible rate for this much hidden skill is <b>'
      + fmtPct(star, 1) + '</b>. ' + (pinned
        ? 'They are parked: the rating moved <b>' + fmtSigned(drift, 0)
          + '</b> over 400 matches while they made <b>' + fmtSigned(main.profitPerMatch(), 3)
          + '</b> a match.'
        : 'Their rating is running away from the bracket they wanted, by <b>'
          + fmtSigned(drift, 0) + '</b>.');
  }

  dIn.addEventListener('input', render);
  tIn.addEventListener('input', render);
  solve.addEventListener('click', function () {
    tIn.value = Math.round(100 * SH.pinningTankRate(0.5, +dIn.value));
    render();
  });
  render();
})();

/* ====== 2. OUTCOME-ONLY SPRT ====== */
(function () {
  var svg = document.getElementById('sprt_svg');
  if (!svg) return;
  var playBtn = document.getElementById('sprt_play');
  var resetBtn = document.getElementById('sprt_reset');
  var cap = document.getElementById('sprt_cap'), tag = document.getElementById('sprt_tag');
  var MATCHES = 400, cursor = 0, timer = null, runs = null;

  function build() {
    runs = { honest: [], blatant: null, pinned: null };
    for (var i = 0; i < 20; i++) {
      var r = new Rng(70000 + i * 613);
      var res = SH.simulateCareer(new SH.HonestPlayer('typical', 1200), MATCHES, r,
        { detectors: { o: new SH.OutcomeSPRT({}) } });
      runs.honest.push(res.detectors.o.history);
    }
    runs.blatant = SH.simulateCareer(
      new SH.Shark({ hiddenDelta: 400, tankRate: 0 }), MATCHES, new Rng(31337),
      { detectors: { o: new SH.OutcomeSPRT({}) } }).detectors.o.history;
    runs.pinned = SH.simulateCareer(
      new SH.Shark({ hiddenDelta: 400, betCorrelation: 1.0 }), MATCHES, new Rng(4242),
      { detectors: { o: new SH.OutcomeSPRT({}) } }).detectors.o.history;
  }

  function clip(hist, n) {
    // Once a sequential test crosses a boundary it has reached a verdict; the
    // walk after that point is not something a real system would keep watching.
    var out = [];
    for (var i = 0; i <= Math.min(n, hist.length - 1); i++) {
      out.push([i, hist[i]]);
      if (hist[i] >= UPPER || hist[i] <= LOWER) break;
    }
    return out;
  }

  function render() {
    var f = frame(svg, {
      x0: 0, x1: MATCHES, y0: -14, y1: 14,
      yTicks: [
        { v: UPPER, label: 'accuse', strong: true, dash: true, inline: true, color: 'var(--coral)' },
        { v: 0, label: '0', strong: true },
        { v: LOWER, label: 'clear', strong: true, dash: true, inline: true, color: 'var(--sage)' },
        { v: 12, label: '+12' }, { v: -12, label: '-12' }
      ],
      xTicks: [0, 100, 200, 300, 400],
      xLabel: 'matches played', yLabel: 'log-likelihood ratio'
    });
    runs.honest.forEach(function (h) { line(f, clip(h, cursor), 'var(--sage)', 1, 0.34); });
    line(f, clip(runs.blatant, cursor), 'var(--coral)', 2.2);
    line(f, clip(runs.pinned, cursor), 'var(--amber-bright)', 2.2);

    tag.textContent = 'match ' + cursor;
    tag.className = 'tag ' + (cursor >= MATCHES ? 'idle' : 'warn');

    var pin = runs.pinned[Math.min(cursor, runs.pinned.length - 1)];
    var bl = runs.blatant[Math.min(cursor, runs.blatant.length - 1)];
    var caught = runs.blatant.findIndex(function (v) { return v >= UPPER; });
    cap.innerHTML = 'After <b>' + cursor + '</b> matches: the <span class="coral">unconcealed shark</span> is at <b>'
      + bl.toFixed(2) + '</b>' + (caught >= 0 && cursor >= caught
        ? ' and was accused at match <b>' + caught + '</b>' : '')
      + '. The <span class="amber">shark tanking at tau*</span> is at <b>' + pin.toFixed(2)
      + '</b> - ' + (pin < 0 ? 'heading for the <span class="sage">all-clear</span>, alongside the honest players'
        : 'indistinguishable from the honest cluster') + '.';
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    playBtn.textContent = 'Play'; playBtn.classList.remove('on');
  }

  playBtn.addEventListener('click', function () {
    if (timer) { stop(); return; }
    if (cursor >= MATCHES) { cursor = 0; }
    playBtn.textContent = 'Pause'; playBtn.classList.add('on');
    timer = setInterval(function () {
      cursor = Math.min(MATCHES, cursor + 8);
      render();
      if (cursor >= MATCHES) stop();
    }, 60);
  });
  resetBtn.addEventListener('click', function () { stop(); cursor = 0; render(); });

  build(); cursor = MATCHES; render();
})();

/* ====== 3. CALIBRATION TABLE ====== */
(function () {
  var tbl = document.getElementById('calib_tbl');
  if (!tbl || !D.detectors) return;
  var data = D.detectors.calibration_by_betting_habit;
  var names = { 'honest-cautious': 'cautious bettor', 'honest-typical': 'typical bettor',
                'honest-aggressive': 'aggressive bettor', 'whale': 'whale (always max bet)' };
  // Deliberately only two score columns. A third showing the fix would read as
  // a spoiler here, and worse, would name a detector the article has not built
  // yet - the reader would meet "joint model" as an undefined term with a
  // suspiciously perfect score.
  var html = '<thead><tr><th>honest player</th><th>wager-weighted score</th>'
    + '<th>the outcome-only test, for comparison</th></tr></thead><tbody>';
  Object.keys(names).forEach(function (key) {
    var row = data[key];
    if (!row) return;
    var w = row.weighted;
    var cls = w < 0.2 ? 'bad' : (w > 0.8 ? 'hi' : '');
    html += '<tr><td class="nm">' + names[key] + '</td>'
      + '<td class="' + cls + '">' + fmtPct(w, 1) + '</td>'
      + '<td>' + fmtPct(row.outcome, 1) + '</td></tr>';
  });
  tbl.innerHTML = html + '</tbody>';
})();

/* ====== 4. SIX-CELL EVIDENCE GRID ====== */
(function () {
  var svg = document.getElementById('cells_svg');
  if (!svg) return;
  var eIn = document.getElementById('cells_e'), eOut = document.getElementById('cells_e_v');
  var dIn = document.getElementById('cells_d'), dOut = document.getElementById('cells_d_v');
  var cap = document.getElementById('cells_cap'), tag = document.getElementById('cells_tag');

  function render() {
    var e = +eIn.value / 100, delta = +dIn.value;
    eOut.textContent = fmtPct(e);
    dOut.textContent = '+' + delta;
    var tau = SH.pinningTankRate(e, delta);
    var table = SH.jointCellTable(e, delta, tau);

    clear(svg);
    var X0 = 132, CW = 148, Y0 = 44, CH = 52;
    var mult = [0.1, 0.4, 1.0];
    txt(svg, X0 + CW * 0.5, 28, 'LOSS', 'var(--text-muted)', 'middle', 11, 600);
    txt(svg, X0 + CW * 1.5, 28, 'WIN', 'var(--text-muted)', 'middle', 11, 600);
    txt(svg, 12, 16, 'joint model · log-evidence per match (accuse line: 4.60)', 'var(--text-muted)', 'start', 9.5);

    var maxAbs = 0;
    table.forEach(function (r) { r.forEach(function (v) { maxAbs = Math.max(maxAbs, Math.abs(v)); }); });

    for (var t = 0; t < 3; t++) {
      var y = Y0 + t * CH;
      txt(svg, X0 - 12, y + CH / 2 + 4, SH.TIER_NAMES[t] + ' bet', 'var(--text-primary)', 'end', 11);
      txt(svg, X0 - 12, y + CH / 2 + 17, '×' + mult[t].toFixed(1) + ' weighted',
        'var(--text-muted)', 'end', 9);
      for (var o = 0; o < 2; o++) {
        var v = table[t][o];
        var x = X0 + o * CW;
        var strength = Math.min(1, Math.abs(v) / Math.max(maxAbs, 0.4));
        svg.appendChild(el('rect', {
          x: x + 4, y: y + 4, width: CW - 8, height: CH - 8, rx: 4,
          fill: v > 0 ? 'rgba(255,94,58,' + (0.10 + 0.55 * strength) + ')'
                      : 'rgba(125,154,106,' + (0.10 + 0.45 * strength) + ')',
          stroke: 'var(--border)', 'stroke-width': 1
        }));
        txt(svg, x + CW / 2, y + CH / 2 + 2, fmtSigned(v, 2),
          v > 0 ? '#FFB9A6' : '#BFD6AF', 'middle', 15, 600);
        // What the wager-weighted score would have made of the same event.
        var w = (o === 1 ? Math.log(0.5 / e) : Math.log(0.5 / (1 - e)));
        var det = new SH.WeightedLLR({ delta: delta, tau: tau });
        w = det.increment(new SH.Observation(e, o === 1, t));
        txt(svg, x + CW / 2, y + CH - 9, 'weighted: ' + fmtSigned(w, 2),
          'var(--text-muted)', 'middle', 9);
      }
    }
    var minLoss = table[0][0];
    tag.className = 'tag ' + (minLoss > 0 ? 'warn' : 'idle');
    tag.textContent = fmtPct(e) + ' matchup · tau* ' + fmtPct(tau, 1);
    cap.innerHTML = 'Hiding <b>' + delta + '</b> Elo in a <b>' + fmtPct(e)
      + '</b> matchup. A <b class="coral">minimum-bet loss</b> is worth <b>' + fmtSigned(minLoss, 2)
      + '</b> under the joint model and <b>'
      + fmtSigned(new SH.WeightedLLR({ delta: delta, tau: tau })
          .increment(new SH.Observation(e, false, 0)), 2)
      + '</b> under the weighted score. A <b class="sage">minimum-bet win</b> is <b>'
      + fmtSigned(table[0][1], 2) + '</b>: sharks do not win the games they staked nothing on.';
  }

  eIn.addEventListener('input', render);
  dIn.addEventListener('input', render);
  render();
})();

/* ====== 5. DETECTOR COMPARISON ====== */
(function () {
  var svg = document.getElementById('duel_svg');
  if (!svg) return;
  var tabsEl = document.getElementById('duel_tabs');
  var legendEl = document.getElementById('duel_legend');
  var cap = document.getElementById('duel_cap'), tag = document.getElementById('duel_tag');
  var desc = document.getElementById('duel_desc');
  var reseed = document.getElementById('duel_reseed');
  var MATCHES = 400;

  var CASES = [
    { id: 'honest', label: 'honest', desc: 'an ordinary player, typical betting habits',
      make: function () { return new SH.HonestPlayer('typical', 1200); }, shark: false },
    { id: 'whale', label: 'whale', desc: 'honest, and always bets the table maximum',
      make: function () { return new SH.Whale(1200); }, shark: false },
    { id: 'tilter', label: 'tilter', desc: 'honest, but bets bigger after a loss',
      make: function () { return new SH.Tilter(1200); }, shark: false },
    { id: 'blatant', label: 'blatant shark', desc: '+400 Elo hidden, never throws a match',
      make: function () { return new SH.Shark({ hiddenDelta: 400, tankRate: 0 }); }, shark: true },
    { id: 'pinned', label: 'hiding shark', desc: '+400 Elo hidden, tanking at tau*, bets tracking intent',
      make: function () { return new SH.Shark({ hiddenDelta: 400, betCorrelation: 1.0 }); }, shark: true },
    { id: 'evasive', label: 'evasive shark', desc: '+400 Elo hidden, bets deliberately decorrelated',
      make: function () { return new SH.Shark({ hiddenDelta: 400, betCorrelation: 0 }); }, shark: true }
  ];
  var SERIES = [
    { key: 'outcome', label: 'outcomes only', color: 'var(--sage)' },
    { key: 'weighted', label: 'wager-weighted', color: 'var(--terra)' },
    { key: 'joint', label: 'joint, population habits', color: 'var(--steel-bright)' },
    { key: 'hierarchical', label: 'joint, own habits', color: 'var(--amber-bright)' },
    { key: 'mixture', label: 'mixture over shark types', color: 'var(--mauve)' }
  ];

  var active = 4, seed = 4242;

  legendEl.innerHTML = SERIES.map(function (s) {
    return '<span class="k"><i class="sw" style="background:' + s.color + '"></i>' + s.label + '</span>';
  }).join('');

  CASES.forEach(function (c, i) {
    var b = document.createElement('button');
    b.className = 'toggle' + (i === active ? ' active' : '');
    b.textContent = c.label;
    b.addEventListener('click', function () {
      active = i;
      [].slice.call(tabsEl.children).forEach(function (n, j) {
        n.classList.toggle('active', j === i);
      });
      render();
    });
    tabsEl.appendChild(b);
  });

  function render() {
    var c = CASES[active];
    var res = SH.simulateCareer(c.make(), MATCHES, new Rng(seed));
    var dets = res.detectors;

    // A fixed window rather than an auto-range. The decisive detectors reach a
    // verdict inside a dozen matches and their scores then run into the
    // hundreds, so auto-ranging squashes every interesting line into the bottom
    // of an empty chart. The thresholds are what the reader is comparing
    // against, so the frame is built around them and anything that leaves is
    // marked at the edge. Holding it fixed also makes the six tabs comparable.
    var TOP = UPPER * 2.6, BOT = LOWER * 2.6;

    var f = frame(svg, {
      x0: 0, x1: MATCHES, y0: BOT, y1: TOP,
      yTicks: [
        { v: UPPER, label: 'accuse', strong: true, dash: true, inline: true, color: 'var(--coral)' },
        { v: 0, label: '0', strong: true },
        { v: LOWER, label: 'clear', strong: true, dash: true, inline: true, color: 'var(--sage)' }
      ],
      xTicks: [0, 100, 200, 300, 400],
      xLabel: 'matches played', yLabel: 'evidence for "shark"'
    });

    SERIES.forEach(function (s) {
      var h = dets[s.key].history, pts = [], exit = null, verdict = null;
      for (var i = 0; i < h.length; i++) {
        var v = h[i];
        if (v > TOP || v < BOT) {
          pts.push([i, v > TOP ? TOP : BOT]);
          exit = { x: i, up: v > TOP };
          break;
        }
        pts.push([i, v]);
        // Once a sequential test crosses a boundary it has returned a verdict;
        // what the walk does afterwards is not something anyone would watch.
        if (v >= UPPER || v <= LOWER) { verdict = { x: i, y: v }; break; }
      }
      line(f, pts, s.color, 1.9);
      // The decisive detectors settle inside a dozen of the 400 matches, so
      // their line is a near-vertical stub at the origin. Mark the moment the
      // verdict lands, or the edge it left through, so it reads as an event.
      if (verdict) {
        f.svg.appendChild(el('circle', {
          cx: f.fx(verdict.x), cy: f.fy(verdict.y), r: 4,
          fill: s.color, stroke: '#0d0c07', 'stroke-width': 1.4
        }));
      }
      if (exit) {
        var ey = f.fy(exit.up ? TOP : BOT), dy = exit.up ? 7 : -7;
        f.svg.appendChild(el('path', {
          d: 'M' + (f.fx(exit.x) - 4.5) + ' ' + (ey + dy)
            + 'L' + (f.fx(exit.x) + 4.5) + ' ' + (ey + dy)
            + 'L' + f.fx(exit.x) + ' ' + ey + 'Z',
          fill: s.color
        }));
      }
    });

    var flagged = SERIES.filter(function (s) { return dets[s.key].flaggedAt !== null; });
    desc.textContent = c.desc;
    tag.className = 'tag ' + (c.shark
      ? (flagged.length ? 'ok' : 'no')
      : (flagged.length ? 'no' : 'ok'));
    tag.textContent = c.shark
      ? (flagged.length ? flagged.length + '/5 caught them' : 'all 5 missed')
      : (flagged.length ? flagged.length + '/5 false accusations' : 'correctly left alone');

    var parts = SERIES.map(function (s) {
      var at = dets[s.key].flaggedAt;
      return '<b>' + s.label + '</b> ' + (at !== null
        ? '<span class="coral">accused at ' + at + '</span>'
        : (dets[s.key].cleared() ? '<span class="sage">cleared</span>' : 'undecided'));
    });
    cap.innerHTML = parts.join(' &nbsp;·&nbsp; ')
      + ' <span style="color:var(--text-muted)">— arrows mark a score leaving the frame.</span>';
  }

  reseed.addEventListener('click', function () { seed = (seed * 1103515245 + 12345) & 0x7fffffff; render(); });
  render();
})();

/* ====== 6. PRIOR STRENGTH SWEEP ====== */
(function () {
  var svg = document.getElementById('prior_svg');
  if (!svg || !D.detectors) return;
  var rows = D.detectors.prior_strength_sweep;
  var cap = document.getElementById('prior_cap'), tag = document.getElementById('prior_tag');

  var f = frame(svg, {
    x0: rows[0].strength, x1: rows[rows.length - 1].strength, logx: true,
    y0: 0, y1: 1,
    yTicks: [{ v: 0, label: '0%' }, { v: 0.25, label: '25%' }, { v: 0.5, label: '50%' },
             { v: 0.75, label: '75%' }, { v: 1, label: '100%' }],
    xTicks: [{ v: 0.25, label: '0.25' }, { v: 1, label: '1' }, { v: 4, label: '4' },
             { v: 16, label: '16' }, { v: 64, label: '64' }, { v: 400, label: '400' }],
    xLabel: 'prior strength (trust the population →)', yLabel: 'rate'
  });

  // The band where nothing is broken: sharks all caught, nobody falsely accused.
  var good = rows.filter(function (r) {
    return r.shark_caught > 0.99 && r.whale_accused < 0.01 && r.honest_accused < 0.01;
  });
  if (good.length) {
    var x1 = f.fx(good[0].strength), x2 = f.fx(good[good.length - 1].strength);
    f.svg.insertBefore(el('rect', {
      x: x1, y: f.T, width: Math.max(2, x2 - x1), height: f.H - f.B - f.T,
      fill: 'rgba(125,154,106,.09)'
    }), f.svg.firstChild);
    txt(svg, (x1 + x2) / 2, f.T + 14, 'usable', 'var(--sage)', 'middle', 9.5, 600);
  }

  line(f, rows.map(function (r) { return [r.strength, r.shark_caught]; }), 'var(--amber-bright)', 2);
  line(f, rows.map(function (r) { return [r.strength, r.whale_accused]; }), 'var(--coral)', 2);
  line(f, rows.map(function (r) { return [r.strength, r.honest_accused]; }), 'var(--steel-bright)', 2);

  var worstHonest = rows[0], worstWhale = rows[rows.length - 1];
  tag.className = 'tag idle';
  tag.textContent = good.length ? good[0].strength + '–' + good[good.length - 1].strength + ' works' : '';
  cap.innerHTML = 'At the far left the prior is too weak to stabilise anyone: <b class="coral">'
    + fmtPct(worstHonest.honest_accused, 1) + '</b> of ordinary honest players are accused. '
    + 'At the far right it overrides the individual entirely and <b class="coral">'
    + fmtPct(worstWhale.whale_accused, 1) + '</b> of whales are. '
    + 'Sharks stay at <b>100%</b> caught throughout - the prior is a false-positive control, not a detection one.';
})();

/* ====== 7. FIELD STUDY TABLE ====== */
(function () {
  var tbl = document.getElementById('field_tbl');
  if (!tbl || !D.detectors) return;
  var fs = D.detectors.field_study;
  var order = [
    ['outcome', 'outcomes only'],
    ['weighted', 'wager-weighted'],
    ['joint', 'joint, population habits'],
    ['hierarchical', 'joint, own habits'],
    ['mixture', 'mixture over shark types']
  ];
  var html = '<thead><tr><th>detector</th><th>sharks caught</th>'
    + '<th>honest accused</th><th>median matches to catch</th></tr></thead><tbody>';
  order.forEach(function (o) {
    var r = fs.rates[o[0]];
    if (!r) return;
    var tpCls = r.true_positive_rate > 0.99 ? 'good' : (r.true_positive_rate < 0.1 ? 'bad' : '');
    var fpCls = r.false_positive_rate > 0.02 ? 'bad' : 'good';
    var mark = o[0] === 'mixture' ? ' class="mark"' : '';
    html += '<tr' + mark + '><td class="nm">' + o[1] + '</td>'
      + '<td class="' + tpCls + '">' + fmtPct(r.true_positive_rate, 1) + '</td>'
      + '<td class="' + fpCls + '">' + fmtPct(r.false_positive_rate, 2) + '</td>'
      + '<td class="hi">' + (r.median_matches_to_catch === null ? '—' : r.median_matches_to_catch)
      + '</td></tr>';
  });
  tbl.innerHTML = html + '</tbody>';
})();

/* ====== 8. POSTERIOR RECOVERY ====== */
(function () {
  var svg = document.getElementById('fit_svg');
  if (!svg) return;
  var cap = document.getElementById('fit_cap'), tag = document.getElementById('fit_tag');
  if (!D.fits) {
    clear(svg);
    txt(svg, 310, 120, 'population fit not exported yet', 'var(--text-muted)', 'middle', 12);
    tag.className = 'tag idle'; tag.textContent = 'pending';
    return;
  }
  var fit = D.fits.game_a, truth = D.fits.truth;
  var groups = [
    { label: 'bets when playing to win', got: fit.play_profile, want: truth.play_profile },
    { label: 'bets when throwing the match', got: fit.tank_profile, want: truth.tank_profile }
  ];

  clear(svg);
  var X0 = 60, GW = 250, BW = 26, Y0 = 176;
  txt(svg, 12, 16, 'probability of each bet tier', 'var(--text-muted)', 'start', 9.5);
  groups.forEach(function (g, gi) {
    var gx = X0 + gi * (GW + 40);
    txt(svg, gx + GW / 2 - 20, Y0 + 34, g.label, 'var(--text-primary)', 'middle', 10.5);
    for (var t = 0; t < 3; t++) {
      var x = gx + t * 74;
      var hGot = g.got[t] * 130, hWant = g.want[t] * 130;
      svg.appendChild(el('rect', {
        x: x, y: Y0 - hWant, width: BW + 16, height: hWant, rx: 2,
        fill: 'none', stroke: 'var(--steel)', 'stroke-width': 1.2, 'stroke-dasharray': '3 3'
      }));
      svg.appendChild(el('rect', {
        x: x + 8, y: Y0 - hGot, width: BW, height: hGot, rx: 2, fill: 'var(--amber-bright)'
      }));
      txt(svg, x + BW / 2 + 8, Y0 + 15, SH.TIER_NAMES[t], 'var(--text-muted)', 'middle', 9.5);
      txt(svg, x + BW / 2 + 8, Y0 - hGot - 6, g.got[t].toFixed(2),
        'var(--amber-bright)', 'middle', 9.5);
    }
  });

  tag.className = 'tag ok';
  tag.textContent = 'L1 error ' + fit.profile_error.total_l1.toFixed(3);
  cap.innerHTML = 'Fitted on <b>' + fit.n_players + '</b> players × <b>' + fit.matches
    + '</b> matches with <b>no labels at all</b> - the model was never told which '
    + fit.n_sharks + ' were cheating. It recovered the shark betting signature to a total '
    + 'absolute error of <b>' + fit.profile_error.total_l1.toFixed(3) + '</b> across six probabilities, '
    + 'put prevalence at <b>' + fmtPct(fit.prevalence, 1) + '</b>, and estimated the prior strength '
    + 'I had been setting by hand at <b>' + fit.prior_strength.toFixed(1) + '</b>. '
    // Deliberately no acronym here: this is the reader's first encounter with
    // the ranking score, and it is defined properly one section later where it
    // becomes an axis. Say it in words now, name it then.
    + (fit.auc === null || fit.auc === undefined ? ''
       : fit.auc > 0.9995
         ? 'Ranked by how strongly the posterior suspects them, <b>every one of the '
           + fit.n_sharks + ' sharks lands above every honest player in the game</b>.'
         : 'Ranked by how strongly the posterior suspects them, a randomly chosen shark outranks a '
           + 'randomly chosen honest player <b>' + fmtPct(fit.auc, 1) + '</b> of the time.');
})();

/* ====== 9. TRANSFER ====== */
(function () {
  var svg = document.getElementById('tr_svg');
  if (!svg) return;
  var cap = document.getElementById('tr_cap'), tag = document.getElementById('tr_tag');
  if (!D.fits || !D.fits.transfer || !D.fits.transfer.length) {
    clear(svg);
    txt(svg, 310, 120, 'transfer study not exported yet', 'var(--text-muted)', 'middle', 12);
    tag.className = 'tag idle'; tag.textContent = 'pending';
    return;
  }
  // A population that happened to contain no sharks has an undefined separation
  // score; drop those points rather than plotting a hole.
  var rows = D.fits.transfer.filter(function (r) {
    return r.cold && r.warm && r.cold.auc !== null && r.warm.auc !== null;
  });
  if (!rows.length) {
    clear(svg);
    txt(svg, 310, 120, 'no comparable transfer runs', 'var(--text-muted)', 'middle', 12);
    tag.className = 'tag idle'; tag.textContent = 'n/a';
    return;
  }
  var xs = rows.map(function (r) { return r.n_players; });
  var f = frame(svg, {
    x0: xs[0], x1: xs[xs.length - 1], logx: true, y0: 0, y1: 1.0,
    yTicks: [{ v: 0, label: '0' }, { v: 0.25, label: '0.25' },
             { v: 0.5, label: 'coin flip', dash: true, strong: true, inline: true },
             { v: 0.75, label: '0.75' }, { v: 1.0, label: '1.00' }],
    xTicks: xs.map(function (v) { return { v: v, label: String(v) }; }),
    xLabel: 'players in the new title', yLabel: 'shark/honest separation (AUC)'
  });
  line(f, rows.map(function (r) { return [r.n_players, r.warm.auc]; }), 'var(--amber-bright)', 2.2);
  line(f, rows.map(function (r) { return [r.n_players, r.cold.auc]; }), 'var(--steel-bright)', 2.2);
  rows.forEach(function (r) {
    f.svg.appendChild(el('circle', { cx: f.fx(r.n_players), cy: f.fy(r.warm.auc), r: 3,
      fill: 'var(--amber-bright)' }));
    f.svg.appendChild(el('circle', { cx: f.fx(r.n_players), cy: f.fy(r.cold.auc), r: 3,
      fill: 'var(--steel-bright)' }));
  });

  var worstCold = rows.reduce(function (a, b) { return b.cold.auc < a.cold.auc ? b : a; });
  var worstWarm = rows.reduce(function (a, b) { return b.warm.auc < a.warm.auc ? b : a; });
  var below = rows.filter(function (r) { return r.cold.auc < 0.5; }).length;
  tag.className = 'tag no';
  tag.textContent = below + '/' + rows.length + ' cold fits worse than chance';
  cap.innerHTML = 'Each point is a fit on <b>' + rows[0].matches + '</b> matches per player. '
    + 'The transferred prior holds between <b class="amber">' + worstWarm.warm.auc.toFixed(3)
    + '</b> and <b>1.000</b> at every size. From scratch the fit is not merely worse, it is '
    + '<b class="coral">erratic</b> - as low as <b>' + worstCold.cold.auc.toFixed(3) + '</b> at n='
    + worstCold.n_players + ', which is <i>below</i> a coin flip, meaning it confidently ranked the '
    + 'cheaters as the <i>least</i> suspicious players in the game. And it does not improve with '
    + 'more data: the largest cold fit here is worse than the smallest. That is the signature of a '
    + 'model that has found the wrong answer rather than too little of the right one.';
})();

/* ====== 10. THE TAX ====== */
(function () {
  var svg = document.getElementById('tax_svg');
  if (!svg || !D.detectors) return;
  var econ = D.detectors.economics_curve;
  var cap = document.getElementById('tax_cap'), tag = document.getElementById('tax_tag');
  var rows = econ.curve.slice().sort(function (a, b) { return a.bet_correlation - b.bet_correlation; });
  var base = econ.honest_baseline_profit_per_match;

  var maxP = Math.max.apply(null, rows.map(function (r) { return r.profit_per_match; }));
  var f = frame(svg, {
    x0: 0, x1: 1, y0: -0.05, y1: Math.max(0.25, maxP * 1.15),
    yTicks: [{ v: 0, label: '0', strong: true },
             { v: base, label: 'honest baseline', dash: true, inline: true,
               color: 'var(--text-muted)' },
             { v: 0.1, label: '+0.10' }, { v: 0.2, label: '+0.20' }],
    xTicks: [{ v: 0, label: 'decorrelated' }, { v: 0.25, label: '.25' }, { v: 0.5, label: '.5' },
             { v: 0.75, label: '.75' }, { v: 1, label: 'fully correlated' }],
    xLabel: 'how tightly the shark lets bet size track intent',
    yLabel: 'profit per match'
  });

  line(f, rows.map(function (r) { return [r.bet_correlation, r.profit_per_match]; }), 'var(--sage)', 2.4);
  // Detection rate shares the panel on its own 0-1 scale, mapped to the top half.
  var top = f.fy(Math.max(0.25, maxP * 1.15)), bottom = f.fy(0);
  function fyCaught(p) { return bottom + (top - bottom) * p; }
  var d = '';
  rows.forEach(function (r, i) {
    d += (i ? 'L' : 'M') + f.fx(r.bet_correlation).toFixed(2) + ' ' + fyCaught(r.caught_rate).toFixed(2);
  });
  f.svg.appendChild(el('path', { d: d, fill: 'none', stroke: 'var(--coral)', 'stroke-width': 2.4,
    'stroke-dasharray': '5 3' }));
  txt(svg, f.W - f.R - 4, top + 12, 'caught 100%', 'var(--coral)', 'end', 9);

  rows.forEach(function (r) {
    f.svg.appendChild(el('circle', { cx: f.fx(r.bet_correlation), cy: f.fy(r.profit_per_match),
      r: 3, fill: 'var(--sage)' }));
  });

  var hi = rows[rows.length - 1], lo = rows[0];
  tag.className = 'tag ok';
  tag.textContent = 'no safe profitable setting';
  cap.innerHTML = 'Fully correlated: <b class="sage">' + fmtSigned(hi.profit_per_match, 3)
    + '</b> per match, caught <b class="coral">' + fmtPct(hi.caught_rate)
    + '</b> of the time in a median of <b>' + hi.median_matches_to_flag + '</b> matches. '
    + 'Fully decorrelated: caught <b>' + fmtPct(lo.caught_rate) + '</b> of the time - and earning <b>'
    + fmtSigned(lo.profit_per_match, 3) + '</b>, against an honest player’s <b>'
    + fmtSigned(base, 3) + '</b>. Hiding costs them <b>'
    + (100 * (hi.profit_per_match - lo.profit_per_match)).toFixed(1)
    + '%</b> of a table maximum per match, which is the entire edge.';
})();






/* ====== 11. HIDDEN SKILL: THE RANDOMNESS DIAL ======
   The shark's second lever. Concealing less skill means their intent predicts
   their result less well, so every match is weaker evidence and they survive
   longer - but the edge they came for shrinks faster than the safety grows. */
(function () {
  var svg = document.getElementById('skill_svg');
  if (!svg || !D.detectors || !D.detectors.hidden_skill_sweep) return;
  var rows = D.detectors.hidden_skill_sweep.filter(function (r) {
    return r.median_matches_to_flag !== null;
  });
  if (!rows.length) return;
  var cap = document.getElementById('skill_cap'), tag = document.getElementById('skill_tag');

  var maxP = Math.max.apply(null, rows.map(function (r) { return r.profit_per_match; }));
  var maxM = Math.max.apply(null, rows.map(function (r) { return r.median_matches_to_flag; }));
  var f = frame(svg, {
    x0: rows[0].hidden_delta, x1: rows[rows.length - 1].hidden_delta,
    y0: 0, y1: Math.max(0.3, maxP * 1.15),
    yTicks: [{ v: 0, label: '0', strong: true }, { v: 0.1, label: '+0.10' },
             { v: 0.2, label: '+0.20' }, { v: 0.3, label: '+0.30' }],
    xTicks: rows.map(function (r) { return { v: r.hidden_delta, label: '+' + r.hidden_delta }; }),
    xLabel: 'Elo of skill hidden behind the rating', yLabel: 'profit per match'
  });

  line(f, rows.map(function (r) { return [r.hidden_delta, r.profit_per_match]; }), 'var(--sage)', 2.4);
  // Survival time rides its own scale in the upper half of the panel.
  var top = f.T + 14, bottom = f.fy(0);
  function fyM(m) { return bottom + (top - bottom) * (m / (maxM * 1.1)); }
  var d = '';
  rows.forEach(function (r, i) {
    d += (i ? 'L' : 'M') + f.fx(r.hidden_delta).toFixed(2) + ' '
      + fyM(r.median_matches_to_flag).toFixed(2);
  });
  f.svg.appendChild(el('path', { d: d, fill: 'none', stroke: 'var(--amber-bright)',
    'stroke-width': 2.4, 'stroke-dasharray': '5 3' }));
  rows.forEach(function (r) {
    f.svg.appendChild(el('circle', { cx: f.fx(r.hidden_delta), cy: f.fy(r.profit_per_match),
      r: 3, fill: 'var(--sage)' }));
    f.svg.appendChild(el('circle', { cx: f.fx(r.hidden_delta), cy: fyM(r.median_matches_to_flag),
      r: 3, fill: 'var(--amber-bright)' }));
  });
  txt(svg, f.fx(rows[0].hidden_delta) + 8, fyM(rows[0].median_matches_to_flag) - 8,
    rows[0].median_matches_to_flag + ' matches', 'var(--amber-bright)', 'start', 9.5, 600);
  txt(svg, f.W - f.R - 4, fyM(rows[rows.length - 1].median_matches_to_flag) - 8,
    rows[rows.length - 1].median_matches_to_flag + ' matches', 'var(--amber-bright)', 'end', 9.5, 600);

  var lo = rows[0], hi = rows[rows.length - 1];
  tag.className = 'tag idle';
  tag.textContent = 'caught ' + fmtPct(lo.caught_rate) + ' \u2192 ' + fmtPct(hi.caught_rate);
  cap.innerHTML = 'Hiding only <b>' + lo.hidden_delta + '</b> Elo, the shark survives a median of '
    + '<b class="amber">' + lo.median_matches_to_flag + '</b> matches and is caught <b>'
    + fmtPct(lo.caught_rate) + '</b> of the time rather than the <b>100%</b> every other level of '
    + 'concealment gets - but earns just <b class="sage">' + fmtSigned(lo.profit_per_match, 3)
    + '</b> a match. Hiding <b>'
    + hi.hidden_delta + '</b> Elo pays <b class="sage">' + fmtSigned(hi.profit_per_match, 3)
    + '</b> and lasts <b>' + hi.median_matches_to_flag + '</b>. Chance is a real shield, and it is '
    + 'rented at roughly the price of the thing it protects.';
})();

})();
