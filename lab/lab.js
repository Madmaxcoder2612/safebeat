/* SafeBeat Lab: replays results precomputed by the Python code (data.js).
   No network access, no storage, no cookies. Text is set with textContent only. */
(function () {
  'use strict';

  var D = window.SAFEBEAT_LAB;
  var app = document.getElementById('app');
  if (!D) {
    var warn = document.createElement('p');
    warn.className = 'wrap notice';
    warn.textContent = 'The data file (data.js) is missing, so the lab cannot run.';
    document.body.insertBefore(warn, app);
    return;
  }

  var FAULT_LABEL = {
    noise: 'Electrical noise',
    wander: 'Slow baseline drift (breathing, sweat)',
    dropout: 'Lost samples (zeros)',
    electrode_off: 'Electrode lifts off (flat line)',
    clipping: 'Amplifier saturates (clipping)',
    motion: 'Movement artefact'
  };
  var REASON_TEXT = {
    flat_frac: 'the signal is flat in places, so the electrode may be off',
    hf_noise: 'there is too much fast noise',
    peak_amp_cv: 'the beat heights are too uneven',
    beat_dev: 'the beats do not look alike, which often means movement',
    n_beats: 'the number of detected beats is not plausible'
  };
  var WARN_TEXT = { wander: 'slow baseline drift', clip_frac: 'clipped peaks' };
  var STRICT_LABEL = ['Strictest (safest)', 'Strict', 'Middle', 'Loose', 'Loosest (answers most)'];
  var CLASS_NAME = { regular: 'Regular', irregular: 'Irregular' };

  var $ = function (id) { return document.getElementById(id); };
  var personSel = $('person'), faultSel = $('fault'), sevIn = $('sev'), strictIn = $('strict');

  function add(parent, tag, text, attrs) {
    var el = document.createElement(tag);
    if (text) el.textContent = text;
    if (attrs) Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    parent.appendChild(el);
    return el;
  }

  D.windows.forEach(function (w) { add(personSel, 'option', w.name, { value: w.id }); });
  D.meta.faults.forEach(function (f) { add(faultSel, 'option', FAULT_LABEL[f] || f, { value: f }); });
  faultSel.value = 'electrode_off';
  sevIn.value = '7';

  // grid lines for the waveform
  var grid = $('grid');
  var NS = 'http://www.w3.org/2000/svg';
  for (var s = 0; s <= 10; s++) {
    var gl = document.createElementNS(NS, 'line');
    gl.setAttribute('x1', s * 100); gl.setAttribute('x2', s * 100);
    gl.setAttribute('y1', 0); gl.setAttribute('y2', 220);
    grid.appendChild(gl);
  }

  function truthOf(wid) {
    return D.windows.filter(function (w) { return w.id === wid; })[0].truth;
  }
  function pct(v) { return Math.round(v * 100) + '%'; }

  function drawTrace(sig) {
    var n = sig.length, pts = [];
    for (var i = 0; i < n; i++) {
      var y = 110 - (sig[i] / 1000) * 40;
      pts.push((i / (n - 1) * 1000).toFixed(1) + ',' + y.toFixed(1));
    }
    $('trace').setAttribute('points', pts.join(' '));
  }

  function setCard(cardId, kind) {
    var card = $(cardId);
    card.className = 'card ' + kind;
  }
  function setChip(id, kind, text) {
    var chip = $(id);
    chip.className = 'chip ' + kind;
    chip.textContent = text;
  }

  function update() {
    var wid = personSel.value, fault = faultSel.value;
    var sevIdx = parseInt(sevIn.value, 10), strictIdx = parseInt(strictIn.value, 10);
    var truth = truthOf(wid);
    var item = D.cases[wid][fault][sevIdx];
    var sev = D.meta.severities[sevIdx];
    $('sev-out').textContent = sev.toFixed(1);
    $('strict-out').textContent = STRICT_LABEL[strictIdx];
    $('truth').textContent = 'True rhythm of this person: ' + CLASS_NAME[truth].toLowerCase() +
      '. The systems below do not know this; we use it only to mark answers right or wrong.';
    drawTrace(item.sig);

    // naive model
    var nName = D.meta.classes[item.naive.label];
    var nRight = nName === truth;
    $('naive-verdict').textContent = CLASS_NAME[nName];
    setChip('naive-chip', nRight ? 'good' : 'bad', nRight ? 'Correct' : 'Wrong, and it did not hesitate');
    setCard('naive-card', nRight ? 'good' : 'bad');
    $('naive-detail').textContent = 'Always answers. Says ' + pct(item.naive.p_irregular) + ' irregular.';

    // SafeBeat
    var d = item.safe[String(D.meta.margins[strictIdx])];
    var story;
    if (d.label === null) {
      $('safe-verdict').textContent = "Can't tell";
      setChip('safe-chip', 'hold', 'Held back (safe)');
      setCard('safe-card', 'hold');
      var why;
      if (d.reason.indexOf('bad_signal:') === 0) {
        why = d.reason.slice(11).split('+').map(function (k) { return REASON_TEXT[k] || k; }).join('; ');
        $('safe-detail').textContent = 'Bad signal: ' + why + '.';
      } else {
        why = 'the signal is fine but the rhythm is too close to call';
        $('safe-detail').textContent = 'The signal looks usable, but ' + why + '.';
      }
      story = nRight
        ? 'The naive model happened to be right this time, but nothing told it the signal was damaged. SafeBeat held back: ' + why + '.'
        : 'The naive model confidently gave a wrong answer. SafeBeat held back: ' + why + '.';
    } else {
      var sName = D.meta.classes[d.label];
      var sRight = sName === truth;
      $('safe-verdict').textContent = CLASS_NAME[sName];
      setChip('safe-chip', sRight ? 'good' : 'bad', sRight ? 'Correct' : 'Wrong');
      setCard('safe-card', sRight ? 'good' : 'bad');
      var w = d.warn.map(function (k) { return WARN_TEXT[k] || k; });
      $('safe-detail').textContent = w.length
        ? 'Answered. Warning: ' + w.join(', ') + ' (does not affect beat timing).'
        : 'Answered. The signal passed the quality checks.';
      story = sRight
        ? (sev === 0 ? 'Clean signal: both systems answer correctly.' : 'Both systems answer correctly here. The damage did not hurt this example.')
        : 'SafeBeat answered wrongly here. This is a real limit of the gate: see "Limits" below.';
    }
    $('story').textContent = story;
    updateBars(fault, sevIdx);
  }

  function updateBars(fault, sevIdx) {
    var sev = D.meta.severities[sevIdx];
    var row = D.sweep.filter(function (r) { return r.fault === fault && Math.abs(r.severity - sev) < 1e-6; })[0];
    var bars = $('bars');
    while (bars.firstChild) bars.removeChild(bars.firstChild);
    if (!row) return;
    $('score-intro').textContent = FAULT_LABEL[fault] + ', strength ' + sev.toFixed(1) + ', across many people:';
    [
      ['Naive: wrong answers', row.naive_wrong_rate, 'bad'],
      ['SafeBeat: wrong answers', row.safebeat_wrong_rate, 'bad'],
      ['SafeBeat: answered at all', row.safebeat_answer_rate, 'neutral']
    ].forEach(function (b) {
      var r = add(bars, 'div', null, { 'class': 'bar-row' });
      add(r, 'span', b[0]);
      var track = add(r, 'div', null, { 'class': 'bar-track' });
      var fill = add(track, 'div', null, { 'class': 'bar-fill ' + b[2] });
      fill.style.width = Math.max(0, Math.min(100, b[1] * 100)) + '%';
      add(r, 'span', pct(b[1]));
    });
  }

  // trade-off scatter: share answered (x) vs share answered wrong (y)
  function drawTradeoff() {
    var svg = $('trade'), L = 56, R = 500, T = 16, B = 276;
    var series = [
      { fault: 'noise', sev: 0.3, color: '#2c7a4b', name: 'Noise, strength 0.3' },
      { fault: 'noise', sev: 0.5, color: '#1f5f8b', name: 'Noise, strength 0.5' },
      { fault: 'motion', sev: 0.3, color: '#b9770e', name: 'Movement, strength 0.3' },
      { fault: 'motion', sev: 0.5, color: '#a23b8c', name: 'Movement, strength 0.5' }
    ];
    var maxWrong = 0.2;
    series.forEach(function (s) {
      D.margin.forEach(function (r) {
        if (r.fault === s.fault && Math.abs(r.severity - s.sev) < 1e-6) maxWrong = Math.max(maxWrong, r.safebeat_wrong_rate);
      });
    });
    maxWrong = Math.ceil(maxWrong * 10) / 10;
    function X(v) { return L + v * (R - L); }
    function Y(v) { return B - (v / maxWrong) * (B - T); }
    function line(x1, y1, x2, y2, color) {
      var l = document.createElementNS(NS, 'line');
      l.setAttribute('x1', x1); l.setAttribute('y1', y1); l.setAttribute('x2', x2); l.setAttribute('y2', y2);
      l.setAttribute('stroke', color); l.setAttribute('stroke-width', '1');
      svg.appendChild(l);
    }
    function text(x, y, t, anchor) {
      var e = document.createElementNS(NS, 'text');
      e.setAttribute('x', x); e.setAttribute('y', y); e.setAttribute('font-size', '12');
      e.setAttribute('fill', 'currentColor'); e.setAttribute('text-anchor', anchor || 'middle');
      e.textContent = t;
      svg.appendChild(e);
    }
    svg.style.color = 'inherit';
    for (var i = 0; i <= 4; i++) {
      var gy = T + (B - T) * i / 4;
      line(L, gy, R, gy, 'rgba(128,128,128,0.3)');
      text(L - 8, gy + 4, Math.round((maxWrong * (1 - i / 4)) * 100) + '%', 'end');
    }
    for (var j = 0; j <= 4; j++) {
      var gx = L + (R - L) * j / 4;
      line(gx, B, gx, B + 5, 'rgba(128,128,128,0.6)');
      text(gx, B + 20, Math.round(j * 25) + '%');
    }
    text((L + R) / 2, B + 42, 'share of windows answered (availability)');
    var yl = document.createElementNS(NS, 'text');
    yl.setAttribute('transform', 'translate(14,' + ((T + B) / 2) + ') rotate(-90)');
    yl.setAttribute('text-anchor', 'middle'); yl.setAttribute('font-size', '12'); yl.setAttribute('fill', 'currentColor');
    yl.textContent = 'share answered WRONG';
    svg.appendChild(yl);
    var legend = $('trade-legend');
    series.forEach(function (s) {
      var rows = D.margin.filter(function (r) { return r.fault === s.fault && Math.abs(r.severity - s.sev) < 1e-6; })
        .sort(function (a, b) { return a.margin - b.margin; });
      var pts = rows.map(function (r) { return X(r.safebeat_answer_rate).toFixed(1) + ',' + Y(r.safebeat_wrong_rate).toFixed(1); });
      var pl = document.createElementNS(NS, 'polyline');
      pl.setAttribute('points', pts.join(' ')); pl.setAttribute('fill', 'none');
      pl.setAttribute('stroke', s.color); pl.setAttribute('stroke-width', '2');
      svg.appendChild(pl);
      rows.forEach(function (r) {
        var c = document.createElementNS(NS, 'circle');
        c.setAttribute('cx', X(r.safebeat_answer_rate)); c.setAttribute('cy', Y(r.safebeat_wrong_rate));
        c.setAttribute('r', 4.5); c.setAttribute('fill', s.color);
        svg.appendChild(c);
      });
      var li = add(legend, 'li', s.name);
      var dot = document.createElement('i');
      dot.style.background = s.color;
      li.insertBefore(dot, li.firstChild);
    });
  }

  [personSel, faultSel, sevIn, strictIn].forEach(function (el) {
    el.addEventListener('input', update);
    el.addEventListener('change', update);
  });
  drawTradeoff();
  app.hidden = false;
  update();
})();
