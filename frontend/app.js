/* ==========================================================================
   autobranch — frontend
   Vanilla ES2020. No build step, no dependencies, no network except the
   same-origin JSON API described in DESIGN.md §6.

   Two knobs, both read off `window` so they can be set from the console or
   from a one-line <script> before this file:

     window.AUTOBRANCH_MOCK     = true           // never touch the network
     window.AUTOBRANCH_API_BASE = "http://host"  // point at another origin

   If the API is unreachable (network error, 404, 5xx) the client latches into
   mock mode and serves a built-in fixture, so the UI is fully reviewable
   before the backend lands.
   ========================================================================== */
(function () {
  "use strict";

  /* ───────────────────────────── config ───────────────────────────── */

  var API_BASE = String(window.AUTOBRANCH_API_BASE || "").replace(/\/+$/, "");
  var MOCK = window.AUTOBRANCH_MOCK === true;
  var POLL_MS = 3000;

  var LIVE_STATES = { pending: true, running: true, evaluating: true };
  var STATUS_ORDER = {
    running: 0,
    evaluating: 1,
    pending: 2,
    succeeded: 3,
    abandoned: 4,
    failed: 5
  };

  var state = {
    route: null,
    runId: null,
    run: null,
    selected: null,
    sort: { key: "score", dir: "desc" },
    generation: 0,
    pollTimer: null
  };

  /* ───────────────────────────── helpers ───────────────────────────── */

  function el(id) { return document.getElementById(id); }

  function fieldGetter(root) {
    return function (name) { return root.querySelector('[data-f="' + name + '"]'); };
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function clamp(n, lo, hi) { return Math.min(hi, Math.max(lo, n)); }

  function clone(value) {
    if (typeof structuredClone === "function") { return structuredClone(value); }
    return JSON.parse(JSON.stringify(value));
  }

  function fmtScore(v) {
    if (v == null || isNaN(v)) { return "—"; }
    var abs = Math.abs(v);
    if (abs >= 1000) { return v.toFixed(0); }
    if (abs >= 10) { return v.toFixed(2); }
    return v.toFixed(3);
  }

  function fmtUsd(v) {
    if (v == null || isNaN(v)) { return "—"; }
    return "$" + Number(v).toFixed(2);
  }

  function fmtInt(v) {
    if (v == null || isNaN(v)) { return "—"; }
    return Number(v).toLocaleString("en-US");
  }

  function fmtTokens(v) {
    if (v == null || isNaN(v)) { return "—"; }
    if (v >= 1e6) { return (v / 1e6).toFixed(2) + "M"; }
    if (v >= 1e3) { return (v / 1e3).toFixed(1) + "k"; }
    return String(v);
  }

  function fmtDuration(seconds) {
    if (seconds == null || isNaN(seconds)) { return "—"; }
    var s = Math.max(0, Math.floor(seconds));
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = s % 60;
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    if (h > 0) { return h + ":" + pad(m) + ":" + pad(sec); }
    return pad(m) + ":" + pad(sec);
  }

  function fmtBytes(n) {
    if (n == null || isNaN(n)) { return "—"; }
    var units = ["B", "KB", "MB", "GB", "TB"];
    var i = 0;
    var v = Number(n);
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return (i === 0 ? v.toFixed(0) : v.toFixed(v < 10 ? 2 : 1)) + " " + units[i];
  }

  function fmtClock(ts) {
    // `ts` is an ISO string from the mock fixture but a float epoch in
    // SECONDS from the API (registry.api_payload). A bare new Date(seconds)
    // renders every live log line as 1970, so normalise here rather than
    // changing the API contract, which is "exactly api_payload".
    var d = typeof ts === "number"
      ? new Date(ts < 1e12 ? ts * 1000 : ts)
      : new Date(ts);
    if (isNaN(d.getTime())) { return "--:--:--"; }
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  }

  function truncate(text, max) {
    var s = String(text == null ? "" : text);
    if (s.length <= max) { return s; }
    return s.slice(0, Math.max(0, max - 1)) + "…";
  }

  function shortId(id) { return String(id || "").replace(/^b-/, ""); }

  function statusClass(status) {
    return STATUS_ORDER[status] === undefined ? "pending" : status;
  }

  /* Human-readable one-liners for a reward config, used on tree nodes. */
  function weightsSummary(cfg) {
    if (!cfg || !cfg.weights) { return "no weights declared"; }
    var keys = Object.keys(cfg.weights);
    keys.sort(function (a, b) {
      return Math.abs(cfg.weights[b]) - Math.abs(cfg.weights[a]);
    });
    return keys.map(function (k) {
      var v = cfg.weights[k];
      var sign = v > 0 ? "+" : "";
      return k.replace(/^damage_/, "dmg_") + " " + sign + v;
    }).join("  ");
  }

  function opsSummary(cfg) {
    if (!cfg) { return "—"; }
    var bits = [];
    (cfg.shaping || []).forEach(function (s) { bits.push(s.kind + ":" + s.signal); });
    (cfg.transforms || []).forEach(function (t) { bits.push(t.op + ":" + t.signal); });
    if (cfg.expr) { bits.push("expr"); }
    if (cfg.anneal) { bits.push("anneal"); }
    if (!bits.length) { bits.push("no shaping"); }
    return bits.join(" · ");
  }

  function bestScoreOf(branch) {
    if (!branch || !branch.scores || !branch.scores.length) { return null; }
    var direction = (state.run && state.run.objective && state.run.objective.direction) || "maximize";
    var vals = branch.scores.map(function (p) { return p.score; });
    return direction === "minimize" ? Math.min.apply(null, vals) : Math.max.apply(null, vals);
  }

  function lastScoreOf(branch) {
    if (!branch || !branch.scores || !branch.scores.length) { return null; }
    return branch.scores[branch.scores.length - 1];
  }

  /* ───────────────────────────── mock fixture ─────────────────────────────
     A run on the environment DESIGN.md generalises from: nine branches, four
     levels, two abandoned lines, and the 0.000 → 0.990 story the framework
     exists to make cheap.
     ------------------------------------------------------------------------ */

  var FIXTURE = {
    env_name: "street-fighter-iii",
    objective: { name: "win_rate_vs_cpu8", direction: "maximize" },
    constraints: "must posttrain Qwen/Qwen3-4B\nstay under $50\nmust beat 76% win rate vs the level-8 CPU\nno GPU larger than an L40S",
    budget: { usd: 50, hours: 6, max_branches: 8 },
    branches: [
      {
        branch_id: "b-0a1f",
        parent_id: null,
        depth: 0,
        status: "succeeded",
        spend_usd: 3.18,
        tokens: 41200,
        reward_config: {
          weights: { round_win: 1.0, damage_dealt: 0.1, damage_taken: -0.1 }
        },
        scores: [
          { step: 0, score: 0.0 }, { step: 2000, score: 0.0 },
          { step: 4000, score: 0.012 }, { step: 6000, score: 0.104 },
          { step: 8000, score: 0.189 }, { step: 10000, score: 0.214 }
        ],
        rationale: "Baseline straight off the manifest: terminal round_win dominates, with a small symmetric damage trade so the first few thousand steps have some gradient to follow. Nothing clever here on purpose — this is the reference score every child gets measured against."
      },
      {
        branch_id: "b-7c22",
        parent_id: "b-0a1f",
        depth: 1,
        status: "succeeded",
        spend_usd: 5.42,
        tokens: 96400,
        reward_config: {
          weights: { round_win: 1.0, damage_dealt: 0.08, damage_taken: -0.12 },
          shaping: [{ kind: "potential", signal: "health_margin", gamma: 0.99, scale: 0.5 }],
          anneal: { shaping: { from: 1.0, to: 0.0, by_fraction: 0.8 } }
        },
        scores: [
          { step: 10000, score: 0.214 }, { step: 12000, score: 0.301 },
          { step: 14000, score: 0.388 }, { step: 16000, score: 0.502 },
          { step: 18000, score: 0.604 }, { step: 20000, score: 0.651 }
        ],
        rationale: "Potential-based shaping on health_margin, annealed to zero over the first 80% of training. Potential shaping is policy-invariant at the optimum, so this buys early signal without changing what is optimal — and the anneal means the last 20% of training sees the objective-aligned reward alone."
      },
      {
        branch_id: "b-3e90",
        parent_id: "b-0a1f",
        depth: 1,
        status: "abandoned",
        spend_usd: 4.07,
        tokens: 73100,
        reward_config: {
          weights: { damage_dealt: 1.0, damage_taken: -0.2, round_win: 0.2 },
          transforms: [{ op: "scale", signal: "damage_dealt", by: 2.0 }]
        },
        scores: [
          { step: 10000, score: 0.021 }, { step: 12000, score: 0.031 },
          { step: 14000, score: 0.029 }, { step: 16000, score: 0.031 },
          { step: 18000, score: 0.028 }
        ],
        rationale: "Invert the baseline's emphasis: make chip damage the primary reward and the round outcome a tiebreak. The hypothesis was that dense damage reward would bootstrap faster than sparse round outcomes.",
        abandoned_reason: "Four consecutive evaluations with no objective improvement (0.031 → 0.029 → 0.031 → 0.028) while self-play win rate climbed to 0.812. The policy learned to farm the reward, not to win: it pokes at range and never commits. Recorded in losers.json so no sibling pays for this again."
      },
      {
        branch_id: "b-9b41",
        parent_id: "b-0a1f",
        depth: 1,
        status: "running",
        spend_usd: 2.96,
        tokens: 58800,
        reward_config: {
          weights: { round_win: 1.0 }
        },
        scores: [
          { step: 10000, score: 0.188 }, { step: 12000, score: 0.242 },
          { step: 14000, score: 0.333 }, { step: 16000, score: 0.371 },
          { step: 18000, score: 0.407 }
        ],
        rationale: "The control arm: reward *is* the objective, nothing else. Slower early, but it cannot be farmed, and if it eventually catches the shaped branches then every shaping term in the tree is overhead rather than help. Worth one branch to know."
      },
      {
        branch_id: "b-c118",
        parent_id: "b-7c22",
        depth: 2,
        status: "succeeded",
        spend_usd: 6.31,
        tokens: 128500,
        reward_config: {
          weights: { round_win: 1.0, damage_dealt: 0.08, damage_taken: -0.12 },
          shaping: [{ kind: "potential", signal: "health_margin", gamma: 0.99, scale: 0.5 }],
          transforms: [{ op: "clip", signal: "damage_dealt", lo: 0.0, hi: 0.2 }],
          anneal: { shaping: { from: 1.0, to: 0.0, by_fraction: 0.8 } }
        },
        scores: [
          { step: 20000, score: 0.651 }, { step: 23000, score: 0.712 },
          { step: 26000, score: 0.780 }, { step: 29000, score: 0.841 },
          { step: 32000, score: 0.883 }
        ],
        rationale: "Keep the parent exactly, add one clip: damage_dealt saturates at 0.2 per step. read_tree shows b-3e90 died farming chip damage, so cap the ceiling on that term rather than remove it. One variable changed from the parent, which is the only way the delta means anything."
      },
      {
        branch_id: "b-2f55",
        parent_id: "b-7c22",
        depth: 2,
        status: "abandoned",
        spend_usd: 3.74,
        tokens: 81900,
        reward_config: {
          weights: { round_win: 1.0, damage_taken: -0.4 },
          shaping: [{ kind: "potential", signal: "health_margin", gamma: 0.995, scale: 1.6 }],
          expr: "0.3 * max(health_margin, 0) ** 2"
        },
        scores: [
          { step: 20000, score: 0.112 }, { step: 23000, score: 0.109 },
          { step: 26000, score: 0.112 }
        ],
        rationale: "Push the margin story much harder: a heavy shaping scale plus a quadratic bonus on positive health margin through the expr escape hatch, and a steep penalty for taking damage.",
        abandoned_reason: "Per-branch GPU-minute budget exhausted at 0.112, flat across three evaluations. The quadratic margin term dominates the terminal round_win by roughly 6x, so the policy optimises for a healthy stalemate — it blocks, retreats, and times out rounds it is winning on health."
      },
      {
        branch_id: "b-5a07",
        parent_id: "b-7c22",
        depth: 2,
        status: "evaluating",
        spend_usd: 5.08,
        tokens: 104200,
        reward_config: {
          weights: { round_win: 1.0, damage_dealt: 0.08, damage_taken: -0.12 },
          shaping: [{ kind: "potential", signal: "health_margin", gamma: 0.99, scale: 0.5 }],
          anneal: { shaping: { from: 1.0, to: 0.0, by_fraction: 0.4 } }
        },
        scores: [
          { step: 20000, score: 0.651 }, { step: 23000, score: 0.698 },
          { step: 26000, score: 0.744 }, { step: 29000, score: 0.790 }
        ],
        rationale: "Same as the parent but anneal the shaping away twice as fast (40% of training rather than 80%). If the shaping is only a bootstrap, dropping it earlier should cost nothing and hand the policy more objective-aligned steps."
      },
      {
        branch_id: "b-8d3c",
        parent_id: "b-9b41",
        depth: 2,
        status: "running",
        spend_usd: 2.11,
        tokens: 44600,
        reward_config: {
          weights: { round_win: 1.0 },
          transforms: [{ op: "scale", signal: "round_win", by: 3.0 }]
        },
        scores: [
          { step: 18000, score: 0.407 }, { step: 21000, score: 0.431 },
          { step: 24000, score: 0.455 }
        ],
        rationale: "Control arm with a larger advantage scale on the only reward term, to test whether the sparse line is merely under-scaled relative to the shaped branches rather than genuinely slower."
      },
      {
        branch_id: "b-4d7e",
        parent_id: "b-c118",
        depth: 3,
        status: "succeeded",
        spend_usd: 7.86,
        tokens: 161300,
        reward_config: {
          weights: { round_win: 1.0, damage_dealt: 0.06, damage_taken: -0.14 },
          shaping: [{ kind: "potential", signal: "health_margin", gamma: 0.99, scale: 0.35 }],
          transforms: [
            { op: "clip", signal: "damage_dealt", lo: 0.0, hi: 0.2 },
            { op: "clip", signal: "damage_taken", lo: -0.25, hi: 0.0 }
          ],
          anneal: { shaping: { from: 1.0, to: 0.0, by_fraction: 0.6 } }
        },
        scores: [
          { step: 32000, score: 0.883 }, { step: 35000, score: 0.921 },
          { step: 38000, score: 0.954 }, { step: 41000, score: 0.972 },
          { step: 44000, score: 0.986 }, { step: 47000, score: 0.990 }
        ],
        rationale: "Clip both damage terms symmetrically, lean the shaping scale down to 0.35, and shorten the anneal to 0.6. The parent's clip proved the ceiling mattered on the dealing side; this tests whether the same ceiling on the taking side removes the residual flinching behaviour. It does — 0.883 → 0.990 against the level-8 CPU."
      }
    ],
    events: [
      { t: 0, level: "info", branch_id: null, message: "bundle accepted: street-fighter-iii v1 · 4 signals · objective win_rate_vs_cpu8 (maximize)" },
      { t: 4, level: "info", branch_id: null, message: "constraints parsed: base_model=Qwen/Qwen3-4B · usd<=50 · objective floor 0.760 · gpu<=L40S" },
      { t: 11, level: "info", branch_id: null, message: "image built in 94s · container cached for the rest of the run" },
      { t: 18, level: "info", branch_id: "b-0a1f", message: "propose_reward accepted · 3 weights, no shaping · config hash 4f1c9a" },
      { t: 640, level: "info", branch_id: "b-0a1f", message: "evaluate() → 0.214 over 200 matches vs cpu8" },
      { t: 700, level: "info", branch_id: "b-7c22", message: "fork from b-0a1f · snapshot_directory(/workspace) mounted in 2.4s" },
      { t: 706, level: "info", branch_id: "b-3e90", message: "fork from b-0a1f · snapshot_directory(/workspace) mounted in 2.1s" },
      { t: 712, level: "info", branch_id: "b-9b41", message: "fork from b-0a1f · snapshot_directory(/workspace) mounted in 2.6s" },
      { t: 2180, level: "warn", branch_id: "b-3e90", message: "self-play win rate 0.812 but objective flat at 0.031 — reward and objective have decoupled" },
      { t: 2260, level: "warn", branch_id: "b-3e90", message: "abandon(): no objective improvement over 4 evaluations · written to losers.json" },
      { t: 2410, level: "info", branch_id: "b-7c22", message: "evaluate() → 0.651 · best in tree, forking 3 children" },
      { t: 3600, level: "info", branch_id: "b-c118", message: "read_tree(): b-3e90 died farming damage_dealt — clipping that term instead of dropping it" },
      { t: 4120, level: "warn", branch_id: "b-2f55", message: "abandon(): GPU-minute budget exhausted at 0.112 · expr term dominating round_win ~6x" },
      { t: 5240, level: "info", branch_id: "b-c118", message: "evaluate() → 0.883 · objective floor 0.760 cleared" },
      { t: 5320, level: "info", branch_id: "b-4d7e", message: "fork from b-c118 · symmetric clip on damage_taken" },
      { t: 6910, level: "info", branch_id: "b-4d7e", message: "evaluate() → 0.954 · successive halving keeps this line, prunes 1 sibling" },
      { t: 7280, level: "info", branch_id: "b-4d7e", message: "evaluate() → 0.990 · new best in tree" },
      { t: 7390, level: "info", branch_id: null, message: "consolidation staged for artifacts/run_8fc2a1 · weights 175.8 MB" }
    ]
  };

  /* ───────────────────────────── mock backend ───────────────────────────── */

  var Mock = {
    runs: {},
    order: [],
    seq: 0,

    newId: function () {
      this.seq += 1;
      var hex = (0x100000 + this.seq * 0x9e37 + 0x8fc2a1) .toString(16).slice(-6);
      return "run_" + hex;
    },

    record: function (id, opts) {
      var o = opts || {};
      var branches = clone(FIXTURE.branches).map(function (b) {
        var final = b.status;
        b.final_status = final;
        b.full_scores = b.scores;
        b.shown = 0;
        return b;
      });

      var seeded = o.seeded === true;
      branches.forEach(function (b) {
        if (!seeded) { b.shown = 0; return; }
        b.shown = LIVE_STATES[b.final_status]
          ? Math.max(1, b.full_scores.length - 2)
          : b.full_scores.length;
      });
      if (!seeded) { branches[0].shown = 1; }

      return {
        run_id: id,
        seeded: seeded,
        env_name: o.env_name || FIXTURE.env_name,
        objective: clone(FIXTURE.objective),
        constraints: o.constraints || FIXTURE.constraints,
        budget: o.budget || clone(FIXTURE.budget),
        state: seeded ? "running" : "pending",
        elapsed_s: seeded ? 7412 : 0,
        spend_usd: seeded ? 31.4 : 0,
        created_ms: Date.now() - (seeded ? 7412 : 0) * 1000,
        revealed: seeded ? branches.length : 1,
        ticks: 0,
        branches: branches,
        events: seeded
          ? clone(FIXTURE.events)
          : [
              { t: 0, level: "info", branch_id: null, message: "bundle accepted · manifest parsed · " + (o.signals || 4) + " signals declared" },
              { t: 1, level: "info", branch_id: null, message: "constraints parsed · budget cap " + fmtUsd((o.budget || FIXTURE.budget).usd) + " / " + (o.budget || FIXTURE.budget).hours + "h / " + (o.budget || FIXTURE.budget).max_branches + " concurrent branches" },
              { t: 2, level: "info", branch_id: "b-0a1f", message: "master agent wrote the baseline reward config · training started" }
            ]
      };
    },

    seed: function () {
      if (this.runs.run_8fc2a1) { return; }
      var rec = this.record("run_8fc2a1", { seeded: true });
      this.runs[rec.run_id] = rec;
      this.order.unshift(rec.run_id);
    },

    activeCount: function (rec) {
      var n = 0;
      rec.branches.slice(0, rec.revealed).forEach(function (b) {
        if (b.shown < b.full_scores.length) { n += 1; }
      });
      return n;
    },

    push: function (rec, level, branchId, message) {
      rec.events.push({ t: rec.elapsed_s, level: level, branch_id: branchId, message: message });
      if (rec.events.length > 400) { rec.events.splice(0, rec.events.length - 400); }
    },

    /* Advance the fixture by one poll interval so the UI has something to be
       live about: branches gain evaluations, new branches fork in, and the
       run eventually consolidates. */
    tick: function (rec) {
      if (!LIVE_STATES[rec.state]) { return; }
      rec.ticks += 1;
      rec.elapsed_s += POLL_MS / 1000;

      if (rec.state === "pending" && rec.ticks >= 1) { rec.state = "running"; }

      var active = this.activeCount(rec);
      rec.spend_usd = Math.round((rec.spend_usd + 0.018 * Math.max(1, active)) * 100) / 100;

      var self = this;
      if (rec.ticks % 2 === 0) {
        rec.branches.slice(0, rec.revealed).forEach(function (b) {
          if (b.shown >= b.full_scores.length) { return; }
          b.shown += 1;
          var point = b.full_scores[b.shown - 1];
          if (b.shown === b.full_scores.length) {
            if (b.final_status === "abandoned") {
              self.push(rec, "warn", b.branch_id, "abandon(): " + truncate(b.abandoned_reason || "line stopped paying", 120));
            } else {
              self.push(rec, "info", b.branch_id, "evaluate() → " + fmtScore(point.score) + " · branch " + b.final_status);
            }
          } else if (b.shown % 2 === 0) {
            self.push(rec, "info", b.branch_id, "evaluate() → " + fmtScore(point.score) + " at step " + fmtInt(point.step));
          }
        });
      }

      if (rec.ticks % 3 === 0 && rec.revealed < rec.branches.length && active < rec.budget.max_branches) {
        var next = rec.branches[rec.revealed];
        rec.revealed += 1;
        next.shown = 1;
        this.push(rec, "info", next.branch_id, "fork from " + (next.parent_id || "root") + " · snapshot_directory(/workspace) mounted");
      }

      var done = rec.revealed >= rec.branches.length && this.activeCount(rec) === 0;
      if (done) {
        rec.state = "succeeded";
        this.push(rec, "info", null, "tree exhausted · consolidating best branch into artifacts/" + rec.run_id);
      } else if (rec.spend_usd >= rec.budget.usd) {
        rec.state = "succeeded";
        this.push(rec, "warn", null, "spend cap " + fmtUsd(rec.budget.usd) + " reached · stopping branches and consolidating the best line");
      } else if (rec.elapsed_s / 3600 >= rec.budget.hours) {
        rec.state = "succeeded";
        this.push(rec, "warn", null, "wall-clock cap " + rec.budget.hours + "h reached · consolidating the best line");
      }
    },

    visibleBranches: function (rec) {
      return rec.branches.slice(0, rec.revealed).map(function (b) {
        var scores = b.full_scores.slice(0, b.shown);
        /* A branch still producing evaluations shows a live status: its own if
           that status is already live, otherwise "running" until it lands on
           its terminal one. */
        var status = b.shown >= b.full_scores.length
          ? b.final_status
          : (LIVE_STATES[b.final_status] ? b.final_status : "running");
        var frac = b.full_scores.length ? b.shown / b.full_scores.length : 1;
        return {
          branch_id: b.branch_id,
          parent_id: b.parent_id,
          depth: b.depth,
          status: status,
          reward_config: b.reward_config,
          scores: scores,
          spend_usd: Math.round(b.spend_usd * frac * 100) / 100,
          tokens: Math.round(b.tokens * frac),
          rationale: b.rationale,
          abandoned_reason: status === "abandoned" ? b.abandoned_reason : null
        };
      });
    },

    best: function (branches) {
      var best = null;
      branches.forEach(function (b) {
        if (!b.scores.length) { return; }
        var top = Math.max.apply(null, b.scores.map(function (p) { return p.score; }));
        if (!best || top > best.score) { best = { branch_id: b.branch_id, score: top }; }
      });
      return best;
    },

    serialize: function (rec) {
      var branches = this.visibleBranches(rec);
      var visible = {};
      branches.forEach(function (b) { visible[b.branch_id] = true; });
      var events = rec.events
        .filter(function (e) { return !e.branch_id || visible[e.branch_id]; })
        .map(function (e) {
          return {
            ts: new Date(rec.created_ms + e.t * 1000).toISOString(),
            level: e.level,
            branch_id: e.branch_id,
            message: e.message
          };
        });
      return {
        run_id: rec.run_id,
        env_name: rec.env_name,
        objective: rec.objective,
        state: rec.state,
        created_at: new Date(rec.created_ms).toISOString(),
        elapsed_s: Math.round(rec.elapsed_s),
        spend_usd: rec.spend_usd,
        budget: rec.budget,
        constraints: rec.constraints,
        best: this.best(branches),
        branches: branches,
        events: events
      };
    },

    artifact: function (rec) {
      var branches = this.visibleBranches(rec);
      var finished = branches.filter(function (b) { return b.status === "succeeded"; });
      var best = this.best(finished);
      if (!best) {
        return { ready: false, files: [], run_command: "", best: null };
      }

      var byId = {};
      branches.forEach(function (b) { byId[b.branch_id] = b; });

      var lineage = [];
      var cursor = byId[best.branch_id];
      while (cursor) {
        lineage.unshift({
          branch_id: cursor.branch_id,
          parent_id: cursor.parent_id,
          reward_config: cursor.reward_config,
          objective_score: Math.max.apply(null, cursor.scores.map(function (p) { return p.score; })),
          rationale: cursor.rationale
        });
        cursor = cursor.parent_id ? byId[cursor.parent_id] : null;
      }

      var losers = branches
        .filter(function (b) { return b.status === "abandoned" || b.status === "failed"; })
        .map(function (b) {
          return {
            branch_id: b.branch_id,
            parent_id: b.parent_id,
            reward_config: b.reward_config,
            best_objective: b.scores.length ? Math.max.apply(null, b.scores.map(function (p) { return p.score; })) : null,
            spend_usd: b.spend_usd,
            reason: b.abandoned_reason
          };
        });

      var manifest = {
        run_id: rec.run_id,
        env: rec.env_name,
        base_model: "Qwen/Qwen3-4B",
        objective: { name: rec.objective.name, direction: rec.objective.direction, score: best.score },
        winning_branch: best.branch_id,
        branches_explored: branches.length,
        spend_usd: rec.spend_usd,
        gpu: "L40S (allocated)",
        trainer: "grpo_lora"
      };

      var runCommand =
        "modal volume get autobranch-artifacts /artifacts/" + rec.run_id + " ./" + rec.run_id + "\n" +
        "modal endpoint create ./" + rec.run_id + "/serve.py \\\n" +
        "  --custom-volume-name autobranch-artifacts \\\n" +
        "  --custom-volume-path /artifacts/" + rec.run_id;

      function jsonFile(name, data, desc) {
        var text = JSON.stringify(data, null, 2);
        return {
          name: name,
          url: "data:application/json;charset=utf-8," + encodeURIComponent(text),
          bytes: new TextEncoder().encode(text).length,
          desc: desc
        };
      }

      var weightsNote =
        "This is mock mode: the real artifact here is the winning branch's checkpoint\n" +
        "(" + best.branch_id + ", objective " + fmtScore(best.score) + "). Start the backend to download the\n" +
        "actual weights bundle.\n";

      return {
        ready: true,
        best: { branch_id: best.branch_id, score: best.score, objective: rec.objective.name },
        run_command: runCommand,
        files: [
          {
            name: "weights-" + best.branch_id + ".tar.zst",
            url: "data:text/plain;charset=utf-8," + encodeURIComponent(weightsNote),
            bytes: 184320112,
            desc: "winning branch's checkpoint, optimizer state stripped"
          },
          jsonFile("manifest.json", manifest, "env, base model, objective score, commands, spend"),
          jsonFile("reward_lineage.json", lineage, "root → winner: every mutation and the score it bought"),
          jsonFile("losers.json", losers, "every abandoned branch and why — this is the science"),
          {
            name: "serve.py",
            url: "data:text/x-python;charset=utf-8," + encodeURIComponent(
              "# runnable inference entrypoint for " + rec.run_id + "\n" +
              "# generated by autobranch — serves " + best.branch_id + " (objective " + fmtScore(best.score) + ")\n"
            ),
            bytes: 1842,
            desc: "runnable inference entrypoint"
          }
        ]
      };
    },

    handle: function (method, path, body) {
      this.seed();
      var m;

      if (method === "POST" && path === "/api/runs") {
        var id = this.newId();
        var rec = this.record(id, {
          constraints: (body && body.constraints) || "",
          budget: (body && body.budget) || clone(FIXTURE.budget),
          env_name: FIXTURE.env_name
        });
        this.runs[id] = rec;
        this.order.unshift(id);
        return { run_id: id };
      }

      if (method === "GET" && path === "/api/runs") {
        var self = this;
        return {
          runs: this.order.map(function (rid) {
            var r = self.runs[rid];
            var branches = self.visibleBranches(r);
            var best = self.best(branches);
            return {
              run_id: r.run_id,
              env_name: r.env_name,
              state: r.state,
              objective_name: r.objective.name,
              best_score: best ? best.score : null,
              spend_usd: r.spend_usd,
              created_at: new Date(r.created_ms).toISOString()
            };
          })
        };
      }

      m = /^\/api\/runs\/([^/]+)\/artifact$/.exec(path);
      if (method === "GET" && m) {
        var ar = this.runs[decodeURIComponent(m[1])];
        if (!ar) { throw new ApiError(404, "unknown run"); }
        return this.artifact(ar);
      }

      m = /^\/api\/runs\/([^/]+)\/stop$/.exec(path);
      if (method === "POST" && m) {
        var sr = this.runs[decodeURIComponent(m[1])];
        if (!sr) { throw new ApiError(404, "unknown run"); }
        if (LIVE_STATES[sr.state]) {
          sr.branches.slice(0, sr.revealed).forEach(function (b) {
            if (b.shown < b.full_scores.length && b.shown > 0) { b.shown = b.full_scores.length; }
          });
          sr.state = "succeeded";
          this.push(sr, "warn", null, "operator stop requested · branches halted, consolidating the best line");
        }
        return { ok: true };
      }

      m = /^\/api\/runs\/([^/]+)$/.exec(path);
      if (method === "GET" && m) {
        var gr = this.runs[decodeURIComponent(m[1])];
        if (!gr) { throw new ApiError(404, "unknown run"); }
        this.tick(gr);
        return this.serialize(gr);
      }

      throw new ApiError(404, "no mock route for " + method + " " + path);
    }
  };

  /* ───────────────────────────── api client ───────────────────────────── */

  function ApiError(status, message) {
    this.name = "ApiError";
    this.status = status;
    this.message = message || "request failed";
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  function enableMock(reason) {
    if (MOCK) { return; }
    MOCK = true;
    var badge = el("mock-badge");
    if (badge) {
      badge.hidden = false;
      badge.title = "No backend reachable (" + reason + ") — rendering the built-in fixture.";
    }
  }

  function request(method, path, body) {
    if (MOCK) {
      return Promise.resolve().then(function () { return Mock.handle(method, path, body); });
    }

    var opts = { method: method, headers: { Accept: "application/json" } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }

    return fetch(API_BASE + path, opts).then(function (res) {
      if (res.status === 404 || res.status >= 500) {
        throw { fallback: true, reason: "HTTP " + res.status };
      }
      if (!res.ok) {
        return res.text().then(function (text) {
          var message = text;
          try { message = JSON.parse(text).detail || text; } catch (err) { /* plain text */ }
          throw new ApiError(res.status, message || ("HTTP " + res.status));
        });
      }
      return res.json();
    }).catch(function (err) {
      if (err && err.fallback) {
        enableMock(err.reason);
        return Mock.handle(method, path, body);
      }
      if (err instanceof ApiError) { throw err; }
      enableMock(err && err.message ? err.message : "network error");
      return Mock.handle(method, path, body);
    });
  }

  var api = {
    createRun: function (body) { return request("POST", "/api/runs", body); },
    listRuns: function () { return request("GET", "/api/runs"); },
    getRun: function (id) { return request("GET", "/api/runs/" + encodeURIComponent(id)); },
    getArtifact: function (id) { return request("GET", "/api/runs/" + encodeURIComponent(id) + "/artifact"); },
    stopRun: function (id) { return request("POST", "/api/runs/" + encodeURIComponent(id) + "/stop", {}); }
  };

  /* ───────────────────────────── theme ───────────────────────────── */

  var THEME_KEY = "autobranch:theme";
  var THEMES = ["system", "light", "dark"];

  function applyTheme(theme) {
    var root = document.documentElement;
    if (theme === "light" || theme === "dark") {
      root.setAttribute("data-theme", theme);
    } else {
      root.removeAttribute("data-theme");
    }
    var btn = el("theme-toggle");
    if (btn) { btn.textContent = "Theme: " + theme; }
  }

  function initTheme() {
    var stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch (err) { stored = null; }
    var theme = THEMES.indexOf(stored) >= 0 ? stored : "system";
    applyTheme(theme);

    var btn = el("theme-toggle");
    if (!btn) { return; }
    btn.addEventListener("click", function () {
      var current = document.documentElement.getAttribute("data-theme") || "system";
      var next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (err) { /* private mode */ }
    });
  }

  /* ───────────────────────────── chrome ───────────────────────────── */

  function updateNav() {
    var links = document.querySelectorAll(".topnav a[data-nav]");
    for (var i = 0; i < links.length; i += 1) {
      var link = links[i];
      var kind = link.getAttribute("data-nav");
      if (kind === "new") {
        link.hidden = false;
        link.href = "#/new";
      } else if (state.runId) {
        link.hidden = false;
        link.href = "#/" + kind + "/" + encodeURIComponent(state.runId);
      } else {
        link.hidden = true;
      }
      if (state.route && kind === state.route.name) {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  }

  function setApiOrigin() {
    var node = el("api-origin");
    if (!node) { return; }
    node.textContent = "api: " + (API_BASE || "same-origin");
  }

  function statusBadge(status) {
    var s = statusClass(status);
    var live = LIVE_STATES[s] ? " pulse" : "";
    return '<span class="badge badge-' + s + '"><span class="dot st-' + s + live + '"></span>' + esc(s) + "</span>";
  }

  function mount(node) {
    var view = el("view");
    view.textContent = "";
    view.appendChild(node);
  }

  function mountMessage(title, body, kind, link) {
    var wrap = document.createElement("div");
    wrap.className = "page";
    wrap.innerHTML =
      '<div class="notice' + (kind ? " is-" + kind : "") + '">' +
      '<p class="notice-title">' + esc(title) + "</p>" +
      "<p>" + esc(body) + "</p>" +
      (link ? '<p class="back"><a href="' + esc(link.href) + '">' + esc(link.text) + "</a></p>" : "") +
      "</div>";
    mount(wrap);
  }

  /* Re-rendering a container by innerHTML drops focus. Polls happen every
     three seconds, so keyboard focus inside the tree or the table has to
     survive them. */
  function withFocusPreserved(container, render) {
    var active = document.activeElement;
    var key = active && container.contains(active) ? active.getAttribute("data-branch") : null;
    render();
    if (!key) { return; }
    var again = container.querySelector('[data-branch="' + key.replace(/["\\]/g, "") + '"]');
    if (again && typeof again.focus === "function") { again.focus(); }
  }

  /* ───────────────────────────── sparkline ───────────────────────────── */

  function sparkline(scores, opts) {
    var o = opts || {};
    var w = o.width || 320;
    var h = o.height || 48;
    var pad = 4;

    if (!scores || !scores.length) {
      return '<p class="spark-empty">no evaluations yet</p>';
    }
    var vals = scores.map(function (p) { return p.score; });
    var lo = Math.min.apply(null, vals);
    var hi = Math.max.apply(null, vals);
    if (hi - lo < 1e-9) { hi = lo + 1e-9; }

    var n = scores.length;
    var pts = scores.map(function (p, i) {
      var x = pad + (n === 1 ? (w - pad * 2) / 2 : (i / (n - 1)) * (w - pad * 2));
      var y = h - pad - ((p.score - lo) / (hi - lo)) * (h - pad * 2);
      return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
    });

    var line = pts.map(function (p) { return p[0] + "," + p[1]; }).join(" ");
    var area = line + " " + pts[n - 1][0] + "," + (h - pad) + " " + pts[0][0] + "," + (h - pad);
    var last = pts[n - 1];

    return '<svg class="spark' + (o.faint ? " spark-faint" : "") + '" viewBox="0 0 ' + w + " " + h +
      '" preserveAspectRatio="none" role="img" aria-label="Objective score history, ' + n +
      " evaluations, latest " + fmtScore(vals[n - 1]) + '">' +
      '<polygon class="spark-area" points="' + area + '"></polygon>' +
      '<line class="spark-base" x1="' + pad + '" y1="' + (h - pad) + '" x2="' + (w - pad) + '" y2="' + (h - pad) + '"></line>' +
      '<polyline class="spark-line" points="' + line + '"></polyline>' +
      '<circle class="spark-dot" cx="' + last[0] + '" cy="' + last[1] + '" r="2.5"></circle>' +
      "</svg>";
  }

  /* ───────────────────────────── tree layout ─────────────────────────────
     A tidy layout in two passes. Pass one walks the forest depth-first and
     hands every leaf the next free column; an internal node takes the mean of
     its children's columns. Because leaf columns are unique and a subtree owns
     a contiguous column range, parents centred over their children can never
     collide with a sibling subtree — no contour shifting required for the
     shapes a reward search produces. Pass two multiplies columns and depth by
     fixed pitches to get pixels.
     ------------------------------------------------------------------------ */

  var NODE_W = 212;
  var NODE_H = 86;
  var GAP_X = 30;
  var GAP_Y = 54;
  var PAD = 18;

  function layoutTree(branches) {
    var byId = {};
    branches.forEach(function (b) { byId[b.branch_id] = b; });

    var children = {};
    var roots = [];
    branches.forEach(function (b) {
      if (b.parent_id && byId[b.parent_id]) {
        if (!children[b.parent_id]) { children[b.parent_id] = []; }
        children[b.parent_id].push(b.branch_id);
      } else {
        roots.push(b.branch_id);
      }
    });

    var columns = {};
    var nextColumn = 0;
    var seen = {};

    function assign(id) {
      if (seen[id]) { return columns[id]; }
      seen[id] = true;
      var kids = children[id] || [];
      if (!kids.length) {
        columns[id] = nextColumn;
        nextColumn += 1;
        return columns[id];
      }
      var cols = kids.map(assign);
      columns[id] = (Math.min.apply(null, cols) + Math.max.apply(null, cols)) / 2;
      return columns[id];
    }

    roots.forEach(assign);
    /* Defensive: a cycle in parent_id would leave nodes unassigned. */
    branches.forEach(function (b) {
      if (columns[b.branch_id] === undefined) {
        columns[b.branch_id] = nextColumn;
        nextColumn += 1;
      }
    });

    var maxDepth = 0;
    var nodes = branches.map(function (b) {
      var depth = typeof b.depth === "number" ? b.depth : 0;
      maxDepth = Math.max(maxDepth, depth);
      return {
        branch: b,
        x: PAD + columns[b.branch_id] * (NODE_W + GAP_X),
        y: PAD + depth * (NODE_H + GAP_Y)
      };
    });

    var byNode = {};
    nodes.forEach(function (n) { byNode[n.branch.branch_id] = n; });

    var edges = [];
    branches.forEach(function (b) {
      if (!b.parent_id || !byNode[b.parent_id]) { return; }
      edges.push({ from: byNode[b.parent_id], to: byNode[b.branch_id] });
    });

    return {
      nodes: nodes,
      edges: edges,
      width: PAD * 2 + Math.max(1, nextColumn) * (NODE_W + GAP_X) - GAP_X,
      height: PAD * 2 + (maxDepth + 1) * (NODE_H + GAP_Y) - GAP_Y
    };
  }

  function renderTree(svg, run) {
    var branches = run.branches || [];
    if (!branches.length) {
      svg.setAttribute("width", "320");
      svg.setAttribute("height", "80");
      svg.innerHTML = '<text class="node-meta" x="18" y="40">no branches yet — the master agent is writing the first reward config</text>';
      return;
    }

    var tree = layoutTree(branches);
    var bestId = run.best && run.best.branch_id;
    svg.setAttribute("width", String(tree.width));
    svg.setAttribute("height", String(tree.height));
    svg.setAttribute("viewBox", "0 0 " + tree.width + " " + tree.height);

    var parts = [];

    tree.edges.forEach(function (e) {
      var x1 = e.from.x + NODE_W / 2;
      var y1 = e.from.y + NODE_H;
      var x2 = e.to.x + NODE_W / 2;
      var y2 = e.to.y;
      var mid = y1 + GAP_Y / 2;
      var faded = e.to.branch.status === "abandoned" || e.to.branch.status === "failed";
      parts.push('<path class="tree-edge' + (faded ? " is-faded" : "") + '" d="M' + x1 + " " + y1 +
        " V" + mid + " H" + x2 + " V" + y2 + '"></path>');
    });

    tree.nodes.forEach(function (n) {
      var b = n.branch;
      var s = statusClass(b.status);
      var abandoned = s === "abandoned" || s === "failed";
      var best = bestId === b.branch_id;
      var score = bestScoreOf(b);
      var classes = ["node", "status-" + s];
      if (abandoned) { classes.push("is-abandoned"); }
      if (best) { classes.push("is-best"); }
      if (state.selected === b.branch_id) { classes.push("is-selected"); }

      var label = b.branch_id + ", " + s + ", objective " + fmtScore(score) +
        (abandoned ? ", abandoned line" : "");

      parts.push(
        '<g class="' + classes.join(" ") + '" tabindex="0" role="button" ' +
        'data-branch="' + esc(b.branch_id) + '" aria-label="' + esc(label) + '">' +
        '<rect class="node-ring" rx="5" x="' + (n.x - 3) + '" y="' + (n.y - 3) + '" width="' + (NODE_W + 6) + '" height="' + (NODE_H + 6) + '"></rect>' +
        '<rect class="node-box" rx="3" x="' + n.x + '" y="' + n.y + '" width="' + NODE_W + '" height="' + NODE_H + '"></rect>' +
        '<line class="node-stripe stripe-' + s + '" x1="' + (n.x + 1.25) + '" y1="' + (n.y + 2) + '" x2="' + (n.x + 1.25) + '" y2="' + (n.y + NODE_H - 2) + '"></line>' +
        '<text class="node-id" x="' + (n.x + 14) + '" y="' + (n.y + 21) + '">' + esc(b.branch_id) + (abandoned ? " ✕" : "") + "</text>" +
        (best ? '<text class="best-tag" x="' + (n.x + NODE_W - 12) + '" y="' + (n.y + 21) + '" text-anchor="end">BEST</text>' : "") +
        '<text class="node-score" x="' + (n.x + 14) + '" y="' + (n.y + 44) + '">' + esc(fmtScore(score)) + "</text>" +
        '<text class="node-status text-' + s + '" x="' + (n.x + NODE_W - 12) + '" y="' + (n.y + 44) + '" text-anchor="end">' + esc(s.toUpperCase()) + "</text>" +
        '<text class="node-meta" x="' + (n.x + 14) + '" y="' + (n.y + 62) + '">' + esc(truncate(weightsSummary(b.reward_config), 29)) + "</text>" +
        '<text class="node-meta" x="' + (n.x + 14) + '" y="' + (n.y + 76) + '">' + esc(truncate(opsSummary(b.reward_config), 29)) + "</text>" +
        "</g>"
      );
    });

    svg.innerHTML = parts.join("");
    svg.__layout = tree;
  }

  function scrollNodeIntoView(scroller, branchId) {
    var svg = scroller.querySelector(".tree-svg");
    if (!svg || !svg.__layout) { return; }
    var hit = null;
    svg.__layout.nodes.forEach(function (n) {
      if (n.branch.branch_id === branchId) { hit = n; }
    });
    if (!hit) { return; }
    var targetLeft = hit.x + NODE_W / 2 - scroller.clientWidth / 2;
    var targetTop = hit.y + NODE_H / 2 - scroller.clientHeight / 2;
    scroller.scrollLeft = clamp(targetLeft, 0, Math.max(0, svg.__layout.width - scroller.clientWidth));
    scroller.scrollTop = clamp(targetTop, 0, Math.max(0, svg.__layout.height - scroller.clientHeight));
  }

  /* ───────────────────────────── detail panel ───────────────────────────── */

  function lineageOf(run, branchId) {
    var byId = {};
    (run.branches || []).forEach(function (b) { byId[b.branch_id] = b; });
    var path = [];
    var cursor = byId[branchId];
    var guard = 0;
    while (cursor && guard < 64) {
      path.unshift(cursor);
      cursor = cursor.parent_id ? byId[cursor.parent_id] : null;
      guard += 1;
    }
    return path;
  }

  function renderDetail(panel, run) {
    if (!state.selected) {
      panel.innerHTML =
        '<div class="detail-head"><div><p class="eyebrow">Branch detail</p>' +
        '<p class="detail-path">Click a node in the tree, or a row in the branch table.</p></div></div>' +
        '<div class="detail-section"><p class="rationale">Every branch keeps its reward config, its score history, its spend, and the rationale the agent wrote before it forked. The abandoned ones keep theirs too — that is what losers.json is for.</p></div>';
      return;
    }

    var branch = null;
    (run.branches || []).forEach(function (b) {
      if (b.branch_id === state.selected) { branch = b; }
    });
    if (!branch) {
      panel.innerHTML = '<p class="empty mono">branch ' + esc(state.selected) + " is no longer in the tree.</p>";
      return;
    }

    var s = statusClass(branch.status);
    var abandoned = s === "abandoned" || s === "failed";
    var score = bestScoreOf(branch);
    var last = lastScoreOf(branch);
    var path = lineageOf(run, branch.branch_id).map(function (b) { return b.branch_id; });

    var html =
      '<div class="detail-head">' +
      "<div>" +
      '<p class="eyebrow">Branch detail</p>' +
      '<p class="detail-id' + (abandoned ? " is-abandoned" : "") + '">' + esc(branch.branch_id) + "</p>" +
      '<p class="detail-path">' + esc(path.join(" → ")) + "</p>" +
      "</div>" +
      statusBadge(s) +
      "</div>" +

      '<div class="detail-section">' +
      '<dl class="kv-grid">' +
      "<div><dt>Best objective</dt><dd" + (run.best && run.best.branch_id === branch.branch_id ? ' class="accent"' : "") + ">" + esc(fmtScore(score)) + "</dd></div>" +
      "<div><dt>Latest eval</dt><dd>" + esc(last ? fmtScore(last.score) : "—") + "</dd></div>" +
      "<div><dt>Depth</dt><dd>" + esc(String(branch.depth == null ? "—" : branch.depth)) + "</dd></div>" +
      "<div><dt>Evaluations</dt><dd>" + esc(String((branch.scores || []).length)) + "</dd></div>" +
      "<div><dt>Spend</dt><dd>" + esc(fmtUsd(branch.spend_usd)) + "</dd></div>" +
      "<div><dt>Agent tokens</dt><dd>" + esc(fmtTokens(branch.tokens)) + "</dd></div>" +
      "</dl>" +
      "</div>" +

      '<div class="detail-section">' +
      '<p class="eyebrow">Objective score history</p>' +
      sparkline(branch.scores, { faint: abandoned, width: 320, height: 48 }) +
      '<p class="detail-path">' +
      esc((branch.scores || []).length
        ? "step " + fmtInt(branch.scores[0].step) + " → " + fmtInt(last.step)
        : "no evaluations recorded") +
      "</p>" +
      "</div>" +

      '<div class="detail-section">' +
      '<p class="eyebrow">Reward config</p>' +
      '<pre class="code mono">' + esc(JSON.stringify(branch.reward_config || {}, null, 2)) + "</pre>" +
      "</div>" +

      '<div class="detail-section">' +
      '<p class="eyebrow">Agent rationale</p>' +
      '<p class="rationale">' + esc(branch.rationale || "No rationale recorded for this branch.") + "</p>" +
      (branch.abandoned_reason
        ? '<p class="note-abandoned"><strong>Abandoned:</strong> ' + esc(branch.abandoned_reason) + "</p>"
        : "") +
      "</div>";

    panel.innerHTML = html;
  }

  /* ───────────────────────────── branch table ───────────────────────────── */

  function sortValue(branch, key) {
    if (key === "score") {
      var v = bestScoreOf(branch);
      return v == null ? -Infinity : v;
    }
    if (key === "status") { return STATUS_ORDER[statusClass(branch.status)]; }
    if (key === "depth") { return branch.depth == null ? 0 : branch.depth; }
    if (key === "spend_usd") { return branch.spend_usd || 0; }
    if (key === "tokens") { return branch.tokens || 0; }
    return String(branch.branch_id || "");
  }

  function renderTable(table, run) {
    var branches = (run.branches || []).slice();
    var key = state.sort.key;
    var dir = state.sort.dir === "asc" ? 1 : -1;

    branches.sort(function (a, b) {
      var va = sortValue(a, key);
      var vb = sortValue(b, key);
      if (va < vb) { return -1 * dir; }
      if (va > vb) { return 1 * dir; }
      return String(a.branch_id).localeCompare(String(b.branch_id));
    });

    var heads = table.querySelectorAll("th");
    for (var i = 0; i < heads.length; i += 1) {
      var btn = heads[i].querySelector(".th-sort");
      if (!btn) { continue; }
      if (btn.getAttribute("data-sort") === key) {
        heads[i].setAttribute("aria-sort", state.sort.dir === "asc" ? "ascending" : "descending");
      } else {
        heads[i].removeAttribute("aria-sort");
      }
    }

    var bestId = run.best && run.best.branch_id;
    var rows = branches.map(function (b) {
      var s = statusClass(b.status);
      var abandoned = s === "abandoned" || s === "failed";
      var cls = ["is-clickable"];
      if (abandoned) { cls.push("is-abandoned"); }
      if (state.selected === b.branch_id) { cls.push("is-selected"); }
      return '<tr class="' + cls.join(" ") + '" data-branch="' + esc(b.branch_id) + '" tabindex="0">' +
        '<td class="cell-id">' + esc(b.branch_id) + "</td>" +
        '<td class="num">' + esc(String(b.depth == null ? "—" : b.depth)) + "</td>" +
        "<td><span class=\"dot st-" + s + '"></span> ' + esc(s) + "</td>" +
        '<td class="num cell-score' + (bestId === b.branch_id ? " is-best" : "") + '">' + esc(fmtScore(bestScoreOf(b))) + "</td>" +
        '<td class="num">' + esc(fmtUsd(b.spend_usd)) + "</td>" +
        '<td class="num">' + esc(fmtTokens(b.tokens)) + "</td>" +
        "</tr>";
    });

    table.querySelector("tbody").innerHTML = rows.join("") ||
      '<tr><td colspan="6">no branches yet</td></tr>';
  }

  /* ───────────────────────────── log ───────────────────────────── */

  function renderLog(list, run) {
    var events = (run.events || []).slice(-120);
    var pinned = list.scrollHeight - list.scrollTop - list.clientHeight < 24;

    list.innerHTML = events.map(function (e) {
      var level = e.level === "warn" || e.level === "error" ? e.level : "info";
      return '<li class="log-level-' + level + '">' +
        '<span class="log-ts">' + esc(fmtClock(e.ts)) + "</span>" +
        (e.branch_id ? '<span class="log-branch">' + esc(e.branch_id) + "</span>" : '<span class="log-branch">master</span>') +
        '<span class="log-msg">' + esc(e.message) + "</span>" +
        "</li>";
    }).join("") || '<li><span class="log-msg">no events yet</span></li>';

    if (pinned) { list.scrollTop = list.scrollHeight; }
  }

  /* ───────────────────────────── new run view ───────────────────────────── */

  function renderNew() {
    var frag = el("tpl-new").content.cloneNode(true);
    var root = frag.firstElementChild;
    mount(frag);

    var form = root.querySelector("#new-run-form");
    var dropzone = root.querySelector("#dropzone");
    var fileInput = root.querySelector("#bundle-file");
    var fileChip = root.querySelector("#file-chip");
    var fileName = root.querySelector("#file-name");
    var fileBytes = root.querySelector("#file-bytes");
    var fileClear = root.querySelector("#file-clear");
    var urlInput = root.querySelector("#bundle-url");
    var paneZip = root.querySelector("#pane-zip");
    var paneGit = root.querySelector("#pane-git");
    var tabs = root.querySelectorAll(".seg");
    var errorNode = root.querySelector("#form-error");
    var submit = root.querySelector("#submit-run");

    var source = "zip";
    var picked = null;

    function showError(message) {
      if (!message) {
        errorNode.hidden = true;
        errorNode.textContent = "";
        return;
      }
      errorNode.hidden = false;
      errorNode.textContent = message;
    }

    /* The two bundle sources are mutually exclusive: switching clears the other. */
    function setSource(next) {
      source = next;
      for (var i = 0; i < tabs.length; i += 1) {
        tabs[i].setAttribute("aria-selected", String(tabs[i].getAttribute("data-source") === next));
      }
      paneZip.hidden = next !== "zip";
      paneGit.hidden = next !== "git";
      if (next === "zip") {
        urlInput.value = "";
      } else {
        clearFile();
      }
      showError("");
    }

    function clearFile() {
      picked = null;
      fileInput.value = "";
      fileChip.hidden = true;
      fileName.textContent = "—";
      fileBytes.textContent = "";
    }

    function setFile(file) {
      if (!file) { return; }
      if (!/\.zip$/i.test(file.name)) {
        showError("Bundle must be a .zip — got " + file.name);
        return;
      }
      picked = file;
      fileName.textContent = file.name;
      fileBytes.textContent = fmtBytes(file.size);
      fileChip.hidden = false;
      if (source !== "zip") { setSource("zip"); }
      showError("");
    }

    for (var i = 0; i < tabs.length; i += 1) {
      tabs[i].addEventListener("click", function (ev) {
        setSource(ev.currentTarget.getAttribute("data-source"));
      });
    }

    dropzone.addEventListener("click", function () { fileInput.click(); });
    dropzone.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        fileInput.click();
      }
    });
    ["dragenter", "dragover"].forEach(function (name) {
      dropzone.addEventListener(name, function (ev) {
        ev.preventDefault();
        dropzone.classList.add("is-over");
      });
    });
    ["dragleave", "dragend"].forEach(function (name) {
      dropzone.addEventListener(name, function () { dropzone.classList.remove("is-over"); });
    });
    dropzone.addEventListener("drop", function (ev) {
      ev.preventDefault();
      dropzone.classList.remove("is-over");
      if (ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files.length) {
        setFile(ev.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files.length) { setFile(fileInput.files[0]); }
    });
    fileClear.addEventListener("click", function (ev) {
      ev.stopPropagation();
      clearFile();
    });
    urlInput.addEventListener("input", function () {
      if (urlInput.value && source !== "git") { setSource("git"); }
    });

    function readBase64(file) {
      return new Promise(function (resolve, reject) {
        var reader = new FileReader();
        reader.onerror = function () { reject(new Error("could not read " + file.name)); };
        reader.onload = function () {
          var result = String(reader.result || "");
          var comma = result.indexOf(",");
          resolve(comma >= 0 ? result.slice(comma + 1) : result);
        };
        reader.readAsDataURL(file);
      });
    }

    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      showError("");

      var constraints = root.querySelector("#constraints").value.trim();
      var usd = Number(root.querySelector("#budget-usd").value);
      var hours = Number(root.querySelector("#budget-hours").value);
      var maxBranches = Number(root.querySelector("#budget-branches").value);

      if (source === "git" && !urlInput.value.trim()) {
        showError("Give a git URL, or switch to a .zip upload.");
        urlInput.focus();
        return;
      }
      if (source === "zip" && !picked) {
        showError("Drop a .zip bundle, or switch to a git URL.");
        dropzone.focus();
        return;
      }
      if (!constraints) {
        showError("Write at least one constraint — it is what bounds the search.");
        root.querySelector("#constraints").focus();
        return;
      }
      if (!(usd > 0) || !(hours > 0) || !(maxBranches >= 1)) {
        showError("Budget caps must all be positive.");
        return;
      }

      submit.disabled = true;
      submit.textContent = "Submitting…";

      var payload = {
        constraints: constraints,
        budget: { usd: usd, hours: hours, max_branches: Math.round(maxBranches) }
      };

      var prepared = source === "git"
        ? Promise.resolve().then(function () {
            payload.bundle_url = urlInput.value.trim();
            return payload;
          })
        : readBase64(picked).then(function (b64) {
            payload.bundle_b64 = b64;
            return payload;
          });

      prepared
        .then(function (body) { return api.createRun(body); })
        .then(function (res) {
          if (!res || !res.run_id) { throw new Error("API returned no run_id"); }
          window.location.hash = "#/run/" + encodeURIComponent(res.run_id);
        })
        .catch(function (err) {
          submit.disabled = false;
          submit.textContent = "Start run";
          showError(err && err.message ? err.message : "submission failed");
        });
    });

    /* Recent runs — a convenience, and the only consumer of GET /api/runs. */
    var recentBlock = root.querySelector("#recent-block");
    var recentTable = root.querySelector("#recent-table");
    api.listRuns().then(function (res) {
      var runs = (res && res.runs) || [];
      if (!runs.length) { return; }
      recentBlock.hidden = false;
      recentTable.querySelector("tbody").innerHTML = runs.map(function (r) {
        return "<tr>" +
          '<td><a href="#/run/' + encodeURIComponent(r.run_id) + '">' + esc(r.run_id) + "</a></td>" +
          "<td>" + esc(r.env_name || "—") + "</td>" +
          "<td>" + esc(r.objective_name || "—") + "</td>" +
          '<td><span class="dot st-' + statusClass(r.state) + '"></span> ' + esc(statusClass(r.state)) + "</td>" +
          '<td class="num">' + esc(fmtScore(r.best_score)) + "</td>" +
          '<td class="num">' + esc(fmtUsd(r.spend_usd)) + "</td>" +
          "</tr>";
      }).join("");
    }).catch(function () { /* a listing failure is not worth a banner */ });
  }

  /* ───────────────────────────── run view ───────────────────────────── */

  function renderRun(runId) {
    var frag = el("tpl-run").content.cloneNode(true);
    var root = frag.firstElementChild;
    var f = fieldGetter(root);
    mount(frag);

    var generation = state.generation;
    var painted = false;
    var treeScroll = f("tree-scroll");
    var svg = f("tree-svg");
    var detail = f("detail");
    var table = f("branch-table");
    var log = f("log");

    function paintTree() {
      withFocusPreserved(svg, function () { renderTree(svg, state.run); });
    }

    function paintTable() {
      withFocusPreserved(table, function () { renderTable(table, state.run); });
    }

    function select(branchId, scroll) {
      state.selected = branchId;
      if (!state.run) { return; }
      paintTree();
      paintTable();
      renderDetail(detail, state.run);
      if (scroll) { scrollNodeIntoView(treeScroll, branchId); }
    }

    svg.addEventListener("click", function (ev) {
      var node = ev.target.closest(".node");
      if (node) { select(node.getAttribute("data-branch"), false); }
    });
    svg.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" && ev.key !== " ") { return; }
      var node = ev.target.closest(".node");
      if (!node) { return; }
      ev.preventDefault();
      select(node.getAttribute("data-branch"), false);
    });

    table.addEventListener("click", function (ev) {
      var row = ev.target.closest("tr[data-branch]");
      if (ev.target.closest(".th-sort")) { return; }
      if (row) { select(row.getAttribute("data-branch"), true); }
    });
    table.addEventListener("keydown", function (ev) {
      var row = ev.target.closest("tr[data-branch]");
      if (!row || (ev.key !== "Enter" && ev.key !== " ")) { return; }
      ev.preventDefault();
      select(row.getAttribute("data-branch"), true);
    });

    var sorters = table.querySelectorAll(".th-sort");
    for (var i = 0; i < sorters.length; i += 1) {
      sorters[i].addEventListener("click", function (ev) {
        var key = ev.currentTarget.getAttribute("data-sort");
        if (state.sort.key === key) {
          state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
        } else {
          state.sort.key = key;
          state.sort.dir = key === "branch_id" || key === "depth" || key === "status" ? "asc" : "desc";
        }
        if (state.run) {
          paintTable();
          f("sort-note").textContent = "sorted by " + key + " " + state.sort.dir;
        }
      });
    }

    f("stop").addEventListener("click", function (ev) {
      var btn = ev.currentTarget;
      btn.disabled = true;
      btn.textContent = "Stopping…";
      api.stopRun(runId)
        .then(function () { return api.getRun(runId); })
        .then(function (run) {
          if (state.generation !== generation) { return; }
          paint(run);
        })
        .catch(function () { /* the next poll will show the truth */ })
        .then(function () {
          btn.textContent = "Stop run";
          if (state.run && LIVE_STATES[state.run.state]) { btn.disabled = false; }
        });
    });

    function paint(run) {
      state.run = run;
      state.runId = run.run_id || runId;

      f("run_id").textContent = run.run_id || runId;
      f("env_name").textContent = run.env_name || "unknown environment";
      f("objective").textContent = (run.objective && run.objective.name) || "objective";
      f("direction").textContent = (run.objective && run.objective.direction) || "maximize";
      f("state-badge").innerHTML = statusBadge(run.state);
      f("elapsed").textContent = fmtDuration(run.elapsed_s);
      f("best-score").textContent = run.best ? fmtScore(run.best.score) : "—";
      f("branch-count").textContent = String((run.branches || []).length);

      var budget = run.budget || {};
      var cap = Number(budget.usd || 0);
      var spend = Number(run.spend_usd || 0);
      var pct = cap > 0 ? clamp((spend / cap) * 100, 0, 100) : 0;
      f("spend-text").textContent = fmtUsd(spend) + " / " + fmtUsd(cap);
      f("meter-fill").style.width = pct.toFixed(1) + "%";
      var meter = f("meter");
      meter.classList.toggle("is-warn", pct >= 75 && pct < 100);
      meter.classList.toggle("is-over", pct >= 100);
      meter.setAttribute("aria-valuenow", String(Math.round(pct)));
      f("budget-foot").textContent =
        pct.toFixed(0) + "% of cap · wall clock " + fmtDuration(run.elapsed_s) + " of " +
        (budget.hours != null ? budget.hours + "h" : "—") +
        " · max " + (budget.max_branches != null ? budget.max_branches : "—") + " concurrent branches";

      var live = Boolean(LIVE_STATES[run.state]);
      f("stop").disabled = !live;
      f("stop").textContent = live ? "Stop run" : "Run finished";
      f("artifact-link").href = "#/artifact/" + encodeURIComponent(state.runId);

      if (!state.selected && run.best) { state.selected = run.best.branch_id; }

      paintTree();
      paintTable();
      renderDetail(detail, run);
      renderLog(log, run);
      f("log-count").textContent = (run.events || []).length + " events";
      updateNav();

      /* First paint only: bring the best branch into view rather than opening
         hard against a corner of a wide canvas. */
      if (!painted) {
        painted = true;
        if (state.selected) {
          scrollNodeIntoView(treeScroll, state.selected);
        } else if (svg.__layout) {
          treeScroll.scrollLeft = Math.max(0, (svg.__layout.width - treeScroll.clientWidth) / 2);
        }
      }
    }

    function poll() {
      api.getRun(runId).then(function (run) {
        if (state.generation !== generation) { return; }
        paint(run);
        if (LIVE_STATES[run.state]) {
          state.pollTimer = window.setTimeout(poll, POLL_MS);
        }
      }).catch(function (err) {
        if (state.generation !== generation) { return; }
        mountMessage(
          "Run " + runId + " could not be loaded",
          (err && err.message ? err.message : "unknown error") +
            ". Start a new run, or set window.AUTOBRANCH_MOCK = true to review the UI against the built-in fixture.",
          "error",
          { href: "#/new", text: "← New run" }
        );
      });
    }

    poll();
  }

  /* ───────────────────────────── artifact view ───────────────────────────── */

  var FILE_NOTES = {
    "manifest.json": "env, base model, objective score, commands, spend",
    "reward_lineage.json": "root → winner: every mutation and the score it bought",
    "losers.json": "every abandoned branch and why — this is the science",
    "serve.py": "runnable inference entrypoint",
    "README.md": "how to run it"
  };

  function renderArtifact(runId) {
    var frag = el("tpl-artifact").content.cloneNode(true);
    var root = frag.firstElementChild;
    var f = fieldGetter(root);
    mount(frag);

    var generation = state.generation;
    state.runId = runId;
    updateNav();

    f("run_id").textContent = runId;
    f("back").href = "#/run/" + encodeURIComponent(runId);

    Promise.all([
      api.getArtifact(runId),
      api.getRun(runId).catch(function () { return null; })
    ]).then(function (results) {
      if (state.generation !== generation) { return; }
      var art = results[0] || {};
      var run = results[1];

      if (!art.ready) {
        mountMessage(
          "Artifact for " + runId + " is not ready",
          "No branch has finished training and cleared the evaluator yet. Consolidation walks the tree, ranks by objective, and writes the weights, the reward lineage, and the losers to the artifacts volume — come back when a branch has succeeded.",
          "warn",
          { href: "#/run/" + encodeURIComponent(runId), text: "← Watch the run" }
        );
        return;
      }

      var best = art.best || (run && run.best) || {};
      var objectiveName = best.objective || (run && run.objective && run.objective.name) || "objective";

      f("objective-label").textContent = objectiveName;
      f("score").textContent = fmtScore(best.score);
      f("score-meta").textContent =
        (run && run.objective ? run.objective.direction : "maximize") +
        " · scored by the immutable evaluator, not by any reward the agents wrote";
      f("branch").textContent = best.branch_id || "—";
      f("env").textContent = (run && run.env_name) || "—";
      f("spend").textContent = run ? fmtUsd(run.spend_usd) : "—";
      f("explored").textContent = run ? String((run.branches || []).length) : "—";

      if (run && run.branches && best.branch_id) {
        var path = lineageOf(run, best.branch_id);
        if (path.length) {
          f("lineage-block").hidden = false;
          var previous = null;
          f("lineage").innerHTML = path.map(function (b, idx) {
            var score = bestScoreOf(b);
            var delta = previous == null || score == null ? null : score - previous;
            previous = score;
            return '<li class="' + (idx === path.length - 1 ? "is-winner" : "") + '">' +
              '<span class="lineage-step">' + (idx === 0 ? "ROOT" : "FORK " + idx) + "</span>" +
              '<span class="lineage-id">' + esc(b.branch_id) + "</span>" +
              '<span class="lineage-delta">' + esc(truncate(opsSummary(b.reward_config), 40)) + "</span>" +
              '<span class="lineage-score">' + esc(fmtScore(score)) +
              (delta == null ? "" : " <span class=\"lineage-delta\">" + (delta >= 0 ? "+" : "") + fmtScore(delta) + "</span>") +
              "</span>" +
              "</li>";
          }).join("");
        }
      }

      var files = art.files || [];
      f("files").innerHTML = files.map(function (file) {
        var note = file.desc || FILE_NOTES[file.name] || "artifact file";
        return "<li>" +
          '<span class="file-main">' +
          '<span class="file-title">' + esc(file.name) + "</span>" +
          '<span class="file-desc">' + esc(note) + "</span>" +
          "</span>" +
          '<span class="file-size">' + esc(fmtBytes(file.bytes)) + "</span>" +
          '<a class="btn btn-sm" href="' + esc(file.url || "#") + '" download="' + esc(file.name) + '">Download</a>' +
          "</li>";
      }).join("") || "<li><span class=\"file-desc\">no files published</span></li>";

      var command = art.run_command || "";
      f("run-command").textContent = command;
      f("copy").addEventListener("click", function (ev) {
        var btn = ev.currentTarget;
        copyText(command).then(function (ok) {
          btn.textContent = ok ? "Copied" : "Copy failed";
          window.setTimeout(function () { btn.textContent = "Copy"; }, 1600);
        });
      });
    }).catch(function (err) {
      if (state.generation !== generation) { return; }
      mountMessage(
        "Artifact for " + runId + " could not be loaded",
        (err && err.message ? err.message : "unknown error") + ".",
        "error"
      );
    });
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text)
        .then(function () { return true; })
        .catch(function () { return legacyCopy(text); });
    }
    return Promise.resolve(legacyCopy(text));
  }

  function legacyCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "readonly");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    document.body.removeChild(area);
    return ok;
  }

  /* ───────────────────────────── router ───────────────────────────── */

  function parseHash() {
    var raw = window.location.hash.replace(/^#/, "");
    var parts = raw.split("/").filter(function (p) { return p.length > 0; });
    if (!parts.length) { return { name: "new" }; }
    if (parts[0] === "run" && parts[1]) { return { name: "run", id: decodeURIComponent(parts[1]) }; }
    if (parts[0] === "artifact" && parts[1]) { return { name: "artifact", id: decodeURIComponent(parts[1]) }; }
    return { name: "new" };
  }

  function route() {
    var hash = window.location.hash;
    if (state.route && hash === state.lastHash) { return; }
    state.lastHash = hash;

    state.generation += 1;
    if (state.pollTimer) {
      window.clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }

    var next = parseHash();
    var changedRun = !state.route || next.id !== state.route.id;
    state.route = next;

    if (next.name === "run") {
      if (changedRun) {
        state.selected = null;
        state.run = null;
      }
      state.runId = next.id;
      updateNav();
      renderRun(next.id);
      return;
    }

    if (next.name === "artifact") {
      state.runId = next.id;
      updateNav();
      renderArtifact(next.id);
      return;
    }

    updateNav();
    renderNew();
  }

  /* ───────────────────────────── boot ───────────────────────────── */

  function boot() {
    initTheme();
    setApiOrigin();
    if (MOCK) {
      var badge = el("mock-badge");
      if (badge) {
        badge.hidden = false;
        badge.title = "window.AUTOBRANCH_MOCK is true — rendering the built-in fixture.";
      }
    }
    if (!window.location.hash) {
      window.location.replace("#/new");
    }
    window.addEventListener("hashchange", route);
    route();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
