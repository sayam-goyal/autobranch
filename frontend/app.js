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

   Four screens, hash-routed: #/runs (the default landing), #/new,
   #/run/<id>, #/artifact/<id>.
   ========================================================================== */
(function () {
  "use strict";

  /* ───────────────────────────── config ───────────────────────────── */

  var API_BASE = String(window.AUTOBRANCH_API_BASE || "").replace(/\/+$/, "");
  var MOCK = window.AUTOBRANCH_MOCK === true;
  var POLL_MS = 3000;
  var TICK_MS = 1000;

  var LIVE_STATES = { pending: true, running: true, evaluating: true };
  var STATUS_ORDER = {
    running: 0,
    evaluating: 1,
    pending: 2,
    succeeded: 3,
    abandoned: 4,
    failed: 5
  };

  /* One poller at a time, ever. `generation` invalidates in-flight responses
     from a view that has since been replaced; `pollTimer` is the single
     scheduled fetch and `tickTimer` is a clock, not a request — it repaints
     elapsed/age text from values already in hand. */
  var state = {
    route: null,
    lastHash: null,
    runId: null,
    run: null,
    selected: null,
    sort: { key: "score", dir: "desc" },
    runsSort: { key: "created_at", dir: "desc" },
    runMeta: {},
    generation: 0,
    pollTimer: null,
    tickTimer: null
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

  function escapeRe(value) {
    return String(value == null ? "" : value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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

  /* Coarse, for "how long ago" — never longer than two units. */
  function fmtAgo(seconds) {
    if (seconds == null || isNaN(seconds)) { return "—"; }
    var s = Math.max(0, Math.round(seconds));
    if (s < 60) { return s + "s"; }
    var m = Math.floor(s / 60);
    if (m < 60) { return m + "m " + (s % 60) + "s"; }
    var h = Math.floor(m / 60);
    if (h < 24) { return h + "h " + (m % 60) + "m"; }
    return Math.floor(h / 24) + "d " + (h % 24) + "h";
  }

  function fmtBytes(n) {
    if (n == null || isNaN(n)) { return "—"; }
    var units = ["B", "KB", "MB", "GB", "TB"];
    var i = 0;
    var v = Number(n);
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return (i === 0 ? v.toFixed(0) : v.toFixed(v < 10 ? 2 : 1)) + " " + units[i];
  }

  /* `ts` is an ISO string from the mock fixture but a float epoch in SECONDS
     from the API (registry.api_payload). A bare new Date(seconds) renders
     every live log line as 1970, so normalise here rather than changing the
     API contract, which is "exactly api_payload". */
  function tsToMs(ts) {
    if (ts == null || ts === "") { return NaN; }
    if (typeof ts === "number") { return ts < 1e12 ? ts * 1000 : ts; }
    var parsed = new Date(ts).getTime();
    return isNaN(parsed) ? NaN : parsed;
  }

  function fmtClock(ts) {
    var ms = tsToMs(ts);
    if (isNaN(ms)) { return "--:--:--"; }
    var d = new Date(ms);
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  }

  function fmtStamp(ts) {
    var ms = tsToMs(ts);
    if (isNaN(ms)) { return "—"; }
    var d = new Date(ms);
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) +
      " " + pad(d.getHours()) + ":" + pad(d.getMinutes());
  }

  function ageText(ts) {
    var ms = tsToMs(ts);
    if (isNaN(ms)) { return "—"; }
    return fmtAgo((Date.now() - ms) / 1000);
  }

  function truncate(text, max) {
    var s = String(text == null ? "" : text);
    if (s.length <= max) { return s; }
    return s.slice(0, Math.max(0, max - 1)) + "…";
  }

  function plural(n, word) { return n + " " + word + (n === 1 ? "" : "s"); }

  /* `plural` just adds an "s"; a few words in this UI need a real plural. */
  function pluralise(n, one, many) { return n + " " + (n === 1 ? one : many); }

  function statusClass(status) {
    return STATUS_ORDER[status] === undefined ? "pending" : status;
  }

  function isTerminalRun(runState) { return !LIVE_STATES[runState]; }

  /* Human-readable one-liners for a reward config, used on tree nodes. */
  function weightsSummary(cfg) {
    if (!cfg || !cfg.weights) { return "no weights declared"; }
    var keys = Object.keys(cfg.weights);
    keys.sort(function (a, b) {
      return Math.abs(cfg.weights[b]) - Math.abs(cfg.weights[a]);
    });
    if (!keys.length) { return "no weights declared"; }
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

  function branchById(run, branchId) {
    var hit = null;
    ((run && run.branches) || []).forEach(function (b) {
      if (b.branch_id === branchId) { hit = b; }
    });
    return hit;
  }

  /* A `pending` branch means "sandbox starting" — but only while the run is
     still going. On a run that failed before it ever launched, the same
     status means the sandbox never started at all, and pulsing it as though
     something were happening would be a lie. */
  function isLiveBranch(run, status) {
    return Boolean(LIVE_STATES[statusClass(status)]) &&
      Boolean(LIVE_STATES[run && run.state]);
  }

  function statusLabel(run, status) {
    var s = statusClass(status);
    if (s !== "pending") { return s; }
    return LIVE_STATES[run && run.state] ? "starting" : "never started";
  }

  function statusCounts(branches) {
    var counts = {};
    (branches || []).forEach(function (b) {
      var s = statusClass(b.status);
      counts[s] = (counts[s] || 0) + 1;
    });
    return counts;
  }

  /* ───────────────────────── sandboxes & activity ─────────────────────────
     A node in the tree is a Modal sandbox, and the two things a watcher wants
     off it are its id and what it is doing. Both have a published source and
     a derived fallback, because the published ones only exist once the
     backend has written them.
     ------------------------------------------------------------------------ */

  var SANDBOX_RE = /\bsb-[A-Za-z0-9]{6,}/;

  /* `branch.sandbox_id` when the registry has it; otherwise the last sandbox
     id this branch's events mentioned — `launch_root` and `fork` both name the
     sandbox they just created in the log. */
  function sandboxOf(run, branch) {
    if (!branch) { return null; }
    if (branch.sandbox_id) { return String(branch.sandbox_id); }
    var found = null;
    ((run && run.events) || []).forEach(function (e) {
      if (e.branch_id !== branch.branch_id) { return; }
      var m = SANDBOX_RE.exec(String(e.message || ""));
      if (m) { found = m[0]; }
    });
    return found;
  }

  /* Modal sandbox ids are ULID-ish: the prefix is a timestamp and is nearly
     identical across siblings, so a useful short form keeps both ends. */
  function shortSandbox(id) {
    if (!id) { return null; }
    var body = String(id).replace(/^sb-/, "");
    if (body.length <= 12) { return "sb-" + body; }
    return "sb-" + body.slice(0, 4) + "…" + body.slice(-4);
  }

  var IDLE_ACTIVITY = {
    pending: "sandbox starting",
    running: "running",
    evaluating: "evaluating against the objective",
    succeeded: "finished",
    abandoned: "abandoned",
    failed: "failed"
  };

  function lastEventFor(run, branchId) {
    var newest = null;
    ((run && run.events) || []).forEach(function (e) {
      if (e.branch_id === branchId) { newest = e; }
    });
    return newest;
  }

  function eventsFor(run, branchId) {
    return ((run && run.events) || []).filter(function (e) {
      return e.branch_id === branchId;
    });
  }

  /* The log lines are written for a log, where the branch id is a useful
     prefix. On a node already labelled with that id it is noise, and a few
     shapes read better rewritten. Anything unrecognised passes through
     untouched — a message we do not understand is still the truth. */
  function humanizeEvent(message, branchId) {
    /* A branch that dies logs its traceback, newlines and all. An activity
       line is one line, so whitespace is collapsed here — the untouched
       message is still in the feed. */
    var text = String(message == null ? "" : message).replace(/\s+/g, " ").trim();
    text = text.replace(/^root branch\s+/, "");
    if (branchId) {
      text = text.replace(new RegExp("^branch\\s+" + escapeRe(branchId) + "\\b[:,]?\\s*"), "");
      text = text.replace(new RegExp("^" + escapeRe(branchId) + "\\b[:,]?\\s+"), "");
    }
    text = text.replace(/^branch\s+(b[\w.]*)\s+/, "$1 ");

    var m;
    m = /^running in sandbox\s+(sb-[A-Za-z0-9]+);?\s*deadline in\s+(.+)$/.exec(text);
    if (m) { return "sandbox booted · agent loop starting · deadline in " + m[2]; }

    if (/^queued for\s+/.test(text)) { return "queued — sandbox not started yet"; }
    if (/^started\s*\(depth/.test(text)) { return "agent loop started"; }

    m = /^objective\s+(\S+)\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s+at step\s+([\d,]+)/.exec(text);
    if (m) {
      return "scored " + m[1] + " = " + m[2] + " at step " + fmtInt(Number(String(m[3]).replace(/,/g, "")));
    }

    m = /^trained\s+([\d,]+)\s+steps in\s+([\d.]+)s\s*\(total\s+([\d,]+)\)/.exec(text);
    if (m) {
      return "trained " + fmtInt(Number(String(m[1]).replace(/,/g, ""))) + " steps in " +
        m[2] + "s · " + fmtInt(Number(String(m[3]).replace(/,/g, ""))) + " total";
    }

    m = /^reward\s*->\s*(.+)$/.exec(text);
    if (m) { return "wrote a new reward · " + m[1]; }

    m = /^baseline reward:\s*(.+)$/.exec(text);
    if (m) { return "baseline reward · " + m[1]; }

    m = /^forked\s+(\S+?):\s*(.*)$/.exec(text);
    if (m) { return "forked child " + m[1] + (m[2] ? " · " + m[2] : ""); }

    m = /^rolled back to snapshot\s+(.+)$/.exec(text);
    if (m) { return "rolled back to snapshot " + m[1]; }

    if (/^rejected\s+/.test(text)) { return "last agent action rejected · " + text.replace(/^rejected\s+/, ""); }

    return text;
  }

  /* The one question a watcher actually has: what is this sandbox doing?
     Three sources, in order of trust.

       1. `branch.activity` — written by the sandbox *before* it starts the
          work, so it stays correct across a long train call, where the newest
          event still describes the previous step. Present only once the
          backend publishes the field.
       2. the newest event carrying this branch_id. Always available, one step
          behind during long operations.
       3. the branch status. A `pending` branch has not run anything yet, and
          "sandbox starting" is the honest reading of that. */
  function activityOf(run, branch) {
    if (!branch) { return null; }
    if (branch.activity) {
      return {
        text: String(branch.activity),
        since: branch.activity_since || null,
        source: "sandbox",
        level: "info"
      };
    }
    var newest = lastEventFor(run, branch.branch_id);
    if (newest) {
      return {
        text: humanizeEvent(newest.message, branch.branch_id),
        since: newest.ts,
        source: "event",
        level: newest.level === "warn" || newest.level === "error" ? newest.level : "info"
      };
    }
    var s = statusClass(branch.status);
    return {
      text: (s === "pending" && !LIVE_STATES[run && run.state])
        ? "sandbox never started"
        : (IDLE_ACTIVITY[s] || "no activity recorded"),
      since: null,
      source: "status",
      level: "info"
    };
  }

  function activityLine(run, branch) {
    var act = activityOf(run, branch);
    if (!act) { return ""; }
    var age = act.since == null ? null : ageText(act.since);
    return act.text + (age && age !== "—" ? " · " + age + " ago" : "");
  }

  /* ───────────────────────────── mock fixture ─────────────────────────────
     A run on the environment DESIGN.md generalises from: nine sandboxes, four
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
        sandbox_id: "sb-01M7F2QZ8K4XR9D3A6WTYB0A1F",
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
        sandbox_id: "sb-01M7F3R4A8K4XR9D3A6WTYB7C22",
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
        sandbox_id: "sb-01M7F3R9C2K4XR9D3A6WTYB3E90",
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
        sandbox_id: "sb-01M7F3RD7FK4XR9D3A6WTYB9B41",
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
        sandbox_id: "sb-01M7F4T1E6K4XR9D3A6WTYBC118",
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
        sandbox_id: "sb-01M7F4T6B3K4XR9D3A6WTYB2F55",
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
        sandbox_id: "sb-01M7F4TA9JK4XR9D3A6WTYB5A07",
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
        sandbox_id: "sb-01M7F4TG2PK4XR9D3A6WTYB8D3C",
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
        sandbox_id: "sb-01M7F5V3H1K4XR9D3A6WTYB4D7E",
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
      { t: 18, level: "info", branch_id: "b-0a1f", message: "root branch running in sandbox sb-01M7F2QZ8K4XR9D3A6WTYB0A1F; deadline in 6.0h" },
      { t: 24, level: "info", branch_id: "b-0a1f", message: "reward -> round_win +1.0, damage_dealt +0.1, damage_taken -0.1" },
      { t: 640, level: "info", branch_id: "b-0a1f", message: "objective win_rate_vs_cpu8 = 0.2140 at step 10000" },
      { t: 700, level: "info", branch_id: "b-7c22", message: "forked b-7c22: potential shaping on health_margin, annealed to 0 by 0.8" },
      { t: 706, level: "info", branch_id: "b-3e90", message: "forked b-3e90: damage_dealt primary, round_win as tiebreak" },
      { t: 712, level: "info", branch_id: "b-9b41", message: "forked b-9b41: reward is the objective, nothing else" },
      { t: 2180, level: "warn", branch_id: "b-3e90", message: "self-play win rate 0.812 but objective flat at 0.031 — reward and objective have decoupled" },
      { t: 2260, level: "warn", branch_id: "b-3e90", message: "branch b-3e90 abandoned: no objective improvement over 4 evaluations (best 0.0310)" },
      { t: 2410, level: "info", branch_id: "b-7c22", message: "objective win_rate_vs_cpu8 = 0.6510 at step 20000" },
      { t: 3600, level: "info", branch_id: "b-c118", message: "read_tree(): b-3e90 died farming damage_dealt — clipping that term instead of dropping it" },
      { t: 4120, level: "warn", branch_id: "b-2f55", message: "branch b-2f55 abandoned: GPU-minute budget exhausted, expr term dominating round_win ~6x (best 0.1120)" },
      { t: 5240, level: "info", branch_id: "b-c118", message: "objective win_rate_vs_cpu8 = 0.8830 at step 32000" },
      { t: 5320, level: "info", branch_id: "b-4d7e", message: "forked b-4d7e: symmetric clip on damage_taken" },
      { t: 6910, level: "info", branch_id: "b-4d7e", message: "trained 3000 steps in 182s (total 38000)" },
      { t: 7280, level: "info", branch_id: "b-4d7e", message: "objective win_rate_vs_cpu8 = 0.9900 at step 47000" },
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
      var hex = (0x100000 + this.seq * 0x9e37 + 0x8fc2a1).toString(16).slice(-6);
      return "run_" + hex;
    },

    record: function (id, opts) {
      var o = opts || {};
      var branches = clone(FIXTURE.branches).map(function (b) {
        b.final_status = b.status;
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
              { t: 2, level: "info", branch_id: "b-0a1f", message: "root branch running in sandbox " + FIXTURE.branches[0].sandbox_id + "; deadline in " + (o.budget || FIXTURE.budget).hours + ".0h" }
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
       live about: sandboxes gain evaluations, new ones fork in, and the run
       eventually consolidates. */
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
              self.push(rec, "warn", b.branch_id, "branch " + b.branch_id + " abandoned: " + truncate(b.abandoned_reason || "line stopped paying", 120));
            } else {
              self.push(rec, "info", b.branch_id, "branch " + b.branch_id + " " + b.final_status + " (best " + point.score.toFixed(4) + ")");
            }
          } else {
            self.push(rec, "info", b.branch_id,
              "objective " + rec.objective.name + " = " + point.score.toFixed(4) + " at step " + point.step);
          }
        });
      }

      if (rec.ticks % 3 === 0 && rec.revealed < rec.branches.length && active < rec.budget.max_branches) {
        var next = rec.branches[rec.revealed];
        rec.revealed += 1;
        next.shown = 1;
        this.push(rec, "info", next.branch_id,
          "forked " + next.branch_id + ": " + weightsSummary(next.reward_config));
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
      var nowS = rec.created_ms / 1000 + rec.elapsed_s;
      return rec.branches.slice(0, rec.revealed).map(function (b) {
        var scores = b.full_scores.slice(0, b.shown);
        /* A branch still producing evaluations shows a live status: its own if
           that status is already live, otherwise "running" until it lands on
           its terminal one. */
        var running = b.shown < b.full_scores.length;
        var status = running
          ? (LIVE_STATES[b.final_status] ? b.final_status : "running")
          : b.final_status;
        var frac = b.full_scores.length ? b.shown / b.full_scores.length : 1;
        var point = b.full_scores[Math.max(0, b.shown - 1)];
        var activity;
        if (!running) {
          activity = { succeeded: "done", abandoned: "abandoned", failed: "failed" }[b.final_status] || b.final_status;
        } else if (status === "evaluating") {
          activity = "evaluating " + rec.objective.name + " at step " + fmtInt(point.step);
        } else if (status === "pending") {
          activity = "starting up";
        } else {
          activity = "training " + fmtInt(2000) + " steps";
        }
        return {
          branch_id: b.branch_id,
          parent_id: b.parent_id,
          depth: b.depth,
          status: status,
          sandbox_id: status === "pending" ? null : b.sandbox_id,
          activity: activity,
          activity_since: nowS - (running ? 7 + (b.depth * 3) : 0),
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

  /* Two permanent tabs, plus two that only exist while a run is in context.
     `runs` is the landing tab, so the first thing the app answers is "what
     have I submitted", not "submit something". */
  function updateNav() {
    var links = document.querySelectorAll(".topnav a[data-nav]");
    var contextual = false;
    for (var i = 0; i < links.length; i += 1) {
      var link = links[i];
      var kind = link.getAttribute("data-nav");
      if (kind === "runs" || kind === "new") {
        link.hidden = false;
        link.href = kind === "runs" ? "#/runs" : "#/new";
      } else if (state.runId) {
        link.hidden = false;
        contextual = true;
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
    var sep = el("nav-sep");
    if (sep) { sep.hidden = !contextual; }
  }

  function setApiOrigin() {
    var node = el("api-origin");
    if (!node) { return; }
    node.textContent = "api: " + (API_BASE || "same-origin");
  }

  /* `live` is passed explicitly where the caller knows better than the status
     alone does — a pending branch inside a finished run is not live. */
  function statusBadge(status, live, label) {
    var s = statusClass(status);
    var moving = live === undefined ? Boolean(LIVE_STATES[s]) : Boolean(live);
    return '<span class="badge badge-' + s + '"><span class="dot st-' + s +
      (moving ? " pulse" : "") + '"></span>' + esc(label || s) + "</span>";
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
     three seconds, so keyboard focus inside the tree or a table has to
     survive them. */
  function withFocusPreserved(container, render) {
    var active = document.activeElement;
    var inside = active && container.contains(active);
    var attr = null;
    var key = null;
    if (inside) {
      attr = active.hasAttribute("data-branch") ? "data-branch"
        : (active.hasAttribute("data-run") ? "data-run" : null);
      key = attr ? active.getAttribute(attr) : null;
    }
    var scrolled = container.scrollTop;
    render();
    if (container.scrollTop !== scrolled) { container.scrollTop = scrolled; }
    if (!key) { return; }
    var again = container.querySelector("[" + attr + '="' + key.replace(/["\\]/g, "") + '"]');
    if (again && typeof again.focus === "function") { again.focus(); }
  }

  /* A clock, not a poller: it repaints text from numbers already fetched, so
     an open page shows a moving elapsed without asking the API anything. */
  function startTicker(fn) {
    stopTicker();
    var generation = state.generation;
    state.tickTimer = window.setInterval(function () {
      if (state.generation !== generation) { stopTicker(); return; }
      try { fn(); } catch (err) { stopTicker(); }
    }, TICK_MS);
  }

  function stopTicker() {
    if (state.tickTimer) {
      window.clearInterval(state.tickTimer);
      state.tickTimer = null;
    }
  }

  function schedulePoll(fn) {
    if (state.pollTimer) { window.clearTimeout(state.pollTimer); }
    state.pollTimer = window.setTimeout(fn, POLL_MS);
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

  /* ───────────────────────────── zip reader ─────────────────────────────
     Enough of the zip format to pull one small text file out of a bundle the
     user just picked, without reading the whole archive into memory: scan the
     tail for the end-of-central-directory record, walk the central directory,
     then slice exactly the one local entry we want. Inflation uses the
     platform's own `DecompressionStream("deflate-raw")` — no vendored zlib,
     no CDN.

     This exists so the New Run screen can show what the system understood
     *before* anything is submitted. It is a preview, never a gate: every
     failure path here degrades to "the server will parse this on submit".
     ---------------------------------------------------------------------- */

  var ZIP_EOCD = 0x06054b50;
  var ZIP_CD = 0x02014b50;
  var ZIP_LOCAL = 0x04034b50;

  function ZipError(message) {
    this.name = "ZipError";
    this.message = message;
  }
  ZipError.prototype = Object.create(Error.prototype);
  ZipError.prototype.constructor = ZipError;

  function sliceBuffer(file, start, end) {
    var from = Math.max(0, start);
    var to = Math.min(file.size, end);
    var blob = file.slice(from, to);
    if (typeof blob.arrayBuffer === "function") {
      return blob.arrayBuffer().then(function (buf) { return new DataView(buf); });
    }
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onerror = function () { reject(new ZipError("could not read the archive")); };
      reader.onload = function () { resolve(new DataView(reader.result)); };
      reader.readAsArrayBuffer(blob);
    });
  }

  function inflateRaw(bytes) {
    if (typeof DecompressionStream !== "function") {
      return Promise.reject(new ZipError("this browser cannot inflate deflate streams"));
    }
    return Promise.resolve().then(function () {
      var stream = new DecompressionStream("deflate-raw");
      var writer = stream.writable.getWriter();
      writer.write(bytes);
      writer.close();
      return new Response(stream.readable).arrayBuffer();
    }).then(function (buf) { return new Uint8Array(buf); });
  }

  /* The central directory, newest-format-first. A zip64 archive is refused
     rather than mis-read: a 64 MiB cap means we will never legitimately see
     one, and a wrong offset would silently decode garbage. */
  function readCentralDirectory(file) {
    var tailSize = Math.min(file.size, 66560);
    return sliceBuffer(file, file.size - tailSize, file.size).then(function (tail) {
      var eocd = -1;
      for (var i = tail.byteLength - 22; i >= 0; i -= 1) {
        if (tail.getUint32(i, true) === ZIP_EOCD) { eocd = i; break; }
      }
      if (eocd < 0) {
        throw new ZipError("no zip end-of-central-directory record — not a zip archive");
      }
      var count = tail.getUint16(eocd + 10, true);
      var cdSize = tail.getUint32(eocd + 12, true);
      var cdOffset = tail.getUint32(eocd + 16, true);
      if (cdOffset === 0xffffffff || cdSize === 0xffffffff || count === 0xffff) {
        throw new ZipError("zip64 archives are not previewed in the browser");
      }
      return sliceBuffer(file, cdOffset, cdOffset + cdSize).then(function (cd) {
        var entries = [];
        var p = 0;
        var decoder = new TextDecoder("utf-8");
        for (var n = 0; n < count && p + 46 <= cd.byteLength; n += 1) {
          if (cd.getUint32(p, true) !== ZIP_CD) {
            throw new ZipError("corrupt central directory at entry " + n);
          }
          var method = cd.getUint16(p + 10, true);
          var compressed = cd.getUint32(p + 20, true);
          var uncompressed = cd.getUint32(p + 24, true);
          var nameLen = cd.getUint16(p + 28, true);
          var extraLen = cd.getUint16(p + 30, true);
          var commentLen = cd.getUint16(p + 32, true);
          var localOffset = cd.getUint32(p + 42, true);
          var nameBytes = new Uint8Array(cd.buffer, cd.byteOffset + p + 46, nameLen);
          entries.push({
            name: decoder.decode(nameBytes),
            method: method,
            compressed: compressed,
            uncompressed: uncompressed,
            offset: localOffset
          });
          p += 46 + nameLen + extraLen + commentLen;
        }
        return entries;
      });
    });
  }

  function readZipEntry(file, entry) {
    if (entry.uncompressed > 4 * 1024 * 1024) {
      return Promise.reject(new ZipError(entry.name + " is too large to preview"));
    }
    return sliceBuffer(file, entry.offset, entry.offset + 30).then(function (head) {
      if (head.getUint32(0, true) !== ZIP_LOCAL) {
        throw new ZipError("corrupt local header for " + entry.name);
      }
      var nameLen = head.getUint16(26, true);
      var extraLen = head.getUint16(28, true);
      var start = entry.offset + 30 + nameLen + extraLen;
      return sliceBuffer(file, start, start + entry.compressed);
    }).then(function (view) {
      var bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
      if (entry.method === 0) { return bytes; }
      if (entry.method === 8) { return inflateRaw(bytes); }
      throw new ZipError("unsupported compression method " + entry.method + " for " + entry.name);
    }).then(function (bytes) {
      return new TextDecoder("utf-8").decode(bytes);
    });
  }

  var MANIFEST_NAME = "autobranch.yaml";

  /* Mirrors `server.find_manifest`: the bundle root, or exactly one directory
     below it, because a GitHub zip always nests everything under
     `repo-branch/`. Deeper is refused there, so it is refused here. */
  function pickManifest(entries) {
    var direct = null;
    var nested = [];
    entries.forEach(function (e) {
      var parts = e.name.split("/").filter(function (p) { return p.length > 0; });
      if (parts.length === 1 && parts[0] === MANIFEST_NAME) { direct = e; }
      if (parts.length === 2 && parts[1] === MANIFEST_NAME) { nested.push(e); }
    });
    if (direct) { return direct; }
    nested.sort(function (a, b) { return a.name < b.name ? -1 : 1; });
    return nested.length ? nested[0] : null;
  }

  function readManifestFromZip(file) {
    return readCentralDirectory(file).then(function (entries) {
      var entry = pickManifest(entries);
      if (!entry) {
        throw new ZipError(
          "no " + MANIFEST_NAME + " at the bundle root or one directory below it"
        );
      }
      return readZipEntry(file, entry).then(function (text) {
        return { path: entry.name, text: text, entries: entries };
      });
    });
  }

  /* ───────────────────────────── a small YAML reader ─────────────────────
     Enough YAML to read an autobranch manifest and nothing more: block
     mappings, block sequences, `>` and `|` block scalars, one-line flow
     sequences and mappings, quoted and plain scalars, comments. Anchors,
     aliases, tags, multiple documents and complex keys throw rather than
     being half-supported — a manifest needing them is one the server should
     be the only thing parsing.

     One deliberate deviation from the spec, for display's sake: a folded
     (`>`) scalar comes back without its trailing newline.
     ---------------------------------------------------------------------- */

  function YamlError(message, line) {
    this.name = "YamlError";
    this.line = line || 0;
    this.message = line ? "line " + line + ": " + message : message;
  }
  YamlError.prototype = Object.create(Error.prototype);
  YamlError.prototype.constructor = YamlError;

  function parseYaml(text) {
    var src = String(text == null ? "" : text)
      .replace(/^﻿/, "")
      .replace(/\r\n?/g, "\n")
      .split("\n");
    var cursor = 0;

    function indentOf(line) {
      var n = 0;
      while (n < line.length && line.charAt(n) === " ") { n += 1; }
      return n;
    }

    function significant(from) {
      for (var k = from; k < src.length; k += 1) {
        var line = src[k];
        var trimmed = line.trim();
        if (trimmed === "" || trimmed.charAt(0) === "#") { continue; }
        if (/^\t/.test(line) || /^ *\t/.test(line)) {
          throw new YamlError("tabs cannot be used for indentation", k + 1);
        }
        if (trimmed === "---" || trimmed === "...") {
          throw new YamlError("multi-document manifests are not supported", k + 1);
        }
        return k;
      }
      return -1;
    }

    function peek() {
      var k = significant(cursor);
      if (k < 0) { return null; }
      var indent = indentOf(src[k]);
      return { line: k, indent: indent, text: src[k].slice(indent).replace(/\s+$/, "") };
    }

    function stripComment(line) {
      var quote = null;
      for (var k = 0; k < line.length; k += 1) {
        var ch = line.charAt(k);
        if (quote) {
          if (ch === "\\" && quote === '"') { k += 1; continue; }
          if (ch === quote) { quote = null; }
          continue;
        }
        if (ch === '"' || ch === "'") { quote = ch; continue; }
        if (ch === "#" && (k === 0 || /\s/.test(line.charAt(k - 1)))) { return line.slice(0, k); }
      }
      return line;
    }

    function splitKey(text, line) {
      var quote = null;
      for (var k = 0; k < text.length; k += 1) {
        var ch = text.charAt(k);
        if (quote) {
          if (ch === "\\" && quote === '"') { k += 1; continue; }
          if (ch === quote) { quote = null; }
          continue;
        }
        if (ch === '"' || ch === "'") { quote = ch; continue; }
        if (ch === "#" && k > 0 && /\s/.test(text.charAt(k - 1))) { break; }
        if (ch === "[" || ch === "{") { break; }
        if (ch === ":" && (k === text.length - 1 || /\s/.test(text.charAt(k + 1)))) {
          var key = text.slice(0, k).trim();
          if (!key) { throw new YamlError("empty mapping key", line + 1); }
          return { key: scalarToken(key, line, true), rest: text.slice(k + 1).replace(/^\s+/, "") };
        }
      }
      return null;
    }

    function unquote(token, line) {
      var quote = token.charAt(0);
      if (token.length < 2 || token.charAt(token.length - 1) !== quote) {
        throw new YamlError("unterminated quoted string", line + 1);
      }
      var body = token.slice(1, -1);
      if (quote === "'") { return body.replace(/''/g, "'"); }
      return body.replace(/\\(["\\/nrtb])/g, function (_all, ch) {
        return { n: "\n", r: "\r", t: "\t", b: "\b" }[ch] || ch;
      });
    }

    /* A single scalar token: null, boolean, number, quoted or plain string.
       `asKey` keeps mapping keys as strings, which is what every manifest
       key is. */
    function scalarToken(token, line, asKey) {
      var t = String(token).trim();
      if (t.charAt(0) === "&" || t.charAt(0) === "*" || t.charAt(0) === "!") {
        throw new YamlError("anchors, aliases and tags are not supported", line + 1);
      }
      if (t.charAt(0) === '"' || t.charAt(0) === "'") { return unquote(t, line); }
      if (asKey) { return t; }
      if (t === "" || t === "~" || /^(null|Null|NULL)$/.test(t)) { return null; }
      if (/^(true|True|TRUE|yes|Yes|on|On)$/.test(t)) { return true; }
      if (/^(false|False|FALSE|no|No|off|Off)$/.test(t)) { return false; }
      if (/^[-+]?\d+$/.test(t)) { return parseInt(t, 10); }
      if (/^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$/.test(t)) { return Number(t); }
      if (/^[-+]?0x[0-9a-fA-F]+$/.test(t)) { return parseInt(t, 16); }
      if (/^[-+]?\.(inf|Inf|INF)$/.test(t)) { return t.charAt(0) === "-" ? -Infinity : Infinity; }
      return t;
    }

    function parseFlow(text, line) {
      var pos = 0;

      function ws() { while (pos < text.length && /\s/.test(text.charAt(pos))) { pos += 1; } }

      function token() {
        ws();
        var ch = text.charAt(pos);
        if (ch === '"' || ch === "'") {
          var start = pos;
          pos += 1;
          while (pos < text.length) {
            if (ch === '"' && text.charAt(pos) === "\\") { pos += 2; continue; }
            if (text.charAt(pos) === ch) { pos += 1; break; }
            pos += 1;
          }
          return unquote(text.slice(start, pos), line);
        }
        var from = pos;
        while (pos < text.length && ",]}:".indexOf(text.charAt(pos)) < 0) { pos += 1; }
        return scalarToken(text.slice(from, pos), line);
      }

      function value() {
        ws();
        var ch = text.charAt(pos);
        if (ch === "[") {
          pos += 1;
          var arr = [];
          ws();
          if (text.charAt(pos) === "]") { pos += 1; return arr; }
          for (;;) {
            arr.push(value());
            ws();
            var sep = text.charAt(pos);
            pos += 1;
            if (sep === "]") { return arr; }
            if (sep !== ",") { throw new YamlError("malformed flow sequence", line + 1); }
          }
        }
        if (ch === "{") {
          pos += 1;
          var obj = {};
          ws();
          if (text.charAt(pos) === "}") { pos += 1; return obj; }
          for (;;) {
            var key = String(token());
            ws();
            if (text.charAt(pos) !== ":") { throw new YamlError("malformed flow mapping", line + 1); }
            pos += 1;
            obj[key] = value();
            ws();
            var stop = text.charAt(pos);
            pos += 1;
            if (stop === "}") { return obj; }
            if (stop !== ",") { throw new YamlError("malformed flow mapping", line + 1); }
          }
        }
        return token();
      }

      var out = value();
      ws();
      if (pos < text.length && stripComment(text.slice(pos)).trim() !== "") {
        throw new YamlError("trailing text after a flow value", line + 1);
      }
      return out;
    }

    function blockScalar(header, parentIndent, line) {
      var folded = header.charAt(0) === ">";
      var chomp = header.indexOf("-") >= 0 ? "strip" : (header.indexOf("+") >= 0 ? "keep" : "clip");
      if (!/^[>|][-+]?\d*$/.test(header.trim())) {
        throw new YamlError("unsupported block scalar header " + header, line + 1);
      }
      var lines = [];
      var base = -1;
      while (cursor < src.length) {
        var raw = src[cursor];
        if (raw.trim() === "") { lines.push(""); cursor += 1; continue; }
        var indent = indentOf(raw);
        if (indent <= parentIndent) { break; }
        if (base < 0) { base = indent; }
        lines.push(raw.slice(Math.min(base, indent)).replace(/\s+$/, ""));
        cursor += 1;
      }
      while (lines.length && lines[lines.length - 1] === "") { lines.pop(); }

      if (!folded) {
        var literal = lines.join("\n");
        return chomp === "strip" ? literal : literal + "\n";
      }
      var paragraphs = [];
      var current = [];
      lines.forEach(function (text) {
        if (text === "") {
          paragraphs.push(current.join(" "));
          current = [];
          return;
        }
        current.push(text.trim());
      });
      paragraphs.push(current.join(" "));
      return paragraphs.join("\n").replace(/\s+$/, "");
    }

    function parseValue(rest, parentIndent, line) {
      if (rest === "" || rest.charAt(0) === "#") {
        var next = peek();
        if (!next || next.indent <= parentIndent) { return null; }
        return parseNode(next.indent);
      }
      if (rest.charAt(0) === ">" || rest.charAt(0) === "|") {
        return blockScalar(rest, parentIndent, line);
      }
      if (rest.charAt(0) === "[" || rest.charAt(0) === "{") {
        return parseFlow(rest, line);
      }
      return scalarToken(stripComment(rest), line);
    }

    function parseMapInto(out, indent) {
      for (;;) {
        var p = peek();
        if (!p || p.indent < indent) { return out; }
        if (p.indent > indent) { throw new YamlError("unexpected indentation", p.line + 1); }
        if (/^-( |$)/.test(p.text)) { return out; }
        var pair = splitKey(p.text, p.line);
        if (!pair) { throw new YamlError("expected 'key: value'", p.line + 1); }
        if (Object.prototype.hasOwnProperty.call(out, pair.key)) {
          throw new YamlError("duplicate key " + pair.key, p.line + 1);
        }
        cursor = p.line + 1;
        out[pair.key] = parseValue(pair.rest, indent, p.line);
      }
    }

    function parseSeq(indent) {
      var out = [];
      for (;;) {
        var p = peek();
        if (!p || p.indent < indent) { return out; }
        if (p.indent > indent) { throw new YamlError("unexpected indentation in a list", p.line + 1); }
        if (!/^-( |$)/.test(p.text)) { return out; }

        var after = p.text.slice(1);
        var lead = after.length - after.replace(/^ +/, "").length;
        var rest = after.replace(/^ +/, "");
        var itemIndent = p.indent + 1 + lead;
        cursor = p.line + 1;

        if (rest === "" || rest.charAt(0) === "#") {
          var next = peek();
          out.push(next && next.indent > p.indent ? parseNode(next.indent) : null);
          continue;
        }
        var pair = splitKey(rest, p.line);
        if (pair) {
          var item = {};
          /* The inline key's own column is the parent indent for its value,
             exactly as in a block mapping — pass anything smaller and a `>`
             block scalar swallows the item's sibling keys. */
          item[pair.key] = parseValue(pair.rest, itemIndent, p.line);
          out.push(parseMapInto(item, itemIndent));
          continue;
        }
        if (rest.charAt(0) === ">" || rest.charAt(0) === "|") {
          out.push(blockScalar(rest, p.indent, p.line));
          continue;
        }
        if (rest.charAt(0) === "[" || rest.charAt(0) === "{") {
          out.push(parseFlow(rest, p.line));
          continue;
        }
        out.push(scalarToken(stripComment(rest), p.line));
      }
    }

    function parseNode(indent) {
      var p = peek();
      if (!p || p.indent < indent) { return null; }
      if (/^-( |$)/.test(p.text)) { return parseSeq(p.indent); }
      return parseMapInto({}, p.indent);
    }

    var root = parseNode(0);
    var leftover = peek();
    if (leftover) { throw new YamlError("unexpected content", leftover.line + 1); }
    return root;
  }

  /* ───────────────────── the manifest, as the system reads it ─────────────
     A client-side mirror of `autobranch/spec.py`. Same regexes, same rules,
     same `path: message` shape for every problem, so a problem found here
     reads identically to one the server returns. Two differences, both
     deliberate: this collects every problem instead of raising on the first,
     and it is advisory — `POST /api/runs` re-validates and its answer wins.
     ---------------------------------------------------------------------- */

  var NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
  var SIGNAL_RE = /^[a-z][a-z0-9_]{0,47}$/;
  var ENTRYPOINT_RE = /^[A-Za-z_][\w.]*:[A-Za-z_]\w*$/;
  var SIGNAL_TYPES = ["float", "int", "bool"];

  function isMap(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function previewManifest(raw) {
    var problems = [];
    function fail(where, message) { problems.push(where + ": " + message); }

    if (!isMap(raw)) {
      return { spec: null, problems: ["manifest: expected a mapping at the top level"] };
    }

    var spec = {
      name: raw.name == null ? "" : String(raw.name),
      version: raw.version == null ? 1 : raw.version,
      description: raw.description == null ? "" : String(raw.description),
      entrypoint: raw.entrypoint == null ? "" : String(raw.entrypoint),
      trainer: raw.trainer == null ? "ppo" : String(raw.trainer),
      signals: [],
      objective: { name: "", evaluator: "", direction: "maximize", budget: {} },
      image: { base: "debian_slim", python: "3.11", apt: [], pip: [], commands: [], env: {}, secrets: [], volumes: {} },
      resources: { gpu: null, cpu: 2, memory: 8192 },
      policy: { kind: "discrete", n_actions: null, obs_dim: null, base_model: null }
    };

    if (raw.name == null) { fail("manifest", "missing required key 'name'"); }
    else if (!NAME_RE.test(spec.name)) {
      fail("manifest.name", "'" + spec.name + "' must match " + NAME_RE.source);
    }

    if (raw.entrypoint == null) { fail("manifest", "missing required key 'entrypoint'"); }
    else if (!ENTRYPOINT_RE.test(spec.entrypoint)) {
      fail("manifest.entrypoint", "expected 'module:callable', got '" + spec.entrypoint + "'");
    }

    if (raw.trainer != null &&
        spec.trainer !== "ppo" && spec.trainer !== "grpo_lora" &&
        spec.trainer.indexOf("custom:") !== 0) {
      fail("manifest.trainer", "expected ppo|grpo_lora|custom:module:fn, got '" + spec.trainer + "'");
    }

    /* -- signals: the vocabulary a reward may be built from -- */
    if (raw.signals == null) {
      fail("manifest", "missing required key 'signals'");
    } else if (!Array.isArray(raw.signals) || !raw.signals.length) {
      fail("manifest.signals",
        "at least one signal is required — a reward cannot be composed from nothing");
    } else {
      var seen = {};
      raw.signals.forEach(function (entry, index) {
        var where = "manifest.signals[" + index + "]";
        if (!isMap(entry)) {
          fail(where, "expected a mapping, got " + (Array.isArray(entry) ? "a list" : typeof entry));
          return;
        }
        var signal = {
          index: index,
          name: entry.name == null ? "" : String(entry.name),
          type: entry.type == null ? "float" : String(entry.type),
          range: null,
          doc: entry.doc == null ? "" : String(entry.doc)
        };
        if (entry.name == null) { fail(where, "missing required key 'name'"); }
        else if (!SIGNAL_RE.test(signal.name)) {
          fail(where, "signal name '" + signal.name + "' must match " + SIGNAL_RE.source);
        }
        if (SIGNAL_TYPES.indexOf(signal.type) < 0) {
          fail(where + ".type", "unknown type '" + signal.type + "'");
        }
        if (entry.range != null) {
          if (!Array.isArray(entry.range) || entry.range.length !== 2) {
            fail(where + ".range", "expected [lo, hi]");
          } else {
            var lo = Number(entry.range[0]);
            var hi = Number(entry.range[1]);
            signal.range = [entry.range[0], entry.range[1]];
            if (isNaN(lo) || isNaN(hi)) { fail(where + ".range", "expected two numbers"); }
            else if (lo > hi) { fail(where + ".range", "lo " + lo + " exceeds hi " + hi); }
          }
        }
        if (signal.name && seen[signal.name]) {
          fail("manifest.signals", "duplicate signal '" + signal.name + "'");
        }
        seen[signal.name] = true;
        spec.signals.push(signal);
      });
    }

    /* -- objective: the thing being optimised, and the one block an agent
          may never write -- */
    if (raw.objective == null) {
      fail("manifest", "missing required key 'objective'");
    } else if (!isMap(raw.objective)) {
      fail("manifest.objective", "expected a mapping");
    } else {
      var obj = raw.objective;
      spec.objective.name = obj.name == null ? "" : String(obj.name);
      spec.objective.evaluator = obj.evaluator == null ? "" : String(obj.evaluator);
      spec.objective.direction = obj.direction == null ? "maximize" : String(obj.direction);
      spec.objective.budget = isMap(obj.budget) ? obj.budget : {};
      if (obj.name == null) { fail("manifest.objective", "missing required key 'name'"); }
      if (obj.evaluator == null) { fail("manifest.objective", "missing required key 'evaluator'"); }
      else if (!ENTRYPOINT_RE.test(spec.objective.evaluator)) {
        fail("manifest.objective.evaluator",
          "expected 'module:callable', got '" + spec.objective.evaluator + "'");
      }
      if (spec.objective.direction !== "maximize" && spec.objective.direction !== "minimize") {
        fail("manifest.objective.direction",
          "expected maximize|minimize, got '" + spec.objective.direction + "'");
      }
      if (obj.budget != null && !isMap(obj.budget)) {
        fail("manifest.objective.budget", "expected a mapping");
      }
    }

    /* -- image -- */
    if (raw.image != null && !isMap(raw.image)) {
      fail("manifest.image", "expected a mapping");
    } else if (isMap(raw.image)) {
      var img = raw.image;
      spec.image.base = img.base == null ? "debian_slim" : String(img.base);
      spec.image.python = img.python == null ? "3.11" : String(img.python);
      ["apt", "pip", "commands", "secrets"].forEach(function (key) {
        if (img[key] == null) { return; }
        if (!Array.isArray(img[key])) { fail("manifest.image." + key, "expected a list"); return; }
        spec.image[key] = img[key].map(function (v) { return String(v); });
      });
      if (img.env != null) {
        if (!isMap(img.env)) { fail("manifest.image.env", "expected {KEY: value}"); }
        else { spec.image.env = img.env; }
      }
      if (img.volumes != null) {
        if (!isMap(img.volumes)) { fail("manifest.image.volumes", "expected {path: volume_name}"); }
        else {
          spec.image.volumes = img.volumes;
          Object.keys(img.volumes).forEach(function (mount) {
            if (String(mount).charAt(0) !== "/") {
              fail("manifest.image.volumes", "'" + mount + "' must be absolute");
            }
          });
        }
      }
    }

    /* -- resources: what this run will ask Modal for -- */
    if (raw.resources != null && !isMap(raw.resources)) {
      fail("manifest.resources", "expected a mapping");
    } else if (isMap(raw.resources)) {
      var res = raw.resources;
      var gpu = res.gpu;
      spec.resources.gpu = (gpu == null || gpu === "" || gpu === "none") ? null : String(gpu);
      /* A value that fails validation is still shown as declared: a panel
         that quietly displayed the default instead would be describing a run
         that is not going to happen. */
      var cpu = res.cpu == null ? 2 : Number(res.cpu);
      var memory = res.memory == null ? 8192 : Number(res.memory);
      spec.resources.cpu = isNaN(cpu) ? String(res.cpu) : cpu;
      spec.resources.memory = isNaN(memory) ? String(res.memory) : memory;
      if (isNaN(cpu) || !(cpu > 0)) { fail("manifest.resources.cpu", "must be positive"); }
      if (isNaN(memory) || memory < 512) {
        fail("manifest.resources.memory", "must be at least 512 MiB");
      }
    }

    /* -- policy -- */
    if (raw.policy != null && !isMap(raw.policy)) {
      fail("manifest.policy", "expected a mapping");
    } else if (isMap(raw.policy)) {
      var pol = raw.policy;
      spec.policy.kind = pol.kind == null ? "discrete" : String(pol.kind);
      if (spec.policy.kind !== "discrete" && spec.policy.kind !== "continuous") {
        fail("manifest.policy.kind", "got '" + spec.policy.kind + "'");
      }
      if (spec.policy.kind === "discrete" && pol.n_actions == null) {
        fail("manifest.policy.n_actions", "required for discrete policies");
      }
      if (pol.n_actions != null) {
        var n = Number(pol.n_actions);
        spec.policy.n_actions = isNaN(n) ? String(pol.n_actions) : n;
        if (isNaN(n) || n < 2) { fail("manifest.policy.n_actions", "need at least 2 actions"); }
      }
      if (pol.obs_dim != null) { spec.policy.obs_dim = Number(pol.obs_dim); }
      if (pol.base_model != null) { spec.policy.base_model = String(pol.base_model); }
    } else if (raw.policy == null) {
      fail("manifest.policy.n_actions", "required for discrete policies");
    }

    if (raw.version != null && (isNaN(Number(raw.version)) || Number(raw.version) % 1 !== 0)) {
      fail("manifest.version", "expected an integer, got '" + raw.version + "'");
    }

    return { spec: spec, problems: problems };
  }

  /* ───────────────────── problems, field-adjacent ─────────────────────
     The API answers a bad submission with HTTP 400 and the whole problem
     list, one per line. Every line carries a path — `manifest.signals[2]`,
     `manifest.objective.direction`, `budget`, `entrypoint` — so each one can
     be put next to the thing it is about rather than in one undifferentiated
     red block.
     ---------------------------------------------------------------------- */

  var PROBLEM_ROUTES = [
    [/^manifest\.objective/, "manifest.objective"],
    [/^manifest\.signals/, "manifest.signals"],
    [/^manifest\.resources/, "manifest.resources"],
    [/^manifest\.image/, "manifest.resources"],
    [/^manifest\.policy/, "manifest.policy"],
    [/^manifest\b/, "manifest.identity"],
    [/^budget\b/, "budget"],
    [/^constraints\b/, "constraints"],
    [/^(bundle|zip |no autobranch\.yaml|could not fetch|unsupported compression)/i, "bundle"],
    [/^(entrypoint|env\b|env\.|reset\(|step\(|validate_env|environment not stepped|signal\b)/, "env"]
  ];

  function problemRoute(problem) {
    var text = String(problem).trim();
    for (var i = 0; i < PROBLEM_ROUTES.length; i += 1) {
      if (PROBLEM_ROUTES[i][0].test(text)) { return PROBLEM_ROUTES[i][1]; }
    }
    return null;
  }

  function problemSignalIndex(problem) {
    var m = /^manifest\.signals\[(\d+)\]/.exec(String(problem).trim());
    return m ? Number(m[1]) : null;
  }

  /* Split the 400 body back into the list the server built. `SubmissionError`
     joins its problems with newlines, so this is the exact inverse. */
  function splitProblems(detail) {
    return String(detail == null ? "" : detail)
      .split("\n")
      .map(function (line) { return line.trim(); })
      .filter(function (line) { return line.length > 0; });
  }

  function clearProblems(root) {
    root.querySelectorAll(".field-problem").forEach(function (node) {
      node.hidden = true;
      node.textContent = "";
    });
    root.querySelectorAll(".sec-problems").forEach(function (node) {
      node.hidden = true;
      node.innerHTML = "";
    });
    root.querySelectorAll(".has-problem").forEach(function (node) {
      node.classList.remove("has-problem");
      node.removeAttribute("title");
    });
    root.querySelectorAll("[aria-invalid]").forEach(function (node) {
      node.removeAttribute("aria-invalid");
    });
  }

  var PROBLEM_FIELDS = {
    bundle: ["#bundle-file", "#bundle-url"],
    constraints: ["#constraints"],
    budget: ["#budget-usd", "#budget-hours", "#budget-branches"]
  };

  function showProblems(root, problems, opts) {
    var f = fieldGetter(root);
    var options = opts || {};
    clearProblems(root);

    var box = f("problems");
    var list = f("problems-list");
    if (!problems || !problems.length) {
      box.hidden = true;
      list.innerHTML = "";
      return;
    }

    f("problems-head").textContent = (options.source === "server"
      ? "The server refused this bundle — " + plural(problems.length, "problem")
      : plural(problems.length, "problem") + " in this manifest") +
      ". Nothing was launched.";
    list.innerHTML = problems.map(function (p) {
      var route = problemRoute(p);
      return '<li><span class="problem-path mono">' + esc(route || "submission") + "</span>" +
        '<span class="problem-text">' + esc(p) + "</span></li>";
    }).join("");
    box.hidden = false;

    var grouped = {};
    problems.forEach(function (p) {
      var route = problemRoute(p);
      if (!route) { return; }
      if (!grouped[route]) { grouped[route] = []; }
      grouped[route].push(p);

      var index = problemSignalIndex(p);
      if (index != null) {
        var row = root.querySelector('[data-signal-row="' + index + '"]');
        if (row) {
          row.classList.add("has-problem");
          row.setAttribute("title", p);
        }
      }
    });

    Object.keys(grouped).forEach(function (route) {
      var lines = grouped[route];

      var field = root.querySelector('.field-problem[data-problem="' + route + '"]');
      if (field) {
        field.textContent = lines.join(" · ");
        field.hidden = false;
      }
      (PROBLEM_FIELDS[route] || []).forEach(function (selector) {
        var input = root.querySelector(selector);
        if (input) { input.setAttribute("aria-invalid", "true"); }
      });

      var section = root.querySelector('.pf-sec[data-problem="' + route + '"]');
      if (!section && route === "env") { section = f("pf-env"); }
      if (!section) { return; }
      var holder = section.querySelector(".sec-problems");
      if (!holder) {
        holder = document.createElement("ul");
        holder.className = "sec-problems";
        section.appendChild(holder);
      }
      holder.innerHTML = lines.map(function (line) {
        return "<li>" + esc(line) + "</li>";
      }).join("");
      holder.hidden = false;
      section.classList.add("has-problem");
    });
  }

  /* ───────────────────── preflight render ───────────────────── */

  function kvRow(label, value, cls) {
    return "<div><dt>" + esc(label) + "</dt><dd" + (cls ? ' class="' + cls + '"' : "") +
      ">" + esc(value == null || value === "" ? "—" : value) + "</dd></div>";
  }

  function chipList(values) {
    if (!values || !values.length) { return '<span class="dim mono">none</span>'; }
    return values.map(function (v) {
      return '<span class="chip mono">' + esc(v) + "</span>";
    }).join("");
  }

  function renderPreflight(root, parsed) {
    var f = fieldGetter(root);
    var body = f("pf-body");
    var placeholder = f("pf-placeholder");
    var badge = f("pf-state");
    if (!body) { return; }

    if (!parsed) {
      body.hidden = true;
      body.innerHTML = "";
      placeholder.hidden = false;
      badge.hidden = true;
      return;
    }

    placeholder.hidden = true;

    if (parsed.error) {
      showProblems(root, [], {});
      body.hidden = false;
      body.innerHTML =
        '<div class="notice is-warn">' +
        '<p class="notice-title">Manifest not previewed</p>' +
        "<p>" + esc(parsed.error) + "</p>" +
        "<p>This blocks nothing: the server parses the bundle itself on submit and " +
        "answers with the full problem list.</p>" +
        "</div>";
      badge.hidden = false;
      badge.className = "badge badge-evaluating";
      badge.textContent = "not previewed";
      return;
    }

    var spec = parsed.spec;
    var problems = parsed.problems || [];

    badge.hidden = false;
    if (problems.length) {
      badge.className = "badge badge-failed";
      badge.textContent = plural(problems.length, "problem");
    } else {
      badge.className = "badge badge-succeeded";
      badge.textContent = parsed.example ? "example manifest" : "manifest parsed";
    }

    var objective = spec.objective;
    var budgetKeys = Object.keys(objective.budget || {});
    var signals = spec.signals || [];

    var html =
      '<div class="pf-source mono">' +
      esc(parsed.example
        ? "examples/gridworld/" + MANIFEST_NAME + " — a built-in example, nothing is submitted"
        : parsed.path + " — parsed in this browser, " + plural(parsed.entryCount || 0, "file") + " in the archive") +
      "</div>" +

      '<div class="pf-grid">' +

      '<section class="pf-sec" data-problem="manifest.identity">' +
      '<p class="eyebrow">Environment</p>' +
      '<p class="pf-headline mono">' + esc(spec.name || "unnamed") +
      '<span class="pf-version">v' + esc(spec.version) + "</span></p>" +
      '<dl class="kv-grid pf-kv">' +
      kvRow("Trainer", spec.trainer, "mono") +
      kvRow("Entrypoint", spec.entrypoint, "mono") +
      "</dl>" +
      (spec.description
        ? '<p class="pf-desc">' + esc(spec.description) + "</p>"
        : '<p class="pf-desc dim">No description in the manifest.</p>') +
      '<ul class="sec-problems" hidden></ul>' +
      "</section>" +

      '<section class="pf-sec pf-objective" data-problem="manifest.objective">' +
      '<p class="eyebrow">Objective — the measurement</p>' +
      '<p class="pf-headline mono">' + esc(objective.name || "unnamed") +
      '<span class="dir-chip mono">' + esc(objective.direction) + "</span></p>" +
      '<dl class="kv-grid pf-kv">' +
      kvRow("Evaluator", objective.evaluator, "mono") +
      kvRow("Eval budget", budgetKeys.length ? budgetKeys.map(function (k) {
        return k + "=" + objective.budget[k];
      }).join(" · ") : "defaults", "mono") +
      "</dl>" +
      '<p class="pf-desc">Agents compose rewards from the signals below. This block is the ' +
      "score, and no agent — and no reward — can change it.</p>" +
      '<ul class="sec-problems" hidden></ul>' +
      "</section>" +

      '<section class="pf-sec" data-problem="manifest.resources">' +
      '<p class="eyebrow">Resources per sandbox</p>' +
      '<dl class="kv-grid pf-kv">' +
      kvRow("GPU", spec.resources.gpu || "none (CPU only)", "mono") +
      kvRow("CPU", spec.resources.cpu, "mono num") +
      kvRow("Memory", typeof spec.resources.memory === "number"
        ? fmtBytes(spec.resources.memory * 1024 * 1024)
        : String(spec.resources.memory), "mono num") +
      kvRow("Image", spec.image.base + " · python " + spec.image.python, "mono") +
      "</dl>" +
      '<p class="pf-sub eyebrow">pip</p>' +
      '<p class="chips">' + chipList(spec.image.pip) + "</p>" +
      (spec.image.apt.length
        ? '<p class="pf-sub eyebrow">apt</p><p class="chips">' + chipList(spec.image.apt) + "</p>"
        : "") +
      (spec.image.secrets.length
        ? '<p class="pf-sub eyebrow">secrets</p><p class="chips">' + chipList(spec.image.secrets) + "</p>"
        : "") +
      '<ul class="sec-problems" hidden></ul>' +
      "</section>" +

      '<section class="pf-sec" data-problem="manifest.policy">' +
      '<p class="eyebrow">Policy</p>' +
      '<dl class="kv-grid pf-kv">' +
      kvRow("Kind", spec.policy.kind, "mono") +
      kvRow("Actions", spec.policy.n_actions == null ? "—" : spec.policy.n_actions, "mono num") +
      kvRow("Obs dim", spec.policy.obs_dim == null ? "—" : spec.policy.obs_dim, "mono num") +
      kvRow("Base model", spec.policy.base_model || "from the constraints, if any", "mono") +
      "</dl>" +
      '<ul class="sec-problems" hidden></ul>' +
      "</section>" +

      "</div>" +

      '<section class="pf-sec pf-signals" data-problem="manifest.signals">' +
      '<div class="pf-sec-head">' +
      '<p class="eyebrow">Signals</p>' +
      '<span class="region-note mono">' + esc(plural(signals.length, "signal")) +
      " declared · the whole vocabulary a reward may be built from</span>" +
      "</div>" +
      '<div class="table-scroll">' +
      '<table class="table table-dense pf-table">' +
      "<thead><tr>" +
      '<th scope="col">Signal</th>' +
      '<th scope="col">Type</th>' +
      '<th scope="col" class="num">Range</th>' +
      '<th scope="col" class="cell-wide">What the agent is told it means</th>' +
      "</tr></thead><tbody>" +
      (signals.length
        ? signals.map(function (s) {
            return '<tr data-signal-row="' + s.index + '">' +
              '<td class="cell-id">' + esc(s.name || "—") + "</td>" +
              "<td>" + esc(s.type) + "</td>" +
              '<td class="num">' + esc(s.range ? "[" + s.range[0] + ", " + s.range[1] + "]" : "—") + "</td>" +
              '<td class="cell-wide cell-doc">' +
              (s.doc ? esc(s.doc) : '<span class="dim">no doc — the agent will have to guess what this means</span>') +
              "</td></tr>";
          }).join("")
        : '<tr><td colspan="4"><span class="dim">no signals declared</span></td></tr>') +
      "</tbody></table></div>" +
      '<ul class="sec-problems" hidden></ul>' +
      "</section>" +

      '<section class="pf-sec" data-f="pf-env" data-problem="env" hidden>' +
      '<p class="eyebrow">Environment check</p>' +
      '<p class="pf-desc">The server imports the entrypoint and steps the environment before ' +
      "anything is created. These came back from that pass.</p>" +
      '<ul class="sec-problems" hidden></ul>' +
      "</section>";

    body.innerHTML = html;
    body.hidden = false;

    if (problems.length) {
      showProblems(root, problems, { source: "client" });
    } else {
      showProblems(root, [], {});
    }
  }

  /* The reference manifest, so the screen can be read without a bundle in
     hand. Byte-for-byte the shipped `examples/gridworld/autobranch.yaml`
     header fields — it is parsed by the same reader a real upload goes
     through, which is the point of showing it. */
  var EXAMPLE_MANIFEST = [
    "name: gridworld",
    "version: 1",
    "description: >",
    "  8x8 locked-door gridworld. The agent starts in a corner, must collect a key",
    "  from the far corner, then reach the goal; lava kills and a 5% slip",
    "  probability makes hugging it expensive. Reward design decides whether this",
    "  is solved or not, which is the point.",
    "",
    "image:",
    "  base: debian_slim",
    '  python: "3.11"',
    "  pip:",
    "    - numpy>=1.26",
    "    - torch>=2.2",
    "    - pyyaml>=6.0",
    "",
    "entrypoint: grid_env:make_env",
    "",
    "resources:",
    "  gpu: null",
    "  cpu: 2",
    "  memory: 2048",
    "",
    "signals:",
    "  - name: reached_goal",
    "    type: bool",
    "    range: [0, 1]",
    "    doc: >",
    "      1 on the step the agent enters the goal tile. The goal is locked: it can",
    "      only be entered while holding the key, so this is also the success",
    "      condition of the objective.",
    "  - name: stepped_in_lava",
    "    type: bool",
    "    range: [0, 1]",
    '    doc: "1 on the step the agent enters a lava tile, which ends the episode in failure."',
    "  - name: picked_up_key",
    "    type: bool",
    "    range: [0, 1]",
    '    doc: "1 on the single step the key is collected (walking onto the key tile). 0 thereafter."',
    "  - name: distance_to_goal",
    "    type: float",
    "    range: [0, 1]",
    "    doc: >",
    "      Manhattan distance from the agent to the goal, divided by 2*(size-1).",
    "      Cannot reach 0 without the key, because a locked goal acts as a wall.",
    "  - name: distance_to_key",
    "    type: float",
    "    range: [0, 1]",
    "    doc: >",
    "      Manhattan distance from the agent to the key, divided by 2*(size-1), and",
    "      exactly 0 once the key is held -- it measures distance to what is still",
    "      needed.",
    "  - name: step_cost",
    "    type: float",
    "    range: [0, 1]",
    '    doc: "Always 1.0, one per step. Give it a negative weight to pay for time."',
    "  - name: revisited_tile",
    "    type: bool",
    "    range: [0, 1]",
    "    doc: >",
    "      1 if the tile the agent occupies after moving was already visited this",
    "      episode (including standing still against a wall or a locked door).",
    "",
    "objective:",
    "  name: success_rate",
    "  direction: maximize",
    "  evaluator: grid_env:evaluate",
    "  budget:",
    "    episodes: 256",
    "    lanes: 32",
    "    max_steps: 100",
    "    slip_prob: 0.05",
    "",
    "policy:",
    "  kind: discrete",
    "  n_actions: 4",
    "  obs_dim: 81",
    "",
    "trainer: ppo",
    ""
  ].join("\n");

  /* ───────────────────────────── tree layout ─────────────────────────────
     Two guarantees, in this order of importance.

     (1) NOTHING FLOATS. A branch id encodes its own lineage by construction:
         b0 → b0.1 → b0.1.2. A parent id is therefore always derivable by
         stripping the last ".N" segment, and `resolveParent` tries, in order:

           a. `parent_id`, when that branch is actually present in the payload;
           b. the id-derived ancestor chain — b0.1.1 → b0.1 → b0 — walking up
              until it reaches a branch that IS present, synthesising a visibly
              marked "record missing" placeholder for any intermediate id that
              is genuinely absent, so the chain still reaches all the way down;
           c. the single root, for an id with no derivable ancestor at all.

         Exactly one node ends up without a parent. A second root is
         impossible: any extra root candidate is re-parented onto the first
         with a dashed "unlinked" edge rather than left loose on the canvas.
         `failed` and `abandoned` branches are ordinary members of the tree —
         same position, same edge to their parent, only de-emphasised paint —
         so the floating boxes a watcher used to see for dead branches cannot
         be drawn from here. The invariant check at the bottom of this function
         says so out loud, in the console, if the code above it ever regresses.

     (2) THE CANVAS IS THE CONTENT. Row depth comes from the resolved tree, not
         from `branch.depth`, so an edge can never point upwards; and the svg
         is sized to the node bounding box plus PAD, never to a fixed height,
         so a two-node tree is a two-node-tall strip.

     Layout itself is a tidy two-pass: every leaf takes the next free column,
     an internal node takes the midpoint of its children's columns. Leaf
     columns are unique and a subtree owns a contiguous range, so a parent
     centred over its children cannot collide with a sibling subtree.
     ------------------------------------------------------------------------ */

  var NODE_W = 236;
  var NODE_H = 76;
  var GAP_X = 26;
  var GAP_Y = 38;
  var PAD = 14;
  var MAX_CHAIN = 64;

  /* b0.1.1 → b0.1 → b0 → null. An id with no "." has no derivable parent. */
  function parentFromId(id) {
    var text = String(id == null ? "" : id);
    var cut = text.lastIndexOf(".");
    if (cut <= 0) { return null; }
    return text.slice(0, cut);
  }

  function layoutTree(branches) {
    var byId = {};
    var order = [];
    var notes = [];
    var synthesised = 0;

    function addNode(branch, placeholder) {
      var id = String(branch.branch_id);
      if (byId[id]) { return byId[id]; }
      var node = {
        id: id,
        branch: branch,
        placeholder: Boolean(placeholder),
        parent: null,
        link: null,
        kids: [],
        level: 0,
        column: null,
        x: 0,
        y: 0
      };
      byId[id] = node;
      order.push(node);
      return node;
    }

    (branches || []).forEach(function (b) {
      if (b && b.branch_id != null) { addNode(b, false); }
    });

    /* `chain` runs from a node's immediate derived parent upwards; `anchor` is
       the first id that really exists, or null when the chain runs off the top
       of the tree (the root pass then links the topmost placeholder). */
    function attachChain(chain, anchor) {
      var child = null;
      for (var i = 0; i < chain.length; i += 1) {
        var ph = addNode({
          branch_id: chain[i],
          parent_id: null,
          depth: null,
          status: "missing",
          scores: [],
          reward_config: null
        }, true);
        synthesised += 1;
        notes.push("no branch record for " + chain[i] + " — drawn as a placeholder so its children stay attached");
        if (child) { child.parent = ph; child.link = "derived"; }
        child = ph;
      }
      if (child) {
        child.parent = anchor;
        child.link = anchor ? "derived" : null;
      }
      return byId[chain[0]];
    }

    function resolveParent(node) {
      var pid = node.branch && node.branch.parent_id != null
        ? String(node.branch.parent_id)
        : "";

      /* (a) the recorded parent, when it is really in the payload */
      if (pid && pid !== node.id && byId[pid]) {
        node.link = "recorded";
        return byId[pid];
      }
      if (pid && pid !== node.id) {
        notes.push("branch " + node.id + " names parent " + pid + ", which is not in this payload — parent derived from the id");
      }

      /* (b) the id-derived ancestor chain */
      var chain = [];
      var cursor = parentFromId(node.id);
      while (cursor && chain.length < MAX_CHAIN) {
        if (byId[cursor]) {
          node.link = "derived";
          return chain.length ? attachChain(chain, byId[cursor]) : byId[cursor];
        }
        chain.push(cursor);
        cursor = parentFromId(cursor);
      }
      if (chain.length) {
        node.link = "derived";
        return attachChain(chain, null);
      }

      /* (c) nothing derivable: the root pass below attaches it to the root */
      return null;
    }

    /* Only real branches are resolved; placeholders are linked as they are
       created, and `order` grows underneath us, hence the snapshot. */
    order.slice().forEach(function (node) {
      node.parent = resolveParent(node);
    });

    /* A parent_id cycle would leave the forest rootless. Cut it rather than
       drop the nodes it contains. */
    order.forEach(function (node) {
      var seen = {};
      var cursor = node;
      var guard = 0;
      while (cursor && cursor.parent && guard < 512) {
        if (seen[cursor.id]) {
          notes.push("parent cycle through " + cursor.id + " — one link cut to keep the tree a tree");
          cursor.parent = null;
          cursor.link = null;
          break;
        }
        seen[cursor.id] = true;
        cursor = cursor.parent;
        guard += 1;
      }
    });

    /* One root, always. Prefer a real branch over a placeholder, then the
       shortest id — b0 is a likelier root than b1.2. */
    var candidates = order.filter(function (n) { return !n.parent; });
    var root = null;
    candidates.forEach(function (n) {
      if (!root) { root = n; return; }
      var rank = (Number(n.placeholder) - Number(root.placeholder)) ||
        (n.id.length - root.id.length);
      if (rank < 0) { root = n; }
    });
    candidates.forEach(function (n) {
      if (n === root) { return; }
      n.parent = root;
      n.link = "unlinked";
      notes.push("branch " + n.id + " has no derivable parent — attached to the root " + root.id + " with a dashed edge rather than floated");
    });

    order.forEach(function (n) { n.kids = []; });
    order.forEach(function (n) { if (n.parent) { n.parent.kids.push(n); } });

    /* Depth from the resolved tree, so a node always sits below its parent. */
    var reached = [];
    if (root) {
      root.level = 0;
      reached.push(root);
      for (var head = 0; head < reached.length; head += 1) {
        var here = reached[head];
        for (var k = 0; k < here.kids.length; k += 1) {
          here.kids[k].level = here.level + 1;
          reached.push(here.kids[k]);
        }
      }
    }

    var nextColumn = 0;
    function assignColumns(node) {
      if (node.column !== null) { return node.column; }
      if (!node.kids.length) {
        node.column = nextColumn;
        nextColumn += 1;
        return node.column;
      }
      var cols = node.kids.map(assignColumns);
      node.column = (Math.min.apply(null, cols) + Math.max.apply(null, cols)) / 2;
      return node.column;
    }
    if (root) { assignColumns(root); }
    order.forEach(function (n) {
      if (n.column === null) { n.column = nextColumn; nextColumn += 1; }
    });

    order.forEach(function (n) {
      n.x = PAD + n.column * (NODE_W + GAP_X);
      n.y = PAD + n.level * (NODE_H + GAP_Y);
    });

    /* Size to the actual bounding box plus PAD, and shift the content so the
       box starts at PAD — no empty acreage under a short tree. */
    var minX = Infinity;
    var minY = Infinity;
    var maxX = -Infinity;
    var maxY = -Infinity;
    order.forEach(function (n) {
      minX = Math.min(minX, n.x);
      minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x + NODE_W);
      maxY = Math.max(maxY, n.y + NODE_H);
    });
    if (!order.length) { minX = minY = maxX = maxY = 0; }
    var dx = PAD - minX;
    var dy = PAD - minY;
    order.forEach(function (n) { n.x += dx; n.y += dy; });

    var edges = [];
    order.forEach(function (n) {
      if (!n.parent) { return; }
      edges.push({ from: n.parent, to: n, kind: n.link || "derived" });
    });

    /* The invariant, asserted rather than assumed: one root, and every other
       node owns exactly one edge. Breaking it is the bug that used to show up
       as a floating box, so it belongs in the console, not on the canvas. */
    var incoming = {};
    edges.forEach(function (e) { incoming[e.to.id] = (incoming[e.to.id] || 0) + 1; });
    var orphans = order.filter(function (n) {
      return n !== root && incoming[n.id] !== 1;
    }).map(function (n) { return n.id; });

    var audit = {
      nodes: order.length,
      edges: edges.length,
      root: root ? root.id : null,
      roots: order.filter(function (n) { return !n.parent; }).length,
      placeholders: synthesised,
      inferred: edges.filter(function (e) { return e.kind !== "recorded"; }).length,
      orphans: orphans,
      notes: notes,
      ok: true
    };
    if (order.length && (orphans.length || audit.roots > 1 || edges.length !== order.length - 1)) {
      audit.ok = false;
      if (window.console && window.console.error) {
        window.console.error(
          "autobranch: sandbox-tree invariant broken — a node would render with no edge to its parent. " +
          "This is a bug in layoutTree(), not in the data.", audit
        );
      }
    }
    /* Parked on `window` so the invariant is inspectable from the console. */
    window.AUTOBRANCH_TREE_AUDIT = audit;

    return {
      nodes: order,
      edges: edges,
      root: root,
      audit: audit,
      nodeW: NODE_W,
      nodeH: NODE_H,
      width: Math.round(order.length ? maxX + dx + PAD : 0),
      height: Math.round(order.length ? maxY + dy + PAD : 0)
    };
  }

  /* One node, one Modal sandbox, three rows: what it is and its status; the
     objective score it reached and the sandbox it ran in; what it is doing.
     The reward weights used to be printed here too and were unreadable at
     35 truncated characters — they live in the Sandbox tab, in full. */
  function nodeMarkup(run, node, bestId) {
    var b = node.branch;
    var x = node.x;
    var y = node.y;

    if (node.placeholder) {
      return '<g class="node is-placeholder" aria-hidden="true">' +
        '<rect class="node-box" rx="3" x="' + x + '" y="' + y + '" width="' + NODE_W + '" height="' + NODE_H + '"></rect>' +
        '<text class="node-id" x="' + (x + 14) + '" y="' + (y + 28) + '">' + esc(node.id) + "</text>" +
        '<text class="node-status" x="' + (x + NODE_W - 14) + '" y="' + (y + 28) + '" text-anchor="end">MISSING</text>' +
        '<text class="node-meta" x="' + (x + 14) + '" y="' + (y + 52) + '">record missing · inferred from a child id</text>' +
        "</g>";
    }

    var s = statusClass(b.status);
    var spent = s === "abandoned" || s === "failed";
    var best = bestId === node.id;
    var score = bestScoreOf(b);
    var sandbox = sandboxOf(run, b);
    var act = activityOf(run, b);
    var live = isLiveBranch(run, s);
    var label = statusLabel(run, s);

    var classes = ["node", "status-" + s];
    if (spent) { classes.push("is-spent"); }
    if (best) { classes.push("is-best"); }
    if (state.selected === node.id) { classes.push("is-selected"); }

    var aria = "sandbox " + (sandbox || "not yet created") + ", branch " + node.id +
      ", " + label + ", objective " + fmtScore(score) + ", " + act.text +
      (best ? ", best in this run" : "") + (spent ? ", this line is finished" : "");

    var sandboxText = sandbox
      ? shortSandbox(sandbox)
      : (s === "pending" && live ? "sandbox starting…" : "no sandbox id");

    return '<g class="' + classes.join(" ") + '" tabindex="0" role="button" ' +
      'data-branch="' + esc(node.id) + '" aria-label="' + esc(aria) + '">' +
      '<rect class="node-ring" rx="5" x="' + (x - 3) + '" y="' + (y - 3) + '" width="' + (NODE_W + 6) + '" height="' + (NODE_H + 6) + '"></rect>' +
      '<rect class="node-box" rx="3" x="' + x + '" y="' + y + '" width="' + NODE_W + '" height="' + NODE_H + '"></rect>' +
      '<line class="node-stripe stripe-' + s + '" x1="' + (x + 1.5) + '" y1="' + (y + 2) + '" x2="' + (x + 1.5) + '" y2="' + (y + NODE_H - 2) + '"></line>' +

      '<text class="node-id" x="' + (x + 14) + '" y="' + (y + 20) + '">' + esc(node.id) + "</text>" +
      '<text class="node-status text-' + s + '" x="' + (x + NODE_W - 14) + '" y="' + (y + 20) + '" text-anchor="end">' +
      esc(label.toUpperCase()) + "</text>" +

      '<text class="node-score" x="' + (x + 14) + '" y="' + (y + 44) + '">' + esc(fmtScore(score)) + "</text>" +
      '<text class="node-meta" x="' + (x + NODE_W - 14) + '" y="' + (y + 44) + '" text-anchor="end">' +
      esc(sandboxText) + "</text>" +

      '<circle class="node-pip pip-' + s + (live ? " pulse" : "") + '" cx="' + (x + 17) + '" cy="' + (y + 61) + '" r="3"></circle>' +
      '<text class="node-activity' + (act.level === "warn" || act.level === "error" ? " is-" + act.level : "") +
      '" x="' + (x + 27) + '" y="' + (y + 64) + '">' + esc(truncate(act.text, best ? 24 : 30)) + "</text>" +
      (best
        ? '<text class="best-tag" x="' + (x + NODE_W - 14) + '" y="' + (y + 64) + '" text-anchor="end">BEST</text>'
        : "") +
      "</g>";
  }

  function renderTree(svg, run) {
    var branches = (run && run.branches) || [];

    if (!branches.length) {
      /* A compact strip, not a canvas waiting for content. */
      svg.setAttribute("width", "340");
      svg.setAttribute("height", "36");
      svg.setAttribute("viewBox", "0 0 340 36");
      svg.innerHTML = '<text class="node-meta" x="14" y="22">no sandboxes yet — the first one is being created</text>';
      svg.__layout = null;
      return;
    }

    var tree = layoutTree(branches);
    var bestId = run.best && run.best.branch_id;

    svg.setAttribute("width", String(tree.width));
    svg.setAttribute("height", String(tree.height));
    svg.setAttribute("viewBox", "0 0 " + tree.width + " " + tree.height);

    var parts = [];

    /* Edges first so a node box always paints over its own elbow. Dashed means
       "inferred" (derived from the id, or re-parented onto the root); a spent
       line is thinner and lighter but never invisible — an edge no one can see
       is what makes a dead branch look like it is floating. */
    tree.edges.forEach(function (e) {
      var x1 = e.from.x + tree.nodeW / 2;
      var y1 = e.from.y + tree.nodeH;
      var x2 = e.to.x + tree.nodeW / 2;
      var y2 = e.to.y;
      var mid = y1 + Math.max(10, (y2 - y1) / 2);
      var cls = ["tree-edge", "edge-" + e.kind];
      var status = statusClass(e.to.branch && e.to.branch.status);
      if (status === "abandoned" || status === "failed") { cls.push("is-spent"); }
      parts.push('<path class="' + cls.join(" ") + '" d="M' + x1 + " " + y1 +
        " V" + mid + " H" + x2 + " V" + y2 + '"></path>');
    });

    tree.nodes.forEach(function (n) {
      parts.push(nodeMarkup(run, n, bestId));
    });

    svg.innerHTML = parts.join("");
    svg.__layout = tree;
  }

  function scrollNodeIntoView(scroller, branchId) {
    var svg = scroller.querySelector(".tree-svg");
    if (!svg || !svg.__layout) { return; }
    var layout = svg.__layout;
    var hit = null;
    layout.nodes.forEach(function (n) {
      if (n.id === branchId) { hit = n; }
    });
    if (!hit) { return; }
    var targetLeft = hit.x + layout.nodeW / 2 - scroller.clientWidth / 2;
    var targetTop = hit.y + layout.nodeH / 2 - scroller.clientHeight / 2;
    scroller.scrollLeft = clamp(targetLeft, 0, Math.max(0, layout.width - scroller.clientWidth));
    scroller.scrollTop = clamp(targetTop, 0, Math.max(0, layout.height - scroller.clientHeight));
  }

  /* ───────────────────────────── status chips ───────────────────────────── */

  function renderStatusChips(list, run) {
    var branches = (run && run.branches) || [];
    var counts = statusCounts(branches);
    var order = ["running", "evaluating", "pending", "succeeded", "abandoned", "failed"];
    var shown = order.filter(function (s) { return counts[s]; });
    if (!shown.length) {
      list.innerHTML = '<li class="chip-status"><span class="dim mono">no sandboxes yet</span></li>';
      return;
    }
    list.innerHTML = shown.map(function (s) {
      return '<li class="chip-status st-text-' + s + '" title="' +
        esc(pluralise(counts[s], "sandbox", "sandboxes") + " " + statusLabel(run, s)) + '">' +
        '<span class="dot st-' + s + (isLiveBranch(run, s) ? " pulse" : "") + '" aria-hidden="true"></span>' +
        '<span class="mono num">' + counts[s] + "</span>" +
        '<span class="chip-status-label">' + esc(statusLabel(run, s)) + "</span>" +
        "</li>";
    }).join("");
  }

  /* ───────────────────────────── detail panel ───────────────────────────── */

  /* Root → this branch, under the same lineage rule the tree draws: the
     recorded parent_id when that branch is present, otherwise the parent the
     id itself implies, so a missing record never truncates the path. */
  function lineageOf(run, branchId) {
    var byId = {};
    ((run && run.branches) || []).forEach(function (b) { byId[b.branch_id] = b; });
    var path = [];
    var seen = {};
    var cursor = byId[branchId];
    var guard = 0;
    while (cursor && guard < 64 && !seen[cursor.branch_id]) {
      seen[cursor.branch_id] = true;
      path.unshift(cursor);
      var pid = cursor.parent_id == null ? "" : String(cursor.parent_id);
      var next = pid && byId[pid] ? byId[pid] : null;
      if (!next) {
        var derived = parentFromId(cursor.branch_id);
        while (derived && !byId[derived]) { derived = parentFromId(derived); }
        next = derived ? byId[derived] : null;
      }
      cursor = next;
      guard += 1;
    }
    return path;
  }

  function renderDetail(panel, run) {
    if (!state.selected) {
      panel.innerHTML =
        '<div class="detail-head"><div><p class="eyebrow">Sandbox detail</p>' +
        '<p class="detail-path">Pick a node in the tree above, or a row under “All sandboxes”.</p></div></div>' +
        '<div class="detail-section"><p class="rationale">Every sandbox keeps its reward config, its score history, its spend, and the rationale the agent wrote before it forked. The abandoned ones keep theirs too — that is what losers.json is for.</p></div>';
      return;
    }

    var branch = branchById(run, state.selected);
    if (!branch) {
      /* A fork is announced in the event log by its parent, and its own branch
         record lands a poll or two later. Say so rather than implying the
         sandbox vanished. */
      panel.innerHTML =
        '<div class="detail-head"><div><p class="eyebrow">Sandbox</p>' +
        '<p class="detail-id">' + esc(state.selected) + "</p>" +
        '<p class="detail-path">no branch record published yet</p></div></div>' +
        '<div class="detail-section"><p class="rationale">' + esc(state.selected) +
        " appears in the activity feed but the registry has not published its branch record yet." +
        " Forks are announced by their parent and register themselves once their sandbox is up;" +
        " this node will appear in the tree on one of the next refreshes.</p>" +
        (eventsFor(run, state.selected).length
          ? '<ol class="timeline" tabindex="0">' + eventsFor(run, state.selected).slice(-10).reverse().map(function (e) {
              var level = e.level === "warn" || e.level === "error" ? e.level : "info";
              return '<li class="timeline-' + level + '">' +
                '<span class="timeline-ts mono">' + esc(fmtClock(e.ts)) + "</span>" +
                '<span class="timeline-msg">' + esc(humanizeEvent(e.message, state.selected)) + "</span></li>";
            }).join("") + "</ol>"
          : "") +
        "</div>";
      return;
    }

    var s = statusClass(branch.status);
    var abandoned = s === "abandoned" || s === "failed";
    var score = bestScoreOf(branch);
    var last = lastScoreOf(branch);
    var path = lineageOf(run, branch.branch_id).map(function (b) { return b.branch_id; });
    var sandbox = sandboxOf(run, branch);
    var act = activityOf(run, branch);
    var timeline = eventsFor(run, branch.branch_id).slice(-40).reverse();

    var html =
      '<div class="detail-head">' +
      "<div>" +
      '<p class="eyebrow">Sandbox</p>' +
      '<p class="detail-id' + (abandoned ? " is-abandoned" : "") + '">' +
      esc(sandbox || (s === "pending" ? "not created yet" : "id unknown")) + "</p>" +
      '<p class="detail-path">branch ' + esc(branch.branch_id) + " · lineage " + esc(path.join(" → ")) + "</p>" +
      "</div>" +
      statusBadge(s, isLiveBranch(run, s), statusLabel(run, s)) +
      "</div>" +

      '<div class="detail-section detail-now">' +
      '<p class="eyebrow">Doing now</p>' +
      '<p class="now-line' + (act.level === "warn" || act.level === "error" ? " is-" + act.level : "") +
      '" title="' + esc(act.text) + '">' +
      '<span class="dot st-' + s + (isLiveBranch(run, s) ? " pulse" : "") + '" aria-hidden="true"></span>' +
      '<span>' + esc(truncate(act.text, 200)) + "</span></p>" +
      '<p class="detail-path">' +
      esc((act.since == null ? "no timestamp" : "since " + fmtClock(act.since) + " · " + ageText(act.since) + " ago") +
        " · " + (act.source === "sandbox"
          ? "published by the sandbox"
          : act.source === "event" ? "newest event for this branch" : "inferred from status")) +
      "</p>" +
      "</div>" +

      /* Numbers and the score history belong together: two adjacent boxes
         saying one thing become one box saying it. */
      '<div class="detail-section detail-split">' +
      "<div>" +
      '<dl class="kv-grid">' +
      "<div><dt>Best objective</dt><dd" + (run.best && run.best.branch_id === branch.branch_id ? ' class="accent"' : "") + ">" + esc(fmtScore(score)) + "</dd></div>" +
      "<div><dt>Latest eval</dt><dd>" + esc(last ? fmtScore(last.score) : "—") + "</dd></div>" +
      "<div><dt>Depth</dt><dd>" + esc(String(branch.depth == null ? "—" : branch.depth)) + "</dd></div>" +
      "<div><dt>Evaluations</dt><dd>" + esc(String((branch.scores || []).length)) + "</dd></div>" +
      "<div><dt>Spend</dt><dd>" + esc(fmtUsd(branch.spend_usd)) + "</dd></div>" +
      "<div><dt>Agent tokens</dt><dd>" + esc(fmtTokens(branch.tokens)) + "</dd></div>" +
      "</dl>" +
      "</div>" +
      "<div>" +
      '<p class="eyebrow">Objective score history</p>' +
      sparkline(branch.scores, { faint: abandoned, width: 320, height: 48 }) +
      '<p class="detail-path">' +
      esc((branch.scores || []).length
        ? "step " + fmtInt(branch.scores[0].step) + " → " + fmtInt(last.step) +
          " · scored by the evaluator, never by the reward"
        : "no evaluations recorded yet") +
      "</p>" +
      "</div>" +
      "</div>" +

      '<div class="detail-section detail-split">' +
      "<div>" +
      '<p class="eyebrow">Agent rationale</p>' +
      '<p class="rationale">' + esc(branch.rationale || "No rationale recorded for this branch.") + "</p>" +
      (branch.abandoned_reason
        ? '<p class="note-abandoned"><strong>' +
          (s === "failed" ? "Failed:" : "Abandoned:") + "</strong> " + esc(branch.abandoned_reason) + "</p>"
        : "") +
      "</div>" +
      "<div>" +
      '<p class="eyebrow">Reward config</p>' +
      '<pre class="code mono" tabindex="0">' + esc(JSON.stringify(branch.reward_config || {}, null, 2)) + "</pre>" +
      "</div>" +
      "</div>" +

      '<div class="detail-section">' +
      '<p class="eyebrow">This sandbox’s timeline</p>' +
      (timeline.length
        ? '<ol class="timeline" tabindex="0">' + timeline.map(function (e) {
            var level = e.level === "warn" || e.level === "error" ? e.level : "info";
            return '<li class="timeline-' + level + '">' +
              '<span class="timeline-ts mono">' + esc(fmtClock(e.ts)) + "</span>" +
              '<span class="timeline-msg">' + esc(humanizeEvent(e.message, branch.branch_id)) + "</span>" +
              "</li>";
          }).join("") + "</ol>"
        : '<p class="spark-empty">no events recorded for this sandbox yet</p>') +
      '<p class="detail-path">newest first' + (timeline.length >= 40 ? " · last 40" : "") + "</p>" +
      "</div>";

    panel.innerHTML = html;
  }

  /* ───────────────────────────── sandbox table ───────────────────────────── */

  function sortValue(branch, key) {
    if (key === "score") {
      var v = bestScoreOf(branch);
      return v == null ? -Infinity : v;
    }
    if (key === "status") { return STATUS_ORDER[statusClass(branch.status)]; }
    if (key === "depth") { return branch.depth == null ? 0 : branch.depth; }
    if (key === "spend_usd") { return branch.spend_usd || 0; }
    if (key === "tokens") { return branch.tokens || 0; }
    if (key === "sandbox") { return String(sandboxOf(state.run, branch) || "zzz"); }
    return String(branch.branch_id || "");
  }

  function markSort(table, sort) {
    var heads = table.querySelectorAll("th");
    for (var i = 0; i < heads.length; i += 1) {
      var btn = heads[i].querySelector(".th-sort");
      if (!btn) { continue; }
      if (btn.getAttribute("data-sort") === sort.key) {
        heads[i].setAttribute("aria-sort", sort.dir === "asc" ? "ascending" : "descending");
      } else {
        heads[i].removeAttribute("aria-sort");
      }
    }
  }

  function renderTable(table, run) {
    var branches = ((run && run.branches) || []).slice();
    var key = state.sort.key;
    var dir = state.sort.dir === "asc" ? 1 : -1;

    branches.sort(function (a, b) {
      var va = sortValue(a, key);
      var vb = sortValue(b, key);
      if (va < vb) { return -1 * dir; }
      if (va > vb) { return 1 * dir; }
      return String(a.branch_id).localeCompare(String(b.branch_id));
    });

    markSort(table, state.sort);

    var bestId = run.best && run.best.branch_id;
    var rows = branches.map(function (b) {
      var s = statusClass(b.status);
      var abandoned = s === "abandoned" || s === "failed";
      var sandbox = sandboxOf(run, b);
      var act = activityOf(run, b);
      var cls = ["is-clickable"];
      if (abandoned) { cls.push("is-abandoned"); }
      if (state.selected === b.branch_id) { cls.push("is-selected"); }
      if (LIVE_STATES[s]) { cls.push("is-live"); }
      return '<tr class="' + cls.join(" ") + '" data-branch="' + esc(b.branch_id) + '" tabindex="0">' +
        '<td class="cell-id">' + esc(b.branch_id) + "</td>" +
        '<td class="cell-sandbox" title="' + esc(sandbox || "no sandbox id yet") + '">' +
        esc(sandbox ? shortSandbox(sandbox) : "—") + "</td>" +
        '<td class="num">' + esc(String(b.depth == null ? "—" : b.depth)) + "</td>" +
        '<td><span class="dot st-' + s + (isLiveBranch(run, s) ? " pulse" : "") + '"></span> ' +
        esc(statusLabel(run, s)) + "</td>" +
        '<td class="num cell-score' + (bestId === b.branch_id ? " is-best" : "") + '">' + esc(fmtScore(bestScoreOf(b))) + "</td>" +
        '<td class="num">' + esc(fmtUsd(b.spend_usd)) + "</td>" +
        '<td class="num">' + esc(fmtTokens(b.tokens)) + "</td>" +
        '<td class="cell-wide cell-activity' + (act.level === "warn" || act.level === "error" ? " is-" + act.level : "") +
        '" title="' + esc(act.text) + '">' + esc(truncate(act.text, 52)) + "</td>" +
        "</tr>";
    });

    table.querySelector("tbody").innerHTML = rows.join("") ||
      '<tr><td colspan="8">no sandboxes yet</td></tr>';
  }

  /* ───────────────────────────── activity feed ─────────────────────────────
     The whole run, newest first. The branch id is a chip rather than a label
     because the first thing anyone wants after reading a line is to look at
     the sandbox that wrote it.
     ------------------------------------------------------------------------ */

  function renderFeed(list, run) {
    var events = ((run && run.events) || []).slice(-200).reverse();
    var scrolled = list.scrollTop;

    list.innerHTML = events.map(function (e, index) {
      var level = e.level === "warn" || e.level === "error" ? e.level : "info";
      var cls = ["feed-item", "feed-" + level];
      if (index === 0) { cls.push("is-newest"); }
      if (e.branch_id && e.branch_id === state.selected) { cls.push("is-selected"); }
      return '<li class="' + cls.join(" ") + '">' +
        '<span class="feed-ts mono">' + esc(fmtClock(e.ts)) + "</span>" +
        (e.branch_id
          ? '<button class="chip-branch mono" type="button" data-branch="' + esc(e.branch_id) +
            '" title="Select this sandbox in the tree">' + esc(e.branch_id) + "</button>"
          : '<span class="chip-master mono">master</span>') +
        '<span class="feed-msg">' + esc(e.message) + "</span>" +
        "</li>";
    }).join("") || '<li class="feed-item feed-info"><span class="feed-msg">no events yet</span></li>';

    list.scrollTop = scrolled;
  }

  /* ───────────────────────────── training runs ─────────────────────────────
     The landing screen: one row per run you submitted. `GET /api/runs` has no
     branch count in it, so the tree size is filled in from the run document —
     but at most ONE run document per refresh cycle, picked live-runs-first.
     A list of twenty runs therefore costs two requests per cycle, not
     twenty-one.
     ------------------------------------------------------------------------ */

  var HYDRATE_STALE_MS = POLL_MS * 2;

  function runsSortValue(row, key) {
    var meta = state.runMeta[row.run_id];
    if (key === "best_score") { return row.best_score == null ? -Infinity : row.best_score; }
    if (key === "spend_usd") { return row.spend_usd || 0; }
    if (key === "created_at") { var ms = tsToMs(row.created_at); return isNaN(ms) ? 0 : ms; }
    if (key === "state") { return STATUS_ORDER[statusClass(row.state)]; }
    if (key === "branches") { return meta && meta.branches ? meta.branches.length : -1; }
    return String(row[key] || "");
  }

  function pickHydrate(rows) {
    var now = Date.now();
    var stale = null;
    var missing = null;
    rows.forEach(function (row) {
      var meta = state.runMeta[row.run_id];
      if (!meta) {
        if (!missing) { missing = row.run_id; }
        return;
      }
      if (meta.error) { return; }
      if (LIVE_STATES[row.state] && now - meta.fetched > HYDRATE_STALE_MS && !stale) {
        stale = row.run_id;
      }
    });
    return stale || missing;
  }

  function miniDots(meta, runState) {
    if (!meta || !meta.branches) { return '<span class="dim">—</span>'; }
    var fakeRun = { state: runState };
    var branches = meta.branches.slice().sort(function (a, b) {
      return STATUS_ORDER[statusClass(a.status)] - STATUS_ORDER[statusClass(b.status)];
    });
    var shown = branches.slice(0, 10);
    var dots = shown.map(function (b) {
      var s = statusClass(b.status);
      return '<span class="dot st-' + s + (isLiveBranch(fakeRun, s) ? " pulse" : "") + '" title="' +
        esc(b.branch_id + " · " + statusLabel(fakeRun, s)) + '"></span>';
    }).join("");
    return '<span class="mini-dots">' + dots +
      (branches.length > shown.length ? '<span class="dim mono">+' + (branches.length - shown.length) + "</span>" : "") +
      '</span><span class="mono num mini-count">' + branches.length + "</span>";
  }

  function renderRuns() {
    var frag = el("tpl-runs").content.cloneNode(true);
    var root = frag.firstElementChild;
    var f = fieldGetter(root);
    mount(frag);

    var generation = state.generation;
    var table = f("runs-table");
    var rows = [];

    function paint() {
      var sorted = rows.slice();
      var key = state.runsSort.key;
      var dir = state.runsSort.dir === "asc" ? 1 : -1;
      sorted.sort(function (a, b) {
        var va = runsSortValue(a, key);
        var vb = runsSortValue(b, key);
        if (va < vb) { return -1 * dir; }
        if (va > vb) { return 1 * dir; }
        return String(a.run_id).localeCompare(String(b.run_id));
      });

      markSort(table, state.runsSort);

      var live = 0;
      var spend = 0;
      var best = null;
      var bestRun = null;
      rows.forEach(function (row) {
        if (LIVE_STATES[row.state]) { live += 1; }
        spend += Number(row.spend_usd || 0);
        if (row.best_score != null && (best == null || row.best_score > best)) {
          best = row.best_score;
          bestRun = row.run_id;
        }
      });
      f("total").textContent = String(rows.length);
      f("live").textContent = String(live);
      f("spend").textContent = fmtUsd(spend);
      f("best").textContent = fmtScore(best);
      f("best").title = bestRun ? "best objective score so far, in " + bestRun : "no run has scored yet";
      f("poll-note").textContent = live
        ? "refreshing every " + (POLL_MS / 1000) + "s · " + plural(live, "run") + " live"
        : "all runs are finished · refresh paused";

      withFocusPreserved(table, function () {
        table.querySelector("tbody").innerHTML = sorted.map(function (row) {
          var s = statusClass(row.state);
          var meta = state.runMeta[row.run_id];
          var cls = ["is-clickable"];
          if (LIVE_STATES[s]) { cls.push("is-live"); }
          if (row.run_id === state.runId) { cls.push("is-current"); }
          var href = "#/run/" + encodeURIComponent(row.run_id);
          /* Environment and objective are one fact about the run, not two
             columns' worth: one cell, the objective dimmed behind it. */
          return '<tr class="' + cls.join(" ") + '" data-run="' + esc(row.run_id) + '" tabindex="0">' +
            '<td class="cell-id"><a href="' + esc(href) + '">' + esc(row.run_id) + "</a></td>" +
            "<td>" + esc(row.env_name || "—") +
            '<span class="cell-aside">' + esc(row.objective_name || "no objective") + "</span></td>" +
            '<td><span class="dot st-' + s + (LIVE_STATES[s] ? " pulse" : "") + '"></span> ' +
            esc(s === "pending" ? "starting" : s) + "</td>" +
            '<td class="num cell-score' + (row.best_score != null ? " has-score" : "") + '">' +
            esc(fmtScore(row.best_score)) + "</td>" +
            '<td class="cell-sandboxes">' + miniDots(meta, row.state) + "</td>" +
            '<td class="num">' + esc(fmtUsd(row.spend_usd)) + "</td>" +
            '<td class="num" data-age="' + esc(String(tsToMs(row.created_at))) + '" title="' +
            esc(fmtStamp(row.created_at)) + '">' + esc(ageText(row.created_at)) + "</td>" +
            "</tr>";
        }).join("") || '<tr><td colspan="7">No runs yet. <a href="#/new">Submit a bundle</a> and this table fills in.</td></tr>';
      });
    }

    function openRun(runId) {
      window.location.hash = "#/run/" + encodeURIComponent(runId);
    }

    table.addEventListener("click", function (ev) {
      if (ev.target.closest(".th-sort")) { return; }
      if (ev.target.closest("a")) { return; }
      var row = ev.target.closest("tr[data-run]");
      if (row) { openRun(row.getAttribute("data-run")); }
    });
    table.addEventListener("keydown", function (ev) {
      var row = ev.target.closest("tr[data-run]");
      if (!row || (ev.key !== "Enter" && ev.key !== " ")) { return; }
      ev.preventDefault();
      openRun(row.getAttribute("data-run"));
    });

    var sorters = table.querySelectorAll(".th-sort");
    for (var i = 0; i < sorters.length; i += 1) {
      sorters[i].addEventListener("click", function (ev) {
        var key = ev.currentTarget.getAttribute("data-sort");
        if (state.runsSort.key === key) {
          state.runsSort.dir = state.runsSort.dir === "asc" ? "desc" : "asc";
        } else {
          state.runsSort.key = key;
          state.runsSort.dir = (key === "best_score" || key === "spend_usd" ||
            key === "created_at" || key === "branches") ? "desc" : "asc";
        }
        f("sort-note").textContent = "sorted by " + key.replace(/_/g, " ") + " " + state.runsSort.dir;
        paint();
      });
    }

    /* One cycle: the list, then at most one run document. Never more. */
    function cycle() {
      api.listRuns().then(function (res) {
        if (state.generation !== generation) { return; }
        rows = (res && res.runs) || [];
        paint();

        var target = pickHydrate(rows);
        if (!target) {
          if (rows.some(function (r) { return LIVE_STATES[r.state]; })) { schedulePoll(cycle); }
          return;
        }
        api.getRun(target).then(function (run) {
          if (state.generation !== generation) { return; }
          state.runMeta[target] = {
            fetched: Date.now(),
            branches: run.branches || [],
            best: run.best || null,
            state: run.state
          };
          paint();
        }).catch(function () {
          if (state.generation !== generation) { return; }
          state.runMeta[target] = { fetched: Date.now(), error: true, branches: null };
        }).then(function () {
          if (state.generation !== generation) { return; }
          var more = rows.some(function (r) { return LIVE_STATES[r.state]; }) || pickHydrate(rows);
          if (more) { schedulePoll(cycle); }
        });
      }).catch(function (err) {
        if (state.generation !== generation) { return; }
        mountMessage(
          "The run list could not be loaded",
          (err && err.message ? err.message : "unknown error") +
            ". Set window.AUTOBRANCH_MOCK = true to review the UI against the built-in fixture.",
          "error",
          { href: "#/new", text: "→ New run" }
        );
      });
    }

    startTicker(function () {
      root.querySelectorAll("td[data-age]").forEach(function (cell) {
        var ms = Number(cell.getAttribute("data-age"));
        if (!isNaN(ms)) { cell.textContent = fmtAgo((Date.now() - ms) / 1000); }
      });
    });

    cycle();
  }

  /* ───────────────────────────── new run view ───────────────────────────── */

  function setFieldProblem(root, route, message) {
    var node = root.querySelector('.field-problem[data-problem="' + route + '"]');
    if (!node) { return; }
    if (!message) {
      node.hidden = true;
      node.textContent = "";
      return;
    }
    node.textContent = message;
    node.hidden = false;
  }

  function renderNew() {
    var frag = el("tpl-new").content.cloneNode(true);
    var root = frag.firstElementChild;
    var f = fieldGetter(root);
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
    var submit = root.querySelector("#submit-run");

    var source = "zip";
    var picked = null;
    var parsed = null;

    function note(text) { f("submit-note").textContent = text; }

    function resetPreflight() {
      parsed = null;
      renderPreflight(root, null);
      showProblems(root, [], {});
      note("The bundle is validated before a single container is created.");
    }

    /* Read `autobranch.yaml` out of the archive the user just picked and show
       what it says. Any failure — not a zip, zip64, no manifest, a YAML
       feature this reader does not do — becomes a note, never a block. */
    function preflight(file) {
      note("Reading " + file.name + "…");
      readManifestFromZip(file).then(function (found) {
        if (picked !== file) { return; }
        var raw;
        try {
          raw = parseYaml(found.text);
        } catch (err) {
          parsed = {
            error: MANIFEST_NAME + " could not be read here — " +
              (err && err.message ? err.message : "unsupported YAML")
          };
          renderPreflight(root, parsed);
          note("Manifest not previewed. Submit and the server will answer with the full list.");
          return;
        }
        var preview = previewManifest(raw);
        parsed = {
          spec: preview.spec,
          problems: preview.problems,
          path: found.path,
          entryCount: found.entries.length
        };
        renderPreflight(root, parsed);
        note(preview.problems.length
          ? plural(preview.problems.length, "problem") + " found in the manifest. The server checks again on submit."
          : "Manifest reads clean. The server checks the environment itself on submit.");
      }).catch(function (err) {
        if (picked !== file) { return; }
        parsed = { error: err && err.message ? err.message : "the archive could not be read" };
        renderPreflight(root, parsed);
        note("Manifest not previewed. Submit and the server will answer with the full list.");
      });
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
      setFieldProblem(root, "bundle", "");
    }

    function clearFile() {
      picked = null;
      fileInput.value = "";
      fileChip.hidden = true;
      fileName.textContent = "—";
      fileBytes.textContent = "";
      resetPreflight();
    }

    function setFile(file) {
      if (!file) { return; }
      if (!/\.zip$/i.test(file.name)) {
        setFieldProblem(root, "bundle", "Bundle must be a .zip — got " + file.name);
        return;
      }
      picked = file;
      fileName.textContent = file.name;
      fileBytes.textContent = fmtBytes(file.size);
      fileChip.hidden = false;
      if (source !== "zip") { setSource("zip"); }
      setFieldProblem(root, "bundle", "");
      preflight(file);
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

    f("pf-example").addEventListener("click", function () {
      var preview = previewManifest(parseYaml(EXAMPLE_MANIFEST));
      parsed = {
        spec: preview.spec,
        problems: preview.problems,
        example: true
      };
      renderPreflight(root, parsed);
      note("That is the reference bundle, shown as an example. Nothing has been submitted.");
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
      clearProblems(root);

      var constraints = root.querySelector("#constraints").value.trim();
      var usd = Number(root.querySelector("#budget-usd").value);
      var hours = Number(root.querySelector("#budget-hours").value);
      var maxBranches = Number(root.querySelector("#budget-branches").value);

      if (source === "git" && !urlInput.value.trim()) {
        setFieldProblem(root, "bundle", "Give a git URL, or switch to a .zip upload.");
        urlInput.focus();
        return;
      }
      if (source === "zip" && !picked) {
        setFieldProblem(root, "bundle", "Drop a .zip bundle, or switch to a git URL.");
        dropzone.focus();
        return;
      }
      if (!constraints) {
        setFieldProblem(root, "constraints",
          "Write at least one constraint — it is what bounds the search.");
        root.querySelector("#constraints").focus();
        return;
      }
      if (!(usd > 0) || !(hours > 0) || !(maxBranches >= 1)) {
        setFieldProblem(root, "budget", "Budget caps must all be positive.");
        return;
      }

      submit.disabled = true;
      submit.textContent = "Submitting…";
      note("Validating the bundle server-side: manifest, then the environment itself.");

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
          var problems = splitProblems(err && err.message ? err.message : "submission failed");
          showProblems(root, problems, { source: "server" });
          note("Nothing was launched. Fix the bundle and submit again.");
          var box = f("problems");
          if (box && typeof box.scrollIntoView === "function") {
            /* No `behavior: "smooth"`: the default is instant, which is what
               a reduced-motion preference would ask for anyway. */
            box.scrollIntoView({ block: "center" });
          }
        });
    });

    resetPreflight();
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
    var clock = { elapsed: 0, at: Date.now(), live: false };

    /* ── the one lower region ──
       Exactly one pane is visible at a time. That is the organising idea of
       this page: identity, the tree, and one thing underneath — the selected
       sandbox, the whole table, or the run's activity — rather than three
       permanent panels competing for the same glance. */
    var TABS = ["sandbox", "all", "activity"];
    var tabButtons = {};
    var tabPanes = {
      sandbox: root.querySelector("#pane-sandbox"),
      all: root.querySelector("#pane-all"),
      activity: root.querySelector("#pane-activity")
    };
    var activeTab = "sandbox";

    function paintRegionNote() {
      var note = f("region-note");
      if (!note) { return; }
      var run = state.run;
      if (activeTab === "sandbox") {
        note.textContent = state.selected ? "branch " + state.selected : "nothing selected";
      } else if (activeTab === "all") {
        note.textContent = pluralise(((run && run.branches) || []).length, "sandbox", "sandboxes") +
          " · sorted by " + state.sort.key.replace(/_/g, " ") + " " + state.sort.dir;
      } else {
        note.textContent = plural(((run && run.events) || []).length, "event") +
          (run && LIVE_STATES[run.state] ? " · live, newest first" : " · final, newest first");
      }
    }

    function showTab(name, moveFocus) {
      if (TABS.indexOf(name) < 0) { return; }
      activeTab = name;
      TABS.forEach(function (key) {
        var btn = tabButtons[key];
        var pane = tabPanes[key];
        var on = key === name;
        if (btn) {
          btn.setAttribute("aria-selected", String(on));
          btn.tabIndex = on ? 0 : -1;
          if (on && moveFocus) { btn.focus(); }
        }
        if (pane) { pane.hidden = !on; }
      });
      paintRegionNote();
    }

    TABS.forEach(function (key) {
      var btn = root.querySelector('.seg[data-tab="' + key + '"]');
      tabButtons[key] = btn;
      if (!btn) { return; }
      btn.addEventListener("click", function () { showTab(key, false); });
      btn.addEventListener("keydown", function (ev) {
        var step = ev.key === "ArrowRight" ? 1 : (ev.key === "ArrowLeft" ? -1 : 0);
        if (!step) { return; }
        ev.preventDefault();
        var at = TABS.indexOf(key);
        showTab(TABS[(at + step + TABS.length) % TABS.length], true);
      });
    });

    function paintTree() {
      withFocusPreserved(svg, function () { renderTree(svg, state.run); });
    }

    /* The tree footer says what the canvas is — and, when the payload forced
       the layout to infer a link, says that too, so a dashed edge is never
       left unexplained. */
    function paintTreeFoot() {
      var node = f("tree-foot");
      if (!node) { return; }
      var audit = svg.__layout && svg.__layout.audit;
      var base = "Every node is one Modal sandbox, attached to the sandbox that spawned it — " +
        "failed and abandoned lines included.";
      if (!audit || !audit.inferred) {
        node.textContent = base +
          " Select one for its reward config, its score history and its own timeline.";
        return;
      }
      node.textContent = base + " " + plural(audit.inferred, "link") +
        " here had to be inferred from the branch ids and are drawn dashed" +
        (audit.placeholders
          ? ", including " + pluralise(audit.placeholders, "sandbox", "sandboxes") +
            " with no published branch record"
          : "") + ".";
    }

    function paintTable() {
      withFocusPreserved(table, function () { renderTable(table, state.run); });
    }

    /* Picking a sandbox anywhere — tree, table, activity chip — means "show me
       this one", so it always brings the Sandbox pane forward. */
    function select(branchId, scroll) {
      state.selected = branchId;
      showTab("sandbox", false);
      if (!state.run) { return; }
      paintTree();
      paintTable();
      renderDetail(detail, state.run);
      renderFeed(log, state.run);
      paintRegionNote();
      if (scroll) { scrollNodeIntoView(treeScroll, branchId); }
    }

    svg.addEventListener("click", function (ev) {
      var node = ev.target.closest(".node");
      if (node && node.hasAttribute("data-branch")) {
        select(node.getAttribute("data-branch"), false);
      }
    });
    svg.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" && ev.key !== " ") { return; }
      var node = ev.target.closest(".node");
      if (!node) { return; }
      ev.preventDefault();
      select(node.getAttribute("data-branch"), false);
    });

    table.addEventListener("click", function (ev) {
      if (ev.target.closest(".th-sort")) { return; }
      var row = ev.target.closest("tr[data-branch]");
      if (row) { select(row.getAttribute("data-branch"), true); }
    });
    table.addEventListener("keydown", function (ev) {
      var row = ev.target.closest("tr[data-branch]");
      if (!row || (ev.key !== "Enter" && ev.key !== " ")) { return; }
      ev.preventDefault();
      select(row.getAttribute("data-branch"), true);
    });

    /* A branch chip in the feed selects that sandbox — the feed is a way into
       the tree, not just a wall of text. */
    log.addEventListener("click", function (ev) {
      var chip = ev.target.closest(".chip-branch");
      if (!chip) { return; }
      select(chip.getAttribute("data-branch"), true);
    });

    var sorters = table.querySelectorAll(".th-sort");
    for (var i = 0; i < sorters.length; i += 1) {
      sorters[i].addEventListener("click", function (ev) {
        var key = ev.currentTarget.getAttribute("data-sort");
        if (state.sort.key === key) {
          state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
        } else {
          state.sort.key = key;
          state.sort.dir = (key === "branch_id" || key === "depth" ||
            key === "status" || key === "sandbox") ? "asc" : "desc";
        }
        if (state.run) {
          paintTable();
          paintRegionNote();
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
      state.runMeta[state.runId] = {
        fetched: Date.now(),
        branches: run.branches || [],
        best: run.best || null,
        state: run.state
      };

      var live = Boolean(LIVE_STATES[run.state]);
      clock = { elapsed: Number(run.elapsed_s || 0), at: Date.now(), live: live };

      f("run_id").textContent = run.run_id || runId;
      f("env_name").textContent = run.env_name || "unknown environment";
      f("objective").textContent = (run.objective && run.objective.name) || "objective";
      f("direction").textContent = (run.objective && run.objective.direction) || "maximize";
      f("started").textContent = "started " + fmtStamp(run.created_at);
      f("state-badge").innerHTML = statusBadge(run.state);
      f("elapsed").textContent = fmtDuration(run.elapsed_s);
      f("branch-count").textContent = String((run.branches || []).length);

      /* The point of the product, so it is the biggest thing on the page. */
      var objectiveName = (run.objective && run.objective.name) || "objective";
      var scoreNode = f("best-score");
      scoreNode.textContent = run.best ? fmtScore(run.best.score) : "—";
      scoreNode.classList.toggle("is-unset", !run.best);
      f("score-label").textContent = "Best " + objectiveName;
      f("best-sub").textContent = run.best
        ? "branch " + run.best.branch_id + " · " +
          ((run.objective && run.objective.direction) || "maximize") +
          " · scored by the evaluator, never by a reward"
        : "no sandbox has cleared the evaluator yet";

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
        " · max " + (budget.max_branches != null ? budget.max_branches : "—") + " concurrent sandboxes" +
        (budget.max_depth != null ? " · max depth " + budget.max_depth : "");

      f("stop").disabled = !live;
      f("stop").textContent = live ? "Stop run" : "Run finished";
      f("artifact-link").href = "#/artifact/" + encodeURIComponent(state.runId);

      if (!state.selected) {
        /* Open on the thing worth looking at: the best branch if there is one
           *and it is in the tree* — a fork can appear in the event log a poll
           or two before its branch record is published — otherwise whichever
           sandbox is currently doing something. */
        var fallback = null;
        (run.branches || []).forEach(function (b) {
          if (!fallback && LIVE_STATES[statusClass(b.status)]) { fallback = b.branch_id; }
        });
        var bestId = run.best && run.best.branch_id;
        state.selected = (bestId && branchById(run, bestId) ? bestId : null) || fallback ||
          ((run.branches || [])[0] || {}).branch_id || null;
      }

      renderStatusChips(f("status-chips"), run);
      paintTree();
      paintTable();
      renderDetail(detail, run);
      renderFeed(log, run);
      paintRegionNote();
      paintTreeFoot();
      updateNav();

      /* First paint only: bring the selected branch into view rather than
         opening hard against a corner of a wide canvas. */
      if (!painted) {
        painted = true;
        if (state.selected) {
          scrollNodeIntoView(treeScroll, state.selected);
        } else if (svg.__layout) {
          treeScroll.scrollLeft = Math.max(0, (svg.__layout.width - treeScroll.clientWidth) / 2);
        }
      }
    }

    /* Polling stops dead when the run is terminal — a finished run has
       nothing left to say, and a page left open on one should cost nothing. */
    function poll() {
      api.getRun(runId).then(function (run) {
        if (state.generation !== generation) { return; }
        paint(run);
        if (LIVE_STATES[run.state]) { schedulePoll(poll); }
      }).catch(function (err) {
        if (state.generation !== generation) { return; }
        mountMessage(
          "Run " + runId + " could not be loaded",
          (err && err.message ? err.message : "unknown error") +
            ". Open the run list, or set window.AUTOBRANCH_MOCK = true to review the UI against the built-in fixture.",
          "error",
          { href: "#/runs", text: "← Training runs" }
        );
      });
    }

    startTicker(function () {
      if (!clock.live) { return; }
      var node = f("elapsed");
      if (node) {
        node.textContent = fmtDuration(clock.elapsed + (Date.now() - clock.at) / 1000);
      }
    });

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
      if (run) { state.run = run; }

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
              (delta == null ? "" : ' <span class="lineage-delta">' + (delta >= 0 ? "+" : "") + fmtScore(delta) + "</span>") +
              "</span>" +
              "</li>";
          }).join("");
        }
      }

      var files = art.files || [];
      f("files").innerHTML = files.map(function (file) {
        var note = file.desc || FILE_NOTES[file.name] || "artifact file";
        var href = file.url ||
          ("/api/runs/" + encodeURIComponent(runId) + "/artifact/" + encodeURIComponent(file.name));
        return "<li>" +
          '<span class="file-main">' +
          '<span class="file-title">' + esc(file.name) + "</span>" +
          '<span class="file-desc">' + esc(note) + "</span>" +
          "</span>" +
          '<span class="file-size">' + esc(fmtBytes(file.bytes)) + "</span>" +
          '<a class="btn btn-sm" href="' + esc(href) + '" download="' + esc(file.name) + '">Download</a>' +
          "</li>";
      }).join("") || '<li><span class="file-desc">no files published</span></li>';

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
        "error",
        { href: "#/runs", text: "← Training runs" }
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
    if (!parts.length) { return { name: "runs" }; }
    if (parts[0] === "new") { return { name: "new" }; }
    if (parts[0] === "run" && parts[1]) { return { name: "run", id: decodeURIComponent(parts[1]) }; }
    if (parts[0] === "artifact" && parts[1]) { return { name: "artifact", id: decodeURIComponent(parts[1]) }; }
    return { name: "runs" };
  }

  function route() {
    var hash = window.location.hash;
    if (state.route && hash === state.lastHash) { return; }
    state.lastHash = hash;

    /* One generation change kills every in-flight response, the scheduled
       poll and the clock, so no two views can ever be fetching at once. */
    state.generation += 1;
    if (state.pollTimer) {
      window.clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
    stopTicker();

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

    if (next.name === "new") {
      updateNav();
      renderNew();
      return;
    }

    updateNav();
    renderRuns();
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
      window.location.replace("#/runs");
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









