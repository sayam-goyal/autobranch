# SF3 run — on-disk runbook

Absolute epochs only, never a remembered time. Written because monitors die and
a restarted session must be able to continue without re-deriving anything.

## The run

| what | value |
|---|---|
| run id | **`run-a6ffd62451`** |
| launched | 22:07:13 PST, epoch **1794722833** |
| training deadline (4.75 h) | epoch **1794739933** ≈ 02:52 PST |
| 8-hour envelope ends | epoch **1794748800** ≈ 05:40 PST |
| bundle | `examples/sf3` (sf3-ryu v1) |
| objective | `win_rate_vs_cpu8`, maximize |
| base model | `Qwen/Qwen3-VL-8B-Instruct`, **full-parameter** (`grpo_full`) |
| spend cap | $220 |
| max branches | 8 |
| web UI | https://sf-hackathon--autobranch-web-dev.us-east.modal.run |

**Two superseded runs, both stopped, ignore in any analysis:**

- `run-bf5c9b0ae9` (21:37) — launched without `--detach`, so the ephemeral app
  stopped when the local entrypoint returned and killed the root spawn. Sits at
  `pending` with zero branches forever: not a crash, a lifetime.
- `run-12845b4e57` (21:40) — ran the pre-fix runtime. Replaced because its
  evaluator would have loaded an 8.77B policy on **CPU** (never returning), and
  because its env reused emulator lanes across collections, which the emulator
  cannot do and does not complain about. Branch sandbox and nine orphaned lanes
  terminated by hand.
- `run-4342abbaad` (21:49) — **completed one healthy iteration**, and that
  iteration is why this run is sized the way it is. Its numbers are kept in
  `first_iteration.jsonl`: 6 rounds won / 0 lost across 8 lanes, damage dealt
  2.4x damage taken, zero degenerate groups, ratio drift 0.0006 — and
  `wall_seconds` 772 against a 264 s estimate. A decision costs ~1.0 s, not
  0.25 s, because the VL forward adds ~0.75 s on top of the emulator. Replaced
  to pick up evaluation concurrency 24 (one batch instead of three: ~27 min per
  call instead of ~80), 400 decisions per collection, and an export every 5
  iterations instead of every one.

Both are the reason the probes exist: every one of those failures is silent.

## Current state — handoff at 14:20 PST, 15 Nov

The run is **live on Modal** and continues without this session. Deadline
**16:43 PST** (epoch 1794785000-ish; `created_at + budget.hours`).

```
run-25ed75aa0a   4 branches, deadline 16:43
b0        d0  running   0.875 → 0.9167 → 0.9167   (52.8k / 57.6k / 62.4k steps)
b0.1      d1  running   forked: + super_gain 0.05
b0.1.1    d2  running   forked: invalid_move 0.125 → 0
b0.1.1.1  d3  running   at max_depth
```

**Best measured: 0.9167** (22/24, deterministic, level-8 CPU), Wilson 95%
[0.742, 0.977]. Against the predecessor's numbers: clears `r2_hinh` 0.550
[0.452, 0.644] decisively, overlaps `r3_potential` 0.790 [0.700, 0.858].

Three consecutive rises (0.750 → 0.875 → 0.9167 → 0.9167) at n=24 is also what
drift looks like. The last two are identical, which is the first hint of a
plateau. **Nothing here is separated at this n** — the honest next step is one
larger evaluation of the winner, not another training cycle.

### When the deadline passes

```bash
# 1. what the tree ended up as, with intervals and the null/plateau read
python examples/sf3/analyze.py --run-id run-25ed75aa0a --out report.md

# 2. a tighter number on the winner: n=72 is 3 batches, ~75 min, ~±10 points
modal run eval_sf3.py --run-id run-25ed75aa0a --branch-id b0 --matches 72

# 3. the artifact (weights live on the Volume; WEIGHTS.md says where)
modal run app.py::cli_finalize --run-id run-25ed75aa0a

# 4. if a branch died but the clock has not: resumes from its own export
modal run app.py::cli_restart --run-id run-25ed75aa0a
```

### Two numbers not to trust from a laptop

* **Spend.** The UI reads `$60+`; true cost is nearer `$25`. Branches in this
  run store the ledger's *run-wide* total, and `registry._run_spend` sums
  branch records when it cannot reach the ledger — which it cannot from
  off-volume. Fixed in the code (`by_branch()`), but this run carries the old
  build. The **cap was never affected**: the budget guard reads the ledger
  directly inside the sandbox, where the figure is correct.
* **Branch reward configs.** The registry shows each fork carrying its
  parent's reward; the child writes its own once it starts. Read the events
  (`forked b0.1: ... super_gain=0.05`) for what a branch actually ran.

## Commands

```bash
# tree, scores, reward configs, stop reasons
python examples/sf3/watch.py --run-id run-a6ffd62451

# ...plus the per-branch training curves off the Volume
python examples/sf3/watch.py --run-id run-a6ffd62451 --metrics

# loop every two minutes
python examples/sf3/watch.py --run-id run-a6ffd62451 --loop 120 --metrics

# the framework's own status
modal run app.py::cli_status --run-id run-a6ffd62451

# build the artifact (weights + lineage + losers), after training ends
modal run app.py::cli_finalize --run-id run-a6ffd62451
```

## Where the data is

Everything needed for post-hoc analysis is on the `autobranch-data` Volume,
**not** in a sandbox:

```
runs/run-a6ffd62451/<branch>/metrics.jsonl    one line per training iteration
runs/run-a6ffd62451/<branch>/hf/              the full HF checkpoint (~16 GiB)
```

`/workspace` is sandbox-local, so anything left there dies with its branch.
The trainer writes the export and mirrors the metrics to the Volume for exactly
that reason — and keeping 16 GiB out of `/workspace` also matters because every
fork snapshots that directory.

Registry state (tree, reward configs, scores, events, spend) lives in
`modal.Dict`s and is what `watch.py` reads first.

## What to watch, and what each reading means

| reading | healthy | what it means if not |
|---|---|---|
| `first_epoch_ratio_drift` | < 0.05 | collection and the update are not scoring the same thing. This is the padding-side bug; the gradient is meaningless while loss and grad_norm look fine. |
| `zero_variance_group_fraction` | well below 1.0 | a group whose members all score the same has no advantage for any member. Near 1.0 is **not learning**, not learning slowly. |
| `entropy` | > ~0.3 nats | mode collapse. The predecessor policy won 0/100 at its mode while competent while sampling, and that came from the base model. `entropy_coef` is 0.01 to resist it. |
| `invalid_move_rate` | low and falling | the policy keeps picking moves it cannot perform (EX without meter). Real cost, but masked tokens mean nothing is unparseable. |
| `device` in a score's detail | `cuda` | the policy was scored on CPU. For an 8.77B VL model that is not slow, it is "never returns". |
| objective score | rising | a rising *training return* is not evidence. The predecessor saw a policy peak at 0.599 then collapse to 0.353 while its return rose twentyfold. |

## Known failure modes and the response

1. **A branch sandbox dies** — the orchestrator abandons it and the tree keeps
   going. Do not restart it by hand; a fork from its parent is the cheaper
   recovery and the search policy will do it.
2. **An emulator lane fails to boot** — `Sf3Env._ensure_lanes` keeps the lanes
   that did and continues with fewer. A collection with zero lanes raises, and
   that branch abandons.
3. **Rate limits on app create** — the env looks up *one* app
   (`autobranch-sf3-emulator`) and creates sandboxes inside it, so there is no
   app-create storm. The predecessor project hit this by creating an app per
   worker.
4. **Spend approaching the cap** — the cap is enforced centrally, not advised.
   Branches abandon when their share is exhausted whether or not the agent
   agrees. **Two caveats, measured on this run.**

   *The UI under-reports during a long call.* `record_compute` fires when a
   train call **returns**, so GPU time is billed in lumps. During a ~50 minute
   train call the run shows stale spend -- it read `$0.01` 22 minutes into this
   one, which was the agent's LLM tokens alone and nothing else. That is the
   same symptom reported on the Rocket League run, from a different cause, so
   it is worth not mistaking one for the other.

   *The cap is only checked between agent turns.* `SearchPolicy` evaluates the
   budget while the agent is deciding, so a single long call can overrun it.
   Harmless here and not harmless in general -- see the rates below.

   | | |
   |---|---|
   | H100 + 8 CPU + 64 GiB | **$4.84/hr** |
   | one ~750 s iteration | $1.01 |
   | 8 emulator lanes | $1.01/hr |
   | 24 lanes (evaluation) | $3.03/hr |
   | **projected over 4.25 h** | **≈$25** against the $220 cap |

## The comparison this run exists to make

Hinh's hand-tuned reward is expressible in the DSL and was given to the agent
as context, not as a target:

```json
{"weights": {"round_won": 1.0, "round_lost": -1.0,
             "damage_dealt": 0.1, "damage_taken": -0.1,
             "invalid_move": -0.125}}
```

The predecessor project measured, on this same character and difficulty, 55.0%
for that configuration against 79.0% for a potential-shaped one (n=100 each).
So the question is not whether the framework can train — it is whether the
search finds something better than the hand-tuned reward, and the honest answer
has to come from the objective at a fixed difficulty, not from training return.
