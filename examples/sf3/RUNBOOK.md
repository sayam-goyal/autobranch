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
   agrees.

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
