# SF3 run — on-disk runbook

Absolute epochs only, never a remembered time. Written because monitors die and
a restarted session must be able to continue without re-deriving anything.

## The run

| what | value |
|---|---|
| run id | **`run-12845b4e57`** |
| launched | 21:40:28 PST, epoch **1794720028** |
| training deadline (5.5 h) | epoch **1794739828** ≈ 03:10 PST |
| 8-hour envelope ends | epoch **1794748800** ≈ 05:40 PST |
| bundle | `examples/sf3` (sf3-ryu v1) |
| objective | `win_rate_vs_cpu8`, maximize |
| base model | `Qwen/Qwen3-VL-8B-Instruct`, **full-parameter** (`grpo_full`) |
| spend cap | $220 |
| max branches | 8 |
| web UI | https://sf-hackathon--autobranch-web-dev.us-east.modal.run |

**Dead run to ignore: `run-bf5c9b0ae9`.** Launched 21:37 without `--detach`,
so the ephemeral app stopped when the local entrypoint returned and killed the
root spawn. It sits at `pending` with zero branches forever. Not a crash — a
lifetime. `launch.sh` now detaches.

## Commands

```bash
# tree, scores, reward configs, stop reasons
python examples/sf3/watch.py --run-id run-12845b4e57

# ...plus the per-branch training curves off the Volume
python examples/sf3/watch.py --run-id run-12845b4e57 --metrics

# loop every two minutes
python examples/sf3/watch.py --run-id run-12845b4e57 --loop 120 --metrics

# the framework's own status
modal run app.py::cli_status --run-id run-12845b4e57

# build the artifact (weights + lineage + losers), after training ends
modal run app.py::cli_finalize --run-id run-12845b4e57
```

## Where the data is

Everything needed for post-hoc analysis is on the `autobranch-data` Volume,
**not** in a sandbox:

```
runs/run-12845b4e57/<branch>/metrics.jsonl    one line per training iteration
runs/run-12845b4e57/<branch>/hf/              the full HF checkpoint (~16 GiB)
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
