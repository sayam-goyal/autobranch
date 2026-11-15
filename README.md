# autobranch

Hand it an environment and a sentence of intent. It hands back trained weights
and everything needed to run them.

In between, a tree of agents in Modal Sandboxes searches the reward-function
space — writing rewards, training, measuring, and branching like git worktrees
when a line looks promising. Lines that stop paying off get pruned.

```bash
./serve_local.sh          # http://localhost:8080
```

```
┌─ b0      succeeded  success_rate 0.996   reached_goal=10 picked_up_key=3 …
├─ b0.1    abandoned  dominated by a sibling by 0.988
└─ b0.1.1  abandoned  no objective improvement over 3 evaluations
```

## What actually happened when it ran

A Kimi agent on Modal took a locked-door gridworld from nothing to a solved
policy, writing its own reward and forking three levels deep:

| step | objective |
|---|---:|
| 81,920 | 0.0000 |
| 188,416 | 0.0000 |
| 294,912 | 0.0000 |
| 499,712 | **0.9961** |

It chose `reached_goal=20, picked_up_key=5, stepped_in_lava=-5,
revisited_tile=-0.05, step_cost=-0.01` — discovering on its own that the key
matters, which is the whole difficulty of that environment. Cost: **$0.30**.

The artifact it produced:

```
weights/latest.pt      684,414 bytes   loads in plain torch, no framework
manifest.json                          score, provenance, policy shape, spend
reward_lineage.json                    root → winner, every mutation + score
losers.json                            every abandoned branch and why
serve.py, README.md                    how to run it
```

The manifest reports **both policy modes** — deterministic 0.996, stochastic
0.836 — which exposed a 0.16 mode gap that a single-mode number would have
hidden.

## Why this rather than tuning a reward by hand

Most RL effort goes into the reward, and most rewards are written by guessing,
running for hours, squinting at a curve, and guessing again. Three things make
that loop expensive, and this framework attacks each one.

**The reward usually decides the outcome, and the right one is not guessable.**
Measured, on real runs:

| run | finding |
|---|---|
| Rocket League | `approach_ball` at 20 peaked at **0.599** then collapsed to **0.353** while its *training return rose twentyfold*; at 3 it reached **0.922** |
| Street Fighter III | a potential-shaped reward beat a hand-tuned one by **24 points** at level 8 (79.0% vs 55.0%), at matched model size and configuration |
| Tetris | the reward axis was a **null** — six configurations across two rewrites, all exactly **0.000** |

The last row is the honest one. **Searching rewards does not always help**, and
when it does not, the useful result is learning so cheaply: six Tetris arms cost
**$4.51** to establish that the whole axis was flat, which redirected the effort
to the thing that did matter (how the policy represents an action — 0.000 to
0.994 on a *byte-identical* reward). A search that reports a null quickly is
worth more than a hunch defended for a week.

**A human tuning a reward is also choosing the scoreboard.** That is the failure
mode nobody catches in themselves: you adjust the reward, the metric you are
watching improves, and you cannot tell whether the policy got better or the
measurement got friendlier. Here the two are *structurally* separated —
environments emit **signals** (facts about a transition), agents compose rewards
from those signals, and the objective is evaluated by code that imports no
reward module at all. `tests/test_objective_independence.py` walks the import
graph and fails if that is ever violated. An agent cannot make itself look good
without getting good, because it has no path to the thing that judges it.

**Exploring in parallel is usually too expensive to do properly.** A branch here
is a Modal Sandbox forked from a filesystem snapshot, so a promising line is
continued rather than restarted — the child inherits weights, optimizer state
and history while writing nowhere the parent can see, and the parent keeps
training. That makes "try six rewards from this checkpoint" cost six sandboxes
for the remaining steps, not six runs from scratch.

### What it does not do

- It does not change the policy architecture. Branches mutate **rewards**; the
  network comes from the manifest. The Tetris finding — representation beat
  every reward — is exactly the kind of thing this search cannot find, and the
  framework's job there was to rule out the axis it *can* search, fast.
- It does not remove the need for a good objective. Everything rests on the
  evaluator being the number you actually care about; a saturated or gameable
  objective produces a confident, worthless winner.
- It is not cheaper than one good guess. It is cheaper than ten bad ones, and it
  tells you which you had.

## The environment contract

An environment is a bundle: a directory, zip, or git URL with a manifest and
some code.

```yaml
# autobranch.yaml
name: gridworld
entrypoint: grid_env:make_env

signals:                      # FACTS about a transition. Never rewards.
  - {name: reached_goal,    doc: "1 on the step the goal is reached with the key"}
  - {name: picked_up_key,   doc: "1 on the step the key is collected"}
  - {name: stepped_in_lava, doc: "1 when the agent enters lava"}
  - {name: distance_to_key, doc: "normalised distance, 0 when held"}

objective:                    # What we optimise. Agents may NEVER modify this.
  name: success_rate
  evaluator: grid_env:evaluate
  budget: {episodes: 256}

resources: {gpu: null, cpu: 2, memory: 8192}
policy:    {kind: discrete, n_actions: 4, obs_dim: 81}
trainer:   ppo                # ppo | grpo_lora | grpo_full | custom:module:fn
```

```python
class AutobranchEnv(Protocol):
    def reset(self, *, seed=None): ...
    def step(self, action) -> StepResult: ...   # carries SIGNALS, not reward
```

**`step` returns signals, not reward.** This is the design's spine. The
trainer composes reward from a config the agent wrote; a separate evaluator —
which imports no reward code, enforced by a test that walks the import graph —
produces the only number that counts. Agents compose rewards and cannot touch
their own measurement.

Rewards are **data, not code**:

```json
{"weights": {"reached_goal": 1.0, "stepped_in_lava": -0.5},
 "shaping": [{"kind": "potential", "signal": "distance_to_key", "gamma": 0.99}],
 "anneal":  {"shaping": {"from": 1.0, "to": 0.0, "by_fraction": 0.8}}}
```

A fixed op vocabulary over declared signals. Model-authored Python never runs
in the trainer, configs hash for de-duplication and diff to show exactly what
a fork changed, and every term is named so `term_totals()` reports what each
part of the reward actually bought. An `expr` escape hatch parses to an AST
and walks an allowlist — `__import__`, attribute access, lambdas and
comprehensions are all rejected.

## Branching

```
parent.snapshot_directory("/workspace")  →  parent.poll() is None   # still running
child.mount_image("/workspace", snap)    →  child inherits everything
child writes | parent writes             →  isolated
```

A branch is a sandbox. Forking snapshots the workspace and starts a child with
it mounted, so the child inherits weights, optimizer state and history while
writing nowhere the parent can see — and the parent keeps training.
`snapshot_filesystem()` is the intuitive choice and the wrong one: it
terminates its sandbox, which would make every fork a hand-off. Rollback is
the same primitive in reverse: unmount, mount an ancestor.

Sandboxes self-identify through `MODAL_SANDBOX_ID`, so a branch forks itself.
No central supervisor brokers it.

## Who may stop a branch

Two authorities, deliberately:

- the **agent** calls `abandon` when it judges its line hopeless;
- the **orchestrator** abandons it regardless on budget, deadline, depth,
  plateau, or domination by a sibling.

Agent-only would let an optimistic model burn the budget on one dead line.
Orchestrator-only would discard the judgement that makes this better than a
grid sweep.

## Constraints, in your own words

The free text goes to the agent verbatim — a model reads *"be aggressive early
then tighten up"* better than any schema. Only the machine-actionable parts
are extracted:

| you write | effect |
|---|---|
| `must posttrain Qwen/Qwen3-4B` | base model downloaded to the Volume, `grpo_lora` trainer |
| `must beat 76%` | target score the agent is told to clear |
| `nothing bigger than an L40S` | GPU ceiling |
| `stay under $50` | hard spend cap, enforced not advised |

## Three trainers, one method

| trainer | what it writes | when |
|---|---|---|
| `ppo` | a dense MLP from scratch | the environment has a small observation vector |
| `grpo_lora` | an adapter over frozen base weights | post-training that has to fit on one GPU |
| `grpo_full` | **every parameter** of a base model | reproducing a full fine-tune |

The RL method is the same in both post-training paths -- critic-free,
group-relative advantages, a clipped surrogate, and an action that is a masked
token rather than parsed text. They differ in what they are allowed to write
to, which is a question about memory and about fidelity to what is being
reproduced, not about algorithm.

`grpo_full` exists because the thing it reproduces was a full fine-tune. An
8.77B model takes a full-parameter update on **one** H100 -- 49.5 GiB of 79.2,
measured, with an 8-bit optimizer doing the work that makes it fit, since fp32
Adam moments alone would be 64 GiB.

It also carries the vision path: `examples/sf3` feeds
`Qwen/Qwen3-VL-8B-Instruct` the emulator's frame plus a short state line, which
costs ~104 prompt tokens. The action menu is in the prompt, because a policy
sampling over 40 masked tokens with no labels cannot know which one is a
Hadouken and would have to learn the mapping by trial -- discarding everything
the base model already knows about the game.

## Post-training a base model

`must posttrain Qwen/Qwen3-4B` switches the trainer to `grpo_lora`: LoRA
adapters over a base model, critic-free, with group-relative advantages.

**An action is a masked token, never parsed text.** One forward pass, logits at
the last position, restricted to one token per action index, then sampled. The
sampled index *is* the action, so an illegal action is unrepresentable and no
"invalid move" penalty has to exist -- a penalty that would otherwise compete
with the reward terms an agent is actually searching over.

**No critic.** For one state, sample `G` completions and use
`(r - mean) / (std + eps)` within the group. The honest cost is a group where
every sample scores identically, so `zero_variance_group_fraction` is in every
metrics line: a run whose groups are mostly degenerate is not learning slowly,
it is not learning.

Weights land on the Volume once, under `models/<escaped-repo-id>/`. The
download goes to a staging directory and is moved into place with `os.replace`
only after a completion marker is written *inside* it, so a half-finished
download can never be mistaken for a usable model and two branches racing on
the same repo cannot interleave into one corrupt tree.

Scoring a LoRA policy goes through the same `objective.load_checkpoint_policy`
as a PPO one, dispatched on the checkpoint's own `arch.kind`. That needs the
inference half of post-training to be reachable from the measurement path,
which is why it lives in `lm_policy.py` and imports no reward code -- the
import-graph test walks `objective`'s closure and would fail on `posttrain`.
A downloaded artifact rebuilds its adapter from the checkpoint record alone,
because the `adapter/` directory stayed behind in a sandbox that no longer
exists.

## Modal services

| service | role |
|---|---|
| **Sandboxes** | one per branch; `MODAL_SANDBOX_ID` lets it fork itself |
| **Sandbox snapshots** | `snapshot_directory` / `mount_image` — fork and rollback |
| **Volumes** | bundles, the published runtime, ledger shards, artifacts, base models |
| **Dicts** | tree state and event shards, one key per writer |
| **Secrets** | the Kimi proxy token; workspace creds so a sandbox can fork |
| **Shared Endpoints** | `moonshotai/Kimi-K3.1-Pro` — the agents' reasoning |
| **Functions + ASGI** | the web app, run launcher, and consolidation |

## Layout

```
app.py              Modal entrypoint — outside the package, deliberately
autobranch/
  spec.py           the environment contract
  reward.py         the reward language
  orchestrator.py   search policy, constraints, stop authority
  branch_main.py    what runs inside a branch sandbox
  forking.py        snapshot, fork, rollback
  trainer.py        PPO, driven by the spec
  posttrain.py      grpo_lora: LoRA post-training, critic-free
  posttrain_full.py grpo_full: full-parameter post-training, critic-free
  lm_policy.py      running a post-trained policy -- imports no reward code
  objective.py      the immutable measurement
  registry.py       run/branch state, one key per writer
  ledger.py         spend, per-writer shards
  consolidate.py    the artifact
  server.py         the API and frontend, one origin
frontend/           vanilla HTML/CSS/JS, no build step
examples/gridworld/ the reference environment (ppo)
examples/wordguess/ the reference environment for post-training
examples/sf3/       Street Fighter III, a vision policy over a real emulator
```

## Running it

```bash
./serve_local.sh                                   # the UI, against live Modal
modal deploy app.py                                # the backend
modal run app.py::cli_validate  --bundle examples/gridworld
modal run app.py::cli_submit    --bundle examples/gridworld --usd 8 --hours 1
modal run app.py::cli_status    --run-id run-xxxx
modal run app.py::cli_finalize  --run-id run-xxxx   # build the artifact
python3 -m pytest tests/ -q                        # 418 tests
```

## Notes from building it

Section 7 of [DESIGN.md](DESIGN.md) records ten operational traps as guards
rather than comments. The ones that cost the most:

- **A Sandbox is not a Function.** `volume.commit()` and `volume.reload()` both
  raise inside one, which made every branch report `$0` spent while its ledger
  shards were written perfectly. Sandbox writes to a mounted Volume persist on
  their own.
- **`/workspace` is sandbox-local.** Only `/data` comes from the Volume, so
  checkpoints written by the trainer died with their sandbox — the framework's
  deliverable was the one thing it did not keep.
- **A sandbox image is built from the submitted manifest**, so it contains none
  of this framework. `add_local_python_source` cannot help: it needs local
  files at build time and these sandboxes are created at runtime. The runtime
  rides the Volume instead.
- **A Modal entrypoint must not live inside a package the image also mounts** —
  it is uploaded twice and the container imports two `modal.App` objects, which
  kills ASGI routing with a bare HTTP 500 and no traceback anywhere.
- **Serialisation that drops fields fails silently.** Five fields added to a
  dataclass and not its `to_dict()` surfaced as an artifact shipping
  `MISSING.txt` — a symptom pointing nowhere near the cause. A field-coverage
  test now makes that impossible.
- **Poll real state; never trust a completion event.** Every silent failure
  here was caught by comparing counters.
