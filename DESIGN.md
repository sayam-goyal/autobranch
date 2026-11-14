# Autobranch — design

A generalized RL framework on Modal. You hand it an **environment** and a
sentence of intent; it hands back **trained weights and everything needed to
run them**. In between, a tree of agents in Modal Sandboxes searches the
reward-function space, branching like git worktrees and pruning what isn't
working.

Everything after the handoff runs on Modal. Nothing runs locally.

---

## 1. What problem this actually solves

Reward design is the bottleneck in applied RL, and it is searched by hand. The
project this framework generalizes produced two findings worth building on:

* Six reward configurations were trained and compared. **All six scored 0.000
  against a fixed opponent** while self-play win rate read a healthy 0.807 over
  6,705 matches. Self-play progress was not progress.
* The axis that mattered — how the policy represented a decision — had been
  settled by assumption in the plan and never tested. One ablation holding the
  reward fixed and swapping only that axis went **0.000 → 0.990**.

Two design consequences, and they are the spine of this framework:

1. **The objective is immutable and agents cannot touch it.** Agents compose
   rewards; a separate evaluator they cannot modify scores them. Reward is a
   means, never the measurement.
2. **Branching must be cheap enough to test assumptions, not just weights.** A
   fork should cost seconds, so "what if the whole premise is wrong" is an
   affordable experiment rather than a late discovery.

---

## 2. The environment contract

An environment is a **bundle**: a directory, zip, or git URL containing a
manifest plus code. The manifest is the whole interface.

```yaml
# autobranch.yaml
name: street-fighter-iii
version: 1

image:                          # how to build the container
  base: nvidia/cuda:12.8.1-devel-ubuntu22.04
  python: "3.11"
  apt: [git, build-essential]
  pip: [torch==2.8.0]
  commands: ["pip install -e ."]
  secrets: [sf3-rom-auth]       # modal secret names
  volumes: {"/roms": sf3-rom}   # modal volume mounts

entrypoint: sf_env:make_env     # module:callable -> AutobranchEnv

resources: {gpu: null, cpu: 2, memory: 8192}

# The ONLY things a reward may be built from. Facts, not rewards.
signals:
  - {name: damage_dealt,  type: float, range: [0, 1],  doc: "fraction of max health dealt this step"}
  - {name: damage_taken,  type: float, range: [0, 1],  doc: "..."}
  - {name: round_win,     type: int,   range: [-1, 1], doc: "+1 won, -1 lost, 0 ongoing"}
  - {name: health_margin, type: float, range: [-1, 1], doc: "own minus opponent health"}

# What we are ACTUALLY optimizing. Agents may never modify this block.
objective:
  name: win_rate_vs_cpu8
  direction: maximize
  evaluator: sf_env:evaluate    # module:callable -> ObjectiveScore
  budget: {matches: 200}

policy: {kind: discrete, n_actions: 40, obs_dim: 80}
trainer: ppo                    # ppo | grpo_lora | custom:module:fn
```

The Python side is deliberately tiny:

```python
class AutobranchEnv(Protocol):
    def reset(self) -> Obs: ...
    def step(self, action) -> StepResult: ...   # carries SIGNALS, never a reward
    @property
    def n_envs(self) -> int: ...
```

**`step` returns signals, not reward.** The trainer composes the reward from a
config the agent wrote. This is what makes reward search possible without
agents reading or editing environment internals, and it is what makes the
objective un-gameable: the evaluator path imports no reward code, enforced by
an AST test in CI.

### Reward configs are data, not code

```json
{
  "weights": {"round_win": 1.0, "damage_dealt": 0.1, "damage_taken": -0.1},
  "shaping": [{"kind": "potential", "signal": "health_margin", "gamma": 0.99, "scale": 0.5}],
  "transforms": [{"op": "clip", "signal": "damage_dealt", "lo": 0.0, "hi": 0.2}],
  "anneal": {"shaping": {"from": 1.0, "to": 0.0, "by_fraction": 0.8}}
}
```

A fixed op vocabulary over declared signals. JSON means configs are
snapshottable, diffable, hashable (for dedup), and mutatable by an LLM without
ever executing model-authored code inside the trainer. An `expr` escape hatch
evaluates restricted arithmetic over signals for cases the vocabulary misses.

---

## 3. Branching: git semantics on Modal

Verified empirically before committing to it:

```
parent.snapshot_directory("/workspace")  ->  parent.poll() is None   # STILL RUNNING
child.mount_image("/workspace", snap)    ->  child sees parent state
child writes 'child-v2' / parent writes 'parent-v3'  ->  ISOLATED
```

So `snapshot_directory` + `mount_image` gives a child the parent's full state —
weights, optimizer, curriculum, history — with copy-on-write isolation, **while
the parent keeps running**. That is a git worktree.

`snapshot_filesystem()` is the wrong primitive here: the docs note it
terminates the sandbox. Directory snapshots do not. 30-day default TTL,
extendable; snapshot image IDs are stored in a `modal.Dict` because snapshots
are not named objects.

A branch is therefore: `{parent, snapshot_image_id, reward_config, budget,
status, best_objective}`. Rollback is just mounting an ancestor's snapshot —
"go back three checkpoints and try a different reward" is one call.

---

## 4. The agent tree

**Master agent** (Modal Sandbox) receives the bundle ref plus free-text
constraints ("must posttrain Qwen", "stay under $50", "must beat 76%"). It:

1. Parses the manifest; reads signals, objective, and constraints.
2. Writes an initial reward config and training config.
3. Trains a branch; scores it with the **immutable evaluator**.
4. Forks promising branches with mutated rewards; each child is a sandbox that
   inherits everything and interferes with nothing.
5. Children recurse, and **abandon themselves** when their line stops paying.

Agents act through a closed tool set — no shell, no arbitrary code into the
trainer:

| tool | effect |
|---|---|
| `propose_reward(config, rationale)` | validate against declared signals |
| `train(steps)` | train current branch, return metrics |
| `evaluate()` | immutable objective score |
| `fork(new_reward, note)` | snapshot + child sandbox |
| `rollback(checkpoint)` | mount an ancestor snapshot |
| `abandon(reason)` | self-terminate this line |
| `read_tree()` | ancestry + sibling results (shared blackboard) |
| `report(summary)` | surface best branch upward |

`read_tree` matters more than it looks: without it, branches re-explore each
other's dead ends. Reward configs are hashed and exact repeats rejected.

**Pruning.** Successive halving at wall-clock milestones on the objective, plus
per-branch token and GPU-minute budgets decremented centrally. A branch that
exhausts budget or fails to improve over K evaluations is abandoned
automatically, independent of whether its agent chooses to.

**Model.** Agents run on Modal Shared Endpoints with
`moonshotai/Kimi-K3.1-Pro` — verified live: OpenAI-compatible, 1M context,
$2/$10 per Mtok, auth via `Modal-Key`/`Modal-Secret`. It is a reasoning model:
in a probe, **102 of 143 completion tokens were `reasoning_tokens`** and
`content` was empty at `max_tokens=16`. The client budgets for that and reads
`content` and `reasoning_content` separately. `Modal-Routing-Affinity-Key` is
sent per branch so each agent's loop keeps its prefix cache (64 cached tokens
observed immediately).

---

## 5. Consolidation — what comes back

Walk the tree, rank by **objective** (never by reward), then emit to a Volume:

```
artifacts/<run_id>/
  weights/                 best branch's checkpoint
  manifest.json            env, base model, objective score, commands, spend
  reward_lineage.json      root -> winner, every mutation and its score
  losers.json              every abandoned branch + why (this is the science)
  serve.py                 runnable inference entrypoint
  README.md                how to run it
```

Constraints like "must posttrain Qwen" resolve to a `base_model` the sandbox
downloads from Hugging Face into a Modal Volume; the trainer uses the
`grpo_lora` path. The finished adapter can be served directly via
`modal endpoint create --custom-volume-name <vol> --custom-volume-path <path>`.

---

## 6. Frontend

Static page plus API served from **one origin** by a single Modal ASGI app.

* **New Run** — drop a bundle (zip or git URL), constraints textarea, budget caps.
* **Run** — live branch tree, objective per branch, spend, abandoned lines greyed.
* **Artifact** — download weights, copy the run command.

Visual language: Modal's — near-monochrome, tight mono type, generous
whitespace — with the accent moved from green to **blue**. Light and dark both
defined at token level.

---

## 7. Lessons encoded as guards

Paid for in the project this generalizes; each becomes a guard, not a comment.

1. Modal entrypoints live **outside** any package mounted via
   `add_local_python_source` — a double-uploaded module yields two `modal.App`
   objects and websocket routes die with a bare HTTP 500.
2. No `from __future__ import annotations` in modules defining FastAPI
   endpoints whose annotation types are imported lazily — the route silently
   stops dispatching.
3. Pin `fastapi==0.115.12`, `starlette<0.42`, `uvicorn==0.34.3`,
   `websockets==12.0`. Modal's ASGI bridge cannot serialize Starlette 1.x
   close messages.
4. `modal app stop` silently no-ops without `--yes`.
5. `modal run --detach` **creates** an app rather than replacing one. Relaunch
   must resolve and stop the predecessor, and refuse to start if that fails —
   two trainers on one directory silently corrupt the artifact.
6. Modal enforces an app-create rate limit. Stagger ≥20s and verify dispatch by
   comparing live app count to expected; a dropped job is invisible otherwise.
7. Never aggregate a ledger in one shared file under concurrent writers —
   read-modify-write loses entries (observed: 29 entries kept, one writer of
   six). Per-writer shards, aggregated at read.
8. **Poll real state; never trust a completion event.** Every silent failure in
   the predecessor project was caught by comparing counters, never by a
   notification.
9. Record the **actual** allocated GPU, not the requested one.
10. Degenerate tensor shapes trip native libraries (`n_sim=1` was rejected by a
    contiguity check that a size-1 leading stride cannot satisfy). Prefer ≥2.

---

## 8. Build order

| phase | deliverable | gate |
|---|---|---|
| **P0** | package scaffold, spec + validation, Kimi client, branch primitives, ledger shards | unit tests green, no GPU |
| **P1** | orchestrator + PPO trainer end-to-end on a trivial CPU env: master → fork → prune → consolidate | a real artifact from a real tree |
| **P2** | ASGI backend + frontend | start a run from the browser, watch the tree, download weights |
| **P3** | port a real environment (Tetris or SF3) | generality demonstrated on a non-toy |
| **P4** | docs, cost report, hardening | README reproducible from scratch |

P1's gate is the one that matters: a tree that forks, prunes, and consolidates
on a toy environment proves the framework. Everything after is breadth.
