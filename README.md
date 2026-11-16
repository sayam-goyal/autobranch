<img src="assets/banner.png" alt="autobranch" width="100%">

Reward search for RL, running on sandboxes. Give it an environment and a
an objective and get back trained weights and final artifacts.

Creating reward functions and verifying no reward hacking occurs is tedious.
This framework handles all of that for you using agents swarms and a tree of sandboxes, each one training a model
with its own reward, forking the ones that work and dropping the ones that
don't. A separate evaluator scores every branch.

## Install

```bash
pip install -e ".[train,serve,dev]"
modal token new          # if you haven't
```

## Run

```bash
./serve_local.sh                                    # UI at localhost:8080
modal deploy app.py                                 # backend

modal run app.py::cli_validate --bundle examples/sf3
modal run app.py::cli_submit   --bundle examples/sf3 --usd 50 --hours 8
modal run app.py::cli_status   --run-id run-xxxx
modal run app.py::cli_finalize --run-id run-xxxx    # build the artifact
```

`examples/sf3` is a full environment: Street Fighter III through Diambra,
Ryu against the built-in CPU.

Scoring a checkpoint yourself, outside the search:

```bash
modal run tools/eval_sf3.py --run-id run-xxxx --branch-id b0 --matches 72
```

## Environments

An environment is a directory with a manifest and some code. The manifest is
the whole interface.

```yaml
# autobranch.yaml
name: street-fighter-iii
version: 1

image:
  base: nvidia/cuda:12.8.1-devel-ubuntu22.04
  python: "3.11"
  pip: [torch==2.8.0]
  secrets: [sf3-rom-auth]
  volumes: {"/roms": sf3-rom}

entrypoint: sf_env:make_env       # module:callable -> env
resources: {gpu: L40S, cpu: 8, memory: 32768}

# Facts about a step. The only things a reward may be built from.
signals:
  - {name: damage_dealt,  type: float, range: [0, 1],  doc: "health dealt this step"}
  - {name: damage_taken,  type: float, range: [0, 1],  doc: "health lost this step"}
  - {name: round_win,     type: int,   range: [-1, 1], doc: "+1 won, -1 lost"}
  - {name: health_margin, type: float, range: [-1, 1], doc: "own minus opponent"}

# What is actually being optimised. Agents may not edit this block.
objective:
  name: win_rate_vs_cpu8
  direction: maximize
  evaluator: sf_env:evaluate
  budget: {matches: 100, difficulty: 8}
```

## Rewards are data

A reward config is JSON, not code, so it can be diffed, hashed, and mutated
safely.

```json
{
  "weights": {"round_win": 1.0},
  "shaping": [{"kind": "potential", "signal": "health_margin", "gamma": 0.99, "scale": 1.0}],
  "anneal":  {"shaping": {"from": 1.0, "to": 0.0, "by_fraction": 0.8}}
}
```

## Branching

Each branch is a sandbox. Forking snapshots the parent's filesystem, so
a child starts from the parent's weights and reward, changes one thing, and
trains on. Branches stop themselves when their line stops paying, and the
reason is recorded.

## Trainers

| trainer | what it is |
|---|---|
| `ppo` | normal PPO for small policies |
| `grpo_lora` | critic-free, group-relative, LoRA over a frozen base model |
| `grpo_full` | same method, every parameter trainable |

For language models an action is one token drawn from a restricted vocabulary,
so an illegal action can't be represented and no penalty term is needed.

## Layout

```
app.py              Modal entrypoints (validate, submit, status, finalize)
autobranch/
  spec.py           manifest parsing and validation
  reward.py         the reward DSL: validate and compile
  objective.py      scoring, independent of anything an agent wrote
  branch_main.py    one branch: train, score, fork, stop
  forking.py        snapshot and spawn a child
  registry.py       run and branch state
  ledger.py         spend, as per-writer shards
  server.py         API and UI
examples/sf3        a full environment, end to end
frontend/           the web UI
tools/              score a checkpoint, snapshot one, watch a run
probes/             the measurements the design rests on
evidence/           charts from the runs that have been done
tests/
```
