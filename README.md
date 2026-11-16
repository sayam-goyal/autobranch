<img src="assets/banner.png" alt="autobranch" width="100%">

Reward search for RL, running on Modal. You give it an environment and a
sentence about what you want. It gives back trained weights and the numbers
behind them.

Writing the reward is usually the slow part, and it is usually done by hand.
This runs that search for you: a tree of agents, each one training a branch
with its own reward, forking the ones that work and dropping the ones that
don't. A separate evaluator scores every branch, and the agents cannot touch
it.

## Why

The project this came from trained six reward configs. All six scored 0.000
against a fixed opponent while self-play win rate sat at 0.807 over 6,705
matches. The self-play number was not progress. The thing that actually
mattered was how the policy represented a decision, which had been assumed in
the plan and never tested. Changing only that, with the reward held fixed,
went from 0.000 to 0.990.

So two rules shape everything here:

- **Agents write rewards. They never write the objective.** The evaluator is
  separate code that imports none of theirs. Reward is a means. The objective
  is the measurement.
- **Forking is cheap**, so testing an assumption costs about as much as
  testing a weight.

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

It reads the weights off the Volume and runs the same evaluator the branch
would. Match seeds come from a bank of eight unrelated families, because one
family is one draw of opponents and a number from it looks tighter than it is.

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

Signals are facts, not rewards. `damage_dealt` is a number the environment
knows. Whether dealing damage is good is the agent's guess, and the evaluator
decides if the guess was right.

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

Validation runs before a branch launches, so a config naming a signal that
doesn't exist costs one correction turn instead of an hour of GPU.

## Branching

Each branch is a Modal Sandbox. Forking snapshots the parent's filesystem, so
a child starts from the parent's weights and reward, changes one thing, and
trains on. Branches stop themselves when their line stops paying, and the
reason is recorded. Abandoned branches ship with the winner: they are the
evidence the winner was a choice.

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
