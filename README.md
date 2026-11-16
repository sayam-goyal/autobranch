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

modal run app.py::cli_validate --bundle examples/gridworld
modal run app.py::cli_submit   --bundle examples/gridworld --usd 8 --hours 1
modal run app.py::cli_status   --run-id run-xxxx
modal run app.py::cli_finalize --run-id run-xxxx    # build the artifact
```

`examples/` has three ready to run: `gridworld`, `wordguess`, `sf3`.

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
examples/           gridworld, wordguess, sf3
frontend/           the web UI
tests/
```

## Modal notes

Things that cost time, written down so they don't cost it again.

1. Keep Modal entrypoints outside any package you mount with
   `add_local_python_source`. A double-uploaded module gives you two
   `modal.App` objects and websocket routes die with a bare 500.
2. No `from __future__ import annotations` in modules defining FastAPI
   endpoints with lazily imported annotation types. The route stops
   dispatching, silently.
3. Pin `fastapi==0.115.12`, `starlette<0.42`, `uvicorn==0.34.3`,
   `websockets==12.0`. Modal's ASGI bridge can't serialise Starlette 1.x close
   messages.
4. `modal app stop` does nothing without `--yes`.
5. `modal run --detach` creates an app, it doesn't replace one. Relaunching
   has to stop the old one first, and refuse to start if it can't. Two
   trainers on one directory corrupt the artifact quietly.
6. There's an app-create rate limit. Stagger launches by 20s and check the
   live app count against what you expect, or a dropped job is invisible.
7. Don't aggregate a ledger in one shared file with concurrent writers.
   Read-modify-write loses entries; we once kept 29 of six writers' records.
   Use one file per writer and sum them at read time.
8. Poll real state. Never trust a completion event. Every silent failure in
   the predecessor project was caught by comparing counters.
9. Record the GPU you were actually given, not the one you asked for.
10. Degenerate shapes break native libraries. `n_sim=1` failed a contiguity
    check that a size-1 leading stride can't satisfy. Use 2 or more.

## Tests

```bash
pytest -q
```
