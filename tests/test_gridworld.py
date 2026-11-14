"""The reference environment, and the objective measured over it.

The interesting assertions here are not "does it run" but "does it
discriminate": a locked-door gridworld whose key sits outside every monotone
path to the goal is the smallest environment where a reward search has real
work to do. `test_reward_configs_are_discriminated` is the one that proves
it, and it is the P1 gate in miniature.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autobranch.env_api import StepResult, load_env, validate_env
from autobranch.objective import ObjectiveScore, evaluate, seed_bank
from autobranch.spec import EnvironmentSpec
from autobranch.trainer import init_checkpoint, train

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "examples" / "gridworld"
MANIFEST = BUNDLE / "autobranch.yaml"

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

#: Cheap evaluation budget: enough episodes to separate 0.0 from 1.0, few
#: enough to keep the suite fast. The manifest's real budget is 256.
CHEAP = {"episodes": 32, "lanes": 16}


@pytest.fixture(scope="module")
def spec() -> EnvironmentSpec:
    return EnvironmentSpec.from_yaml(MANIFEST.read_text())


@pytest.fixture(scope="module")
def grid_env_module():
    sys.path.insert(0, str(BUNDLE))
    try:
        import grid_env

        return grid_env
    finally:
        sys.path.remove(str(BUNDLE))


def make(grid_env_module, **kwargs):
    kwargs.setdefault("n_envs", 4)
    kwargs.setdefault("slip_prob", 0.0)  # scripted tests need an obedient agent
    return grid_env_module.GridWorld(**kwargs)


def drive(env, actions):
    """Run a scripted action sequence, returning every StepResult."""
    return [env.step(np.full(env.n_envs, a, dtype=np.int64)) for a in actions]


# ---------------------------------------------------------------------------
# manifest and contract
# ---------------------------------------------------------------------------


def test_manifest_describes_the_environment_it_ships_with(spec, grid_env_module) -> None:
    env = load_env(spec.entrypoint, search_path=BUNDLE, n_envs=2)
    assert spec.name == "gridworld"
    assert spec.trainer == "ppo"
    assert spec.resources.gpu is None, "the reference env must stay CPU-only"
    assert set(spec.signal_names) == set(grid_env_module.SIGNAL_NAMES)
    assert spec.policy.kind == "discrete"
    assert spec.policy.n_actions == grid_env_module.N_ACTIONS
    assert spec.policy.obs_dim == env.obs_dim
    assert spec.objective.name == "success_rate"
    assert spec.objective.direction == "maximize"
    assert all(s.doc.strip() for s in spec.signals), "a signal without a doc is a signal misused"
    assert all(s.range is not None for s in spec.signals)


def test_validate_env_reports_no_problems(spec) -> None:
    env = load_env(spec.entrypoint, search_path=BUNDLE, n_envs=8)
    assert validate_env(env, spec, steps=8) == []


def test_every_declared_signal_is_emitted_each_step(spec, grid_env_module) -> None:
    env = make(grid_env_module)
    env.reset(seed=0)
    for result in drive(env, [RIGHT, DOWN, LEFT, UP]):
        assert set(result.signals) == set(spec.signal_names)
        for name, value in result.signals.items():
            assert np.asarray(value).shape == (env.n_envs,), name


def test_validate_env_catches_an_env_that_lies_about_its_signals(spec) -> None:
    """The negative half of the gate: a validator that only ever passes is useless."""

    class Liar:
        n_envs = 3

        def reset(self, *, seed=None):
            return np.zeros((3, 81), dtype=np.float32)

        def step(self, action):
            return StepResult(
                obs=np.zeros((3, 81), dtype=np.float32),
                signals={
                    "reached_goal": np.zeros(3),
                    "stepped_in_lava": np.zeros(2),       # wrong shape
                    "picked_up_key": np.zeros(3),
                    "distance_to_goal": np.full(3, 7.0),  # outside its declared range
                    "distance_to_key": np.zeros(3),
                    "step_cost": np.ones(3),
                    # revisited_tile missing entirely
                    "secret_bonus": np.ones(3),           # undeclared
                },
                terminated=np.zeros(3, dtype=bool),
                truncated=np.zeros(3, dtype=bool),
            )

        def close(self) -> None:
            pass

    problems = validate_env(Liar(), spec, steps=2)
    joined = " | ".join(problems)
    assert "revisited_tile" in joined and "not emitted" in joined
    assert "secret_bonus" in joined and "not declared" in joined
    assert "stepped_in_lava" in joined
    assert "distance_to_goal" in joined and "outside declared" in joined


def test_identical_seeds_give_identical_episodes(spec, grid_env_module) -> None:
    """Branch comparisons are meaningless if the same seed replays differently."""
    rng = np.random.default_rng(7)
    actions = rng.integers(0, 4, size=(40, 6))
    traces = []
    for _ in range(2):
        env = grid_env_module.GridWorld(n_envs=6, slip_prob=0.2)
        obs = env.reset(seed=1234)
        trace = [obs.copy()]
        for row in actions:
            result = env.step(row)
            trace.append(result.obs.copy())
            trace.append(np.stack([result.signals[n] for n in sorted(result.signals)]))
        traces.append(trace)
    for a, b in zip(*traces):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# the mechanics that make reward design matter
# ---------------------------------------------------------------------------


def test_goal_is_locked_without_the_key(spec, grid_env_module) -> None:
    """The whole point of the environment: distance_to_goal cannot reach 0.

    Walking the direct route to the goal leaves the agent parked one tile
    away, forever, with no termination. A reward that only chases
    `distance_to_goal` converges to exactly this and scores zero.
    """
    env = make(grid_env_module)
    env.reset(seed=0)
    results = drive(env, [RIGHT] * 5 + [DOWN] * 10)
    for result in results:
        assert not np.any(result.signals["reached_goal"])
        assert not np.any(result.signals["picked_up_key"])
        assert not np.any(np.asarray(result.terminated))
    last = results[-1]
    # One tile from the door, which is 1/(2*(size-1)) = 1/14 away.
    assert np.allclose(last.signals["distance_to_goal"], 1.0 / 14.0)
    assert np.all(np.asarray(last.signals["revisited_tile"]) == 1.0)


def test_key_then_goal_reaches_the_goal(spec, grid_env_module) -> None:
    env = make(grid_env_module)
    env.reset(seed=0)
    route = [RIGHT] * 7 + [DOWN] * 7 + [UP] * 2 + [LEFT] * 2  # 18 steps, the BFS optimum
    results = drive(env, route)
    pickups = [i for i, r in enumerate(results) if np.all(r.signals["picked_up_key"] == 1.0)]
    assert pickups == [13], "the key is collected exactly once, on arrival at its tile"
    assert np.all(results[-1].signals["reached_goal"] == 1.0)
    assert np.all(np.asarray(results[-1].terminated))
    assert not np.any([np.any(r.signals["stepped_in_lava"]) for r in results])
    assert np.allclose(results[13].signals["distance_to_key"], 0.0)
    assert len(route) == 18


def test_lava_ends_the_episode_and_the_lane_auto_resets(spec, grid_env_module) -> None:
    env = make(grid_env_module)
    start_obs = env.reset(seed=0)
    results = drive(env, [DOWN] * 3)
    fatal = results[-1]
    assert np.all(fatal.signals["stepped_in_lava"] == 1.0)
    assert np.all(np.asarray(fatal.terminated))
    assert not np.any(np.asarray(fatal.truncated))
    # The published obs is the new episode's; the pre-reset one survives in
    # info["final_obs"] so a trainer can bootstrap correctly.
    assert np.allclose(fatal.obs, start_obs)
    assert not np.allclose(fatal.info["final_obs"], start_obs)


def test_episodes_truncate_at_the_step_cap(spec, grid_env_module) -> None:
    env = make(grid_env_module, max_steps=12)
    env.reset(seed=0)
    results = drive(env, [RIGHT] * 12)
    assert not np.any(np.asarray(results[-2].truncated))
    assert np.all(np.asarray(results[-1].truncated))
    assert not np.any(np.asarray(results[-1].terminated))


def test_layout_rejects_a_key_inside_the_start_goal_box(grid_env_module) -> None:
    """The guard that keeps the environment discriminating.

    A key on a monotone path from start to goal would be collected for free by
    a distance-only reward, and the env would stop testing reward design.
    """
    with pytest.raises(ValueError, match="bounding box"):
        grid_env_module.GridLayout(key=(2, 2)).validate()
    grid_env_module.GridLayout().validate()  # the shipped layout is legal


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------


def test_random_policy_scores_zero(grid_env_module) -> None:
    """The floor. A random walker dies in lava or runs out of clock."""
    rng = np.random.default_rng(0)

    def random_policy(obs):
        return rng.integers(0, 4, size=np.asarray(obs).shape[0])

    out = grid_env_module.evaluate(
        random_policy, seeds=seed_bank(12345), budget={"episodes": 64, "lanes": 16}
    )
    assert out["n"] == 64
    assert out["value"] == 0.0
    assert out["detail"]["key_rate"] == 0.0
    assert out["detail"]["lava_rate"] > 0.0


def test_evaluate_is_reproducible_for_the_same_checkpoint_and_seed(spec, tmp_path) -> None:
    ckpt = init_checkpoint(spec, tmp_path / "untrained.pt", seed=0)
    first = evaluate(spec, ckpt, search_path=BUNDLE, budget=CHEAP, seed=777)
    second = evaluate(spec, ckpt, search_path=BUNDLE, budget=CHEAP, seed=777)
    assert isinstance(first, ObjectiveScore)
    assert (first.value, first.n) == (second.value, second.n)
    assert first.detail["modes"] == second.detail["modes"]
    assert first.detail["seeds"] == list(seed_bank(777))
    # A different seed bank is a different measurement, and must say so.
    third = evaluate(spec, ckpt, search_path=BUNDLE, budget=CHEAP, seed=778)
    assert third.detail["seeds"] != first.detail["seeds"]


def test_evaluate_reports_both_policy_modes(spec, tmp_path) -> None:
    """An argmax policy on fresh weights is a constant, and must not be mistaken
    for a strategy -- so both modes are always measured, whichever is asked for."""
    ckpt = init_checkpoint(spec, tmp_path / "untrained.pt", seed=1)
    score = evaluate(spec, ckpt, search_path=BUNDLE, budget=CHEAP, deterministic=True)
    assert set(score.detail["modes"]) == {"deterministic", "stochastic"}
    assert score.value == score.detail["modes"]["deterministic"]["value"]
    assert score.detail["mode_gap"] == pytest.approx(
        score.detail["modes"]["deterministic"]["value"]
        - score.detail["modes"]["stochastic"]["value"]
    )
    stochastic = evaluate(
        spec, ckpt, search_path=BUNDLE, budget=CHEAP, deterministic=False
    )
    assert stochastic.value == stochastic.detail["modes"]["stochastic"]["value"]
    assert score.value == 0.0 and stochastic.value == 0.0  # untrained scores zero either way
    assert score.detail["objective"] == "success_rate"
    assert score.detail["arch"]["obs_dim"] == spec.policy.obs_dim


def test_objective_score_carries_a_command_that_reproduces_it(spec, tmp_path) -> None:
    """Every number carries its provenance, and the provenance has to run."""
    ckpt = init_checkpoint(spec, tmp_path / "untrained.pt", seed=2)
    score = evaluate(spec, ckpt, search_path=BUNDLE, budget=CHEAP, seed=99)
    assert "--seed 99" in score.command and str(ckpt) in score.command
    out = subprocess.run(
        score.command.split(), capture_output=True, text=True, cwd=str(REPO), check=False
    )
    assert out.returncode == 0, out.stderr
    replayed = json.loads(out.stdout)
    assert replayed["value"] == score.value
    assert replayed["n"] == score.n


# ---------------------------------------------------------------------------
# the P1 gate, in miniature
# ---------------------------------------------------------------------------

NAIVE_REWARD = {
    "weights": {"step_cost": -0.01},
    "shaping": [{"kind": "potential", "signal": "distance_to_goal", "gamma": 0.99, "scale": -1.0}],
}
SENSIBLE_REWARD = {
    "weights": {
        "reached_goal": 1.0,
        "picked_up_key": 0.5,
        "stepped_in_lava": -0.5,
        "step_cost": -0.01,
        "revisited_tile": -0.01,
    },
    "shaping": [
        {"kind": "potential", "signal": "distance_to_key", "gamma": 0.99, "scale": -1.0},
        {"kind": "potential", "signal": "distance_to_goal", "gamma": 0.99, "scale": -0.5},
    ],
}
TRAIN_HYPER = {
    "n_steps": 32,
    "epochs": 10,
    "minibatches": 4,
    "hidden": (64, 64),
    "env_kwargs": {"n_envs": 64},
    "torch_threads": 1,
}


@pytest.mark.slow
def test_reward_configs_are_discriminated(spec, tmp_path) -> None:
    """The environment must separate a plausible-looking reward from a good one.

    This is the property that makes the reward search worth running: chasing
    `distance_to_goal` is the obvious thing to write, it trains happily, and
    it scores zero because the door is locked and the key is off-path.
    """
    scores = {}
    for tag, reward in (("naive", NAIVE_REWARD), ("sensible", SENSIBLE_REWARD)):
        result = train(
            spec, reward, steps=100_000, workspace=tmp_path / tag, resume=False, seed=0,
            hyper=TRAIN_HYPER, search_path=BUNDLE,
        )
        scores[tag] = evaluate(
            spec, result.checkpoint_path, search_path=BUNDLE, budget=CHEAP, deterministic=False
        )

    assert scores["naive"].value < 0.1, (
        f"a distance-only reward should plateau near zero, got {scores['naive'].value}"
    )
    assert scores["sensible"].value > 0.7, (
        f"a key-aware reward should solve the task, got {scores['sensible'].value}"
    )
    assert spec.objective.better(scores["sensible"].value, scores["naive"].value)
