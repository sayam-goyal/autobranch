"""Trainer tests: learning, resuming, logging, dispatch.

Everything here runs on a four-armed contextual bandit rather than on the
gridworld, so the suite stays fast and a failure means "PPO is broken" rather
than "the toy env is hard". The bandit is trivially learnable -- the
observation *is* the answer -- which makes it a clean tripwire: if PPO cannot
solve this, nothing else measured anywhere else in the framework means
anything.
"""

import json

import numpy as np
import pytest
import torch

from autobranch.env_api import StepResult, validate_env
from autobranch.reward import RewardError
from autobranch.spec import EnvironmentSpec
from autobranch.trainer import (
    ActorCritic,
    PPOHyper,
    Workspace,
    arch_from_spec,
    init_checkpoint,
    train,
    trainer_for,
)

N_ARMS = 4
EPISODE_LEN = 8


class BanditEnv:
    """Contextual bandit: obs is a one-hot cue, the matching action is correct.

    Deliberately degenerate as an RL problem (no credit assignment, no
    exploration difficulty) because its only job is to prove the optimiser
    moves the policy in the right direction.
    """

    def __init__(self, *, n_envs: int = 16, seed: int = 0):
        self._n_envs = int(n_envs)
        self._rng = np.random.default_rng(seed)
        self._target = np.zeros(self._n_envs, dtype=np.int64)
        self._t = np.zeros(self._n_envs, dtype=np.int64)
        self.closed = False

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._target = self._rng.integers(0, N_ARMS, size=self._n_envs)
        self._t[:] = 0
        return self._observe()

    def step(self, action):
        act = np.asarray(action).reshape(self._n_envs).astype(np.int64)
        correct = (act == self._target).astype(np.float32)
        self._t += 1
        # Episodes genuinely end rather than time out, which keeps the value
        # target bounded and makes "the loss fell" a meaningful claim.
        terminated = self._t >= EPISODE_LEN
        signals = {
            "correct": correct,
            "step_cost": np.ones(self._n_envs, dtype=np.float32),
        }
        final_obs = self._observe()
        self._target = self._rng.integers(0, N_ARMS, size=self._n_envs)
        self._t[terminated] = 0
        return StepResult(
            obs=self._observe(),
            signals=signals,
            terminated=terminated,
            truncated=np.zeros(self._n_envs, dtype=bool),
            info={"final_obs": final_obs},
        )

    def close(self) -> None:
        self.closed = True

    def _observe(self) -> np.ndarray:
        """One-hot cue plus the fraction of the episode left.

        The clock is in the observation on purpose: without it the critic
        cannot tell step 0 (worth ~8) from step 7 (worth ~1), and the residual
        value loss would be irreducible -- which would make
        `explained_variance` useless as a check on the critic.
        """
        obs = np.zeros((self._n_envs, N_ARMS + 1), dtype=np.float32)
        obs[np.arange(self._n_envs), self._target] = 1.0
        obs[:, N_ARMS] = 1.0 - self._t / EPISODE_LEN
        return obs


def make_bandit(**kwargs) -> BanditEnv:
    """Entrypoint used by the spec below, so `load_env` is exercised for real."""
    return BanditEnv(**kwargs)


def fake_trainer(*_args, **_kwargs) -> str:
    """Target for the `custom:module:fn` dispatch test."""
    return "custom trainer ran"


BANDIT_MANIFEST = {
    "name": "bandit",
    "entrypoint": "tests.test_trainer:make_bandit",
    "signals": [
        {"name": "correct", "type": "bool", "range": [0, 1], "doc": "action matched the cue"},
        {"name": "step_cost", "type": "float", "range": [0, 1], "doc": "one per step"},
    ],
    "objective": {"name": "accuracy", "evaluator": "tests.test_trainer:make_bandit"},
    "resources": {"gpu": None, "cpu": 1, "memory": 512},
    "policy": {"kind": "discrete", "n_actions": N_ARMS, "obs_dim": N_ARMS + 1},
    "trainer": "ppo",
}

REWARD = {"weights": {"correct": 1.0, "step_cost": -0.01}}

FAST = {
    "n_steps": 16,
    "epochs": 8,
    "minibatches": 4,
    "lr": 3e-3,
    "hidden": (32, 32),
    "env_kwargs": {"n_envs": 16},
    "torch_threads": 1,
}
STEPS_PER_ITER = FAST["n_steps"] * FAST["env_kwargs"]["n_envs"]


@pytest.fixture
def spec() -> EnvironmentSpec:
    return EnvironmentSpec.parse(BANDIT_MANIFEST)


def test_bandit_env_satisfies_the_contract(spec: EnvironmentSpec) -> None:
    problems = validate_env(make_bandit(n_envs=8), spec, steps=4)
    assert problems == []


def test_default_hyper_does_not_starve_learning() -> None:
    """16 gradient steps per collection was the predecessor's first failure.

    The number is a property of the config rather than a buried loop bound, so
    a future tweak that halves it is visible in a diff and in every metrics
    line.
    """
    assert PPOHyper().grad_steps_per_collection == 40
    assert PPOHyper().grad_steps_per_collection > 16
    assert PPOHyper(epochs=4, minibatches=4).grad_steps_per_collection == 16


def test_ppo_learns_a_trivially_learnable_task(spec: EnvironmentSpec, tmp_path) -> None:
    result = train(
        spec, REWARD, steps=20 * STEPS_PER_ITER, workspace=tmp_path, resume=False, seed=0,
        hyper=FAST,
    )
    lines = Workspace(tmp_path).read_metrics()
    assert len(lines) == 20
    assert result.steps_done == 20 * STEPS_PER_ITER

    first = lines[0]["signal_step_means"]["correct"]
    last = lines[-1]["signal_step_means"]["correct"]
    assert first < 0.45, f"a fresh policy should be near chance (0.25), got {first}"
    assert last > 0.80, f"PPO failed to solve a one-hot bandit: {first:.3f} -> {last:.3f}"

    # The policy sharpens (entropy drops from ~ln(4)=1.386) and the critic
    # stops being surprised: the loss comparison is made over the second half,
    # because early on the value target is still growing with the policy and a
    # rising value loss there is learning, not failure.
    assert lines[-1]["entropy"] < lines[0]["entropy"]
    mid = np.mean([l["loss"] for l in lines[10:13]])
    end = np.mean([l["loss"] for l in lines[-3:]])
    assert end < mid, f"loss did not fall over the second half: {mid:.3f} -> {end:.3f}"
    assert np.mean([l["value_loss"] for l in lines[-3:]]) < np.mean(
        [l["value_loss"] for l in lines[10:13]]
    )
    assert abs(lines[-1]["policy_loss"]) < abs(lines[0]["policy_loss"])
    assert lines[-1]["explained_variance"] > 0.3, "the critic learned nothing"
    # 8 steps at ~1 per correct answer, minus the step cost. The reported
    # return is the composed reward only -- no bootstrap folded in.
    assert 6.0 < result.metrics["final"]["episode_return_mean"] < 8.0


def test_truncation_is_bootstrapped_rather_than_treated_as_death(tmp_path) -> None:
    """A time limit must not teach the critic that the clock is fatal.

    The env pays exactly 1 per step forever and truncates every 4 steps. With
    `info["final_obs"]` present the trainer bootstraps the cut-off, so the
    critic should converge towards the infinite-horizon value
    `1/(1-gamma) = 10` at gamma=0.9; without it, the best it can learn is the
    within-window sum, 1+.9+.81+.729 = 3.44. Comparing the two runs makes the
    bootstrap observable from the outside.
    """
    class Treadmill:
        n_envs = 8

        def __init__(self, *, with_final_obs: bool):
            self.with_final_obs = with_final_obs
            self._t = np.zeros(8, dtype=np.int64)

        def reset(self, *, seed: int | None = None):
            self._t[:] = 0
            return np.ones((8, 1), dtype=np.float32)

        def step(self, action):
            self._t += 1
            truncated = self._t >= 4
            self._t[truncated] = 0
            info = {"final_obs": np.ones((8, 1), dtype=np.float32)} if self.with_final_obs else {}
            return StepResult(
                obs=np.ones((8, 1), dtype=np.float32),
                signals={"tick": np.ones(8, dtype=np.float32)},
                terminated=np.zeros(8, dtype=bool),
                truncated=truncated,
                info=info,
            )

        def close(self) -> None:
            pass

    spec = EnvironmentSpec.parse({
        "name": "treadmill",
        "entrypoint": "tests.test_trainer:make_bandit",  # unused: env injected
        "signals": [{"name": "tick", "type": "float", "range": [0, 1], "doc": "always 1"}],
        "objective": {"name": "ticks", "evaluator": "tests.test_trainer:fake_trainer"},
        "policy": {"kind": "discrete", "n_actions": 2, "obs_dim": 1},
    })
    hyper = dict(FAST, gamma=0.9, env_kwargs={}, lr=1e-2, anneal_lr=False)

    values = {}
    for flag in (True, False):
        ws = tmp_path / f"final_obs_{flag}"
        train(
            spec, {"weights": {"tick": 1.0}}, steps=40 * 16 * 8, workspace=ws, resume=False,
            hyper=hyper, env=Treadmill(with_final_obs=flag),
        )
        net = ActorCritic(arch_from_spec(spec, hyper["hidden"]))
        net.load_state_dict(
            torch.load(ws / "checkpoints" / "latest.pt", map_location="cpu",
                       weights_only=False)["policy"]
        )
        with torch.no_grad():
            values[flag] = float(net.value(torch.ones(1, 1)))

    assert values[True] > values[False] + 2.0, (
        f"bootstrapped value {values[True]:.2f} should clearly exceed the truncated-as-terminal "
        f"value {values[False]:.2f} (targets are 10.0 and 3.44)"
    )
    assert values[False] < 6.0


def test_metrics_lines_carry_loss_parts_entropy_kl_clip_and_reward_terms(
    spec: EnvironmentSpec, tmp_path
) -> None:
    train(spec, REWARD, steps=2 * STEPS_PER_ITER, workspace=tmp_path, resume=False, hyper=FAST)
    lines = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert lines
    for line in lines:
        for key in (
            "step", "iteration", "progress", "lr", "loss", "policy_loss", "value_loss",
            "entropy", "approx_kl", "clip_frac", "explained_variance", "grad_steps",
            "grad_steps_per_collection", "reward_terms", "reward_fingerprint", "wall_seconds",
        ):
            assert key in line, f"metrics line is missing {key!r}"
        # Per-term accounting: a reward climbing while the objective falls is
        # invisible without it.
        assert set(line["reward_terms"]) == {"correct", "step_cost"}
        assert line["reward_terms"]["step_cost"] < 0
        assert line["grad_steps"] == 32
        # Throughput is diagnostics, never progress.
        assert "env_steps_per_second" in line["diagnostics"]
        assert 0.0 <= line["progress"] <= 1.0


def test_checkpoints_are_written_per_interval_plus_latest(spec: EnvironmentSpec, tmp_path) -> None:
    hyper = dict(FAST, checkpoint_interval=2 * STEPS_PER_ITER)
    train(spec, REWARD, steps=6 * STEPS_PER_ITER, workspace=tmp_path, resume=False, hyper=hyper)
    ws = Workspace(tmp_path)
    named = sorted(p.name for p in ws.checkpoints.glob("step_*.pt"))
    assert ws.latest.exists()
    assert named == [
        f"step_{2 * STEPS_PER_ITER:09d}.pt",
        f"step_{4 * STEPS_PER_ITER:09d}.pt",
        f"step_{6 * STEPS_PER_ITER:09d}.pt",
    ]


def test_resume_restores_policy_step_and_optimizer_moments(spec: EnvironmentSpec, tmp_path) -> None:
    """A fork is a directory snapshot, so everything must survive on disk.

    Three claims, each separately checkable: (1) re-entering `train` with the
    same budget is a no-op because the step count was restored, and the
    weights are untouched; (2) continuing past that budget appends to the same
    metrics stream with strictly increasing steps; (3) Adam's second-moment
    buffers come back non-zero, which is the part that silently resets if you
    only reload the weights -- and a reset optimiser looks exactly like a bad
    reward for the first few iterations.
    """
    first_target = 4 * STEPS_PER_ITER
    first = train(
        spec, REWARD, steps=first_target, workspace=tmp_path, resume=False, seed=0, hyper=FAST
    )
    ws = Workspace(tmp_path)
    saved = torch.load(ws.latest, map_location="cpu", weights_only=False)
    assert saved["step"] == first_target
    params_before = {k: v.clone() for k, v in saved["policy"].items()}

    noop = train(spec, REWARD, steps=first_target, workspace=tmp_path, resume=True, hyper=FAST)
    assert noop.steps_done == first_target
    assert noop.iterations == first.iterations
    after_noop = torch.load(ws.latest, map_location="cpu", weights_only=False)
    for key, value in params_before.items():
        assert torch.allclose(value, after_noop["policy"][key]), f"{key} changed on a no-op resume"

    second_target = 7 * STEPS_PER_ITER
    second = train(spec, REWARD, steps=second_target, workspace=tmp_path, resume=True, hyper=FAST)
    assert second.steps_done == second_target
    assert second.iterations == 7

    steps = [line["step"] for line in ws.read_metrics()]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
    assert steps[0] == STEPS_PER_ITER and steps[-1] == second_target

    resumed = torch.load(ws.latest, map_location="cpu", weights_only=False)
    moments = [
        s["exp_avg_sq"] for s in resumed["optimizer"]["state"].values() if "exp_avg_sq" in s
    ]
    assert moments, "no Adam second moments in the resumed checkpoint"
    assert all(float(m.abs().sum()) > 0 for m in moments)
    assert any(
        not torch.allclose(params_before[k], resumed["policy"][k]) for k in params_before
    ), "continuing training changed nothing"

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["steps_done"] == second_target
    assert state["reward_fingerprint"] == second.reward_fingerprint
    assert state["wall_seconds"] >= first.wall_seconds


def test_resume_records_a_reward_change_as_lineage(spec: EnvironmentSpec, tmp_path) -> None:
    """A child branch inherits weights and swaps the reward. That is a fork."""
    train(spec, REWARD, steps=STEPS_PER_ITER, workspace=tmp_path, resume=False, hyper=FAST)
    mutated = {"weights": {"correct": 2.0}}
    train(spec, mutated, steps=2 * STEPS_PER_ITER, workspace=tmp_path, resume=True, hyper=FAST)
    lineage = json.loads((tmp_path / "state.json").read_text())["reward_lineage"]
    assert len(lineage) == 2
    assert lineage[0]["fingerprint"] != lineage[1]["fingerprint"]
    assert lineage[1]["at_step"] == STEPS_PER_ITER


def test_malformed_reward_surfaces_as_reward_error(spec: EnvironmentSpec, tmp_path) -> None:
    """The agent loop feeds this back as a correction turn, so it must not be swallowed."""
    with pytest.raises(RewardError) as excinfo:
        train(
            spec, {"weights": {"not_a_signal": 1.0}}, steps=STEPS_PER_ITER,
            workspace=tmp_path, resume=False, hyper=FAST,
        )
    assert "not_a_signal" in str(excinfo.value)


def test_init_checkpoint_is_loadable_and_untrained(spec: EnvironmentSpec, tmp_path) -> None:
    path = init_checkpoint(spec, tmp_path / "root.pt", seed=3)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 0
    assert payload["optimizer"] is None
    assert payload["arch"]["obs_dim"] == N_ARMS + 1
    assert payload["arch"]["n_actions"] == N_ARMS


def test_trainer_for_dispatches_on_the_manifest(spec: EnvironmentSpec) -> None:
    assert trainer_for(spec) is train

    import dataclasses

    grpo = dataclasses.replace(spec, trainer="grpo_lora")
    with pytest.raises(NotImplementedError) as excinfo:
        trainer_for(grpo)
    assert "P3" in str(excinfo.value), "the error must name the phase it arrives in"

    custom = dataclasses.replace(spec, trainer="custom:tests.test_trainer:fake_trainer")
    assert trainer_for(custom) is fake_trainer


def test_continuous_policies_train_too(tmp_path) -> None:
    """Dispatch is driven by `spec.policy`, so a continuous env needs no new code."""

    class Reacher:
        """Reward is maximal when the action matches the observed target."""

        n_envs = 8

        def __init__(self) -> None:
            self._rng = np.random.default_rng(0)
            self._target = np.zeros((8, 2), dtype=np.float32)

        def reset(self, *, seed: int | None = None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._target = self._rng.normal(size=(8, 2)).astype(np.float32)
            return self._target.copy()

        def step(self, action):
            act = np.asarray(action, dtype=np.float32).reshape(8, 2)
            err = np.abs(act - self._target).sum(axis=1)
            signals = {"closeness": np.exp(-err).astype(np.float32)}
            self._target = self._rng.normal(size=(8, 2)).astype(np.float32)
            return StepResult(
                obs=self._target.copy(),
                signals=signals,
                terminated=np.zeros(8, dtype=bool),
                truncated=np.ones(8, dtype=bool),
            )

        def close(self) -> None:
            pass

    spec = EnvironmentSpec.parse({
        "name": "reacher",
        "entrypoint": "tests.test_trainer:make_bandit",  # unused: env passed in directly
        "signals": [{"name": "closeness", "type": "float", "range": [0, 1], "doc": "exp(-error)"}],
        "objective": {"name": "closeness", "evaluator": "tests.test_trainer:fake_trainer"},
        "policy": {"kind": "continuous", "n_actions": 2, "obs_dim": 2},
    })
    hyper = dict(FAST, env_kwargs={})
    result = train(
        spec, {"weights": {"closeness": 1.0}}, steps=6 * 16 * 8, workspace=tmp_path,
        resume=False, hyper=hyper, env=Reacher(),
    )
    lines = Workspace(tmp_path).read_metrics()
    assert result.steps_done == 6 * 16 * 8
    assert lines[-1]["signal_step_means"]["closeness"] > lines[0]["signal_step_means"]["closeness"]
