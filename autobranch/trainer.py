"""Reward composition plus PPO. The only place a reward exists at all.

Two things live here and they are deliberately separate:

1. **Composition.** The environment hands over signals; `autobranch.reward`
   compiles the agent's JSON config into a callable over those signals; this
   module calls it. The trainer never reads environment internals and the
   environment never sees a reward, which is the whole reason reward search
   can be delegated to a model.
2. **Optimisation.** An env-agnostic PPO driven entirely by `spec.policy`, so
   a new environment needs no trainer changes.

Two guards are baked in because the predecessor project paid for both:

* **Gradient steps per collection are explicit.** `epochs * minibatches`
  gradient steps run on every rollout, exposed as
  `PPOHyper.grad_steps_per_collection` and written into every metrics line.
  The predecessor ran 16 and starved learning; the default here is 40.
* **Per-term reward accounting is logged every iteration.** A training reward
  climbing while the real objective falls is invisible without it -- that is
  precisely the failure where six configs read healthy and all scored 0.000.

Progress is measured in environment steps against the objective, never in
throughput. `env_steps_per_second` is recorded under `diagnostics` so nobody
mistakes it for learning.
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from autobranch.env_api import INFO_FINAL_OBS, StepResult, import_target, load_env
from autobranch.reward import CompiledReward, RewardConfig, compile_reward, config_hash
from autobranch.spec import EnvironmentSpec

CHECKPOINT_FORMAT = 1


# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPOHyper:
    """Everything tunable, with defaults that learn on a toy env on CPU.

    `n_steps` is per lane, so a collection is `n_envs * n_steps` transitions
    and `epochs * minibatches` gradient steps are taken on it. Both numbers
    are reported rather than implied: an under-trained collection looks
    exactly like a bad reward from the outside.
    """

    n_steps: int = 32
    epochs: int = 10
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    anneal_lr: bool = True
    target_kl: float | None = None
    hidden: tuple[int, ...] = (128, 128)
    checkpoint_interval: int = 0  # env steps; 0 -> one checkpoint per collection
    device: str = "cpu"
    #: Intra-op threads. One by default on purpose: the policies here are small
    #: dense nets where BLAS threading costs more than it buys, and a branch
    #: declaring `cpu: 2` in its manifest should not quietly spawn eight
    #: threads and fight every sibling branch on the same host. `None` leaves
    #: the process default alone.
    torch_threads: int | None = 1
    env_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def grad_steps_per_collection(self) -> int:
        """How many optimiser steps each rollout actually buys."""
        return int(self.epochs) * int(self.minibatches)

    @staticmethod
    def parse(raw: "PPOHyper | dict[str, Any] | None") -> "PPOHyper":
        if raw is None:
            return PPOHyper()
        if isinstance(raw, PPOHyper):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"hyper must be PPOHyper, dict or None, got {type(raw).__name__}")
        known = {f for f in PPOHyper.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown hyperparameters {sorted(unknown)}; known: {sorted(known)}")
        kwargs = dict(raw)
        if "hidden" in kwargs:
            kwargs["hidden"] = tuple(int(h) for h in kwargs["hidden"])
        return PPOHyper(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["grad_steps_per_collection"] = self.grad_steps_per_collection
        return d


@dataclass(frozen=True)
class TrainResult:
    """What a branch reports upward. Never a reward, and never a win rate."""

    steps_done: int
    checkpoint_path: str
    metrics: dict[str, Any]
    wall_seconds: float
    iterations: int
    workspace: str
    reward_fingerprint: str


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    """Orthogonal init, the PPO default.

    Not cosmetic: with default PyTorch init the actor's logits start wide
    enough that an argmax policy is effectively a constant, which is how the
    predecessor project convinced itself a fresh network had a strategy.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


def build_mlp(in_dim: int, out_dim: int, hidden: Iterable[int], out_std: float) -> nn.Sequential:
    """A tanh MLP as `nn.Sequential`.

    The layout is fixed (Linear, Tanh, ..., Linear) because `objective.py` has
    to rebuild this network from checkpoint metadata *without importing this
    module* -- that independence is the point of the framework, and the price
    is that both sides construct an identical module by convention. The
    convention is: `actor`/`critic` sequentials, same hidden sizes, tanh.
    """
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in hidden:
        layers.append(_layer_init(nn.Linear(prev, int(h))))
        layers.append(nn.Tanh())
        prev = int(h)
    layers.append(_layer_init(nn.Linear(prev, int(out_dim)), std=out_std))
    return nn.Sequential(*layers)


def arch_from_spec(spec: EnvironmentSpec, hidden: Iterable[int]) -> dict[str, Any]:
    """The self-describing architecture record stored in every checkpoint."""
    policy = spec.policy
    if policy.obs_dim is None:
        raise ValueError(
            f"{spec.name}: policy.obs_dim is required to build a policy; declare it in the manifest"
        )
    if policy.kind == "discrete":
        if not policy.n_actions:
            raise ValueError(f"{spec.name}: discrete policy needs policy.n_actions")
        return {
            "kind": "discrete",
            "obs_dim": int(policy.obs_dim),
            "n_actions": int(policy.n_actions),
            "act_dim": 1,
            "hidden": [int(h) for h in hidden],
        }
    act_dim = int(policy.n_actions or 1)  # for continuous, n_actions means action dimensionality
    return {
        "kind": "continuous",
        "obs_dim": int(policy.obs_dim),
        "n_actions": None,
        "act_dim": act_dim,
        "hidden": [int(h) for h in hidden],
    }


class ActorCritic(nn.Module):
    """Separate actor and critic trunks, discrete or diagonal-Gaussian."""

    def __init__(self, arch: dict[str, Any]):
        super().__init__()
        self.arch = dict(arch)
        obs_dim = int(arch["obs_dim"])
        hidden = arch.get("hidden") or (128, 128)
        self.kind = str(arch["kind"])
        if self.kind == "discrete":
            self.n_actions = int(arch["n_actions"])
            self.actor = build_mlp(obs_dim, self.n_actions, hidden, out_std=0.01)
        else:
            self.act_dim = int(arch["act_dim"])
            self.actor = build_mlp(obs_dim, self.act_dim, hidden, out_std=0.01)
            self.log_std = nn.Parameter(torch.zeros(self.act_dim))
        self.critic = build_mlp(obs_dim, 1, hidden, out_std=1.0)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        out = self.actor(obs)
        if self.kind == "discrete":
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Normal(out, torch.exp(self.log_std).expand_as(out))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, *, deterministic: bool = False):
        dist = self.distribution(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1) if self.kind == "discrete" else dist.mean
        else:
            action = dist.sample()
        logprob = self._logprob(dist, action)
        return action, logprob, self._entropy(dist), self.value(obs)

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.distribution(obs)
        return self._logprob(dist, action), self._entropy(dist), self.value(obs)

    def _logprob(self, dist, action: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(action)
        return lp if self.kind == "discrete" else lp.sum(-1)

    def _entropy(self, dist) -> torch.Tensor:
        ent = dist.entropy()
        return ent if self.kind == "discrete" else ent.sum(-1)


# ---------------------------------------------------------------------------
# workspace bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class Workspace:
    """`/workspace` on a sandbox: the entire resumable state of a branch.

    A fork is a directory snapshot of this, so everything needed to continue
    must be on disk and nothing needed to continue may live only in memory.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def latest(self) -> Path:
        return self.checkpoints / "latest.pt"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    def step_checkpoint(self, step: int) -> Path:
        return self.checkpoints / f"step_{step:09d}.pt"

    def read_state(self) -> dict[str, Any]:
        if not self.state.exists():
            return {}
        try:
            return json.loads(self.state.read_text())
        except json.JSONDecodeError:  # a half-written state is no state
            return {}

    def write_state(self, payload: dict[str, Any]) -> None:
        tmp = self.state.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self.state)  # atomic: a snapshot never catches a torn file

    def append_metrics(self, line: dict[str, Any]) -> None:
        with self.metrics.open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    def rotate_metrics(self) -> Path | None:
        """Move an existing metrics stream aside before a from-scratch run.

        A restart that appends to the old stream makes `step` jump backwards
        halfway down the file, and every reader -- plots, pruning, the
        frontend -- silently misreads it. Keep the old one; just not here.
        """
        if not self.metrics.exists():
            return None
        stamp = time.strftime("%Y%m%dT%H%M%S")
        archived = self.root / f"metrics.{stamp}.jsonl"
        i = 1
        while archived.exists():
            archived = self.root / f"metrics.{stamp}.{i}.jsonl"
            i += 1
        os.replace(self.metrics, archived)
        return archived

    def read_metrics(self) -> list[dict[str, Any]]:
        if not self.metrics.exists():
            return []
        out = []
        for raw in self.metrics.read_text().splitlines():
            raw = raw.strip()
            if raw:
                try:
                    out.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return out


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    if "numpy" in state:
        np.random.set_state(tuple(state["numpy"]))
    if "python" in state:
        random.setstate(tuple(state["python"]))


def save_checkpoint(
    path: Path,
    *,
    net: ActorCritic,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    iteration: int,
    hyper: PPOHyper,
    spec: EnvironmentSpec,
    reward_fingerprint: str,
    wall_seconds: float,
) -> Path:
    """Write a checkpoint that `objective.py` can read on its own.

    `arch` travels with the weights so the evaluator never needs the trainer
    to tell it what shape the policy is -- the independence test forbids that
    import, so the checkpoint has to be self-describing.
    """
    payload = {
        "format": CHECKPOINT_FORMAT,
        "arch": dict(net.arch),
        "policy": net.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": int(step),
        "iteration": int(iteration),
        "rng": _rng_state(),
        "hyper": hyper.to_dict(),
        "env": {"name": spec.name, "objective": spec.objective.name, "entrypoint": spec.entrypoint},
        "reward_fingerprint": reward_fingerprint,
        "wall_seconds": float(wall_seconds),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def init_checkpoint(
    spec: EnvironmentSpec,
    path: str | os.PathLike,
    *,
    seed: int = 0,
    hyper: "PPOHyper | dict[str, Any] | None" = None,
) -> Path:
    """Write an untrained checkpoint.

    Useful on its own: it is the root of a branch lineage, and it is the only
    honest baseline for an objective score -- "what does this score before any
    training" is the number every later score should be read against.
    """
    hyp = PPOHyper.parse(hyper)
    torch.manual_seed(seed)
    net = ActorCritic(arch_from_spec(spec, hyp.hidden))
    return save_checkpoint(
        Path(path),
        net=net,
        optimizer=None,
        step=0,
        iteration=0,
        hyper=hyp,
        spec=spec,
        reward_fingerprint="",
        wall_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


def _to_matrix(obs: Any, n_envs: int) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(n_envs, -1)


def _broadcast(value: Any, n_envs: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_envs, float(arr))
    return arr.reshape(n_envs)


def _flags(value: Any, n_envs: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(n_envs, bool(arr))
    return arr.reshape(n_envs).astype(bool)


def train(
    spec: EnvironmentSpec,
    reward_config: RewardConfig,
    *,
    steps: int,
    workspace: str | os.PathLike,
    resume: bool = True,
    seed: int = 0,
    hyper: "PPOHyper | dict[str, Any] | None" = None,
    env: Any = None,
    search_path: str | os.PathLike | None = None,
) -> TrainResult:
    """Train `spec`'s policy for `steps` environment steps under a reward config.

    `steps` counts environment transitions across all lanes, which is the only
    unit comparable between branches with different `n_envs`. Training runs
    until `steps_done >= steps`, so the last collection may overshoot by up to
    `n_envs * n_steps`.

    Resuming is the default because a branch is a snapshot of `workspace`: a
    child mounts its parent's directory and calls `train` again, possibly with
    a different reward config over the inherited weights. That is a fork, not
    an error, so a changed reward fingerprint is recorded in `state.json`
    rather than rejected.

    `RewardError` from a malformed config propagates untouched -- the agent
    loop feeds it back as a correction turn.
    """
    hyp = PPOHyper.parse(hyper)
    ws = Workspace(Path(workspace))
    device = torch.device(hyp.device)
    if hyp.torch_threads:
        torch.set_num_threads(int(hyp.torch_threads))
    t0 = time.monotonic()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    compiled: CompiledReward = compile_reward(reward_config, spec)
    fingerprint = compiled.fingerprint

    owns_env = env is None
    if owns_env:
        env = load_env(spec.entrypoint, search_path=search_path, **dict(hyp.env_kwargs))
    n_envs = int(env.n_envs)

    arch = arch_from_spec(spec, hyp.hidden)
    net = ActorCritic(arch).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=hyp.lr, eps=1e-5)

    state = ws.read_state() if resume else {}
    if not resume:
        ws.rotate_metrics()
    steps_done = 0
    iteration = 0
    prior_wall = 0.0
    lineage: list[dict[str, Any]] = list(state.get("reward_lineage", []))
    if resume and state and ws.latest.exists():
        ckpt = torch.load(ws.latest, map_location=device, weights_only=False)
        if ckpt.get("arch", arch) != arch:
            raise ValueError(
                f"checkpoint at {ws.latest} was trained with arch {ckpt.get('arch')} "
                f"but this spec builds {arch}; a policy shape change needs a fresh workspace"
            )
        net.load_state_dict(ckpt["policy"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        steps_done = int(ckpt.get("step", state.get("steps_done", 0)))
        iteration = int(ckpt.get("iteration", state.get("iterations", 0)))
        prior_wall = float(state.get("wall_seconds", ckpt.get("wall_seconds", 0.0)))
        _restore_rng(ckpt.get("rng"))
    if not lineage or lineage[-1].get("fingerprint") != fingerprint:
        lineage.append({"fingerprint": fingerprint, "at_step": steps_done,
                        "hash": config_hash(reward_config)})

    # Env RNG cannot be snapshotted generically, so a resumed run restarts
    # episodes from a seed derived from the step count: reproducible, and
    # distinct from the seeds already consumed.
    obs = _to_matrix(env.reset(seed=seed + steps_done), n_envs)
    # Potential shaping keeps the previous signal value, so a new episode must
    # start without one or its first step fires a jolt measured against the
    # last episode's final state. Known limitation: lanes auto-reset
    # independently and `CompiledReward` has no per-lane reset, so a lane that
    # restarts mid-collection takes one such jolt. It is bounded by the signal
    # range and identical for every reward config being compared, which is why
    # it is a comment here rather than a hack in the reward compiler.
    compiled.reset_episode_state()

    obs_dim = obs.shape[1]
    if arch["obs_dim"] != obs_dim:
        raise ValueError(
            f"{spec.name}: env emits obs_dim {obs_dim} but manifest declares "
            f"{arch['obs_dim']}; run env_api.validate_env before training"
        )

    signal_names = list(spec.signal_names)
    act_shape = (hyp.n_steps, n_envs) if arch["kind"] == "discrete" else (
        hyp.n_steps, n_envs, arch["act_dim"])
    buf_obs = torch.zeros((hyp.n_steps, n_envs, obs_dim), dtype=torch.float32, device=device)
    buf_actions = torch.zeros(act_shape, dtype=torch.float32, device=device)
    buf_logprobs = torch.zeros((hyp.n_steps, n_envs), dtype=torch.float32, device=device)
    buf_rewards = torch.zeros((hyp.n_steps, n_envs), dtype=torch.float32, device=device)
    buf_dones = torch.zeros((hyp.n_steps, n_envs), dtype=torch.float32, device=device)
    buf_values = torch.zeros((hyp.n_steps, n_envs), dtype=torch.float32, device=device)

    ep_return = np.zeros(n_envs)
    ep_length = np.zeros(n_envs, dtype=np.int64)
    ep_signal_sums = np.zeros((n_envs, len(signal_names)))

    last_metrics: dict[str, Any] = {}
    next_checkpoint_at = steps_done + hyp.checkpoint_interval if hyp.checkpoint_interval else 0
    total_target = max(int(steps), 1)

    while steps_done < total_target:
        iteration += 1
        iter_start = time.monotonic()
        progress = min(1.0, steps_done / total_target)

        if hyp.anneal_lr:
            # Linear decay to zero at the step budget, the standard PPO schedule.
            for group in optimizer.param_groups:
                group["lr"] = hyp.lr * max(0.0, 1.0 - progress)

        finished_returns: list[float] = []
        finished_lengths: list[int] = []
        finished_signals: list[np.ndarray] = []
        signal_step_means: dict[str, float] = {name: 0.0 for name in signal_names}
        composed_reward_mean = 0.0

        for t in range(hyp.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            buf_obs[t] = obs_t
            with torch.no_grad():
                action, logprob, _, value = net.act(obs_t)
            buf_actions[t] = action.float()
            buf_logprobs[t] = logprob
            buf_values[t] = value

            np_action = action.cpu().numpy()
            result = env.step(np_action if arch["kind"] != "discrete" else np_action.astype(np.int64))
            if not isinstance(result, StepResult):
                raise TypeError(
                    f"{spec.entrypoint} returned {type(result).__name__} from step; "
                    "env_api.StepResult is required"
                )

            step_progress = min(1.0, (steps_done + (t + 1) * n_envs) / total_target)
            reward = _broadcast(compiled(result.signals, progress=step_progress), n_envs)

            terminated = _flags(result.terminated, n_envs)
            truncated = _flags(result.truncated, n_envs)
            # Episode returns are reported from the *composed* reward only. The
            # bootstrap below is a value-target correction, not something the
            # environment paid out, and folding it into the reported return
            # inflates the number a branch is read by.
            composed = reward.copy()
            if truncated.any() and INFO_FINAL_OBS in result.info:
                # A time limit is not a terminal state: bootstrap it, or the
                # value function learns that running out of clock is death.
                final = _to_matrix(result.info[INFO_FINAL_OBS], n_envs)[truncated]
                with torch.no_grad():
                    v_final = net.value(
                        torch.as_tensor(final, dtype=torch.float32, device=device)
                    ).cpu().numpy()
                reward[truncated] = reward[truncated] + hyp.gamma * v_final

            done = np.logical_or(terminated, truncated)
            buf_rewards[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            buf_dones[t] = torch.as_tensor(done.astype(np.float32), device=device)

            step_signals = np.stack(
                [_broadcast(result.signals.get(name, 0.0), n_envs) for name in signal_names], axis=1
            ) if signal_names else np.zeros((n_envs, 0))
            for j, name in enumerate(signal_names):
                signal_step_means[name] += float(step_signals[:, j].mean()) / hyp.n_steps

            composed_reward_mean += float(composed.mean()) / hyp.n_steps
            ep_return += composed
            ep_length += 1
            ep_signal_sums += step_signals
            if done.any():
                for lane in np.flatnonzero(done):
                    finished_returns.append(float(ep_return[lane]))
                    finished_lengths.append(int(ep_length[lane]))
                    finished_signals.append(ep_signal_sums[lane].copy())
                ep_return[done] = 0.0
                ep_length[done] = 0
                ep_signal_sums[done] = 0.0

            obs = _to_matrix(result.obs, n_envs)

        steps_done += hyp.n_steps * n_envs

        # ---- GAE ------------------------------------------------------
        with torch.no_grad():
            next_value = net.value(torch.as_tensor(obs, dtype=torch.float32, device=device))
            advantages = torch.zeros_like(buf_rewards)
            last_gae = torch.zeros(n_envs, device=device)
            for t in reversed(range(hyp.n_steps)):
                nonterminal = 1.0 - buf_dones[t]
                next_v = next_value if t == hyp.n_steps - 1 else buf_values[t + 1]
                delta = buf_rewards[t] + hyp.gamma * next_v * nonterminal - buf_values[t]
                last_gae = delta + hyp.gamma * hyp.gae_lambda * nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + buf_values

        batch = hyp.n_steps * n_envs
        b_obs = buf_obs.reshape(batch, obs_dim)
        b_actions = buf_actions.reshape((batch,) if arch["kind"] == "discrete" else (batch, -1))
        if arch["kind"] == "discrete":
            b_actions = b_actions.long()
        b_logprobs = buf_logprobs.reshape(batch)
        b_advantages = advantages.reshape(batch)
        b_returns = returns.reshape(batch)
        b_values = buf_values.reshape(batch)

        mb_size = max(1, batch // max(1, hyp.minibatches))
        indices = np.arange(batch)
        grad_steps = 0
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "loss": 0.0}
        stop_early = False
        for _ in range(hyp.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch, mb_size):
                mb = torch.as_tensor(indices[start:start + mb_size], dtype=torch.long, device=device)
                new_logprob, entropy, new_value = net.evaluate_actions(b_obs[mb], b_actions[mb])
                log_ratio = new_logprob - b_logprobs[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()  # k3 estimator, unbiased
                    clip_frac = ((ratio - 1.0).abs() > hyp.clip_coef).float().mean()

                mb_adv = b_advantages[mb]
                if hyp.norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - hyp.clip_coef, 1 + hyp.clip_coef),
                ).mean()

                if hyp.clip_vloss:
                    v_clipped = b_values[mb] + torch.clamp(
                        new_value - b_values[mb], -hyp.clip_coef, hyp.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        (new_value - b_returns[mb]) ** 2, (v_clipped - b_returns[mb]) ** 2
                    ).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + hyp.vf_coef * v_loss - hyp.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), hyp.max_grad_norm)
                optimizer.step()
                grad_steps += 1

                stats["policy_loss"] += float(pg_loss)
                stats["value_loss"] += float(v_loss)
                stats["entropy"] += float(entropy_loss)
                stats["approx_kl"] += float(approx_kl)
                stats["clip_frac"] += float(clip_frac)
                stats["loss"] += float(loss)

            if hyp.target_kl is not None and stats["approx_kl"] / max(1, grad_steps) > hyp.target_kl:
                stop_early = True
                break
        if grad_steps:
            stats = {k: v / grad_steps for k, v in stats.items()}

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        wall = prior_wall + (time.monotonic() - t0)
        ep_signals = (
            {name: float(np.mean([s[j] for s in finished_signals]))
             for j, name in enumerate(signal_names)}
            if finished_signals else {}
        )
        line = {
            "step": steps_done,
            "iteration": iteration,
            "progress": round(min(1.0, steps_done / total_target), 6),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_steps": grad_steps,
            "grad_steps_per_collection": hyp.grad_steps_per_collection,
            "early_stop_kl": stop_early,
            **{k: round(v, 8) for k, v in stats.items()},
            "explained_variance": explained_var,
            "reward_mean_per_step": round(composed_reward_mean, 8),
            "episodes": len(finished_returns),
            "episode_return_mean": float(np.mean(finished_returns)) if finished_returns else None,
            "episode_length_mean": float(np.mean(finished_lengths)) if finished_lengths else None,
            "episode_signal_means": ep_signals,
            "signal_step_means": {k: round(v, 8) for k, v in signal_step_means.items()},
            "reward_terms": compiled.term_totals(),
            "reward_fingerprint": fingerprint,
            "wall_seconds": round(wall, 3),
            # Throughput is diagnostics, never progress: the predecessor project
            # reported a healthy rate while every objective score was 0.000.
            "diagnostics": {
                "env_steps_per_second": round(
                    (hyp.n_steps * n_envs) / max(1e-9, time.monotonic() - iter_start), 1
                ),
                "n_envs": n_envs,
                # Includes the truncation bootstrap, so it is the quantity GAE
                # actually saw -- useful for debugging, not a return.
                "value_target_reward_mean": float(buf_rewards.mean()),
            },
        }
        ws.append_metrics(line)
        last_metrics = line

        should_checkpoint = (
            not hyp.checkpoint_interval
            or steps_done >= next_checkpoint_at
            or steps_done >= total_target
        )
        if should_checkpoint:
            save_checkpoint(
                ws.step_checkpoint(steps_done), net=net, optimizer=optimizer, step=steps_done,
                iteration=iteration, hyper=hyp, spec=spec, reward_fingerprint=fingerprint,
                wall_seconds=wall,
            )
            if hyp.checkpoint_interval:
                while next_checkpoint_at <= steps_done:
                    next_checkpoint_at += hyp.checkpoint_interval

        save_checkpoint(
            ws.latest, net=net, optimizer=optimizer, step=steps_done, iteration=iteration,
            hyper=hyp, spec=spec, reward_fingerprint=fingerprint, wall_seconds=wall,
        )
        ws.write_state({
            "env": spec.name,
            "objective": spec.objective.name,
            "steps_done": steps_done,
            "steps_target": total_target,
            "iterations": iteration,
            "wall_seconds": round(wall, 3),
            "seed": seed,
            "hyper": hyp.to_dict(),
            "reward_config": reward_config,
            "reward_fingerprint": fingerprint,
            "reward_lineage": lineage,
            "latest_checkpoint": str(ws.latest),
            "last_metrics": last_metrics,
        })

    if owns_env:
        env.close()

    wall = prior_wall + (time.monotonic() - t0)
    return TrainResult(
        steps_done=steps_done,
        checkpoint_path=str(ws.latest),
        metrics=summarize_metrics(ws.read_metrics()),
        wall_seconds=round(wall, 3),
        iterations=iteration,
        workspace=str(ws.root),
        reward_fingerprint=fingerprint,
    )


def summarize_metrics(lines: list[dict[str, Any]], *, tail: int = 5) -> dict[str, Any]:
    """Compact view of a run: the last line plus a tail mean of the noisy parts."""
    if not lines:
        return {}
    last = lines[-1]
    window = lines[-tail:]

    def mean_of(key: str) -> float | None:
        vals = [l[key] for l in window if isinstance(l.get(key), (int, float))]
        return float(np.mean(vals)) if vals else None

    return {
        "iterations": last.get("iteration"),
        "step": last.get("step"),
        "final": {k: last.get(k) for k in (
            "loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
            "explained_variance", "episode_return_mean", "episode_length_mean",
        )},
        "tail_mean": {k: mean_of(k) for k in (
            "loss", "entropy", "approx_kl", "clip_frac", "episode_return_mean",
            "reward_mean_per_step",
        )},
        "episode_signal_means": last.get("episode_signal_means", {}),
        "reward_terms_last": last.get("reward_terms", {}),
        "wall_seconds": last.get("wall_seconds"),
    }


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

TrainFn = Callable[..., TrainResult]


def trainer_for(spec: EnvironmentSpec) -> TrainFn:
    """Pick the training function a manifest asks for.

    Dispatch lives here rather than inside `train` so an unsupported trainer
    fails at branch launch with a message naming the phase it arrives in,
    instead of at the first gradient step.
    """
    kind = spec.trainer
    if kind == "ppo":
        return train
    if kind == "grpo_lora":
        # Imported here rather than at module scope on purpose: `posttrain`
        # needs torch, transformers and peft, none of which are in the base
        # image, and a bundle training with `ppo` must not pay for them.
        from autobranch import posttrain

        return posttrain.train
    if kind == "grpo_full":
        # Same reason as above, plus bitsandbytes. `grpo_full` updates every
        # parameter of the base model rather than an adapter over it, which is
        # what the reproduction target did; the RL method is identical.
        from autobranch import posttrain_full

        return posttrain_full.train
    if kind.startswith("custom:"):
        target = kind[len("custom:"):]
        fn = import_target(target)
        if not callable(fn):
            raise TypeError(f"custom trainer {target!r} resolved to a non-callable {type(fn)}")
        return fn
    raise ValueError(
        f"unknown trainer {kind!r}; expected ppo | grpo_lora | grpo_full | custom:module:fn"
    )


__all__ = [
    "ActorCritic",
    "PPOHyper",
    "TrainResult",
    "Workspace",
    "arch_from_spec",
    "build_mlp",
    "init_checkpoint",
    "save_checkpoint",
    "summarize_metrics",
    "train",
    "trainer_for",
]
