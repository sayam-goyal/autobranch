"""LoRA post-training with critic-free RL (GRPO).

Selected when a constraint says "must posttrain <repo id>". Same signature and
TrainResult as the PPO trainer, so nothing above it knows which it got.

An action is one masked token: logits at the last position are restricted to
the action vocabulary, so an illegal action cannot be represented and the
reward never needs an invalid-move penalty.

No critic. Advantages are group-relative over G samples of a state. A group
whose samples all score alike teaches nothing, so
`zero_variance_group_fraction` is in every metrics line.

Base weights are snapshotted to the Volume once and reused. torch,
transformers, peft and huggingface_hub are optional, so every import of them
is function-local and names the manifest key to add when missing.
"""

import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from autobranch.env_api import StepResult, load_env
from autobranch.reward import CompiledReward, RewardConfig, compile_reward, config_hash
from autobranch.spec import EnvironmentSpec

# Everything needed to *run* a post-trained policy lives in `lm_policy`, which
# imports no reward code, so `objective.py` can import it without breaching the
# independence wall. Re-exported here so this module's public API -- and every
# caller of it -- is unchanged by that split.
from autobranch.lm_policy import (  # noqa: F401 - re-exported API
    ADAPTER_DIRNAME,
    CHECKPOINT_FORMAT,
    DEFAULT_DATA_ROOT,
    DOWNLOAD_MARKER,
    LABEL_POOL,
    MODELS_DIRNAME,
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    REQUIRED_PIP,
    ActionVocab,
    LoraPolicy,
    MissingDependencyError,
    PostTrainError,
    _best_effort_commit,
    _default_downloader,
    _hf_token,
    _import_or_explain,
    _last_token_logits,
    _prepare_tokenizer,
    _resolve_device,
    _to_numpy,
    _tokenize,
    _torch_dtype,
    action_labels,
    build_action_vocab,
    default_data_root,
    ensure_base_model,
    escape_repo_id,
    is_model_ready,
    load_policy,
    log_softmax,
    model_dir_for,
    require_dependencies,
    restrict_logits,
    sample_action_indices,
    serialize_observation,
)

if TYPE_CHECKING:  # the runtime import is lazy: `trainer` imports torch at module scope
    from autobranch.trainer import TrainResult














# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# base model weights on the Volume
# ---------------------------------------------------------------------------


















# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GRPOHyper:
    """Everything tunable, with defaults sized for a small base model.

    Like `PPOHyper`, the number of gradient steps bought per collection is
    explicit (`epochs * minibatches`) and written into every metrics line: an
    under-trained collection looks exactly like a bad reward from the outside,
    and the predecessor project lost six runs to that confusion.
    """

    #: Samples per group. The entire baseline comes from these, so 1 is not a
    #: valid setting -- a group of one has no mean to be relative to.
    group_size: int = 4
    #: Hard cap on a rollout, in environment steps. A group is `group_size`
    #: rollouts from the *same* episode seed, which is what makes their
    #: returns comparable.
    episode_steps: int = 16
    epochs: int = 2
    minibatches: int = 2
    #: Prompts per forward pass during the gradient phase. Separate from
    #: `minibatches` because the limit here is GPU memory, not statistics.
    micro_batch: int = 8
    lr: float = 1e-4
    #: KL(policy || reference) penalty, the GRPO regulariser. The reference is
    #: the base model with the adapter switched off, so it costs no extra
    #: weights -- and that is also why it is cheap enough to always report.
    kl_coef: float = 0.02
    #: Entropy bonus. Zero by default: a pretrained LM starts with a sensible
    #: distribution over a ten-symbol vocabulary and pushing entropy up mostly
    #: un-trains it.
    ent_coef: float = 0.0
    clip_coef: float = 0.2
    adv_eps: float = 1e-6
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    #: A zero-variance group contributes exactly zero advantage to every
    #: sample, so including it only dilutes the gradient mean. Dropped by
    #: default, and counted in the metrics either way.
    drop_zero_variance_groups: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    #: Empty means "peft's own default for this architecture", which is the
    #: right answer far more often than a hand-written list.
    lora_target_modules: tuple[str, ...] = ()
    prompt_precision: int = 2
    max_prompt_tokens: int = 1024
    device: str = "auto"
    dtype: str = "auto"
    checkpoint_interval: int = 0  # env steps; 0 -> one checkpoint per collection
    torch_threads: int | None = 1
    env_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.group_size) < 2:
            raise ValueError(
                f"group_size must be at least 2 -- group-relative advantages are the only "
                f"baseline this trainer has -- got {self.group_size}"
            )
        if int(self.episode_steps) < 1:
            raise ValueError(f"episode_steps must be positive, got {self.episode_steps}")
        if float(self.temperature) <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    @property
    def grad_steps_per_collection(self) -> int:
        """How many optimiser steps each collection actually buys."""
        return int(self.epochs) * int(self.minibatches)

    @staticmethod
    def parse(raw: "GRPOHyper | dict[str, Any] | None") -> "GRPOHyper":
        if raw is None:
            return GRPOHyper()
        if isinstance(raw, GRPOHyper):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"hyper must be GRPOHyper, dict or None, got {type(raw).__name__}")
        known = set(GRPOHyper.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown hyperparameters {sorted(unknown)}; known: {sorted(known)}")
        kwargs = dict(raw)
        if "lora_target_modules" in kwargs:
            kwargs["lora_target_modules"] = tuple(
                str(m) for m in (kwargs["lora_target_modules"] or ())
            )
        return GRPOHyper(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lora_target_modules"] = list(self.lora_target_modules)
        d["grad_steps_per_collection"] = self.grad_steps_per_collection
        return d


# ---------------------------------------------------------------------------
# workspace bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class PostTrainWorkspace:
    """`/workspace` on a sandbox: the whole resumable state of a branch.

    Deliberately a near-copy of `trainer.Workspace` rather than an import of
    it: `autobranch.trainer` imports torch at module scope, and this module
    has to be importable in an image that has none of the heavy dependencies.
    The layout is the same so a fork, a snapshot and the frontend all see one
    shape, plus `adapter/` -- the LoRA weights, which are what is actually
    being trained here.
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
    def adapter(self) -> Path:
        return self.root / ADAPTER_DIRNAME

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    def step_checkpoint(self, step: int) -> Path:
        return self.checkpoints / f"step_{step:09d}.pt"

    def has_adapter(self) -> bool:
        """Peft's own marker file -- the only honest test of a usable adapter."""
        return (self.adapter / "adapter_config.json").is_file()

    def read_state(self) -> dict[str, Any]:
        if not self.state.exists():
            return {}
        try:
            return json.loads(self.state.read_text())
        except json.JSONDecodeError:  # a half-written state is no state
            return {}

    def write_state(self, payload: dict[str, Any]) -> None:
        tmp = self.state.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        os.replace(tmp, self.state)  # atomic: a snapshot never catches a torn file

    def append_metrics(self, line: dict[str, Any]) -> None:
        with self.metrics.open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True, default=str) + "\n")

    def rotate_metrics(self) -> Path | None:
        """Move an existing metrics stream aside before a from-scratch run.

        A restart that appends to the old stream makes `step` jump backwards
        halfway down the file, and every reader -- plots, pruning, the
        frontend -- silently misreads it.
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


@dataclass
class Progress:
    """Where a resumed run picks up. Read from `state.json`, never inferred."""

    steps_done: int = 0
    iteration: int = 0
    wall_seconds: float = 0.0
    reward_lineage: list[dict[str, Any]] = field(default_factory=list)
    resumed: bool = False


def restore_progress(ws: PostTrainWorkspace, *, resume: bool = True) -> Progress:
    """Continue from `state.json`, or start clean and rotate the metrics.

    Separate from `train` and free of heavy imports so the resume contract can
    be tested on a machine with no torch -- the behaviour that matters here is
    "a child mounting its parent's workspace continues the step count", and
    that has nothing to do with tensors.
    """
    if not resume:
        ws.rotate_metrics()
        return Progress()
    state = ws.read_state()
    if not state:
        return Progress()
    return Progress(
        steps_done=int(state.get("steps_done", 0)),
        iteration=int(state.get("iterations", 0)),
        wall_seconds=float(state.get("wall_seconds", 0.0)),
        reward_lineage=list(state.get("reward_lineage", [])),
        resumed=True,
    )


def record_lineage(progress: Progress, reward_config: RewardConfig, fingerprint: str) -> None:
    """Append a reward change to the branch's lineage.

    A fork is a resumed workspace with a *different* reward over inherited
    weights, so a changed fingerprint is history to record rather than an error
    to raise.
    """
    lineage = progress.reward_lineage
    if not lineage or lineage[-1].get("fingerprint") != fingerprint:
        lineage.append(
            {
                "fingerprint": fingerprint,
                "at_step": int(progress.steps_done),
                "hash": config_hash(reward_config),
            }
        )


# ---------------------------------------------------------------------------
# the action vocabulary -- the reason an illegal action cannot happen
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# observations as prompts -- one function, on purpose
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# group-relative advantages -- the whole of "critic-free"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupStats:
    """What a collection's groups looked like, including what went wrong."""

    n_groups: int
    group_size: int
    return_mean: float
    return_std: float
    #: Mean of the per-group standard deviations. Falls to zero exactly when
    #: the groups stop disagreeing, which is when learning stops.
    within_group_std_mean: float
    #: Groups where every sample scored identically. They contribute no
    #: gradient at all; reported because a run can look busy while every
    #: group is degenerate.
    zero_variance_fraction: float
    zero_variance_groups: int


def zero_variance_mask(returns: Any, *, atol: float = 1e-12) -> np.ndarray:
    """Which groups have nothing to teach: every sample scored identically.

    One function so the number in the metrics line and the samples the
    gradient phase actually drops are the same set -- a reported fraction that
    does not match what was skipped is worse than no fraction at all.
    """
    arr = np.asarray(returns, dtype=np.float64)
    return arr.std(axis=1) <= float(atol)


def group_advantages(
    returns: Any, *, eps: float = 1e-6, zero_variance_atol: float = 1e-12
) -> tuple[np.ndarray, GroupStats]:
    """`(r - mean(group)) / (std(group) + eps)` for each sample in each group.

    `returns` is `[n_groups, group_size]`: one row per prompt state, one column
    per sampled completion. No critic, no value targets, no bootstrapping --
    the group *is* the baseline, which is the entire reason this trainer can
    post-train a 4B model without carrying a second one.

    A group whose samples all scored the same gets exactly zero advantage,
    which is correct (it contains no preference) and worth counting, so it
    comes back in `GroupStats` instead of being silently averaged away.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"returns must be [n_groups, group_size], got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"group_size {arr.shape[1]} leaves nothing to be relative to; GRPO needs at least 2"
        )
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)  # population std: the group is the population
    scale = std + float(eps)
    # `where` rather than trusting eps: a caller passing eps=0 on a degenerate
    # group would otherwise get 0/0, and a NaN advantage poisons every
    # parameter it touches instead of simply teaching nothing.
    adv = np.divide(arr - mean, scale, out=np.zeros_like(arr), where=scale > 0)
    degenerate = zero_variance_mask(arr, atol=zero_variance_atol)
    stats = GroupStats(
        n_groups=int(arr.shape[0]),
        group_size=int(arr.shape[1]),
        return_mean=float(arr.mean()),
        return_std=float(arr.std()),
        within_group_std_mean=float(std.mean()),
        zero_variance_fraction=float(degenerate.mean()),
        zero_variance_groups=int(degenerate.sum()),
    )
    return adv, stats


# ---------------------------------------------------------------------------
# checkpoints the evaluator can rebuild a policy from
# ---------------------------------------------------------------------------


def lora_arch(
    spec: EnvironmentSpec,
    hyp: GRPOHyper,
    *,
    base_model: str,
    base_model_path: str,
    vocab: ActionVocab,
    dtype: str,
) -> dict[str, Any]:
    """The self-describing architecture record stored in every checkpoint.

    Same principle as `trainer.arch_from_spec`: everything needed to rebuild
    the policy travels with the weights, so whoever scores it never has to ask
    the trainer what shape it is. For a LoRA policy "everything" is the base
    model repo id, where the adapter sits, the action vocabulary, and the exact
    prompt -- miss any one and you have rebuilt a different policy.
    """
    policy = spec.policy
    if policy.kind != "discrete":
        raise PostTrainError(
            f"{spec.name}: policy.kind is {policy.kind!r}; grpo_lora drives a discrete action "
            "vocabulary (one token per action) and has no continuous head"
        )
    if not policy.n_actions:
        raise PostTrainError(f"{spec.name}: discrete policy needs policy.n_actions")
    if policy.obs_dim is None:
        raise PostTrainError(
            f"{spec.name}: policy.obs_dim is required -- it is the length of the state vector the "
            "prompt serialises; declare it in the manifest"
        )
    return {
        "kind": "lm_lora_discrete",
        "obs_dim": int(policy.obs_dim),
        "n_actions": int(policy.n_actions),
        "act_dim": 1,
        "base_model": str(base_model),
        "base_model_path": str(base_model_path),
        "adapter": ADAPTER_DIRNAME,  # relative to the workspace root
        "action_vocab": vocab.to_dict(),
        "prompt": {
            "version": PROMPT_VERSION,
            "template": PROMPT_TEMPLATE,
            "precision": int(hyp.prompt_precision),
            "env_name": spec.name,
            # As rendered: `serialize_observation` strips the labels, because a
            # tokenizer may have given us " A" where "A" was not a single token.
            "labels": [l.strip() for l in vocab.labels],
            "max_tokens": int(hyp.max_prompt_tokens),
        },
        "dtype": str(dtype),
        "lora": {
            "r": int(hyp.lora_r),
            "alpha": int(hyp.lora_alpha),
            "dropout": float(hyp.lora_dropout),
            "target_modules": list(hyp.lora_target_modules),
        },
        "temperature": float(hyp.temperature),
    }


def checkpoint_payload(
    *,
    arch: dict[str, Any],
    adapter_state: dict[str, Any],
    adapter_path: str,
    step: int,
    iteration: int,
    hyper: GRPOHyper,
    spec: EnvironmentSpec,
    reward_fingerprint: str,
    wall_seconds: float,
    optimizer_state: Any = None,
    rng: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the checkpoint record. Pure, so its shape is testable torch-free.

    `policy` holds the adapter tensors themselves, so the file is standalone:
    the trained parameters are in it, not only a path to them. Note what this
    deliberately is *not*: `objective.load_checkpoint_policy` rebuilds a dense
    MLP from `arch` and loads `policy` with `strict=True`, which these tensors
    will not satisfy -- it raises instead of scoring a randomly initialised
    network. That loud failure is the intended behaviour; `load_policy` below
    is the loader for this kind of checkpoint, and it needs no reward code,
    no trainer state and no manifest.
    """
    return {
        "format": CHECKPOINT_FORMAT,
        "trainer": "grpo_lora",
        "arch": dict(arch),
        "policy": dict(adapter_state),
        "adapter_path": str(adapter_path),
        "optimizer": optimizer_state,
        "step": int(step),
        "iteration": int(iteration),
        "rng": rng or {},
        "hyper": hyper.to_dict(),
        "env": {
            "name": spec.name,
            "objective": spec.objective.name,
            "entrypoint": spec.entrypoint,
        },
        "reward_fingerprint": reward_fingerprint,
        "wall_seconds": float(wall_seconds),
    }


def _save_adapter(model: Any, adapter_dir: Path) -> None:
    """Write the LoRA adapter so an interrupted save cannot be half-read.

    Peft writes several files; a reader that catches the directory mid-write
    sees a config without weights. So the save goes to a scratch directory and
    is swapped in. A crash during the swap leaves no adapter, which `resume`
    detects (no `adapter_config.json`) and handles by starting the adapter
    fresh -- recoverable. A half-written adapter is not.
    """
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = adapter_dir.with_name(f".{adapter_dir.name}.saving-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    model.save_pretrained(str(staging))
    previous = adapter_dir.with_name(f".{adapter_dir.name}.previous-{os.getpid()}")
    shutil.rmtree(previous, ignore_errors=True)
    if adapter_dir.exists():
        os.replace(adapter_dir, previous)
    os.replace(staging, adapter_dir)
    shutil.rmtree(previous, ignore_errors=True)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    """Atomic `torch.save`, so a snapshot never catches a torn checkpoint."""
    torch = _import_or_explain("torch", "torch>=2.2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path






# ---------------------------------------------------------------------------
# torch plumbing, all behind lazy imports
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


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One (state, action) the policy committed to, with its group's identity.

    `lane` and `group` locate it in the `[n_envs, group_size]` return matrix:
    every sample from lane 3's second rollout shares that rollout's return,
    which is how a trajectory-level return becomes a per-action advantage
    without a critic to apportion credit.
    """

    prompt: str
    action: int
    behaviour_logprob: float
    lane: int
    group: int


def _collect(
    *,
    env: Any,
    spec: EnvironmentSpec,
    compiled: CompiledReward,
    model: Any,
    tokenizer: Any,
    vocab: ActionVocab,
    hyp: GRPOHyper,
    device: str,
    episode_seed: int,
    rng: np.random.Generator,
    steps_done: int,
    total_target: int,
) -> dict[str, Any]:
    """Run `group_size` rollouts from one episode seed and return the batch.

    The group is `group_size` rollouts of the *same* episodes: every rollout
    resets the environment with the same seed, so lane `l`'s returns differ
    only because the policy sampled differently. That is what makes
    `(r - mean) / std` within a lane a comparison of actions rather than a
    comparison of episodes -- the usual way a group-relative method gets
    quietly broken is by grouping over different states.
    """
    returns = np.zeros((int(env.n_envs), int(hyp.group_size)), dtype=np.float64)
    samples: list[Sample] = []
    signal_names = list(spec.signal_names)
    signal_sums = {name: 0.0 for name in signal_names}

    # Collection runs in eval mode even though the model is being trained:
    # with dropout live, the behaviour logprob recorded here is not the
    # logprob of the policy the gradient phase computes its ratio against, and
    # the ratio silently stops meaning what the clipping assumes.
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        return _rollouts(
            env=env, spec=spec, compiled=compiled, model=model, tokenizer=tokenizer,
            vocab=vocab, hyp=hyp, device=device, episode_seed=episode_seed, rng=rng,
            steps_done=steps_done, total_target=total_target, returns=returns,
            samples=samples, signal_names=signal_names, signal_sums=signal_sums,
        )
    finally:
        if was_training:
            model.train()


def _rollouts(
    *,
    env: Any,
    spec: EnvironmentSpec,
    compiled: CompiledReward,
    model: Any,
    tokenizer: Any,
    vocab: ActionVocab,
    hyp: GRPOHyper,
    device: str,
    episode_seed: int,
    rng: np.random.Generator,
    steps_done: int,
    total_target: int,
    returns: np.ndarray,
    samples: list,
    signal_names: list,
    signal_sums: dict,
) -> dict[str, Any]:
    """The rollout loop itself. Split out only so `_collect` can own the mode flip."""
    torch = _import_or_explain("torch", "torch>=2.2")
    n_envs = int(env.n_envs)
    env_steps = 0
    episodes_finished = 0
    episode_lengths: list[int] = []
    prompt_chars = 0

    for g in range(int(hyp.group_size)):
        obs = _to_matrix(env.reset(seed=int(episode_seed)), n_envs)
        # Potential shaping keeps the previous signal value, so a new episode
        # must start without one or its first step fires a jolt measured
        # against the last episode's final state.
        compiled.reset_episode_state()
        active = np.ones(n_envs, dtype=bool)
        length = np.zeros(n_envs, dtype=np.int64)

        for t in range(int(hyp.episode_steps)):
            if not active.any():
                break
            # Only live lanes are forwarded. A lane that already finished has
            # been auto-reset into a *new* episode, and this rollout's return
            # must not contain it -- so its prompt is not worth a forward pass
            # either. It still needs an action, because the environment is
            # vectorised; action 0 is a placeholder whose consequences are
            # never read.
            live = np.flatnonzero(active)
            prompts = [
                serialize_observation(
                    obs[i],
                    labels=vocab.labels,
                    env_name=spec.name,
                    precision=hyp.prompt_precision,
                )
                for i in live
            ]
            prompt_chars += sum(len(p) for p in prompts)
            with torch.no_grad():
                logits = _last_token_logits(
                    model, tokenizer, prompts, device=device, max_tokens=hyp.max_prompt_tokens
                )
            action_logits = restrict_logits(logits, vocab)
            sampled = sample_action_indices(action_logits, rng, temperature=hyp.temperature)
            logprobs = log_softmax(action_logits, temperature=hyp.temperature)
            actions = np.zeros(n_envs, dtype=np.int64)
            actions[live] = sampled

            result = env.step(actions)
            if not isinstance(result, StepResult):
                raise TypeError(
                    f"{spec.entrypoint} returned {type(result).__name__} from step; "
                    "env_api.StepResult is required"
                )

            step_progress = min(1.0, (steps_done + env_steps + int(active.sum())) / total_target)
            reward = _broadcast(compiled(result.signals, progress=step_progress), n_envs)
            terminated = _flags(result.terminated, n_envs)
            truncated = _flags(result.truncated, n_envs)
            done = terminated | truncated

            for k, i in enumerate(live):
                returns[i, g] += float(reward[i])
                samples.append(
                    Sample(
                        prompt=prompts[k],
                        action=int(sampled[k]),
                        behaviour_logprob=float(logprobs[k, int(sampled[k])]),
                        lane=int(i),
                        group=g,
                    )
                )
            for name in signal_names:
                values = _broadcast(result.signals.get(name, 0.0), n_envs)
                signal_sums[name] += float(values[active].sum())
            length[active] += 1
            env_steps += int(active.sum())

            finished = active & done
            if finished.any():
                episodes_finished += int(finished.sum())
                episode_lengths.extend(int(x) for x in length[finished])
            # A lane that finished is done contributing to this rollout: the
            # env auto-resets it, and counting the next episode's steps into
            # this group's return would mix two episodes into one number.
            active = active & ~done
            obs = _to_matrix(result.obs, n_envs)

    return {
        "returns": returns,
        "samples": samples,
        "env_steps": env_steps,
        "episodes": episodes_finished,
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
        "signal_means": {
            name: (total / env_steps if env_steps else 0.0) for name, total in signal_sums.items()
        },
        "prompt_chars_mean": prompt_chars / max(1, len(samples)),
    }


def _gradient_phase(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    vocab: ActionVocab,
    hyp: GRPOHyper,
    device: str,
    samples: Sequence[Sample],
    advantages: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """`epochs * minibatches` GRPO steps over one collection.

    The loss is the clipped policy-gradient term on group-relative advantages
    plus `kl_coef * KL(policy || reference)`, where the reference is this same
    model with the adapter switched off. No value loss, because there is no
    value function.
    """
    torch = _import_or_explain("torch", "torch>=2.2")
    n = len(samples)
    token_ids = torch.as_tensor(list(vocab.token_ids), dtype=torch.long, device=device)
    stats = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl_to_reference": 0.0,
        "entropy": 0.0,
        "clip_frac": 0.0,
        "ratio_mean": 0.0,
    }
    grad_steps = 0
    mb_size = max(1, n // max(1, int(hyp.minibatches)))

    for _ in range(int(hyp.epochs)):
        order = rng.permutation(n)
        for start in range(0, n, mb_size):
            chunk = order[start:start + mb_size]
            if chunk.size == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            accumulated = {k: 0.0 for k in stats}
            micro = max(1, int(hyp.micro_batch))
            n_micro = 0
            for m_start in range(0, chunk.size, micro):
                idx = chunk[m_start:m_start + micro]
                prompts = [samples[i].prompt for i in idx]
                inputs = _tokenize(
                    tokenizer, prompts, device=device, max_tokens=hyp.max_prompt_tokens
                )
                act = torch.as_tensor([samples[i].action for i in idx], dtype=torch.long, device=device)
                beh = torch.as_tensor(
                    [samples[i].behaviour_logprob for i in idx], dtype=torch.float32, device=device
                )
                adv = torch.as_tensor(
                    [float(advantages[i]) for i in idx], dtype=torch.float32, device=device
                )

                logits = model(**inputs).logits[:, -1, :]
                action_logits = logits.index_select(-1, token_ids).float() / float(hyp.temperature)
                logp = torch.log_softmax(action_logits, dim=-1)
                chosen = logp.gather(1, act.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    # The reference policy is the base model: same weights,
                    # adapter switched off. A frozen second copy would double
                    # the memory for an identical number.
                    with model.disable_adapter():
                        ref_logits = model(**inputs).logits[:, -1, :]
                    ref_logp = torch.log_softmax(
                        ref_logits.index_select(-1, token_ids).float() / float(hyp.temperature),
                        dim=-1,
                    )

                probs = logp.exp()
                kl = (probs * (logp - ref_logp)).sum(-1).mean()
                entropy = -(probs * logp).sum(-1).mean()
                log_ratio = chosen - beh
                ratio = log_ratio.exp()
                pg = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1.0 - hyp.clip_coef, 1.0 + hyp.clip_coef),
                ).mean()
                loss = pg + hyp.kl_coef * kl - hyp.ent_coef * entropy
                # Scaled so the accumulated micro-batches average rather than
                # sum: otherwise the effective learning rate depends on how
                # the minibatch happened to be split for memory.
                weight = idx.size / max(1, chunk.size)
                (loss * weight).backward()

                accumulated["loss"] += float(loss) * weight
                accumulated["policy_loss"] += float(pg) * weight
                accumulated["kl_to_reference"] += float(kl) * weight
                accumulated["entropy"] += float(entropy) * weight
                accumulated["ratio_mean"] += float(ratio.mean()) * weight
                accumulated["clip_frac"] += float(
                    ((ratio - 1.0).abs() > hyp.clip_coef).float().mean()
                ) * weight
                n_micro += 1

            if n_micro == 0:
                continue
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, hyp.max_grad_norm)
            optimizer.step()
            grad_steps += 1
            for k, v in accumulated.items():
                stats[k] += v

    if grad_steps:
        stats = {k: v / grad_steps for k, v in stats.items()}
    stats["grad_steps"] = grad_steps
    return stats


def train(
    spec: EnvironmentSpec,
    reward_config: RewardConfig,
    *,
    steps: int,
    workspace: str | os.PathLike,
    resume: bool = True,
    seed: int = 0,
    hyper: "GRPOHyper | dict[str, Any] | None" = None,
    env: Any = None,
    search_path: str | os.PathLike | None = None,
) -> "TrainResult":
    """Post-train `spec.policy.base_model` with GRPO under a reward config.

    Signature and return type are identical to `trainer.train` because
    `branch_main` calls whichever one `trainer_for(spec)` hands it, with the
    same keywords. In particular `steps` is a cumulative **target** -- train
    until `steps_done >= steps` -- so a second call on a resumed workspace
    asking for the same number is a no-op rather than a doubling, which is
    what the branch loop relies on.
    """
    hyp = GRPOHyper.parse(hyper)
    ws = PostTrainWorkspace(Path(workspace))

    base_model = spec.policy.base_model
    if not base_model:
        raise PostTrainError(
            f"{spec.name}: trainer 'grpo_lora' needs a base model, but policy.base_model is unset. "
            "Set it in the manifest, or give the run a constraint like "
            "'must posttrain Qwen/Qwen3-4B' -- that is what fills it in."
        )

    # Dependencies first: a run that is going to die of a missing package
    # should die before it loads an environment or downloads 9 GB of weights.
    deps = require_dependencies()
    torch = deps["torch"]
    transformers = deps["transformers"]
    peft = deps["peft"]

    t0 = time.monotonic()
    if hyp.torch_threads:
        torch.set_num_threads(int(hyp.torch_threads))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    compiled: CompiledReward = compile_reward(reward_config, spec)
    fingerprint = compiled.fingerprint

    # Resume state is read before anything heavy, because the one thing it can
    # veto is a base model change -- and discovering that *after* downloading
    # the wrong 9 GB would be an expensive way to learn it.
    progress = restore_progress(ws, resume=resume)
    if progress.resumed:
        inherited = ws.read_state().get("base_model")
        if inherited and inherited != base_model:
            raise PostTrainError(
                f"workspace {ws.root} holds an adapter post-trained from {inherited!r} but this "
                f"spec asks for {base_model!r}; a base model change needs a fresh workspace"
            )

    owns_env = env is None
    if owns_env:
        env = load_env(spec.entrypoint, search_path=search_path, **dict(hyp.env_kwargs))
    n_envs = int(env.n_envs)

    local_path = ensure_base_model(base_model)
    device = _resolve_device(torch, hyp.device)
    dtype = _torch_dtype(torch, hyp.dtype, device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(local_path)
    _prepare_tokenizer(tokenizer)
    vocab = build_action_vocab(tokenizer, int(spec.policy.n_actions or 0), model_name=base_model)
    arch = lora_arch(
        spec,
        hyp,
        base_model=base_model,
        base_model_path=local_path,
        vocab=vocab,
        dtype=str(dtype).replace("torch.", ""),
    )

    base = transformers.AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=dtype)
    base.config.use_cache = False  # a single forward per step; the cache is dead weight

    if progress.resumed and ws.has_adapter():
        model = peft.PeftModel.from_pretrained(base, str(ws.adapter), is_trainable=True)
    else:
        lora_kwargs: dict[str, Any] = {
            "r": int(hyp.lora_r),
            "lora_alpha": int(hyp.lora_alpha),
            "lora_dropout": float(hyp.lora_dropout),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if hyp.lora_target_modules:
            lora_kwargs["target_modules"] = list(hyp.lora_target_modules)
        model = peft.get_peft_model(base, peft.LoraConfig(**lora_kwargs))
    model.to(device)
    # Adam moments on bf16 parameters round small updates to nothing, which
    # looks exactly like a learning rate that is too low. The adapter is a few
    # million parameters, so fp32 for it is free.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise PostTrainError(
            "the LoRA adapter exposes no trainable parameters; check "
            "hyper.lora_target_modules against this architecture"
        )
    optimizer = torch.optim.AdamW(trainable, lr=hyp.lr)

    steps_done = progress.steps_done
    iteration = progress.iteration
    prior_wall = progress.wall_seconds
    if progress.resumed and ws.latest.exists():
        previous = torch.load(ws.latest, map_location=device, weights_only=False)
        if previous.get("optimizer"):
            optimizer.load_state_dict(previous["optimizer"])
        # Belt and braces: the checkpoint's own record, not just state.json.
        recorded = (previous.get("arch") or {}).get("base_model")
        if recorded and recorded != base_model:
            raise PostTrainError(
                f"checkpoint {ws.latest} was post-trained from {recorded!r} but this spec asks "
                f"for {base_model!r}; a base model change needs a fresh workspace"
            )
    record_lineage(progress, reward_config, fingerprint)

    total_target = max(int(steps), 1)
    next_checkpoint_at = steps_done + hyp.checkpoint_interval if hyp.checkpoint_interval else 0
    last_metrics: dict[str, Any] = {}
    n_actions = int(spec.policy.n_actions or 0)

    while steps_done < total_target:
        iteration += 1
        iter_start = time.monotonic()
        # Sampling is a pure function of (seed, iteration): both are in
        # state.json, so a resumed run continues the stream instead of
        # replaying it, with no generator state to serialise.
        rng = np.random.default_rng([int(seed), int(iteration)])

        batch = _collect(
            env=env,
            spec=spec,
            compiled=compiled,
            model=model,
            tokenizer=tokenizer,
            vocab=vocab,
            hyp=hyp,
            device=device,
            episode_seed=int(seed) + steps_done,
            rng=rng,
            steps_done=steps_done,
            total_target=total_target,
        )
        steps_done += int(batch["env_steps"])

        if not batch["env_steps"]:
            raise PostTrainError(
                f"{spec.name}: a collection produced no environment steps, so `steps` could never "
                "be reached -- every lane was already done at reset. Check the env's reset()."
            )

        adv_matrix, gstats = group_advantages(batch["returns"], eps=hyp.adv_eps)
        samples: list[Sample] = batch["samples"]
        degenerate = zero_variance_mask(batch["returns"])
        if hyp.drop_zero_variance_groups:
            keep = [i for i, s in enumerate(samples) if not degenerate[s.lane]]
        else:
            keep = list(range(len(samples)))
        kept = [samples[i] for i in keep]
        sample_adv = np.asarray([adv_matrix[s.lane, s.group] for s in kept], dtype=np.float64)

        if kept:
            stats = _gradient_phase(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                vocab=vocab,
                hyp=hyp,
                device=device,
                samples=kept,
                advantages=sample_adv,
                rng=rng,
            )
        else:
            # Every group was degenerate: nothing to learn from, and saying so
            # is the point. A zero-filled line keeps the metrics stream
            # continuous so the gap is visible rather than missing.
            stats = {
                "loss": 0.0, "policy_loss": 0.0, "kl_to_reference": 0.0, "entropy": 0.0,
                "clip_frac": 0.0, "ratio_mean": 1.0, "grad_steps": 0,
            }

        wall = prior_wall + (time.monotonic() - t0)
        elapsed = max(1e-9, time.monotonic() - iter_start)
        line = {
            "step": steps_done,
            "iteration": iteration,
            "progress": round(min(1.0, steps_done / total_target), 6),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_steps": int(stats.pop("grad_steps", 0)),
            "grad_steps_per_collection": hyp.grad_steps_per_collection,
            **{k: round(float(v), 8) for k, v in stats.items()},
            # The three numbers that say whether GRPO had anything to work
            # with. A zero-variance group teaches nothing, so its share is
            # reported rather than averaged into a healthy-looking mean.
            "group_return_mean": round(gstats.return_mean, 8),
            "group_return_std": round(gstats.return_std, 8),
            "within_group_std_mean": round(gstats.within_group_std_mean, 8),
            "zero_variance_group_fraction": round(gstats.zero_variance_fraction, 6),
            "zero_variance_groups": gstats.zero_variance_groups,
            "groups": gstats.n_groups,
            "group_size": gstats.group_size,
            "samples": len(samples),
            "samples_used": len(kept),
            "episodes": int(batch["episodes"]),
            "episode_length_mean": batch["episode_length_mean"],
            "signal_step_means": {k: round(v, 8) for k, v in batch["signal_means"].items()},
            "reward_terms": compiled.term_totals(),
            "reward_fingerprint": fingerprint,
            "wall_seconds": round(wall, 3),
            # Throughput is diagnostics, never progress: the predecessor
            # project reported a healthy rate while every objective score was
            # 0.000. Tokens per second in particular measures the GPU, not the
            # policy.
            "diagnostics": {
                "env_steps_per_second": round(batch["env_steps"] / elapsed, 2),
                "n_envs": n_envs,
                "n_actions": n_actions,
                "prompt_chars_mean": round(batch["prompt_chars_mean"], 1),
                "device": device,
            },
        }
        ws.append_metrics(line)
        last_metrics = line

        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        payload = checkpoint_payload(
            arch=arch,
            adapter_state=peft.get_peft_model_state_dict(model),
            adapter_path=str(ws.adapter),
            step=steps_done,
            iteration=iteration,
            hyper=hyp,
            spec=spec,
            reward_fingerprint=fingerprint,
            wall_seconds=wall,
            optimizer_state=optimizer.state_dict(),
            rng=rng_state,
        )
        _save_adapter(model, ws.adapter)
        should_keep = (
            not hyp.checkpoint_interval
            or steps_done >= next_checkpoint_at
            or steps_done >= total_target
        )
        if should_keep:
            save_checkpoint(ws.step_checkpoint(steps_done), payload)
            if hyp.checkpoint_interval:
                while next_checkpoint_at <= steps_done:
                    next_checkpoint_at += hyp.checkpoint_interval
        save_checkpoint(ws.latest, payload)

        ws.write_state(
            {
                "env": spec.name,
                "objective": spec.objective.name,
                "trainer": "grpo_lora",
                "base_model": base_model,
                "base_model_path": local_path,
                "adapter": str(ws.adapter),
                "steps_done": steps_done,
                "steps_target": total_target,
                "iterations": iteration,
                "wall_seconds": round(wall, 3),
                "seed": seed,
                # The sampling stream is `default_rng([seed, iteration])`, so
                # these two numbers *are* the RNG state; the torch/numpy
                # generator states travel in the checkpoint beside the weights.
                "rng": {
                    "sampling_seed": [int(seed), int(iteration)],
                    "generators_in": str(ws.latest),
                },
                "hyper": hyp.to_dict(),
                "reward_config": reward_config,
                "reward_fingerprint": fingerprint,
                "reward_lineage": progress.reward_lineage,
                "latest_checkpoint": str(ws.latest),
                "last_metrics": last_metrics,
            }
        )

    if owns_env:
        env.close()

    from autobranch.trainer import TrainResult  # lazy: trainer imports torch at module scope

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
    """Compact view of a run: the last line plus a tail mean of the noisy parts.

    Deliberately not `trainer.summarize_metrics`: that one summarises a value
    loss and an explained variance, neither of which exists here, and a
    summary full of nulls reads as a broken run.
    """
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
        "final": {
            k: last.get(k)
            for k in (
                "loss", "policy_loss", "kl_to_reference", "entropy", "clip_frac",
                "group_return_mean", "group_return_std", "zero_variance_group_fraction",
                "episodes", "episode_length_mean",
            )
        },
        "tail_mean": {
            k: mean_of(k)
            for k in (
                "loss", "kl_to_reference", "entropy", "group_return_mean",
                "within_group_std_mean", "zero_variance_group_fraction",
            )
        },
        "signal_step_means": last.get("signal_step_means", {}),
        "reward_terms_last": last.get("reward_terms", {}),
        "wall_seconds": last.get("wall_seconds"),
    }


__all__ = [
    "ActionVocab",
    "GRPOHyper",
    "GroupStats",
    "LoraPolicy",
    "MissingDependencyError",
    "PostTrainError",
    "PostTrainWorkspace",
    "PROMPT_TEMPLATE",
    "PROMPT_VERSION",
    "Progress",
    "Sample",
    "action_labels",
    "build_action_vocab",
    "checkpoint_payload",
    "ensure_base_model",
    "escape_repo_id",
    "group_advantages",
    "is_model_ready",
    "load_policy",
    "lora_arch",
    "model_dir_for",
    "record_lineage",
    "require_dependencies",
    "restore_progress",
    "restrict_logits",
    "sample_action_indices",
    "serialize_observation",
    "summarize_metrics",
    "train",
    "zero_variance_mask",
]
