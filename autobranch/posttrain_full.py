"""Full-parameter critic-free RL on a vision-language base model.

This is the `grpo_full` trainer. It exists because the thing it reproduces was
a full fine-tune: modal-labs/sf3 trained Qwen3-VL-8B through slime/Megatron on
eight GPUs and exported `iter_*_hf` checkpoints, with no LoRA anywhere in the
repository. `grpo_lora` answers a different question -- how good can an adapter
over frozen weights get -- so using it here would have silently changed the
experiment.

It differs from `posttrain.py` in exactly two ways, and shares the RL method:

**Every parameter is trainable.** The memory arithmetic is what makes this hard
rather than interesting:

    bf16 params           16 GiB
    bf16 grads            16 GiB
    Adam fp32 moments     64 GiB    -> 96 GiB, over one 80 GiB card
    Adam 8-bit moments    16 GiB    -> 48 GiB, fits with room for activations

so the 8-bit optimizer is not a tuning choice, it is the difference between
running on one GPU and needing model parallelism. `probes/probe_full_ft.py` measured
this on the real model with a real frame before any of it was built, because
the estimate omits exactly the term that matters -- activation memory under
hundreds of image tokens.

**The observation is an image plus text.** A VL policy reads the emulator's
frame. `posttrain.serialize_observation` renders a numeric vector into a text
prompt, which for a 384x224x3 frame would be 258,048 comma-separated integers;
the processor turns the same frame into a few hundred image tokens. So prompts
are built through the processor here, and the frame travels in the checkpoint's
prompt record so an evaluator rebuilds the same policy.

What is unchanged, deliberately, is the method: an action is a **masked token**
(one forward, logits at the last position, restricted to the action vocabulary,
sampled -- so an illegal action is unrepresentable and no invalid-action
penalty has to exist in the reward), advantages are **group-relative** with no
critic, and the update is a clipped surrogate. Reward comes from a config the
search wrote; the objective that scores it imports none of this.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from autobranch.env_api import load_env
from autobranch.lm_policy import (
    ActionVocab,
    PostTrainError,
    VisionActionPolicy,
    _import_or_explain,
    build_action_vocab,
    ensure_base_model,
    prepare_vision_processor,
    log_softmax,
    restrict_logits,
    sample_action_indices,
)
from autobranch.reward import RewardConfig, compile_reward, config_hash
from autobranch.spec import EnvironmentSpec

if TYPE_CHECKING:
    from autobranch.trainer import TrainResult

CHECKPOINT_FORMAT = 1

#: Bumped whenever the prompt's shape changes. Travels in the checkpoint: a
#: silently reworded prompt is a silently different policy.
PROMPT_VERSION = 1

HF_EXPORT_DIRNAME = "hf"


def durable_root() -> Path:
    """Volume-backed directory for things that must outlive the sandbox.

    `/workspace` is sandbox-local -- only `/data` comes from the Volume -- and
    this project has already paid for forgetting that once, when checkpoints
    written by the trainer died with their branch and the framework's
    deliverable was the one thing it did not keep.

    For a full fine-tune the stakes are higher in both directions. The HF
    export *is* the trained model, 16 GiB of it, so losing it loses the run;
    and putting 16 GiB inside `/workspace` would also make every fork
    `snapshot_directory` that much, turning the cheap branch primitive into the
    most expensive thing in the system. So the export and the metrics live on
    the Volume, keyed by run and branch, and the small torch record stays in
    the workspace where fork inheritance expects it.
    """
    root = Path(os.environ.get("AUTOBRANCH_DATA_ROOT", "/data"))
    run = os.environ.get("AUTOBRANCH_RUN_ID", "local")
    branch = os.environ.get("AUTOBRANCH_BRANCH_ID", "b0")
    return root / "runs" / run / branch

REQUIRED_PIP = (
    ("torch", "torch>=2.2"),
    ("transformers", "transformers>=4.57"),
    ("bitsandbytes", "bitsandbytes>=0.45"),
    ("PIL", "pillow>=10"),
)


@dataclass
class FullHyper:
    """Knobs for a full-parameter critic-free run.

    Defaults are small on purpose. A full fine-tune of 8B parameters forgets
    fast: the learning rate is two orders of magnitude below a LoRA run's
    because every weight moves, and `max_grad_norm` is tight because one bad
    batch of a 40-way categorical can take the policy somewhere it will not
    come back from inside a budget this short.
    """

    lr: float = 1e-6
    group_size: int = 8
    #: One collection per update. Measured: a lane steps in 0.25 s and lanes
    #: run concurrently, so a 600-decision collection costs ~150 s of wall
    #: clock no matter how many lanes there are. Two collections per update
    #: would double that for a baseline the batch already estimates from eight
    #: concurrent matches, and halve the number of gradient steps a fixed
    #: deadline buys.
    groups_per_iteration: int = 1
    #: Measured, not guessed. probes/probe_full_ft.py saw 2.93 s for a single-sample
    #: forward+backward+step with 30 GiB of headroom left, so a batch of one
    #: would spend 78 minutes updating on a 1600-decision iteration. Prompts
    #: here are ~104 tokens, so 8 fit comfortably in that headroom.
    micro_batch_size: int = 8
    #: Train on a random subset of the decisions a collection produced. The
    #: predecessor project tested this knob explicitly; the reason it is on by
    #: default is that collection is cheap relative to the update once the
    #: emulator lanes run concurrently, so an iteration otherwise spends almost
    #: all of its wall clock in backward passes over correlated samples from
    #: the same few matches.
    train_subsample: int = 256
    #: Hard wall-clock ceiling for one `train()` call, regardless of the step
    #: target it was given.
    #:
    #: `branch_main._cap_steps` tries to do this by converting the deadline
    #: into a step count, but it can only do that once it has measured a rate
    #: -- which means from the *second* call. On the first call it falls back
    #: to a pure step ceiling, and on an environment where a step costs ~0.15 s
    #: of wall clock that ceiling was 194 minutes. One branch spent 112 minutes
    #: of a 5-hour deadline inside its first tool call, never reaching an
    #: evaluation, which is precisely what that function's docstring promises
    #: will not happen.
    #:
    #: Steps are what a trainer counts; time is what a deadline is made of, and
    #: only the trainer knows the exchange rate. So it enforces the bound
    #: itself and returns what it has -- `steps` is a target, not a contract.
    max_seconds: float = 1200.0
    #: Export every N iterations, and always before returning.
    #:
    #: This was 5, reasoning that an export is only *needed* when train()
    #: returns because that is when the evaluator loads it, and that in between
    #: it is mere crash insurance worth ~60 s of every ~750 s iteration. Then a
    #: run was preempted on iteration 4 and lost all three of its completed
    #: iterations, because no export had been written yet.
    #:
    #: On preemptible hardware checkpoint frequency is not insurance, it is the
    #: bound on how much work a preemption can destroy. At 2 the worst case is
    #: ~25 minutes instead of ~an hour, for 8% of the wall clock rather than 4%.
    save_every: int = 2
    #: Where the HF export and a mirrored metrics log go. Empty means derive it
    #: from AUTOBRANCH_DATA_ROOT/RUN_ID/BRANCH_ID, which is what a real branch
    #: wants; a probe or a local test passes an explicit path.
    durable_root: str = ""
    epochs: int = 1
    clip: float = 0.2
    max_grad_norm: float = 0.5
    #: Small but nonzero. The probe measured 0.41 nats of action entropy once
    #: the menu was in the prompt -- against ln(40) = 3.69 available -- so the
    #: base model arrives with a strong opinion about what to do. That is the
    #: point of starting from a pretrained VL model, but the predecessor
    #: project found its policy was "only competent while sampling" and
    #: collapsed to 0/100 at the mode, so a run with ~54 iterations cannot
    #: afford to sharpen further before it has explored.
    entropy_coef: float = 0.01
    temperature: float = 1.0
    gamma: float = 0.995
    #: Hinh discounted the round outcome on a 3600-frame half-life. Kept as a
    #: default so his configuration is reachable, not imposed: the reward DSL
    #: owns shaping, this only sets the return's horizon.
    discount_half_life_frames: float = 3600.0
    #: 600, measured both ways. At 600 a collection produced 6 round wins
    #: across 8 lanes; at 400 it produced **zero**, and that is the whole
    #: reason this is not tuned lower to buy gradient steps.
    #:
    #: The mistake worth recording: 6 rounds over 8x600 lane-decisions reads as
    #: "one round per 800 lane-decisions", which invites treating it as a rate
    #: and scaling it down. Round length is a *threshold*, not a rate -- a
    #: round takes some minimum number of decisions in a single lane to reach a
    #: KO or the clock, and below that no lane finishes one no matter how many
    #: lanes there are. 400 was below it.
    #:
    #: A collection with no round outcomes trains on shaping alone: round_won
    #: and round_lost are the only terms tied to the objective, and they never
    #: fire. Worse, the agent sees rounds_won 0 in its metrics and may rewrite
    #: a perfectly good reward in response to an artifact of the collection
    #: window -- which corrupts the search rather than just slowing it.
    max_decisions_per_match: int = 600
    grad_checkpointing: bool = True
    device_map: str = ""  # "" = single device; "auto" splits layers across GPUs
    dtype: str = "bfloat16"
    seed: int = 0

    @staticmethod
    def parse(raw: dict[str, Any] | None) -> "FullHyper":
        raw = dict(raw or {})
        known = {f for f in FullHyper.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise PostTrainError(
                f"unknown grpo_full hyperparameters {sorted(unknown)}; known: {sorted(known)}"
            )
        return FullHyper(**raw)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class Decision:
    """One action the policy committed to, with what it needs for an update."""

    prompt_text: str
    frame: Any
    action_index: int
    logprob: float
    group: int
    reward: float = 0.0
    advantage: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    frames_elapsed: float = 1.0


# ---------------------------------------------------------------------------
# returns and advantages -- the whole of "critic-free"
# ---------------------------------------------------------------------------


def discounted_returns(rewards: Sequence[float], frames: Sequence[float],
                       *, half_life_frames: float) -> np.ndarray:
    """Backward discounted return, discounting by *game frames*, not steps.

    A macro action can consume three frames or ninety, so discounting per
    decision would value a jab and a super identically in time. Hinh discounts
    on a frame half-life for the same reason; this keeps that and lets the DSL
    own everything else.
    """
    r = np.asarray(rewards, dtype=np.float64)
    f = np.asarray(frames, dtype=np.float64)
    out = np.zeros_like(r)
    running = 0.0
    for i in range(len(r) - 1, -1, -1):
        decay = 0.5 ** (max(1.0, f[i]) / max(1.0, half_life_frames))
        running = r[i] + decay * running
        out[i] = running
    return out


def group_relative_advantages(returns: Sequence[float], groups: Sequence[int],
                              *, eps: float = 1e-6) -> tuple[np.ndarray, dict]:
    """`(r - mean) / (std + eps)` within each group. No critic anywhere.

    A group whose members all score identically has zero advantage for every
    member and teaches nothing, so the fraction of such groups is reported
    rather than hidden: a run whose groups are mostly degenerate is not
    learning slowly, it is not learning.
    """
    ret = np.asarray(returns, dtype=np.float64)
    grp = np.asarray(groups, dtype=np.int64)
    adv = np.zeros_like(ret)
    degenerate = 0
    seen = 0
    for g in np.unique(grp):
        mask = grp == g
        values = ret[mask]
        std = float(values.std())
        adv[mask] = (values - values.mean()) / (std + eps)
        seen += 1
        if std <= eps:
            degenerate += 1
    stats = {
        "groups": seen,
        "zero_variance_groups": degenerate,
        "zero_variance_group_fraction": (degenerate / seen) if seen else 0.0,
        "return_mean": float(ret.mean()) if ret.size else 0.0,
        "return_std": float(ret.std()) if ret.size else 0.0,
    }
    return adv, stats


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def _torch_dtype(torch, name: str):
    dtype = getattr(torch, str(name), None)
    if dtype is None:
        raise PostTrainError(f"unknown dtype {name!r}")
    return dtype


def load_for_training(base_path: str, hyper: FullHyper):
    """Load the VL model with every parameter trainable, plus 8-bit Adam."""
    torch = _import_or_explain("torch", "torch>=2.2")
    transformers = _import_or_explain("transformers", "transformers>=4.57")
    _import_or_explain("PIL", "pillow>=10")

    processor = prepare_vision_processor(
        transformers.AutoProcessor.from_pretrained(base_path)
    )
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = transformers.AutoModelForImageTextToText
    kwargs: dict[str, Any] = {"dtype": _torch_dtype(torch, hyper.dtype)}
    if hyper.device_map:
        kwargs["device_map"] = hyper.device_map
    model = model_cls.from_pretrained(base_path, **kwargs)

    if hyper.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not hyper.device_map:
        model.to("cuda")
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    bnb = _import_or_explain("bitsandbytes", "bitsandbytes>=0.45")
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=hyper.lr)
    return model, processor, optimizer


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def collect_group(env, policy: VisionActionPolicy, compiled, hyper: FullHyper,
                  *, group: int, rng: np.random.Generator,
                  progress: float = 0.0) -> tuple[list[Decision], dict]:
    """Run every lane to termination once, scoring each decision by the config.

    One group is one batch of concurrent matches. The group is the baseline:
    advantages are computed across its members, so the lanes have to share the
    conditions that make them comparable -- same difficulty, same policy
    weights, same reward config.
    """
    torch = _import_or_explain("torch", "torch>=2.2")
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    lanes = len(obs["text"])
    live = np.ones(lanes, dtype=bool)
    per_lane: list[list[Decision]] = [[] for _ in range(lanes)]
    totals = {"decisions": 0, "invalid": 0, "damage_dealt": 0.0, "damage_taken": 0.0,
              "rounds_won": 0, "rounds_lost": 0}

    for _ in range(hyper.max_decisions_per_match):
        if not live.any():
            break
        with torch.no_grad():
            action_logits, texts, images = policy.logits(obs)
        actions = sample_action_indices(
            action_logits, rng, temperature=hyper.temperature, greedy=False
        )
        logprobs = log_softmax(action_logits, temperature=hyper.temperature)
        result = env.step(actions)

        # One call for every lane, exactly as the PPO trainer composes reward:
        # the compiled config is vectorised, and calling it per lane would both
        # cost N times as much and give any stateful shaping term (potential,
        # delta) a different history per lane than the trainer intends.
        rewards = np.atleast_1d(np.asarray(
            compiled(result.signals, progress=progress), dtype=np.float64
        )).reshape(-1)
        if rewards.size == 1 and lanes > 1:
            rewards = np.repeat(rewards, lanes)

        for i in range(lanes):
            if not live[i]:
                continue
            signals = {k: float(np.asarray(v).reshape(-1)[i])
                       for k, v in result.signals.items()}
            reward = float(rewards[i])
            per_lane[i].append(Decision(
                prompt_text=texts[i],
                frame=images[i],
                action_index=int(actions[i]),
                logprob=float(logprobs[i, int(actions[i])]),
                group=group,
                reward=reward,
                signals=signals,
                frames_elapsed=max(1.0, signals.get("decision_frames", 1.0)),
            ))
            totals["decisions"] += 1
            totals["invalid"] += int(signals.get("invalid_move", 0.0) > 0)
            totals["damage_dealt"] += signals.get("damage_dealt", 0.0)
            totals["damage_taken"] += signals.get("damage_taken", 0.0)
            totals["rounds_won"] += int(signals.get("round_won", 0.0) > 0)
            totals["rounds_lost"] += int(signals.get("round_lost", 0.0) > 0)

        obs = result.obs
        live &= ~result.done

    decisions: list[Decision] = []
    for lane in per_lane:
        if not lane:
            continue
        returns = discounted_returns(
            [d.reward for d in lane], [d.frames_elapsed for d in lane],
            half_life_frames=hyper.discount_half_life_frames,
        )
        for d, ret in zip(lane, returns, strict=True):
            d.advantage = float(ret)  # replaced by the group-relative value below
        decisions.extend(lane)
    return decisions, totals


# ---------------------------------------------------------------------------
# the update
# ---------------------------------------------------------------------------


def gradient_step(model, processor, optimizer, vocab: ActionVocab,
                  decisions: list[Decision], hyper: FullHyper, *, device: str) -> dict:
    """Clipped surrogate over the sampled action's log-probability."""
    torch = _import_or_explain("torch", "torch>=2.2")

    optimizer.zero_grad(set_to_none=True)
    n = len(decisions)
    if n == 0:
        return {"loss": 0.0, "micro_batches": 0}

    total_loss = 0.0
    total_entropy = 0.0
    clipped = 0
    ratios: list[float] = []
    micro = max(1, hyper.micro_batch_size)
    batches = 0

    for start in range(0, n, micro):
        chunk = decisions[start:start + micro]
        batch = processor(
            text=[d.prompt_text for d in chunk],
            images=[d.frame for d in chunk],
            return_tensors="pt",
            padding=True,
        ).to(device)
        out = model(**batch)
        logits = out.logits[:, -1, :]
        action_logits = logits[:, list(vocab.token_ids)]
        logp = torch.log_softmax(action_logits.float() / hyper.temperature, dim=-1)
        idx = torch.tensor([d.action_index for d in chunk], device=logp.device)
        chosen = logp.gather(1, idx.unsqueeze(1)).squeeze(1)
        old = torch.tensor([d.logprob for d in chunk], device=logp.device, dtype=chosen.dtype)
        adv = torch.tensor([d.advantage for d in chunk], device=logp.device, dtype=chosen.dtype)

        ratio = torch.exp(chosen - old)
        clean = ratio * adv
        clamped = torch.clamp(ratio, 1.0 - hyper.clip, 1.0 + hyper.clip) * adv
        surrogate = torch.minimum(clean, clamped)
        entropy = -(logp.exp() * logp).sum(dim=-1)
        loss = -(surrogate.mean()) - hyper.entropy_coef * entropy.mean()
        # Scaled so the accumulated gradient is the mean over all decisions,
        # not the sum over micro-batches.
        (loss * len(chunk) / n).backward()

        total_loss += float(loss.detach()) * len(chunk)
        total_entropy += float(entropy.detach().mean()) * len(chunk)
        clipped += int((clean > clamped).sum())
        ratios.extend([float(x) for x in ratio.detach().flatten()])
        batches += 1

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hyper.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": total_loss / n,
        "entropy": total_entropy / n,
        "grad_norm": float(grad_norm),
        "clip_frac": clipped / n,
        "ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
        "micro_batches": batches,
    }


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def arch_record(spec: EnvironmentSpec, vocab: ActionVocab, hyper: FullHyper,
                *, base_model: str, base_model_path: str, export: str,
                action_names: Sequence[str] = ()) -> dict[str, Any]:
    """Everything needed to rebuild this policy, travelling with the weights."""
    return {
        "kind": "lm_full_discrete",
        "obs_dim": int(spec.policy.obs_dim or 0),
        "n_actions": int(spec.policy.n_actions or len(vocab)),
        "act_dim": 1,
        "base_model": str(base_model),
        "base_model_path": str(base_model_path),
        "hf_export": export,
        "action_vocab": vocab.to_dict(),
        "prompt": {
            "version": PROMPT_VERSION,
            "kind": "vision_chat",
            "env_name": spec.name,
            "labels": [l.strip() for l in vocab.labels],
            # Without these the evaluator would build a menu-less prompt and
            # measure a different policy than the one that was trained.
            "action_names": list(action_names or ()),
        },
        "dtype": str(hyper.dtype),
        "temperature": float(hyper.temperature),
    }


def save_hf_export(model, processor, directory: Path) -> Path:
    """Write a full HF checkpoint, the way the reproduction target does.

    `iter_*_hf` is the shape modal-labs/sf3 exports and serves, so a
    full-parameter run here produces the same artifact rather than something
    only this framework can load. Staged and renamed so a reader cannot catch
    a half-written export.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(f".{directory.name}.saving-{os.getpid()}")
    import shutil

    shutil.rmtree(staging, ignore_errors=True)
    model.save_pretrained(str(staging), safe_serialization=True)
    processor.save_pretrained(str(staging))
    previous = directory.with_name(f".{directory.name}.previous-{os.getpid()}")
    shutil.rmtree(previous, ignore_errors=True)
    if directory.exists():
        os.replace(directory, previous)
    os.replace(staging, directory)
    shutil.rmtree(previous, ignore_errors=True)
    return directory


# ---------------------------------------------------------------------------
# the trainer entry point
# ---------------------------------------------------------------------------


def train(
    spec: EnvironmentSpec,
    reward_config: RewardConfig | dict[str, Any],
    *,
    steps: int,
    workspace: str,
    resume: bool = True,
    seed: int = 0,
    hyper: dict[str, Any] | None = None,
    env: Any = None,
    search_path: Sequence[str] | None = None,
) -> "TrainResult":
    """Full-parameter critic-free RL. `steps` is a cumulative target.

    Same signature and return type as `trainer.train` and `posttrain.train`, so
    nothing above this has to know which trainer it got. `steps` counts
    environment decisions, not gradient steps and never tokens per second: the
    only progress that counts is measured against the objective.
    """
    from autobranch.trainer import TrainResult

    hyp = FullHyper.parse(hyper)
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    metrics_path = ws / "metrics.jsonl"
    state_path = ws / "progress.json"

    # Durable by default, and overridable for a probe that has no volume.
    durable = Path(hyp.durable_root) if hyp.durable_root else durable_root()
    try:
        durable.mkdir(parents=True, exist_ok=True)
        export_dir = durable / HF_EXPORT_DIRNAME
        durable_metrics = durable / "metrics.jsonl"
    except OSError:
        # No volume mounted (a local test). Degrade to the workspace rather
        # than refuse to train, but say so in the metrics so nobody later
        # mistakes a lost export for a missing one.
        export_dir = ws / HF_EXPORT_DIRNAME
        durable_metrics = None

    # RewardConfig is an alias for `dict[str, Any]`, not a class: the DSL's
    # whole point is that a reward is data, so there is nothing to construct.
    config: RewardConfig = dict(reward_config or {})
    compiled = compile_reward(config, spec)
    fingerprint = config_hash(config)

    base_model = spec.policy.base_model
    if not base_model:
        raise PostTrainError(
            f"{spec.name}: grpo_full needs policy.base_model (the weights to start from)"
        )
    base_path = ensure_base_model(base_model)

    # Progress is read from the Volume first and the workspace second. The
    # workspace copy dies with its sandbox, so after a preemption it is the
    # durable one or nothing -- and "nothing" means silently restarting a
    # four-hour run from zero, which is what happened once.
    durable_state = durable / "progress.json"
    progress = {"step": 0, "iteration": 0}
    if resume:
        for candidate in (durable_state, state_path):
            try:
                if candidate.is_file():
                    progress.update(json.loads(candidate.read_text()))
                    break
            except (json.JSONDecodeError, OSError):
                continue
    # Resume from our own export when there is one: a full fine-tune's whole
    # state is its weights, so continuing from the base model would silently
    # discard every gradient step taken so far.
    load_from = str(export_dir) if (resume and (export_dir / "config.json").is_file()) else base_path

    model, processor, optimizer = load_for_training(load_from, hyp)
    device = "cuda" if not hyp.device_map else str(next(model.parameters()).device)
    vocab = build_action_vocab(
        processor.tokenizer, int(spec.policy.n_actions or 0), model_name=base_model
    )
    # `load_env` takes the entrypoint string, not the spec: it is the bundle's
    # "module:callable" and the search path is the bundle root.
    environment = env if env is not None else load_env(
        spec.entrypoint, search_path=search_path,
        n_envs=hyp.group_size, seed=seed,
        difficulty=int(spec.objective.budget.get("difficulty", 8)),
    )
    policy = VisionActionPolicy(
        model, processor, vocab, device=device, temperature=hyp.temperature,
        # The environment owns the action table, so it owns the names. Passing
        # them through is what puts the menu in the prompt.
        action_names=tuple(getattr(environment, "action_names", ()) or ()),
    )

    rng = np.random.default_rng((seed * 1_000_003 + progress["iteration"]) & 0x7FFFFFFF)
    started = time.time()
    last: dict[str, Any] = {}

    try:
        while progress["step"] < int(steps):
            # Checked before collecting rather than after: a collection plus an
            # update is ~12 minutes here, so testing at the bottom of the loop
            # would overshoot the ceiling by a whole iteration.
            if hyp.max_seconds and (time.time() - started) >= hyp.max_seconds:
                print(
                    f"stopping at {progress['step']:,}/{int(steps):,} steps: this call has "
                    f"used {(time.time() - started) / 60:.0f} min of its "
                    f"{hyp.max_seconds / 60:.0f} min ceiling. `steps` is a target, and a "
                    f"branch that never returns never evaluates.",
                    flush=True,
                )
                break
            iteration = progress["iteration"] + 1
            decisions: list[Decision] = []
            totals = {"decisions": 0, "invalid": 0, "damage_dealt": 0.0,
                      "damage_taken": 0.0, "rounds_won": 0, "rounds_lost": 0}
            for g in range(max(1, hyp.groups_per_iteration)):
                got, tot = collect_group(
                    environment, policy, compiled, hyp,
                    group=iteration * 1000 + g, rng=rng,
                    progress=min(1.0, progress["step"] / max(1, int(steps))),
                )
                decisions.extend(got)
                for k, v in tot.items():
                    totals[k] += v

            if not decisions:
                raise PostTrainError(
                    "a collection produced no decisions; every emulator lane failed to "
                    "step, so there is nothing to learn from"
                )

            # Zero round outcomes means the collection window is shorter than a
            # round, so the only objective-aligned terms in the reward
            # contributed nothing and this iteration learned from shaping
            # alone. Said out loud because the metric that reveals it
            # (rounds_won 0) looks like a reward problem and is a window
            # problem, and an agent cannot tell the difference.
            if totals["rounds_won"] == 0 and totals["rounds_lost"] == 0:
                print(
                    f"WARNING: iteration {iteration} saw no round outcomes in "
                    f"{totals['decisions']} decisions "
                    f"({hyp.max_decisions_per_match} per lane). round_won/round_lost "
                    f"contributed nothing, so this update is shaping-only. Raise "
                    f"max_decisions_per_match above one round's length.",
                    flush=True,
                )

            advantages, stats = group_relative_advantages(
                [d.advantage for d in decisions], [d.group for d in decisions]
            )
            for d, a in zip(decisions, advantages, strict=True):
                d.advantage = float(a)

            # Subsample AFTER advantages, never before: the group is the
            # baseline, so computing it on a subset would estimate each group's
            # mean from fewer members and make the advantages noisier for no
            # saving -- the expensive part is the backward pass, not the mean.
            collected = len(decisions)
            if hyp.train_subsample and collected > hyp.train_subsample:
                keep = rng.choice(collected, size=hyp.train_subsample, replace=False)
                decisions = [decisions[i] for i in sorted(int(k) for k in keep)]

            update: dict[str, Any] = {}
            for epoch in range(max(1, hyp.epochs)):
                update = gradient_step(
                    model, processor, optimizer, vocab, decisions, hyp, device=device
                )
                # On the first epoch the weights have not moved since
                # collection, so the importance ratio must be 1. Anything else
                # means the two forwards disagree about what they are scoring
                # -- which is what right-padding did, by reading the action
                # logits out of a PAD position at a width that depended on the
                # batch. Checked rather than trusted, because the gradient
                # still looked perfectly healthy while meaning nothing.
                if epoch == 0:
                    drift = abs(float(update.get("ratio_mean", 1.0)) - 1.0)
                    update["first_epoch_ratio_drift"] = drift
                    if drift > 0.05:
                        print(
                            f"WARNING: first-epoch importance ratio is "
                            f"{update['ratio_mean']:.3f}, expected 1.000. Collection and "
                            f"update are not scoring the same thing; the gradient is not "
                            f"trustworthy. Check the processor's padding side.",
                            flush=True,
                        )

            progress["step"] += totals["decisions"]
            progress["iteration"] = iteration
            line = {
                "step": progress["step"],
                "iteration": iteration,
                "decisions": totals["decisions"],
                "decisions_trained": len(decisions),
                "invalid_move_rate": totals["invalid"] / max(1, totals["decisions"]),
                "damage_dealt_per_decision": totals["damage_dealt"] / max(1, totals["decisions"]),
                "damage_taken_per_decision": totals["damage_taken"] / max(1, totals["decisions"]),
                "rounds_won": totals["rounds_won"],
                "shaping_only": bool(totals["rounds_won"] == 0
                                     and totals["rounds_lost"] == 0),
                "rounds_lost": totals["rounds_lost"],
                "reward_fingerprint": fingerprint,
                "wall_seconds": round(time.time() - started, 1),
                **{k: v for k, v in stats.items()},
                **{k: v for k, v in update.items()},
            }
            payload = json.dumps(line) + "\n"
            with metrics_path.open("a") as fh:
                fh.write(payload)
            # Mirrored rather than moved: the workspace copy is what a fork
            # inherits and what `resume` reads, the Volume copy is what
            # survives the sandbox and what post-hoc analysis reads.
            if durable_metrics is not None:
                with durable_metrics.open("a") as fh:
                    fh.write(payload)
            # The workspace copy advances every iteration: it dies with the
            # sandbox anyway and is only read when the durable one is absent.
            state_path.write_text(json.dumps(progress))

            # The durable copy advances ONLY with the export, and strictly
            # after it. Those two describe the same thing -- "where training
            # got to" -- and letting them disagree is a silent correctness
            # bug: progress written every iteration against an export written
            # every `save_every` means a death on an odd iteration resumes
            # with iteration-4 weights while believing it is at iteration 5,
            # and trains that much less than the step target implies.
            #
            # Export first, then progress. A crash between them leaves
            # progress *behind* the weights, so a resume redoes an iteration.
            # The other order would skip one, and redoing work is recoverable
            # in a way that skipping it is not.
            if hyp.save_every and iteration % hyp.save_every == 0:
                save_hf_export(model, processor, export_dir)
                try:
                    durable_state.write_text(json.dumps(progress))
                except OSError:
                    pass
            last = line
    finally:
        if env is None and environment is not None:
            try:
                environment.close()
            except Exception:  # noqa: BLE001 - teardown must not mask a result
                pass

    # Always export before returning: `train` returning is the signal that a
    # checkpoint exists to be scored, and an evaluator that loads a stale
    # export would measure a policy from several iterations ago. Progress
    # follows it, for the same reason it does inside the loop.
    save_hf_export(model, processor, export_dir)
    try:
        durable_state.write_text(json.dumps(progress))
    except OSError:
        pass

    torch = _import_or_explain("torch", "torch>=2.2")
    checkpoint = ws / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "trainer": "grpo_full",
            "arch": arch_record(
                spec, vocab, hyp,
                base_model=base_model, base_model_path=base_path,
                export=HF_EXPORT_DIRNAME,
                action_names=policy.action_names,
            ),
            # The weights live in the HF export beside this file, not inline: a
            # 16 GiB tensor dict inside a .pt would double the checkpoint and
            # could not be served by anything but this framework.
            "policy": {},
            "hf_export_path": str(export_dir),
            "durable_root": str(durable),
            "step": progress["step"],
            "iteration": progress["iteration"],
            "hyper": hyp.to_dict(),
            "env": {"name": spec.name, "objective": spec.objective.name,
                    "entrypoint": spec.entrypoint},
            "reward_fingerprint": fingerprint,
            "wall_seconds": round(time.time() - started, 1),
        },
        checkpoint,
    )

    # Hand the GPU back before returning. The evaluator loads its own copy of
    # the policy from the HF export -- deliberately, so that scoring imports no
    # trainer state -- and 16 GiB does not fit beside the 49.5 GiB this run
    # reserved. Dropping the references is not enough: torch's caching
    # allocator holds reserved blocks until empty_cache, so an evaluation would
    # OOM with most of the card apparently free.
    del model, optimizer
    try:
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - a failure to tidy must not fail a run
        pass

    return TrainResult(
        steps_done=int(progress["step"]),
        checkpoint_path=str(checkpoint),
        metrics=last,
        wall_seconds=round(time.time() - started, 3),
        iterations=int(progress["iteration"]),
        workspace=str(ws),
        reward_fingerprint=fingerprint,
    )
