"""The immutable measurement.

Every number here is what a branch is ranked by, so two rules hold.

It imports no reward code, directly or transitively.
`tests/test_objective_independence.py` parses the import graph and fails if
that stops being true. Without the wall, an agent searching reward space could
improve its score by editing what "score" means. The cost is visible below:
the policy is rebuilt from the checkpoint's own `arch` record rather than by
importing the trainer's class.

Every number carries its provenance. `ObjectiveScore.command` reproduces the
score, and this module has a `__main__` so that command is real.
"""

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from autobranch.env_api import import_target
from autobranch.spec import EnvironmentSpec

#: Episodes are drawn from this many distinct seeds. Several seeds rather than
#: one because a single layout/seed rewards memorisation; a fixed small bank
#: rather than many because every branch pays this cost on every evaluation.
SEED_BANK_SIZE = 8

PolicyFn = Callable[[np.ndarray], np.ndarray]


class ObjectiveError(RuntimeError):
    """Raised when a measurement cannot be made (never when it is merely bad)."""

@dataclass(frozen=True)
class ObjectiveScore:
    """One measurement of one checkpoint.

    `value` is in the environment's own units (a success rate, a win rate),
    always oriented as the manifest declares; use `spec.objective.better` to
    compare rather than `>`. `n` is the number of episodes behind the number,
    because a 1.0 over 4 episodes and a 1.0 over 400 are different claims.
    """

    value: float
    n: int
    detail: dict = field(default_factory=dict)
    command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "n": self.n, "detail": self.detail, "command": self.command}


# ---------------------------------------------------------------------------
# the seed bank
# ---------------------------------------------------------------------------


def seed_bank(seed: int = 12345, size: int = SEED_BANK_SIZE) -> tuple[int, ...]:
    """Deterministic episode seeds for a given evaluation seed.

    A pure function of `seed`: the same checkpoint scored on two different
    machines, in two different branches, a week apart, sees the same episodes.
    """
    rng = np.random.default_rng(int(seed))
    return tuple(int(x) for x in rng.integers(0, 2**31 - 1, size=int(size)))


# ---------------------------------------------------------------------------
# the policy, rebuilt from the checkpoint alone
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, out_dim: int, hidden: Iterable[int]) -> nn.Sequential:
    """Rebuild the trainer's MLP layout without importing the trainer.

    This duplicates `trainer.build_mlp` on purpose -- the independence test
    forbids the import, and a checkpoint that cannot be loaded without the
    trainer is a checkpoint whose score depends on reward code. The
    convention both sides keep: `nn.Sequential(Linear, Tanh, ..., Linear)`
    with the hidden sizes recorded in the checkpoint's `arch`. The structural
    load below is `strict=True`, so any drift fails loudly here rather than
    producing a silently wrong number.
    """
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in hidden:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(nn.Tanh())
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


class CheckpointPolicy:
    """A checkpoint turned into `obs -> action`, and nothing else.

    The evaluator receives one of these. It cannot read weights, cannot train,
    and has no access to a reward: the only thing an environment's evaluator
    can do with a policy is run it.
    """

    def __init__(self, arch: dict[str, Any], state_dict: dict[str, Any], *, device: str = "cpu"):
        self.arch = dict(arch)
        self.kind = str(arch["kind"])
        self.device = torch.device(device)
        hidden = arch.get("hidden") or (128, 128)
        out_dim = int(arch["n_actions"]) if self.kind == "discrete" else int(arch["act_dim"])
        self.actor = _mlp(int(arch["obs_dim"]), out_dim, hidden)
        self.critic = _mlp(int(arch["obs_dim"]), 1, hidden)
        self.log_std = nn.Parameter(torch.zeros(out_dim)) if self.kind != "discrete" else None

        module = nn.Module()
        module.actor = self.actor
        module.critic = self.critic
        if self.log_std is not None:
            module.log_std = self.log_std
        module.load_state_dict(state_dict, strict=True)
        module.eval().to(self.device)
        self._module = module

        self.deterministic = True
        self._generator = torch.Generator(device="cpu")
        self.reseed(0)

    def reseed(self, seed: int) -> None:
        """Make stochastic action sampling reproducible for a given seed."""
        self._generator.manual_seed(int(seed) & 0x7FFFFFFF)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        with torch.no_grad():
            out = self.actor(torch.as_tensor(arr, device=self.device))
            if self.kind == "discrete":
                if self.deterministic:
                    action = out.argmax(dim=-1)
                else:
                    probs = torch.softmax(out, dim=-1)
                    action = torch.multinomial(probs, 1, generator=self._generator).squeeze(-1)
                result = action.cpu().numpy().astype(np.int64)
            else:
                if self.deterministic:
                    action = out
                else:
                    std = torch.exp(self.log_std).expand_as(out)
                    noise = torch.randn(out.shape, generator=self._generator)
                    action = out + std * noise
                result = action.cpu().numpy().astype(np.float32)
        return result[0] if single else result


#: `arch.kind` values `CheckpointPolicy` can rebuild as a dense MLP.
MLP_ARCH_KINDS = ("discrete", "continuous")

#: What `grpo_lora` writes. A LoRA adapter over a base model is not a dense
#: MLP, so `CheckpointPolicy` cannot rebuild it and `lm_policy` is asked
#: instead. The dispatch is on this value and nothing else -- never on the
#: `trainer` field, which records who *wrote* the checkpoint rather than what
#: shape it is, and which a hand-edited file could disagree with.
LM_LORA_ARCH_KIND = "lm_lora_discrete"

#: What `grpo_full` writes: every parameter updated, weights in an HF export
#: beside the record rather than inline. Same duck type reaches the evaluator.
LM_FULL_ARCH_KIND = "lm_full_discrete"


def load_checkpoint_policy(
    checkpoint_path: str | os.PathLike, *, device: str = "cpu"
) -> "CheckpointPolicy | Any":
    """Load weights only. No optimiser, no reward config, no hyperparameters.

    Two policy shapes reach this function, and the checkpoint's own `arch.kind`
    decides which loader runs. A PPO checkpoint is a dense MLP, rebuilt here. A
    `grpo_lora` checkpoint is a LoRA adapter over a base model, rebuilt by
    `lm_policy.load_policy`, which returns the same duck type -- callable,
    `reseed`, `deterministic` -- so an environment's evaluator never learns
    which trainer produced the thing it is scoring.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise ObjectiveError(f"no checkpoint at {path}")
    # weights_only=False because our own checkpoints carry RNG state tuples;
    # these files are written by the trainer in the same workspace, never
    # fetched from outside it.
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "arch" not in payload or "policy" not in payload:
        raise ObjectiveError(
            f"{path} is not an autobranch checkpoint (expected 'arch' and 'policy' keys)"
        )
    kind = str((payload.get("arch") or {}).get("kind", ""))
    if kind == LM_LORA_ARCH_KIND:
        from autobranch import lm_policy

        return lm_policy.load_policy(path, device=device)
    if kind == LM_FULL_ARCH_KIND:
        from autobranch import lm_policy

        return lm_policy.load_full_policy(path, device=device)
    if kind not in MLP_ARCH_KINDS:
        raise ObjectiveError(
            f"{path} records arch.kind={kind!r}, which this loader does not know how to rebuild "
            f"(expected one of {', '.join(MLP_ARCH_KINDS)}, {LM_LORA_ARCH_KIND!r} or "
            f"{LM_FULL_ARCH_KIND!r}). Refusing "
            "rather than guessing: a policy rebuilt to the wrong shape would score as a broken "
            "one and be read as a bad reward."
        )
    return CheckpointPolicy(payload["arch"], payload["policy"], device=device)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _call_evaluator(
    evaluator: Callable[..., Any],
    policy: CheckpointPolicy,
    *,
    spec: EnvironmentSpec,
    seeds: Sequence[int],
    budget: dict[str, Any],
    deterministic: bool,
) -> Any:
    """Invoke an environment's evaluator, passing only what it accepts.

    The contract an evaluator implements:

        def evaluate(policy, *, seeds, budget, deterministic, spec=None)
            -> float | dict | ObjectiveScore

    where `policy(obs) -> action`. Evaluators are written by environment
    authors, so unknown keywords are filtered rather than forced: a two-
    argument evaluator should not be a submission error.
    """
    offered = {"seeds": tuple(int(s) for s in seeds), "budget": dict(budget),
               "deterministic": bool(deterministic), "spec": spec}
    try:
        sig = inspect.signature(evaluator)
    except (TypeError, ValueError):  # pragma: no cover - builtins, C callables
        return evaluator(policy, **offered)
    params = sig.parameters
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = offered if takes_kwargs else {k: v for k, v in offered.items() if k in params}
    return evaluator(policy, **kwargs)


def _normalise(result: Any, where: str) -> tuple[float, int, dict[str, Any]]:
    """Accept a float, a dict, or an ObjectiveScore from an evaluator."""
    if isinstance(result, ObjectiveScore):
        return float(result.value), int(result.n), dict(result.detail)
    if isinstance(result, (int, float, np.floating, np.integer)):
        return float(result), 0, {}
    if isinstance(result, dict):
        if "value" not in result:
            raise ObjectiveError(f"{where}: evaluator returned a dict without a 'value' key")
        detail = {k: v for k, v in result.items() if k not in ("value", "n")}
        detail.update(result.get("detail") or {})
        detail.pop("detail", None)
        return float(result["value"]), int(result.get("n", 0)), detail
    raise ObjectiveError(
        f"{where}: evaluator returned {type(result).__name__}; expected float, dict or ObjectiveScore"
    )


def _render_command(
    *,
    manifest: str,
    checkpoint: str,
    seed: int,
    mode: str,
    budget: dict[str, Any],
    search_path: str | None,
    bank_size: int,
) -> str:
    parts = [
        sys.executable or "python",
        "-m",
        "autobranch.objective",
        f"--manifest {manifest}",
        f"--checkpoint {checkpoint}",
        f"--seed {seed}",
        f"--mode {mode}",
        f"--bank-size {bank_size}",
    ]
    if budget:
        parts.append("--budget " + json.dumps(budget, sort_keys=True, separators=(",", ":")))
    if search_path:
        parts.append(f"--search-path {search_path}")
    return " ".join(parts)


def _default_device() -> str:
    """A GPU when one exists. See `evaluate`'s device note."""
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - a broken driver is still a CPU answer
        return "cpu"


def evaluate(
    spec: EnvironmentSpec,
    checkpoint_path: str | os.PathLike,
    *,
    budget: dict[str, Any] | None = None,
    seed: int = 12345,
    deterministic: bool = True,
    search_path: str | os.PathLike | None = None,
    manifest: str | os.PathLike | None = None,
    bank_size: int = SEED_BANK_SIZE,
    device: str = "",
) -> ObjectiveScore:
    """Score a checkpoint against the manifest's objective.

    `deterministic` selects which mode's number becomes `value`; **both** are
    always measured and reported under `detail["modes"]`, because the gap
    between them is diagnostic. A branch that looks strong deterministically
    and random stochastically has a sharp policy; the reverse means the argmax
    has collapsed onto one action, which is the failure mode that cost the
    predecessor project six training runs.
    """
    merged_budget = dict(spec.objective.budget)
    merged_budget.update(budget or {})
    seeds = seed_bank(seed, bank_size)
    evaluator = import_target(spec.objective.evaluator, search_path=search_path)
    if not callable(evaluator):
        raise ObjectiveError(
            f"objective.evaluator {spec.objective.evaluator!r} is not callable"
        )

    # "" means pick: a GPU if there is one. The default used to be "cpu",
    # which is harmless for a two-layer MLP and fatal for an 8.77B
    # vision-language policy -- a forward per decision takes seconds on CPU, so
    # a 40-match evaluation would not have finished inside the run's deadline,
    # and the symptom would have been an evaluation that simply never returned.
    resolved = device or _default_device()
    policy = load_checkpoint_policy(checkpoint_path, device=resolved)

    modes: dict[str, dict[str, Any]] = {}
    # Both modes by default, because the gap between them is diagnostic: a
    # policy strong at the argmax and random while sampling is sharp, and the
    # reverse means the argmax has collapsed onto one action. But each mode is
    # a full pass over the match budget, and for an environment where one
    # decision costs a quarter of a second of emulator time that is the
    # difference between a 20-minute screen and a 40-minute one -- so a
    # manifest may ask for one.
    wanted = merged_budget.get("modes") or ("deterministic", "stochastic")
    if isinstance(wanted, str):
        wanted = (wanted,)
    unknown = set(wanted) - {"deterministic", "stochastic"}
    if unknown:
        raise ObjectiveError(
            f"objective.budget.modes has unknown entries {sorted(unknown)}; "
            "expected deterministic and/or stochastic"
        )
    for mode_name, mode_flag in (("deterministic", True), ("stochastic", False)):
        if mode_name not in wanted:
            continue
        policy.deterministic = mode_flag
        policy.reseed(seed)
        raw = _call_evaluator(
            evaluator, policy, spec=spec, seeds=seeds, budget=merged_budget,
            deterministic=mode_flag,
        )
        value, n, detail = _normalise(raw, f"{spec.objective.evaluator} ({mode_name})")
        modes[mode_name] = {"value": value, "n": n, **detail}

    preferred = "deterministic" if deterministic else "stochastic"
    # If the manifest asked for only the other mode, report that rather than
    # raising: the caller asked for a cheaper evaluation, not a failed one.
    #
    # `measured_mode` is the mode the reported value actually came from, which
    # is not always `preferred` -- exactly when this fallback fires, they
    # differ. Recording the preference instead made the SF3 artifact claim its
    # headline was measured deterministically, and reproduced a command with
    # `--mode deterministic`, for a number the same record showed under
    # `modes.stochastic` with `deterministic: false` in its budget. A
    # provenance line that contradicts itself is worse than none.
    measured_mode = preferred if preferred in modes else next(iter(modes))
    chosen = modes[measured_mode]
    manifest_path = str(manifest) if manifest else _guess_manifest(search_path)
    command = _render_command(
        manifest=manifest_path,
        checkpoint=str(Path(checkpoint_path).resolve()),
        seed=seed,
        mode=measured_mode,
        budget=budget or {},
        search_path=None if search_path is None else str(Path(search_path).resolve()),
        bank_size=bank_size,
    )
    detail = {
        "objective": spec.objective.name,
        "direction": spec.objective.direction,
        "evaluator": spec.objective.evaluator,
        "mode": measured_mode,
        "mode_requested": preferred,
        "modes": modes,
        # Only meaningful when both were measured.
        "mode_gap": (
            modes["deterministic"]["value"] - modes["stochastic"]["value"]
            if len(modes) == 2 else None
        ),
        "seeds": list(seeds),
        "budget": merged_budget,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "arch": policy.arch,
        "device": resolved,
    }
    return ObjectiveScore(
        value=float(chosen["value"]), n=int(chosen["n"]), detail=detail, command=command
    )


def _guess_manifest(search_path: str | os.PathLike | None) -> str:
    if search_path:
        candidate = Path(search_path) / "autobranch.yaml"
        if candidate.exists():
            return str(candidate.resolve())
    return "<manifest>"


# ---------------------------------------------------------------------------
# CLI -- so that ObjectiveScore.command is a real command
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autobranch.objective",
        description="Score a checkpoint against an environment's immutable objective.",
    )
    parser.add_argument("--manifest", required=True, help="path to autobranch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--bank-size", type=int, default=SEED_BANK_SIZE)
    parser.add_argument("--budget", default=None, help="JSON object overriding the manifest budget")
    parser.add_argument(
        "--search-path",
        default=None,
        help="bundle root to import the evaluator from (defaults to the manifest's directory)",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    spec = EnvironmentSpec.from_yaml(manifest_path.read_text())
    search_path = args.search_path or str(manifest_path.parent)
    score = evaluate(
        spec,
        args.checkpoint,
        budget=json.loads(args.budget) if args.budget else None,
        seed=args.seed,
        deterministic=args.mode == "deterministic",
        search_path=search_path,
        manifest=str(manifest_path),
        bank_size=args.bank_size,
    )
    print(json.dumps(score.to_dict(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    raise SystemExit(main())
