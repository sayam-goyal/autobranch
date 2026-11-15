"""The Python half of the environment contract.

`spec.py` validates what an environment *claims*; this module checks what it
actually *does*. The two together are the submission gate: a bundle that
passes both can be handed to the trainer without anyone reading its code.

The one rule that matters here: `step` returns **signals**, never a reward.
A reward is composed later, by the trainer, from an agent-authored config over
exactly the signals the manifest declares. That is why `StepResult` has no
`reward` field -- there is nowhere for an environment author (or an agent
editing an environment) to smuggle one in.

`validate_env` is deliberately pedantic and returns *every* problem it finds
rather than raising on the first. In the project this framework generalises,
six training runs were burned before anyone noticed the measurement was
wrong; a validator that reports five problems at submission time is worth
hours.
"""

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

from autobranch.spec import EnvironmentSpec

#: Key under which a vectorised env may expose the pre-reset observation of
#: lanes that finished this step. Auto-resetting envs overwrite `obs` with the
#: new episode's first observation, which destroys the bootstrap target for a
#: *truncated* (time-limited, not terminal) lane. An env that sets this gets
#: correct value bootstrapping; one that does not gets a slightly biased
#: value function at the time limit. See `trainer._bootstrap_truncated`.
INFO_FINAL_OBS = "final_obs"

#: Problems that describe a smell rather than a contract violation are
#: prefixed with this, so a caller can gate on hard failures only.
ADVISORY = "advisory: "


@dataclass
class StepResult:
    """One vectorised transition.

    `signals` maps a declared signal name to either a scalar (same value for
    every lane) or an array of shape `[n_envs]`. Keys MUST be exactly the set
    declared in the manifest: an undeclared key is as much an error as a
    missing one, because an agent composing a reward can only see declared
    names and a silently-extra signal means the manifest is stale.
    """

    obs: Any
    signals: dict[str, Any]
    terminated: Any
    truncated: Any
    info: dict = field(default_factory=dict)

    @property
    def done(self) -> np.ndarray:
        """Lanes that ended this step for either reason."""
        return np.logical_or(
            np.atleast_1d(np.asarray(self.terminated, dtype=bool)),
            np.atleast_1d(np.asarray(self.truncated, dtype=bool)),
        )


@runtime_checkable
class AutobranchEnv(Protocol):
    """What an environment must provide. Four members, on purpose.

    Vectorisation is mandatory rather than wrapped on: every real environment
    in scope can batch, and a framework that fakes batching over a single-lane
    env pays for it in throughput exactly when it matters. Envs are expected
    to **auto-reset** finished lanes inside `step`.
    """

    @property
    def n_envs(self) -> int: ...

    def reset(self, *, seed: int | None = None) -> Any: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


@contextmanager
def bundle_on_path(directory: str | os.PathLike | None) -> Iterator[None]:
    """Make a bundle's modules importable for the duration of the block.

    Bundles are directories, not installed packages, so `grid_env:make_env`
    only resolves with the bundle root on `sys.path`. The entry is removed
    afterwards; the imported module stays in `sys.modules`, which is what we
    want -- repeated loads of the same bundle must not produce two distinct
    module objects (lesson 1 in DESIGN.md, learned the hard way with two
    `modal.App` instances).
    """
    if directory is None:
        yield
        return
    path = str(os.path.abspath(directory))
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:  # pragma: no cover - someone else cleaned up
                pass


def import_target(entrypoint: str, *, search_path: str | os.PathLike | None = None) -> Any:
    """Resolve a `"module:callable"` reference to the object itself.

    Shared by `load_env` and by `objective.evaluate`, which is why it lives
    here: `objective` must not import the trainer or the reward compiler, so
    the only module it is allowed to lean on for loading is this one.
    """
    if ":" not in entrypoint:
        raise ValueError(f"entrypoint {entrypoint!r} must look like 'module:callable'")
    module_name, _, attr = entrypoint.partition(":")
    with bundle_on_path(search_path):
        module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {attr!r} (from entrypoint {entrypoint!r})"
        ) from exc


def load_env(
    entrypoint: str,
    *,
    search_path: str | os.PathLike | None = None,
    **kwargs: Any,
) -> AutobranchEnv:
    """Instantiate an environment from a `"module:callable"` entrypoint.

    `kwargs` are forwarded to the factory; `search_path` is reserved by this
    function and is the bundle root to import from.
    """
    factory = import_target(entrypoint, search_path=search_path)
    if not callable(factory):
        raise TypeError(f"entrypoint {entrypoint!r} resolved to a non-callable {type(factory)}")
    with bundle_on_path(search_path):
        env = factory(**kwargs)
    return env


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def blocking(problems: list[str]) -> list[str]:
    """Drop advisories, keeping only contract violations."""
    return [p for p in problems if not p.startswith(ADVISORY)]


def _as_obs_matrix(obs: Any, n_envs: int, where: str, problems: list[str]) -> np.ndarray | None:
    try:
        arr = np.asarray(obs, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        problems.append(f"{where}: observation is not array-like ({exc})")
        return None
    if arr.ndim == 1:
        if n_envs != 1:
            problems.append(
                f"{where}: observation has shape {arr.shape}; expected [n_envs={n_envs}, obs_dim]"
            )
            return None
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] != n_envs:
            problems.append(
                f"{where}: observation leading dim {arr.shape[0]} != n_envs {n_envs}"
            )
            return None
    else:
        problems.append(
            f"{where}: observation has {arr.ndim} dims {arr.shape}; "
            "flatten it to [n_envs, obs_dim] -- the policy builder is dense-only"
        )
        return None
    if not np.all(np.isfinite(arr)):
        problems.append(f"{where}: observation contains NaN or inf")
    return arr


def _check_flags(value: Any, name: str, n_envs: int, problems: list[str]) -> None:
    arr = np.atleast_1d(np.asarray(value))
    if arr.size != n_envs:
        problems.append(f"{name} has {arr.size} entries; expected n_envs={n_envs}")
        return
    if arr.dtype != np.bool_:
        try:
            numeric = arr.astype(np.float64)
        except (TypeError, ValueError):
            problems.append(f"{name} is not boolean or 0/1 numeric (dtype {arr.dtype})")
            return
        if not np.all(np.isin(numeric, (0.0, 1.0))):
            problems.append(f"{name} must be boolean or 0/1, got values {np.unique(numeric)[:5]}")


def _check_signals(
    spec: EnvironmentSpec, signals: Any, n_envs: int, where: str, problems: list[str]
) -> None:
    if not isinstance(signals, dict):
        problems.append(f"{where}: signals must be a dict, got {type(signals).__name__}")
        return
    declared = set(spec.signal_names)
    emitted = set(signals)
    for missing in sorted(declared - emitted):
        problems.append(f"{where}: declared signal {missing!r} was not emitted")
    for extra in sorted(emitted - declared):
        problems.append(
            f"{where}: signal {extra!r} is emitted but not declared in the manifest "
            "(an agent cannot build a reward from a signal it cannot see)"
        )
    for name in sorted(declared & emitted):
        sig = spec.signal(name)
        try:
            arr = np.asarray(signals[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            problems.append(f"{where}: signal {name!r} is not numeric ({exc})")
            continue
        if arr.ndim == 0:
            arr = np.full(n_envs, float(arr))
        elif arr.ndim == 1:
            if arr.size != n_envs:
                problems.append(
                    f"{where}: signal {name!r} has shape {arr.shape}; expected scalar or ({n_envs},)"
                )
                continue
        else:
            problems.append(
                f"{where}: signal {name!r} has shape {arr.shape}; expected scalar or ({n_envs},)"
            )
            continue
        if not np.all(np.isfinite(arr)):
            problems.append(f"{where}: signal {name!r} contains NaN or inf")
            continue
        if sig.type == "bool" and not np.all(np.isin(arr, (0.0, 1.0))):
            problems.append(f"{where}: signal {name!r} is declared bool but emitted {arr[:3]}")
        if sig.type == "int" and not np.all(arr == np.round(arr)):
            problems.append(f"{where}: signal {name!r} is declared int but emitted fractional values")
        if sig.range is not None:
            lo, hi = sig.range
            if arr.min() < lo - 1e-6 or arr.max() > hi + 1e-6:
                problems.append(
                    f"{where}: signal {name!r} ranged [{arr.min():g}, {arr.max():g}] "
                    f"outside declared [{lo:g}, {hi:g}] -- the agent is told the declared range, "
                    "so a wrong one silently distorts every reward it writes"
                )


def _sample_actions(spec: EnvironmentSpec, n_envs: int, rng: np.random.Generator) -> np.ndarray:
    policy = spec.policy
    if policy.kind == "discrete":
        n_actions = int(policy.n_actions or 2)
        return rng.integers(0, n_actions, size=n_envs)
    act_dim = int(policy.n_actions or 1)
    return rng.normal(size=(n_envs, act_dim)).astype(np.float32)


def validate_env(env: Any, spec: EnvironmentSpec, *, steps: int = 4, seed: int = 0) -> list[str]:
    """Step `env` for real and report everything that disagrees with `spec`.

    Returns human-readable problems, most specific first; an empty list means
    the bundle is safe to train. Entries prefixed with `ADVISORY` are smells,
    not violations -- use `blocking()` to filter.

    What is checked, and why each one is here: the signal set (a typo'd signal
    name is unobservable at runtime because a reward term over a missing
    signal silently contributes zero), shapes against `n_envs` (a broadcast
    bug turns a per-lane reward into a global one and learning quietly
    stops), `obs_dim` against the manifest (the policy is built from the
    manifest, so a mismatch is a crash several minutes into a run), the action
    extremes (an env that rejects its own declared action range fails only
    once exploration finds that action), and determinism under a fixed seed
    (branches are only comparable if runs are reproducible).
    """
    problems: list[str] = []

    try:
        n_envs = int(env.n_envs)
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal and worth reporting
        return [f"env.n_envs raised {type(exc).__name__}: {exc}"]
    if n_envs < 1:
        return [f"env.n_envs is {n_envs}; must be at least 1"]
    if n_envs == 1:
        problems.append(
            ADVISORY + "n_envs == 1; prefer >= 2 -- a size-1 leading dimension has tripped "
            "contiguity checks in native libraries (DESIGN.md lesson 10)"
        )
    if not callable(getattr(env, "close", None)):
        problems.append("env has no callable close(); the trainer must be able to release it")

    try:
        obs0 = env.reset(seed=seed)
    except TypeError as exc:
        problems.append(f"env.reset(seed=...) must accept a keyword seed ({exc})")
        try:
            obs0 = env.reset()
        except Exception as exc2:  # noqa: BLE001
            problems.append(f"env.reset() raised {type(exc2).__name__}: {exc2}")
            return problems
    except Exception as exc:  # noqa: BLE001
        problems.append(f"env.reset(seed={seed}) raised {type(exc).__name__}: {exc}")
        return problems

    matrix = _as_obs_matrix(obs0, n_envs, "reset()", problems)
    obs_dim = None if matrix is None else int(matrix.shape[1])
    declared_dim = spec.policy.obs_dim
    if obs_dim is not None and declared_dim is not None and obs_dim != declared_dim:
        problems.append(
            f"reset(): obs_dim is {obs_dim} but manifest declares policy.obs_dim={declared_dim}"
        )

    rng = np.random.default_rng(seed)
    action_log = [_sample_actions(spec, n_envs, rng) for _ in range(max(1, steps))]
    trace_a: list[tuple[np.ndarray, dict[str, Any]]] = []

    for i, action in enumerate(action_log):
        try:
            result = env.step(action)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"step {i}: env.step raised {type(exc).__name__}: {exc}")
            return problems
        if not isinstance(result, StepResult):
            problems.append(
                f"step {i}: env.step returned {type(result).__name__}, expected env_api.StepResult"
            )
            return problems
        mat = _as_obs_matrix(result.obs, n_envs, f"step {i}", problems)
        if mat is not None and obs_dim is not None and mat.shape[1] != obs_dim:
            problems.append(
                f"step {i}: obs_dim changed from {obs_dim} to {mat.shape[1]} mid-episode"
            )
        _check_signals(spec, result.signals, n_envs, f"step {i}", problems)
        _check_flags(result.terminated, f"step {i}: terminated", n_envs, problems)
        _check_flags(result.truncated, f"step {i}: truncated", n_envs, problems)
        if not isinstance(result.info, dict):
            problems.append(f"step {i}: info must be a dict, got {type(result.info).__name__}")
        trace_a.append(
            (
                np.zeros(0) if mat is None else mat,
                {k: np.atleast_1d(np.asarray(v, dtype=np.float64)) for k, v in result.signals.items()
                 if isinstance(v, (int, float, bool, np.generic, np.ndarray, list, tuple))},
            )
        )

    # Action extremes: exploration will reach them, so reject an env that cannot.
    if spec.policy.kind == "discrete" and spec.policy.n_actions:
        for action_value in (0, int(spec.policy.n_actions) - 1):
            try:
                env.step(np.full(n_envs, action_value, dtype=np.int64))
            except Exception as exc:  # noqa: BLE001
                problems.append(
                    f"env.step rejected the declared action {action_value} "
                    f"({type(exc).__name__}: {exc})"
                )

    # Determinism: replay the identical action log from the identical seed.
    try:
        obs_b = env.reset(seed=seed)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"second env.reset(seed={seed}) raised {type(exc).__name__}: {exc}")
        return problems
    mat_b = _as_obs_matrix(obs_b, n_envs, "reset() replay", problems)
    if matrix is not None and mat_b is not None and not np.allclose(matrix, mat_b):
        problems.append(
            f"reset(seed={seed}) returned different observations on two calls; "
            "branch comparisons require reproducible episodes"
        )
    for i, action in enumerate(action_log):
        try:
            result = env.step(action)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"replay step {i}: env.step raised {type(exc).__name__}: {exc}")
            break
        mat = _as_obs_matrix(result.obs, n_envs, f"replay step {i}", problems)
        ref_obs, ref_signals = trace_a[i]
        if mat is not None and ref_obs.size and not np.allclose(mat, ref_obs, atol=1e-9):
            problems.append(
                f"replay step {i}: observations differ from the first run under the same seed "
                "and the same actions"
            )
            break
        for name, ref in ref_signals.items():
            got = np.atleast_1d(np.asarray(result.signals.get(name, np.nan), dtype=np.float64))
            if ref.shape == got.shape and not np.allclose(ref, got, atol=1e-9):
                problems.append(f"replay step {i}: signal {name!r} differs under the same seed")
                break

    return problems
