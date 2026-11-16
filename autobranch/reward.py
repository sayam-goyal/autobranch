"""Reward configs are data, not code.

An agent writes JSON; this module compiles it into a callable over signals.
That choice is deliberate and load-bearing in three ways:

* **Safety.** Model-authored Python never executes inside the trainer. The op
  vocabulary below is the entire attack surface.
* **Searchability.** A config can be hashed for de-duplication, diffed against
  its parent to show exactly what a fork changed, and stored in a snapshot so
  a branch's lineage is reconstructible months later.
* **Honesty.** Every term is named, so `term_totals()` can report what each
  part of the reward actually contributed. The predecessor project shipped a
  policy whose training reward climbed while its real objective fell; per-term
  accounting is how that gets noticed early rather than at the end.

A config references only signals the environment declared. Anything else is a
compile error, raised at submission rather than discovered mid-run.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

TRANSFORM_OPS = ("clip", "scale", "sign", "tanh", "threshold", "abs", "pow")
SHAPING_KINDS = ("potential", "delta")

RewardConfig = dict[str, Any]


class RewardError(ValueError):
    """Malformed reward config, with a path into the document."""

def _require(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise RewardError(f"{where}: {msg}")


def _as_float(x: Any, where: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        raise RewardError(f"{where}: expected a number, got {x!r}") from None


# --------------------------------------------------------------------------
# numeric helpers that work on both scalars and per-env arrays
# --------------------------------------------------------------------------

def _np():
    import numpy as np

    return np


def _zeros_like_signals(signals: dict[str, Any]):
    """A zero accumulator shaped like whatever the env reports.

    Environments may be scalar (one episode) or vectorised (many), and we do
    not want two code paths, so the accumulator borrows its shape from the
    first array-valued signal and falls back to a plain float.
    """
    np = _np()
    for v in signals.values():
        arr = np.asarray(v)
        if arr.ndim > 0:
            return np.zeros(arr.shape, dtype=np.float64)
    return 0.0


def _apply_transform(value: Any, op: str, params: dict[str, Any], where: str):
    np = _np()
    v = np.asarray(value, dtype=np.float64)
    if op == "clip":
        lo = _as_float(params.get("lo", -math.inf), f"{where}.lo")
        hi = _as_float(params.get("hi", math.inf), f"{where}.hi")
        _require(lo <= hi, where, f"lo {lo} exceeds hi {hi}")
        return np.clip(v, lo, hi)
    if op == "scale":
        return v * _as_float(params.get("by", 1.0), f"{where}.by")
    if op == "sign":
        return np.sign(v)
    if op == "tanh":
        return np.tanh(v * _as_float(params.get("by", 1.0), f"{where}.by"))
    if op == "threshold":
        at = _as_float(params.get("at", 0.0), f"{where}.at")
        return (v >= at).astype(np.float64)
    if op == "abs":
        return np.abs(v)
    if op == "pow":
        p = _as_float(params.get("exp", 1.0), f"{where}.exp")
        # Keep the sign so `pow` stays usable on signed signals.
        return np.sign(v) * np.abs(v) ** p
    raise RewardError(f"{where}: unknown transform op {op!r}")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate(config: RewardConfig, spec) -> None:
    """Raise `RewardError` unless `config` is well formed for `spec`.

    Called before a branch is ever launched: a config referencing a signal
    that does not exist should cost an agent one correction turn, not a GPU
    hour.
    """
    _require(isinstance(config, dict), "reward", "expected a mapping")
    declared = set(spec.signal_names)
    unknown_top = set(config) - {"weights", "shaping", "transforms", "anneal", "expr", "notes"}
    _require(not unknown_top, "reward", f"unknown keys {sorted(unknown_top)}")

    weights = config.get("weights") or {}
    _require(isinstance(weights, dict), "reward.weights", "expected {signal: coefficient}")
    for name, coeff in weights.items():
        _require(name in declared, "reward.weights", f"undeclared signal {name!r}; declared: {sorted(declared)}")
        _as_float(coeff, f"reward.weights.{name}")

    transforms = config.get("transforms") or []
    _require(isinstance(transforms, list), "reward.transforms", "expected a list")
    for i, t in enumerate(transforms):
        where = f"reward.transforms[{i}]"
        _require(isinstance(t, dict), where, "expected a mapping")
        op = t.get("op")
        _require(op in TRANSFORM_OPS, where, f"op must be one of {TRANSFORM_OPS}, got {op!r}")
        sig = t.get("signal")
        _require(sig in declared, where, f"undeclared signal {sig!r}")

    shaping = config.get("shaping") or []
    _require(isinstance(shaping, list), "reward.shaping", "expected a list")
    for i, s in enumerate(shaping):
        where = f"reward.shaping[{i}]"
        _require(isinstance(s, dict), where, "expected a mapping")
        kind = s.get("kind", "potential")
        _require(kind in SHAPING_KINDS, where, f"kind must be one of {SHAPING_KINDS}, got {kind!r}")
        _require(s.get("signal") in declared, where, f"undeclared signal {s.get('signal')!r}")
        _as_float(s.get("scale", 1.0), f"{where}.scale")
        gamma = _as_float(s.get("gamma", 0.99), f"{where}.gamma")
        _require(0.0 < gamma <= 1.0, f"{where}.gamma", f"expected (0, 1], got {gamma}")

    anneal = config.get("anneal") or {}
    _require(isinstance(anneal, dict), "reward.anneal", "expected a mapping")
    for group, a in anneal.items():
        where = f"reward.anneal.{group}"
        _require(group in ("shaping", "weights"), where, "only 'shaping' and 'weights' can be annealed")
        _require(isinstance(a, dict), where, "expected a mapping")
        _as_float(a.get("from", 1.0), f"{where}.from")
        _as_float(a.get("to", 0.0), f"{where}.to")
        frac = _as_float(a.get("by_fraction", 1.0), f"{where}.by_fraction")
        _require(0.0 < frac <= 1.0, f"{where}.by_fraction", f"expected (0, 1], got {frac}")

    expr = config.get("expr")
    if expr is not None:
        _require(isinstance(expr, str), "reward.expr", "expected a string")
        _validate_expr(expr, declared)

    _require(
        bool(weights) or bool(shaping) or expr,
        "reward",
        "a reward must do something: give weights, shaping, or an expr",
    )


# --------------------------------------------------------------------------
# the restricted arithmetic escape hatch
# --------------------------------------------------------------------------

_EXPR_ALLOWED_FUNCS = {"min", "max", "abs", "clip", "tanh", "sign", "sqrt"}


def _validate_expr(expr: str, declared: set[str]) -> None:
    """Allow arithmetic over signals and nothing else.

    `eval` is never used. The expression is parsed to an AST, every node type
    is checked against a allowlist, and names must be declared signals. This
    exists because a fixed op vocabulary will occasionally miss a genuinely
    useful shape, and the alternative -- agents asking a human to extend the
    vocabulary mid-search -- defeats the point of the framework.
    """
    import ast

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise RewardError(f"reward.expr: not a valid expression: {exc}") from None

    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
        ast.Call, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.IfExp,
        ast.Mod,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise RewardError(
                f"reward.expr: {type(node).__name__} is not allowed; "
                "only arithmetic over signals and min/max/abs/clip/tanh/sign/sqrt"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _EXPR_ALLOWED_FUNCS:
                raise RewardError(
                    f"reward.expr: only {sorted(_EXPR_ALLOWED_FUNCS)} may be called"
                )
        if isinstance(node, ast.Name) and node.id not in declared | _EXPR_ALLOWED_FUNCS:
            raise RewardError(
                f"reward.expr: unknown name {node.id!r}; declared signals: {sorted(declared)}"
            )


def _eval_expr(expr: str, signals: dict[str, Any]):
    import ast

    np = _np()
    funcs = {
        "min": np.minimum, "max": np.maximum, "abs": np.abs,
        "clip": np.clip, "tanh": np.tanh, "sign": np.sign, "sqrt": np.sqrt,
    }

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return np.asarray(signals.get(node.id, 0.0), dtype=np.float64)
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / np.where(np.asarray(b) == 0, 1e-8, b)
            if isinstance(node.op, ast.Pow): return np.sign(a) * np.abs(a) ** b
            if isinstance(node.op, ast.Mod): return np.mod(a, b)
        if isinstance(node, ast.Call):
            return funcs[node.func.id](*[ev(a) for a in node.args])
        if isinstance(node, ast.Compare):
            a, b = ev(node.left), ev(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Gt): return (a > b).astype(np.float64)
            if isinstance(op, ast.GtE): return (a >= b).astype(np.float64)
            if isinstance(op, ast.Lt): return (a < b).astype(np.float64)
            if isinstance(op, ast.LtE): return (a <= b).astype(np.float64)
        if isinstance(node, ast.IfExp):
            cond = np.asarray(ev(node.test), dtype=np.float64)
            return np.where(cond > 0, ev(node.body), ev(node.orelse))
        raise RewardError(f"reward.expr: cannot evaluate {type(node).__name__}")

    return ev(ast.parse(expr, mode="eval"))


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------

@dataclass
class CompiledReward:
    """A reward function plus the bookkeeping that keeps it honest."""

    config: RewardConfig
    signal_names: tuple[str, ...]
    _weights: dict[str, float] = field(default_factory=dict)
    _transforms: tuple[dict, ...] = ()
    _shaping: tuple[dict, ...] = ()
    _anneal: dict[str, dict] = field(default_factory=dict)
    _expr: str | None = None
    _totals: dict[str, float] = field(default_factory=dict)
    _prev: dict[str, Any] = field(default_factory=dict)

    # -- annealing ------------------------------------------------------

    def _anneal_factor(self, group: str, progress: float) -> float:
        a = self._anneal.get(group)
        if not a:
            return 1.0
        lo, hi = float(a.get("from", 1.0)), float(a.get("to", 0.0))
        frac = float(a.get("by_fraction", 1.0))
        t = min(1.0, max(0.0, progress / frac)) if frac > 0 else 1.0
        return lo + (hi - lo) * t

    # -- evaluation -----------------------------------------------------

    def __call__(self, signals: dict[str, Any], *, progress: float = 0.0):
        """Compose the reward for one transition.

        `progress` is the fraction of training elapsed, used only for
        annealing. Per-term contributions accumulate into `_totals` on device-
        free floats so the caller can log what each term is actually buying
        without a second pass over the data.
        """
        np = _np()
        total = _zeros_like_signals(signals)
        w_factor = self._anneal_factor("weights", progress)
        s_factor = self._anneal_factor("shaping", progress)

        # Transforms rewrite a signal in place for every later consumer, so a
        # config can clip a spiky signal once and have weights and shaping
        # both see the tamed version.
        values = dict(signals)
        for i, t in enumerate(self._transforms):
            name = t["signal"]
            params = {k: v for k, v in t.items() if k not in ("op", "signal")}
            values[name] = _apply_transform(values.get(name, 0.0), t["op"], params, f"transform[{i}]")

        for name, coeff in self._weights.items():
            term = np.asarray(values.get(name, 0.0), dtype=np.float64) * (coeff * w_factor)
            total = total + term
            self._totals[name] = self._totals.get(name, 0.0) + float(np.sum(term))

        for i, s in enumerate(self._shaping):
            name = s["signal"]
            scale = float(s.get("scale", 1.0)) * s_factor
            cur = np.asarray(values.get(name, 0.0), dtype=np.float64)
            prev = self._prev.get(name)
            if prev is None:
                # No previous value yet: a shaping term contributes nothing on
                # the first step rather than a spurious full-magnitude jolt.
                term = np.zeros_like(cur)
            elif s.get("kind", "potential") == "potential":
                gamma = float(s.get("gamma", 0.99))
                term = (gamma * cur - prev) * scale
            else:
                term = (cur - prev) * scale
            self._prev[name] = cur
            total = total + term
            key = f"shaping:{name}"
            self._totals[key] = self._totals.get(key, 0.0) + float(np.sum(term))

        # Record current values for signals that only shaping reads.
        for s in self._shaping:
            self._prev.setdefault(s["signal"], np.asarray(values.get(s["signal"], 0.0), dtype=np.float64))

        if self._expr:
            term = np.asarray(_eval_expr(self._expr, values), dtype=np.float64)
            total = total + term
            self._totals["expr"] = self._totals.get("expr", 0.0) + float(np.sum(term))

        return total

    # -- bookkeeping ----------------------------------------------------

    def reset_episode_state(self) -> None:
        """Forget shaping history; call on env reset so potentials restart."""
        self._prev.clear()

    def term_totals(self, *, drain: bool = True) -> dict[str, float]:
        out = dict(self._totals)
        if drain:
            self._totals.clear()
        return out

    @property
    def fingerprint(self) -> str:
        return config_hash(self.config)


def compile_reward(config: RewardConfig, spec) -> CompiledReward:
    """Validate then compile. Always validates -- there is no unchecked path."""
    validate(config, spec)
    return CompiledReward(
        config=json.loads(json.dumps(config)),  # defensive deep copy
        signal_names=tuple(spec.signal_names),
        _weights={k: float(v) for k, v in (config.get("weights") or {}).items()},
        _transforms=tuple(config.get("transforms") or []),
        _shaping=tuple(config.get("shaping") or []),
        _anneal=dict(config.get("anneal") or {}),
        _expr=config.get("expr"),
    )


# --------------------------------------------------------------------------
# lineage helpers -- used by the agent tree and the artifact
# --------------------------------------------------------------------------

def config_hash(config: RewardConfig) -> str:
    """Stable short hash, for rejecting a reward the tree already tried."""
    blob = json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _canonical(config: RewardConfig) -> dict:
    """Drop annotations that do not change behaviour before hashing."""
    out = {k: v for k, v in config.items() if k != "notes"}
    if "weights" in out:
        out["weights"] = {k: float(v) for k, v in sorted(out["weights"].items())}
    return out


def diff(parent: RewardConfig, child: RewardConfig) -> dict[str, Any]:
    """What a fork actually changed.

    Surfaced in the UI next to each branch so a reader can see the mutation
    rather than re-reading two blobs of JSON.
    """
    changes: dict[str, Any] = {"weights": {}, "added": [], "removed": [], "other": []}
    pw = {k: float(v) for k, v in (parent.get("weights") or {}).items()}
    cw = {k: float(v) for k, v in (child.get("weights") or {}).items()}
    for name in sorted(set(pw) | set(cw)):
        a, b = pw.get(name), cw.get(name)
        if a is None:
            changes["added"].append({"signal": name, "to": b})
        elif b is None:
            changes["removed"].append({"signal": name, "from": a})
        elif abs(a - b) > 1e-12:
            changes["weights"][name] = {"from": a, "to": b}
    for key in ("shaping", "transforms", "anneal", "expr"):
        if json.dumps(parent.get(key), sort_keys=True) != json.dumps(child.get(key), sort_keys=True):
            changes["other"].append(key)
    return changes


def summarize(config: RewardConfig) -> str:
    """One-line human summary, for tree nodes and log lines."""
    parts = [f"{k}={float(v):g}" for k, v in sorted((config.get("weights") or {}).items())]
    for s in config.get("shaping") or []:
        parts.append(f"shape({s.get('signal')})x{float(s.get('scale', 1.0)):g}")
    if config.get("expr"):
        parts.append("expr")
    if config.get("anneal"):
        parts.append("anneal")
    return " ".join(parts) or "(empty)"
