"""The environment contract.

An environment arrives as a *bundle*: a directory holding `autobranch.yaml`
plus code. The manifest is the entire interface between someone's environment
and this framework, so it is parsed strictly -- a typo in a signal name should
fail here, at submission, not three hours into a training run.

The central rule the whole framework rests on: an environment declares
**signals** (facts about a transition) and an **objective** (the thing we are
actually optimising). Agents may compose rewards from signals. Agents may
never touch the objective. See `reward.py` for the composition side and
`objective.py` for the enforcement.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SIGNAL_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")

SignalType = Literal["float", "int", "bool"]
TrainerKind = str  # "ppo" | "grpo_lora" | "grpo_full" | "custom:module:fn"

#: The built-in trainers. `grpo_full` updates every parameter of a base model;
#: `grpo_lora` updates an adapter over one. Both are critic-free and
#: group-relative -- they differ in what they are allowed to write to, not in
#: the RL method, so a bundle picks between them on memory and fidelity rather
#: than on algorithm.
BUILTIN_TRAINERS = ("ppo", "grpo_lora", "grpo_full")


class SpecError(ValueError):
    """Raised for any malformed manifest, with a path into the document."""

def _require(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise SpecError(f"{where}: {msg}")


def _get(d: Any, key: str, where: str, *, default: Any = None, required: bool = False) -> Any:
    _require(isinstance(d, dict), where, f"expected a mapping, got {type(d).__name__}")
    if key not in d:
        _require(not required, where, f"missing required key {key!r}")
        return default
    return d[key]


@dataclass(frozen=True)
class Signal:
    """One observable fact an environment reports per step.

    `range` is advisory for the agent's prompt, not enforced at runtime: an
    environment that lies about its range produces a worse reward search, not
    a crash. `doc` is load-bearing -- it is the only description the agent
    sees, so a signal without a useful doc is a signal the agent will misuse.
    """

    name: str
    type: SignalType = "float"
    range: tuple[float, float] | None = None
    doc: str = ""

    @staticmethod
    def parse(raw: Any, where: str) -> "Signal":
        name = _get(raw, "name", where, required=True)
        _require(
            isinstance(name, str) and bool(SIGNAL_RE.match(name)),
            where,
            f"signal name {name!r} must match {SIGNAL_RE.pattern}",
        )
        stype = _get(raw, "type", where, default="float")
        _require(stype in ("float", "int", "bool"), f"{where}.type", f"unknown type {stype!r}")
        rng = _get(raw, "range", where)
        if rng is not None:
            _require(
                isinstance(rng, (list, tuple)) and len(rng) == 2,
                f"{where}.range",
                "expected [lo, hi]",
            )
            lo, hi = float(rng[0]), float(rng[1])
            _require(lo <= hi, f"{where}.range", f"lo {lo} exceeds hi {hi}")
            rng = (lo, hi)
        return Signal(name=name, type=stype, range=rng, doc=str(_get(raw, "doc", where, default="")))


@dataclass(frozen=True)
class Objective:
    """The immutable measurement.

    `evaluator` is a `module:callable` that returns a float. It is invoked in a
    context that must not import reward code -- enforced by
    `tests/test_objective_independence.py`, which walks the import graph. That
    test is the reason an agent cannot quietly make itself look good.
    """

    name: str
    evaluator: str
    direction: Literal["maximize", "minimize"] = "maximize"
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def sign(self) -> float:
        """+1 when larger is better, so callers can always just maximise."""
        return 1.0 if self.direction == "maximize" else -1.0

    def better(self, a: float, b: float) -> bool:
        """Is score `a` better than `b`?"""
        return a * self.sign > b * self.sign

    @staticmethod
    def parse(raw: Any, where: str) -> "Objective":
        name = _get(raw, "name", where, required=True)
        evaluator = _get(raw, "evaluator", where, required=True)
        _require(
            isinstance(evaluator, str) and bool(ENTRYPOINT_RE.match(evaluator)),
            f"{where}.evaluator",
            f"expected 'module:callable', got {evaluator!r}",
        )
        direction = _get(raw, "direction", where, default="maximize")
        _require(
            direction in ("maximize", "minimize"),
            f"{where}.direction",
            f"expected maximize|minimize, got {direction!r}",
        )
        budget = _get(raw, "budget", where, default={}) or {}
        _require(isinstance(budget, dict), f"{where}.budget", "expected a mapping")
        return Objective(name=str(name), evaluator=evaluator, direction=direction, budget=budget)


@dataclass(frozen=True)
class ImageSpec:
    """How to build the container this environment runs in.

    Deliberately a thin description rather than a `modal.Image`: the manifest
    is data that travels through a Dict and a web form, and must stay
    JSON-round-trippable. `builder.py` turns this into a real Image.
    """

    base: str = "debian_slim"
    python: str = "3.11"
    apt: tuple[str, ...] = ()
    pip: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    secrets: tuple[str, ...] = ()
    volumes: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def parse(raw: Any, where: str) -> "ImageSpec":
        raw = raw or {}
        _require(isinstance(raw, dict), where, "expected a mapping")
        def _strs(key: str) -> tuple[str, ...]:
            v = _get(raw, key, where, default=[]) or []
            _require(isinstance(v, (list, tuple)), f"{where}.{key}", "expected a list")
            return tuple(str(x) for x in v)

        volumes = _get(raw, "volumes", where, default={}) or {}
        _require(isinstance(volumes, dict), f"{where}.volumes", "expected {path: volume_name}")
        envs = _get(raw, "env", where, default={}) or {}
        _require(isinstance(envs, dict), f"{where}.env", "expected {KEY: value}")
        for mount in volumes:
            _require(str(mount).startswith("/"), f"{where}.volumes", f"{mount!r} must be absolute")
        return ImageSpec(
            base=str(_get(raw, "base", where, default="debian_slim")),
            python=str(_get(raw, "python", where, default="3.11")),
            apt=_strs("apt"),
            pip=_strs("pip"),
            commands=_strs("commands"),
            env={str(k): str(v) for k, v in envs.items()},
            secrets=_strs("secrets"),
            volumes={str(k): str(v) for k, v in volumes.items()},
        )


@dataclass(frozen=True)
class Resources:
    gpu: str | None = None
    cpu: float = 2.0
    memory: int = 8192

    @staticmethod
    def parse(raw: Any, where: str) -> "Resources":
        raw = raw or {}
        _require(isinstance(raw, dict), where, "expected a mapping")
        gpu = _get(raw, "gpu", where)
        cpu = float(_get(raw, "cpu", where, default=2.0))
        memory = int(_get(raw, "memory", where, default=8192))
        _require(cpu > 0, f"{where}.cpu", "must be positive")
        _require(memory >= 512, f"{where}.memory", "must be at least 512 MiB")
        return Resources(gpu=None if gpu in (None, "", "none") else str(gpu), cpu=cpu, memory=memory)


@dataclass(frozen=True)
class PolicySpec:
    kind: Literal["discrete", "continuous"] = "discrete"
    n_actions: int | None = None
    obs_dim: int | None = None
    base_model: str | None = None  # set by a "must posttrain X" constraint

    @staticmethod
    def parse(raw: Any, where: str) -> "PolicySpec":
        raw = raw or {}
        kind = _get(raw, "kind", where, default="discrete")
        _require(kind in ("discrete", "continuous"), f"{where}.kind", f"got {kind!r}")
        n_actions = _get(raw, "n_actions", where)
        obs_dim = _get(raw, "obs_dim", where)
        if kind == "discrete":
            _require(n_actions is not None, f"{where}.n_actions", "required for discrete policies")
        if n_actions is not None:
            _require(int(n_actions) >= 2, f"{where}.n_actions", "need at least 2 actions")
        base_model = _get(raw, "base_model", where)
        return PolicySpec(
            kind=kind,
            n_actions=int(n_actions) if n_actions is not None else None,
            obs_dim=int(obs_dim) if obs_dim is not None else None,
            base_model=str(base_model) if base_model else None,
        )


@dataclass(frozen=True)
class EnvironmentSpec:
    """A fully validated environment bundle manifest."""

    name: str
    entrypoint: str
    signals: tuple[Signal, ...]
    objective: Objective
    image: ImageSpec = field(default_factory=ImageSpec)
    resources: Resources = field(default_factory=Resources)
    policy: PolicySpec = field(default_factory=PolicySpec)
    trainer: TrainerKind = "ppo"
    version: int = 1
    description: str = ""

    @property
    def signal_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.signals)

    def signal(self, name: str) -> Signal:
        for s in self.signals:
            if s.name == name:
                return s
        raise KeyError(name)

    # ---- parsing -------------------------------------------------------

    @staticmethod
    def parse(raw: Any) -> "EnvironmentSpec":
        _require(isinstance(raw, dict), "manifest", "expected a mapping at the top level")

        name = _get(raw, "name", "manifest", required=True)
        _require(
            isinstance(name, str) and bool(NAME_RE.match(name)),
            "manifest.name",
            f"{name!r} must match {NAME_RE.pattern}",
        )

        entrypoint = _get(raw, "entrypoint", "manifest", required=True)
        _require(
            isinstance(entrypoint, str) and bool(ENTRYPOINT_RE.match(entrypoint)),
            "manifest.entrypoint",
            f"expected 'module:callable', got {entrypoint!r}",
        )

        raw_signals = _get(raw, "signals", "manifest", required=True)
        _require(
            isinstance(raw_signals, (list, tuple)) and len(raw_signals) > 0,
            "manifest.signals",
            "at least one signal is required -- a reward cannot be composed from nothing",
        )
        signals = tuple(
            Signal.parse(s, f"manifest.signals[{i}]") for i, s in enumerate(raw_signals)
        )
        seen: set[str] = set()
        for s in signals:
            _require(s.name not in seen, "manifest.signals", f"duplicate signal {s.name!r}")
            seen.add(s.name)

        objective = Objective.parse(
            _get(raw, "objective", "manifest", required=True), "manifest.objective"
        )
        trainer = str(_get(raw, "trainer", "manifest", default="ppo"))
        _require(
            trainer in BUILTIN_TRAINERS or trainer.startswith("custom:"),
            "manifest.trainer",
            f"expected {'|'.join(BUILTIN_TRAINERS)}|custom:module:fn, got {trainer!r}",
        )

        return EnvironmentSpec(
            name=name,
            entrypoint=entrypoint,
            signals=signals,
            objective=objective,
            image=ImageSpec.parse(_get(raw, "image", "manifest"), "manifest.image"),
            resources=Resources.parse(_get(raw, "resources", "manifest"), "manifest.resources"),
            policy=PolicySpec.parse(_get(raw, "policy", "manifest"), "manifest.policy"),
            trainer=trainer,
            version=int(_get(raw, "version", "manifest", default=1)),
            description=str(_get(raw, "description", "manifest", default="")),
        )

    @staticmethod
    def from_yaml(text: str) -> "EnvironmentSpec":
        import yaml

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # pragma: no cover - message passthrough
            raise SpecError(f"manifest: invalid YAML: {exc}") from exc
        return EnvironmentSpec.parse(raw)

    # ---- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-round-trippable form, for Dicts, web payloads and artifacts."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "trainer": self.trainer,
            "signals": [
                {"name": s.name, "type": s.type, "range": list(s.range) if s.range else None,
                 "doc": s.doc}
                for s in self.signals
            ],
            "objective": {
                "name": self.objective.name,
                "evaluator": self.objective.evaluator,
                "direction": self.objective.direction,
                "budget": dict(self.objective.budget),
            },
            "image": {
                "base": self.image.base, "python": self.image.python,
                "apt": list(self.image.apt), "pip": list(self.image.pip),
                "commands": list(self.image.commands), "env": dict(self.image.env),
                "secrets": list(self.image.secrets), "volumes": dict(self.image.volumes),
            },
            "resources": {
                "gpu": self.resources.gpu, "cpu": self.resources.cpu,
                "memory": self.resources.memory,
            },
            "policy": {
                "kind": self.policy.kind, "n_actions": self.policy.n_actions,
                "obs_dim": self.policy.obs_dim, "base_model": self.policy.base_model,
            },
        }

    def agent_brief(self) -> str:
        """The environment as the agent sees it.

        Only signals, objective and constraints -- deliberately no environment
        internals. An agent that cannot read the simulator cannot special-case
        it, which keeps the reward search honest.
        """
        lines = [
            f"environment: {self.name}",
            f"objective: {self.objective.name} ({self.objective.direction})",
            "",
            "signals available to build a reward from:",
        ]
        for s in self.signals:
            rng = f" in [{s.range[0]:g}, {s.range[1]:g}]" if s.range else ""
            lines.append(f"  - {s.name} ({s.type}{rng}): {s.doc or 'no description given'}")
        if self.policy.base_model:
            lines += ["", f"policy must be post-trained from: {self.policy.base_model}"]
        return "\n".join(lines)


def with_base_model(spec: EnvironmentSpec, base_model: str) -> EnvironmentSpec:
    """Apply a 'must posttrain X' constraint without mutating the original."""
    import dataclasses

    policy = dataclasses.replace(spec.policy, base_model=base_model)
    # Only a bundle that declared no post-training at all gets a default, and
    # the default is the adapter: it is the one that fits on a single GPU. A
    # bundle that asked for `grpo_full` keeps it -- "must posttrain X" says
    # which weights to start from, never how much of them to update, and
    # silently turning a full fine-tune into a LoRA run would answer a
    # different question than the one the manifest asked.
    trainer = "grpo_lora" if spec.trainer == "ppo" else spec.trainer
    return dataclasses.replace(spec, policy=policy, trainer=trainer)
