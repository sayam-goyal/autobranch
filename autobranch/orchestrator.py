"""The agent tree: what runs where, and who may stop whom.

A branch is a sandbox. The root receives the spec and the constraints; its
agent proposes a reward, trains, evaluates against the immutable objective and
decides what to do next. Forking snapshots the workspace and starts a child
with it mounted, so the child inherits weights and history while writing
nowhere the parent can see. The parent keeps running.

Two authorities stop a branch, deliberately: the agent may abandon its own
line, and the orchestrator abandons one regardless when budget, deadline,
depth or patience runs out.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from autobranch.reward import RewardError, compile_reward, config_hash, summarize
from autobranch.spec import EnvironmentSpec, with_base_model

WORKSPACE = "/workspace"
# Objective evaluations without improvement before the orchestrator intervenes.
PATIENCE = 3
# Leave enough slack to evaluate and consolidate rather than dying mid-write.
RESERVE_SECONDS = 600


@dataclass
class Budget:
    """Caps the search. Every field is enforced, not advisory."""

    usd: float = 25.0
    hours: float = 2.0
    max_branches: int = 12
    max_depth: int = 3
    max_concurrent: int = 4
    tokens: int = 4_000_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def parse(raw: dict[str, Any] | None) -> "Budget":
        raw = raw or {}
        b = Budget()
        for key in ("usd", "hours", "max_branches", "max_depth", "max_concurrent", "tokens"):
            if key in raw and raw[key] is not None:
                setattr(b, key, type(getattr(b, key))(raw[key]))
        if b.usd <= 0 or b.hours <= 0:
            raise ValueError("budget.usd and budget.hours must be positive")
        return b


@dataclass
class Constraints:
    """What the user asked for, in their own words plus what we parsed out.

    The free text goes to the agent verbatim -- a model reads "stay aggressive
    early then tighten up" better than any schema we could invent. Only the
    machine-actionable parts are extracted here.
    """

    text: str = ""
    base_model: str | None = None
    max_gpu: str | None = None
    target_score: float | None = None

    @staticmethod
    def parse(text: str) -> "Constraints":
        """Pull the few directives that change what we *launch*, not just what the agent thinks."""
        import re

        c = Constraints(text=text or "")
        low = c.text.lower()

        # "must posttrain qwen/qwen3-4b", "post-train Qwen3-8B", "finetune X"
        m = re.search(
            r"(?:post[- ]?train|posttrain|fine[- ]?tune|finetune)\s+(?:from\s+)?([\w./-]+)", low
        )
        if m:
            raw = m.group(1)
            # Recover original casing from the source text; model ids are case
            # sensitive and HF will 404 on a lowercased repo.
            idx = low.find(raw)
            c.base_model = c.text[idx:idx + len(raw)] if idx >= 0 else raw

        m = re.search(r"\b(h100|h200|b200|a100-80gb|a100-40gb|a100|l40s|l4|a10g|t4)\b", low)
        if m:
            c.max_gpu = m.group(1).upper().replace("A100-80GB", "A100-80GB")

        # "must beat 76%", "above 0.9", "at least 90 %"
        m = re.search(r"(?:beat|above|exceed|at least|over)\s+([0-9]*\.?[0-9]+)\s*%", low)
        if m:
            c.target_score = float(m.group(1)) / 100.0
        else:
            m = re.search(r"(?:beat|above|exceed|at least|over)\s+(0\.[0-9]+)\b", low)
            if m:
                c.target_score = float(m.group(1))
        return c

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchEnv:
    """Everything a sandbox needs to know about itself, passed as env vars.

    Serialised rather than imported because a sandbox is a fresh process with
    no access to the orchestrator's memory, and because an explicit payload is
    inspectable when a branch misbehaves.
    """

    run_id: str
    branch_id: str
    parent_id: str | None
    depth: int
    spec: dict[str, Any]
    constraints: dict[str, Any]
    budget: dict[str, Any]
    deadline_ts: float
    reward_config: dict[str, Any] | None = None
    snapshot_id: str | None = None

    def to_env(self) -> dict[str, str]:
        return {
            "AUTOBRANCH_RUN_ID": self.run_id,
            "AUTOBRANCH_BRANCH_ID": self.branch_id,
            "AUTOBRANCH_PARENT_ID": self.parent_id or "",
            "AUTOBRANCH_DEPTH": str(self.depth),
            "AUTOBRANCH_SPEC": json.dumps(self.spec),
            "AUTOBRANCH_CONSTRAINTS": json.dumps(self.constraints),
            "AUTOBRANCH_BUDGET": json.dumps(self.budget),
            "AUTOBRANCH_DEADLINE_TS": f"{self.deadline_ts:.0f}",
            "AUTOBRANCH_REWARD": json.dumps(self.reward_config or {}),
            "AUTOBRANCH_SNAPSHOT": self.snapshot_id or "",
            "AUTOBRANCH_WORKSPACE": WORKSPACE,
            # The sandbox image comes from the submitted manifest and contains
            # none of our code; the runtime rides the volume instead.
            "PYTHONPATH": "/data/lib:/bundle",
            "AUTOBRANCH_BUNDLE": "/bundle",
        }

    @staticmethod
    def from_env(environ: dict[str, str] | None = None) -> "BranchEnv":
        e = environ if environ is not None else os.environ
        return BranchEnv(
            run_id=e["AUTOBRANCH_RUN_ID"],
            branch_id=e["AUTOBRANCH_BRANCH_ID"],
            parent_id=e.get("AUTOBRANCH_PARENT_ID") or None,
            depth=int(e.get("AUTOBRANCH_DEPTH", "0")),
            spec=json.loads(e["AUTOBRANCH_SPEC"]),
            constraints=json.loads(e.get("AUTOBRANCH_CONSTRAINTS") or "{}"),
            budget=json.loads(e.get("AUTOBRANCH_BUDGET") or "{}"),
            deadline_ts=float(e.get("AUTOBRANCH_DEADLINE_TS", "0")),
            reward_config=json.loads(e.get("AUTOBRANCH_REWARD") or "{}") or None,
            snapshot_id=e.get("AUTOBRANCH_SNAPSHOT") or None,
        )


@dataclass
class StopDecision:
    stop: bool
    reason: str = ""


class SearchPolicy:
    """The orchestrator's half of the stop authority.

    Kept as a small object rather than scattered `if`s so the rules are
    readable in one place and testable without Modal.
    """

    def __init__(self, budget: Budget, objective, *, patience: int = PATIENCE):
        self.budget = budget
        self.objective = objective
        self.patience = patience

    def should_stop_branch(
        self,
        *,
        scores: list[float],
        depth: int,
        spend_usd: float,
        tokens: int,
        seconds_left: float,
        best_sibling: float | None,
    ) -> StopDecision:
        if seconds_left <= RESERVE_SECONDS:
            return StopDecision(True, "global deadline reached; reserving time to consolidate")
        if spend_usd >= self.budget.usd:
            return StopDecision(True, f"branch spend ${spend_usd:.2f} reached the cap")
        if tokens >= self.budget.tokens:
            return StopDecision(True, f"token budget exhausted ({tokens})")
        if depth > self.budget.max_depth:
            return StopDecision(True, f"depth {depth} exceeds max_depth {self.budget.max_depth}")

        if len(scores) > self.patience:
            window = scores[-(self.patience + 1):]
            best_before = window[0]
            if not any(self.objective.better(s, best_before) for s in window[1:]):
                return StopDecision(
                    True,
                    f"no objective improvement over {self.patience} evaluations "
                    f"(held at {best_before:.4f})",
                )

        # A branch far behind a sibling, with evidence, is not worth more GPU.
        if best_sibling is not None and len(scores) >= 2:
            mine = max(scores, key=lambda s: s * self.objective.sign)
            gap = (best_sibling - mine) * self.objective.sign
            if gap > 0.25:
                return StopDecision(
                    True, f"dominated by a sibling by {gap:.3f} on the objective"
                )
        return StopDecision(False)

    def can_fork(self, *, n_branches: int, depth: int, n_live: int) -> StopDecision:
        """Inverted sense: `stop=True` means the fork is refused."""
        if n_branches >= self.budget.max_branches:
            return StopDecision(True, f"branch cap {self.budget.max_branches} reached")
        if depth + 1 > self.budget.max_depth:
            return StopDecision(True, f"max_depth {self.budget.max_depth} reached")
        if n_live >= self.budget.max_concurrent:
            return StopDecision(True, f"{n_live} branches already running (max_concurrent)")
        return StopDecision(False)


def plan_run(
    manifest_yaml: str,
    constraints_text: str = "",
    budget: dict[str, Any] | None = None,
) -> tuple[EnvironmentSpec, Constraints, Budget]:
    """Validate everything a submission needs before any resource is created.

    Called by the API on submit so a bad bundle is rejected in the browser
    rather than discovered by a sandbox ten minutes later.
    """
    spec = EnvironmentSpec.from_yaml(manifest_yaml)
    cons = Constraints.parse(constraints_text)
    bud = Budget.parse(budget)
    if cons.base_model:
        spec = with_base_model(spec, cons.base_model)
    return spec, cons, bud


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:10]}"


def child_branch_id(parent_id: str, n_existing_children: int) -> str:
    """`b0`, `b0.1`, `b0.1.2` -- depth and parentage legible at a glance."""
    return f"{parent_id}.{n_existing_children + 1}"


def initial_reward(spec: EnvironmentSpec) -> dict[str, Any]:
    """A deliberately plain starting point.

    The first config is intentionally close to "optimise the objective and
    nothing else": weight whatever signals look terminal, lightly penalise
    anything that reads as a cost, and let the agent discover shaping. Opening
    with elaborate shaping would hand the search a local optimum to defend,
    which is how the predecessor project spent six configurations at 0.000.
    """
    weights: dict[str, float] = {}
    for s in spec.signals:
        name = s.name
        # "won" is not a substring of "win". A signal named `round_won` matched
        # nothing positive while `round_lost` matched the negative list, so the
        # SF3 baseline opened as {round_lost: -1.0} -- a reward with no term for
        # winning at all, which a policy maximises by stalling the clock rather
        # than by fighting. The past tense belongs here for the same reason the
        # present tense does.
        if any(k in name for k in (
            "win", "won", "success", "goal", "reach", "score", "solved", "victor",
        )):
            weights[name] = 1.0
        elif any(k in name for k in ("lose", "lost", "fail", "death", "died", "lava", "crash")):
            weights[name] = -1.0
        elif any(k in name for k in ("cost", "penalty", "time")):
            weights[name] = -0.01
    if not weights:
        # Nothing matched the heuristics: take the first signal at unit weight
        # so the branch is runnable and let the agent correct it immediately.
        weights[spec.signals[0].name] = 1.0
    return {"weights": weights, "notes": "auto-generated baseline; agent should refine"}


class ToolError(RuntimeError):
    """A tool call the agent got wrong; fed back as a correction turn."""

def validate_action(action, spec: EnvironmentSpec) -> None:
    """Shared gate so every call site rejects the same things the same way."""
    tool = getattr(action, "tool", None)
    args = getattr(action, "args", {}) or {}
    known = (
        "propose_reward", "train", "evaluate", "fork",
        "rollback", "abandon", "read_tree", "report",
    )
    if tool not in known:
        raise ToolError(f"unknown tool {tool!r}; choose one of {known}")

    if tool in ("propose_reward", "fork"):
        cfg = args.get("reward") or args.get("config") or args.get("new_reward")
        if not isinstance(cfg, dict):
            raise ToolError(f"{tool} requires a 'reward' object describing the reward config")
        try:
            compile_reward(cfg, spec)
        except RewardError as exc:
            raise ToolError(f"reward config rejected: {exc}") from None

    if tool == "train":
        steps = args.get("steps")
        if steps is None or int(steps) <= 0:
            raise ToolError("train requires a positive integer 'steps'")

    if tool == "rollback" and not args.get("checkpoint") and not args.get("snapshot_id"):
        raise ToolError("rollback requires 'checkpoint' or 'snapshot_id'")

    if tool == "abandon" and not str(args.get("reason", "")).strip():
        raise ToolError("abandon requires a non-empty 'reason' -- it goes in the report")


def dedup_key(reward_config: dict[str, Any]) -> str:
    """Branches must not re-explore a reward the tree already measured."""
    return config_hash(reward_config)


def describe_branch(branch_id: str, reward_config: dict[str, Any], score: float | None) -> str:
    s = "unscored" if score is None else f"{score:.4f}"
    return f"{branch_id} [{s}] {summarize(reward_config)}"


@dataclass
class TreeView:
    """What one agent is told about the rest of the tree.

    Siblings' results are shared deliberately: without them, branches
    rediscover each other's dead ends and the search degenerates into
    parallel random restarts.
    """

    branches: list[dict[str, Any]] = field(default_factory=list)

    def best_score(self, objective) -> float | None:
        scored = [b for b in self.branches if b.get("best_score") is not None]
        if not scored:
            return None
        return max((b["best_score"] for b in scored), key=lambda s: s * objective.sign)

    def tried_fingerprints(self) -> set[str]:
        return {
            dedup_key(b["reward_config"])
            for b in self.branches
            if b.get("reward_config")
        }

    def render(self, objective) -> str:
        if not self.branches:
            return "no other branches yet"
        lines = []
        for b in sorted(self.branches, key=lambda x: x.get("branch_id", "")):
            flag = {"abandoned": "x", "running": ">", "succeeded": "*"}.get(b.get("status", ""), "-")
            lines.append(
                f" {flag} {describe_branch(b.get('branch_id', '?'), b.get('reward_config') or {}, b.get('best_score'))}"
                + (f"  abandoned: {b['abandoned_reason']}" if b.get("abandoned_reason") else "")
            )
        best = self.best_score(objective)
        if best is not None:
            lines.append(f" best objective so far: {best:.4f}")
        return "\n".join(lines)


def seconds_left(deadline_ts: float, *, now: float | None = None) -> float:
    return max(0.0, deadline_ts - (now if now is not None else time.time()))
