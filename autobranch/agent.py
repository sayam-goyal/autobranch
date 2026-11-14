"""The reward-search agent: one branch's decision loop.

The agent acts through a closed tool set expressed as **strict JSON**, not
OpenAI function-calling. Three reasons, in order of how much they cost to
learn the other way:

1. Every action is a plain JSON object, so a run's whole decision history is
   diffable and replayable from `losers.json` without a model in the loop.
2. Validation happens here, against the environment's *declared signals*, so a
   reward referencing a signal the environment never reports is caught before
   a trainer touches a GPU -- and fed back to the model as a correction turn
   instead of crashing the branch.
3. It ports: any endpoint that can emit text can drive this loop.

The agent composes rewards. It never sees or scores itself -- `evaluate()`
calls an evaluator it cannot read, and the objective is the only number that
decides whether a branch lives. That separation is the one lesson the
predecessor project paid the most for; see `LESSON`.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from .kimi import Kimi, extract_json_object
from .spec import EnvironmentSpec

TOOLS = (
    "propose_reward",
    "train",
    "evaluate",
    "fork",
    "rollback",
    "abandon",
    "read_tree",
    "report",
)

LESSON = (
    "Self-play or training-reward improvement is not progress. In the project this "
    "framework generalizes, six reward configurations all scored 0.000 against a fixed "
    "opponent while self-play win rate read 0.807 over 6,705 matches. Only the objective "
    "counts."
)

#: Keys of a reward config whose entries name a signal. Everything a reward is
#: built from must resolve to a declared signal; this is the index of where to
#: look for those names.
_SIGNAL_BEARING = ("shaping", "transforms")

ACTION_SCHEMA_HINT = (
    '{"tool": "<one of ' + "|".join(TOOLS) + '>", '
    '"args": {<tool-specific arguments>}, '
    '"rationale": "<one or two sentences: why this action, now>"}'
)


# ---- errors -------------------------------------------------------------


class AgentError(RuntimeError):
    """Base for agent-loop failures."""


class ActionInvalid(AgentError):
    """A model-authored action that cannot be executed.

    Carries `feedback`: the exact text handed back to the model as a
    correction turn. Keeping the human-readable reason and the model-facing
    correction the same string is deliberate -- a divergence between them is a
    bug nobody notices until a branch loops forever on the same mistake.
    """

    def __init__(self, feedback: str, *, payload: Any = None) -> None:
        super().__init__(feedback)
        self.feedback = feedback
        self.payload = payload


class MalformedAction(ActionInvalid):
    """Not a JSON object, or missing the `tool` field."""


class UnknownTool(ActionInvalid):
    """`tool` is not in `TOOLS`."""


class BadArguments(ActionInvalid):
    """Required args missing, unexpected args present, or wrong types."""


class UndeclaredSignal(ActionInvalid):
    """A reward config references a signal the environment does not report."""


# ---- tools --------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One callable action, as the model is told about it and as it is checked.

    `args` doubles as documentation and as the allow-list: an argument absent
    from it is rejected rather than ignored, because a silently dropped
    `steps` is a branch that trained for the default number of steps and
    reported a number nobody can explain.
    """

    name: str
    args: dict[str, str] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    doc: str = ""
    reward_args: tuple[str, ...] = ()  # args validated against declared signals

    def render(self) -> str:
        lines = [f"- {self.name}({', '.join(self.args) or ''}) -- {self.doc}"]
        for arg, description in self.args.items():
            flag = "required" if arg in self.required else "optional"
            lines.append(f"    {arg} ({flag}): {description}")
        return "\n".join(lines)


TOOL_SPECS: dict[str, ToolSpec] = {
    "propose_reward": ToolSpec(
        name="propose_reward",
        args={
            "config": "reward config object: {weights:{signal:float}, shaping:[...], "
                      "transforms:[...], anneal:{...}}. Only declared signals may appear.",
        },
        required=("config",),
        doc="Validate and install a reward config on this branch. Does not train.",
        reward_args=("config",),
    ),
    "train": ToolSpec(
        name="train",
        args={"steps": "positive integer environment steps to train for"},
        required=("steps",),
        doc="Train the current branch with the installed reward. Returns training metrics "
            "only -- these are not progress.",
    ),
    "evaluate": ToolSpec(
        name="evaluate",
        args={},
        doc="Score this branch with the immutable evaluator. This is the only number that "
            "counts.",
    ),
    "fork": ToolSpec(
        name="fork",
        args={
            "new_reward": "reward config for the child branch (same shape as "
                          "propose_reward.config)",
            "note": "short label explaining the hypothesis this child tests",
        },
        required=("new_reward",),
        doc="Snapshot this branch and start a child that inherits weights, optimizer and "
            "history, then diverges.",
        reward_args=("new_reward",),
    ),
    "rollback": ToolSpec(
        name="rollback",
        args={"checkpoint": "checkpoint id from read_tree() ancestry"},
        required=("checkpoint",),
        doc="Re-mount an ancestor snapshot and continue from there. Cheap: use it to test "
            "whether a premise, not just a weight, was wrong.",
    ),
    "abandon": ToolSpec(
        name="abandon",
        args={"reason": "what was ruled out, in terms of the objective"},
        required=("reason",),
        doc="End this line. The reason is recorded in losers.json and is a real result.",
    ),
    "read_tree": ToolSpec(
        name="read_tree",
        args={},
        doc="Read ancestry and sibling objective scores. Use it before forking so you do "
            "not re-explore a sibling's dead end; exact repeat reward configs are rejected.",
    ),
    "report": ToolSpec(
        name="report",
        args={
            "summary": "what this line established, ranked by objective score",
            "best_branch": "branch id of the best result seen from here",
        },
        required=("summary",),
        doc="Surface the best branch upward. Does not end the branch.",
    ),
}


def render_tools() -> str:
    return "\n".join(TOOL_SPECS[name].render() for name in TOOLS)


# ---- actions and context ------------------------------------------------


@dataclass
class Action:
    """One validated tool call. `rationale` is kept for `reward_lineage.json`."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args), "rationale": self.rationale}


@dataclass(frozen=True)
class BranchSummary:
    """A sibling as this agent sees it: identity, verdict, and nothing else.

    No weights and no trajectories -- a branch that could read a sibling's
    internals would start imitating it, which collapses the search to one
    line.
    """

    branch_id: str
    status: str = "running"
    best_objective: float | None = None
    note: str = ""
    reward_digest: str = ""

    def render(self) -> str:
        score = "unscored" if self.best_objective is None else f"{self.best_objective:.4f}"
        bits = [f"{self.branch_id}: objective={score} status={self.status}"]
        if self.reward_digest:
            bits.append(f"reward={self.reward_digest}")
        if self.note:
            bits.append(f"note={self.note}")
        return "  - " + " ".join(bits)


@dataclass(frozen=True)
class Budget:
    """What is left, not what was granted -- the model only acts on remainder."""

    usd: float = 0.0
    gpu_minutes: float = 0.0
    tokens: int = 0
    wall_seconds: float = 0.0

    @property
    def nearly_exhausted(self) -> bool:
        """True once any axis is under 15% of a usable floor.

        Thresholds are absolute rather than fractional because the agent is
        never told the original grant; "two dollars left" is actionable,
        "15% left" is not.
        """
        return self.usd < 2.0 or self.gpu_minutes < 5.0 or self.tokens < 20_000 or (
            0.0 < self.wall_seconds < 120.0
        )

    def render(self) -> str:
        return (
            f"budget remaining: ${self.usd:.2f}, {self.gpu_minutes:.1f} gpu-minutes, "
            f"{self.tokens} model tokens, {self.wall_seconds / 60:.1f} wall-clock minutes"
        )


@dataclass(frozen=True)
class AgentContext:
    """Everything the agent is allowed to know when choosing its next action."""

    branch_id: str
    parent_id: str | None = None
    depth: int = 0
    step: int = 0
    reward_config: dict[str, Any] | None = None
    recent_metrics: dict[str, Any] = field(default_factory=dict)
    objective_scores: tuple[float, ...] = ()
    siblings: tuple[BranchSummary, ...] = ()
    budget: Budget = field(default_factory=Budget)
    evals_without_improvement: int = 0
    notes: str = ""

    @property
    def best_objective(self) -> float | None:
        return max(self.objective_scores) if self.objective_scores else None

    @property
    def best_sibling_objective(self) -> float | None:
        scored = [s.best_objective for s in self.siblings if s.best_objective is not None]
        return max(scored) if scored else None

    def render(self) -> str:
        lines = [
            f"branch: {self.branch_id} (parent={self.parent_id or 'none'}, "
            f"depth={self.depth}, step={self.step})",
            "",
            "reward config in play:",
            json.dumps(self.reward_config, indent=2) if self.reward_config else "  none yet",
            "",
            "recent training metrics (NOT progress):",
            json.dumps(self.recent_metrics, indent=2) if self.recent_metrics else "  none yet",
            "",
            "objective scores on this branch, oldest first:",
            "  " + (", ".join(f"{s:.4f}" for s in self.objective_scores) or "none yet"),
            f"  evaluations since last improvement: {self.evals_without_improvement}",
        ]
        lines += ["", "sibling branches:"]
        lines += [s.render() for s in self.siblings] or ["  none"]
        lines += ["", self.budget.render()]
        if self.notes:
            lines += ["", f"notes: {self.notes}"]
        lines += [
            "",
            "Reply with exactly one JSON action object. Schema:",
            ACTION_SCHEMA_HINT,
        ]
        return "\n".join(lines)


# ---- reward config validation ------------------------------------------


def validate_reward_config(config: Any, signal_names: tuple[str, ...], *, where: str) -> None:
    """Reject a reward config that is malformed or names an undeclared signal.

    Raises `UndeclaredSignal` or `BadArguments`, both of which the loop turns
    into a correction turn. This is the enforcement point for the framework's
    central rule: a reward is a function of declared signals and nothing else.
    """
    if not isinstance(config, dict):
        raise BadArguments(
            f"{where} must be a JSON object, got {type(config).__name__}.", payload=config
        )
    declared = set(signal_names)
    known = ", ".join(signal_names)

    weights = config.get("weights", {})
    if weights is not None and not isinstance(weights, dict):
        raise BadArguments(f"{where}.weights must be an object of signal -> number.")
    for name, value in (weights or {}).items():
        if name not in declared:
            raise UndeclaredSignal(
                f"{where}.weights names {name!r}, which this environment does not report. "
                f"Declared signals are: {known}. Rewrite the config using only those."
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise BadArguments(f"{where}.weights[{name!r}] must be a number, got {value!r}.")

    for key in _SIGNAL_BEARING:
        entries = config.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise BadArguments(f"{where}.{key} must be a list of objects.")
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise BadArguments(f"{where}.{key}[{i}] must be an object.")
            signal = entry.get("signal")
            if signal is None:
                raise BadArguments(f"{where}.{key}[{i}] is missing 'signal'.")
            if signal not in declared:
                raise UndeclaredSignal(
                    f"{where}.{key}[{i}].signal is {signal!r}, which this environment does "
                    f"not report. Declared signals are: {known}."
                )

    expr = config.get("expr")
    if expr is not None:
        if not isinstance(expr, str):
            raise BadArguments(f"{where}.expr must be a string of restricted arithmetic.")
        for token in _identifiers(expr):
            if token not in declared:
                raise UndeclaredSignal(
                    f"{where}.expr references {token!r}, which is neither a declared signal "
                    f"nor an allowed function. Declared signals are: {known}."
                )

    if not (weights or config.get("shaping") or expr):
        raise BadArguments(
            f"{where} composes nothing: give at least one entry in 'weights', 'shaping', "
            f"or 'expr'. Declared signals are: {known}."
        )


_EXPR_BUILTINS = frozenset({"min", "max", "abs", "clip", "sign", "exp", "log", "tanh", "sqrt"})


def _identifiers(expr: str) -> list[str]:
    out: list[str] = []
    token = ""
    for ch in expr + " ":
        if ch.isalnum() or ch == "_":
            token += ch
        else:
            if token and not token[0].isdigit() and token not in _EXPR_BUILTINS:
                out.append(token)
            token = ""
    return out


# ---- the agent ----------------------------------------------------------


class Agent:
    """One branch's decision loop over the closed tool set.

    Stateful: `transcript` is this branch's memory, and the client's affinity
    key is bound to the branch so that memory stays in one worker's prefix
    cache.
    """

    def __init__(
        self,
        kimi: Kimi,
        spec: EnvironmentSpec,
        *,
        constraints: str = "",
        branch_id: str = "root",
        system_extra: str = "",
        transcript_token_budget: int = 60_000,
        max_corrections: int = 3,
        max_tokens: int = 4096,
    ) -> None:
        self.kimi = kimi
        self.spec = spec
        self.constraints = constraints.strip()
        self.branch_id = branch_id
        self.system_extra = system_extra.strip()
        self.transcript_token_budget = int(transcript_token_budget)
        self.max_corrections = int(max_corrections)
        self.max_tokens = int(max_tokens)
        self.affinity_key = f"{spec.name}:{branch_id}"
        self.transcript: list[dict[str, str]] = []
        self.system_prompt = self._build_system_prompt()
        self.corrections = 0

    # ---- prompt --------------------------------------------------------

    def _build_system_prompt(self) -> str:
        sections = [
            "You are an autobranch reward-search agent. You are one branch of a tree of "
            "agents searching the reward-function space for a reinforcement learning "
            "environment. You act only through the tools below, one action per turn.",
            "",
            self.spec.agent_brief(),
            "",
            f"declared signals (the complete list; a reward may reference nothing else): "
            f"{', '.join(self.spec.signal_names)}",
            "",
            "HOW YOU ARE MEASURED",
            f"The objective is {self.spec.objective.name}, and it must be "
            f"{self.spec.objective.direction}d. It is computed by an evaluator you cannot "
            "see, read, call directly, or modify; `evaluate()` runs it for you and returns "
            "only the score. Reward is a means. The objective is the only measure of "
            "success, and the only number used to rank or prune branches.",
            "",
            f"LESSON: {LESSON}",
            "Training loss falling, episode return rising, and self-play win rate climbing "
            "are all compatible with an objective score of exactly zero. Treat training "
            "metrics as evidence that your reward is learnable, never as evidence that it "
            "is right. Call evaluate() often enough that you never run more than one "
            "training block on an unmeasured hypothesis.",
            "",
            "TOOLS",
            render_tools(),
            "",
            "OUTPUT FORMAT",
            "Reply with exactly one JSON object and no other text, no markdown fences, and "
            "no commentary: ",
            ACTION_SCHEMA_HINT,
            "Omit args you do not need; do not invent args that are not listed. If an "
            "action of yours is rejected, you will be told exactly why and must reply with "
            "a corrected action.",
            "",
            "WHEN TO ABANDON",
            "Abandoning is a result, not a failure, and its reason is published in "
            "losers.json. Call abandon(reason) when any of these holds:",
            "  - the objective has not improved over the last 3 evaluations on this branch, "
            "and rollback to an ancestor plus a different reward has already been tried;",
            "  - the budget is nearly exhausted (little money, gpu-time, token or "
            "wall-clock left) and this branch's best objective is well below the best "
            "sibling -- spending the remainder here cannot change the run's outcome;",
            "  - the hypothesis this branch was forked to test has been answered, "
            "including when the answer is no.",
            "Before abandoning, prefer read_tree() to check you are not abandoning the best "
            "line, and prefer rollback() over abandon() when the reward is suspect but the "
            "weights are not.",
            "",
            "STRATEGY",
            "  - read_tree() before forking, so you do not repeat a sibling's experiment; "
            "exact duplicate reward configs are rejected.",
            "  - Change one thing per fork. A fork that changes three weights at once "
            "teaches nothing about any of them.",
            "  - A fork is cheap (seconds) and inherits everything. Use it to test a "
            "premise, not only a weight.",
        ]
        if self.constraints:
            sections += [
                "",
                "USER CONSTRAINTS (hard requirements on the run; they do not change the "
                "objective or the signals):",
                self.constraints,
            ]
        if self.system_extra:
            sections += ["", self.system_extra]
        return "\n".join(sections)

    # ---- the loop ------------------------------------------------------

    def next_action(self, context: AgentContext) -> Action:
        """Ask for one action and validate it, correcting the model in place.

        Validation failures are *conversational*: the rejected action and the
        reason are both appended to the transcript, so the model sees its own
        mistake and the correction in the same context on the retry. After
        `max_corrections` attempts the error is raised, because a model that
        cannot name a declared signal in four tries will not in five and the
        orchestrator should prune the branch instead of paying for more turns.
        """
        turn = context.render()
        messages = [{"role": "system", "content": self.system_prompt}] + self.transcript + [
            {"role": "user", "content": turn}
        ]
        pending: list[dict[str, str]] = [{"role": "user", "content": turn}]
        last: ActionInvalid | None = None

        for attempt in range(self.max_corrections + 1):
            reply = self.kimi.chat(
                messages,
                max_tokens=self.max_tokens,
                json_object=True,
                affinity_key=self.affinity_key,
            )
            raw_text = reply.content
            try:
                action = self.parse_action(raw_text)
            except ActionInvalid as exc:
                last = exc
                if attempt >= self.max_corrections:
                    break
                self.corrections += 1
                correction = (
                    f"Your action was rejected: {exc.feedback}\n"
                    f"Reply with one corrected JSON action object and nothing else."
                )
                messages = messages + [
                    {"role": "assistant", "content": raw_text},
                    {"role": "user", "content": correction},
                ]
                pending += [
                    {"role": "assistant", "content": raw_text},
                    {"role": "user", "content": correction},
                ]
                continue
            self.transcript += pending + [{"role": "assistant", "content": raw_text}]
            self._compact()
            return action

        assert last is not None
        # Keep the failed exchange: a branch pruned for unparseable actions must
        # be explainable afterwards.
        self.transcript += pending
        self._compact()
        raise last

    def parse_action(self, text: str) -> Action:
        """Parse and fully validate one model-authored action."""
        try:
            blob = extract_json_object(text)
        except ValueError:
            raise MalformedAction(
                "I could not find a JSON object in your reply. Reply with exactly one JSON "
                f"object matching: {ACTION_SCHEMA_HINT}",
                payload=text[:400],
            ) from None
        try:
            raw = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise MalformedAction(
                f"Your JSON did not parse ({exc.msg} at character {exc.pos}). Reply with one "
                "syntactically valid JSON object and nothing else.",
                payload=blob[:400],
            ) from exc
        return self.validate_action(raw)

    def validate_action(self, raw: Any) -> Action:
        if not isinstance(raw, dict):
            raise MalformedAction(
                f"An action must be a JSON object, not a {type(raw).__name__}.", payload=raw
            )
        tool = raw.get("tool")
        if not isinstance(tool, str) or not tool:
            raise MalformedAction(
                "Your action is missing the required string field 'tool'. It must be one of: "
                + ", ".join(TOOLS),
                payload=raw,
            )
        if tool not in TOOL_SPECS:
            raise UnknownTool(
                f"{tool!r} is not a tool. Choose exactly one of: " + ", ".join(TOOLS),
                payload=raw,
            )
        spec = TOOL_SPECS[tool]

        args = raw.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise BadArguments(
                f"'args' must be a JSON object for {tool}, got {type(args).__name__}.",
                payload=raw,
            )

        missing = [a for a in spec.required if a not in args or args[a] is None]
        if missing:
            raise BadArguments(
                f"{tool} is missing required argument(s): {', '.join(missing)}. "
                f"Expected: {spec.render()}",
                payload=raw,
            )
        unexpected = [a for a in args if a not in spec.args]
        if unexpected:
            raise BadArguments(
                f"{tool} does not take argument(s): {', '.join(sorted(unexpected))}. "
                f"Allowed: {', '.join(spec.args) or 'none'}.",
                payload=raw,
            )

        if tool == "train":
            steps = args.get("steps")
            if isinstance(steps, bool) or not isinstance(steps, (int, float)) or int(steps) < 1:
                raise BadArguments(
                    f"train.steps must be a positive integer, got {steps!r}.", payload=raw
                )
            args = dict(args, steps=int(steps))

        for arg in spec.reward_args:
            if arg in args:
                validate_reward_config(
                    args[arg], self.spec.signal_names, where=f"{tool}.{arg}"
                )

        rationale = raw.get("rationale") or ""
        if not isinstance(rationale, str):
            rationale = json.dumps(rationale)
        return Action(tool=tool, args=dict(args), rationale=rationale.strip())

    # ---- memory --------------------------------------------------------

    def observe(self, tool: str, result: dict) -> None:
        """Record a tool result as the next thing the model reads."""
        body = json.dumps(result, default=str, indent=2)
        if len(body) > 4000:
            body = body[:4000] + f"\n... [truncated, {len(body)} chars total]"
        self.transcript.append(
            {"role": "user", "content": f"result of {tool}:\n{body}"}
        )
        self._compact()

    def transcript_tokens(self) -> int:
        return sum(_estimate_tokens(m["content"]) for m in self.transcript)

    def _compact(self) -> None:
        """Fold the oldest turns into one digest instead of dropping them.

        Summarising is done locally, with no model call: a summariser that
        itself costs tokens makes the budget non-monotonic, and the thing worth
        keeping -- which tool ran, with what reward, scoring what -- is
        mechanically extractable. The digest is explicit in the transcript so
        the model can tell compression from amnesia.
        """
        # One fold may not be enough if the surviving half is itself large, so
        # this repeats -- but each pass must consume at least one real turn or
        # it would re-fold the digest into itself forever.
        while self.transcript_tokens() > self.transcript_token_budget:
            lines, turns = self._split_digest()
            if len(turns) <= 1:
                break  # a single live turn over budget is the caller's problem
            fold = turns[: max(1, len(turns) // 2)]
            keep = turns[len(fold) :]
            lines += [line for line in (_digest_line(m) for m in fold) if line]
            digest = _DIGEST_HEADER + "\n" + "\n".join(
                _cap_digest(lines, max(1, self.transcript_token_budget // 2))
            )
            self.transcript = [{"role": "user", "content": digest}] + keep

    def _split_digest(self) -> tuple[list[str], list[dict[str, str]]]:
        """Separate an existing digest's lines from the un-summarised turns."""
        if self.transcript and self.transcript[0]["content"].startswith(_DIGEST_HEADER):
            body = self.transcript[0]["content"][len(_DIGEST_HEADER) :]
            return [line for line in body.split("\n") if line.strip()], self.transcript[1:]
        return [], list(self.transcript)


_DIGEST_HEADER = (
    "[compacted history -- earlier turns on this branch, summarised, nothing dropped]"
)
_ELISION = "  [... {n} older summary lines elided to stay inside the token budget ...]"


def _cap_digest(lines: list[str], token_budget: int) -> list[str]:
    """Bound the digest, eliding the oldest lines *visibly* if it overflows.

    An unbounded digest just moves the context wall. When it does overflow,
    the elision is stated in the text: the model must be able to tell that
    history was compressed, because a model that believes it has the full
    record will re-run a sibling's experiment.
    """
    kept: list[str] = []
    used = 0
    for line in reversed(lines):  # newest summary lines are the useful ones
        cost = _estimate_tokens(line)
        if used + cost > token_budget and kept:
            return [_ELISION.format(n=len(lines) - len(kept))] + kept
        used += cost
        kept.insert(0, line)
    return kept


def _digest_line(message: dict[str, str]) -> str:
    content = (message.get("content") or "").strip()
    if not content:
        return ""
    if message.get("role") == "assistant":
        try:
            raw = json.loads(extract_json_object(content))
            tool = raw.get("tool", "?")
            args = json.dumps(raw.get("args", {}), sort_keys=True)
            return f"  action {tool} args={_clip(args, 160)} rationale={_clip(raw.get('rationale', ''), 120)}"
        except (ValueError, json.JSONDecodeError):
            return f"  assistant: {_clip(content, 160)}"
    first = content.splitlines()[0]
    return f"  {_clip(first, 100)} -> {_clip(content[len(first):].strip(), 140)}"


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _estimate_tokens(text: str) -> int:
    """~4 chars per token. Deliberately an estimate: the point is to compact
    well before the 1M context wall, and a tokenizer dependency here would
    break the zero-dependency guarantee that `kimi.py` exists to keep."""
    return max(1, len(text) // 4)
