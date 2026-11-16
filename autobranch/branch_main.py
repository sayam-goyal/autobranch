"""What runs inside a branch sandbox.

The loop: the agent proposes a reward, trains, measures against the immutable
objective, and decides whether to refine, fork, roll back or give up. It is
the only module that both talks to a model and spends GPU, so the guards live
here.

The agent never scores itself -- `evaluate` calls `autobranch.objective`,
which imports no reward code, so an agent can write any reward it likes and
cannot move its own measurement.

Two parties can stop a branch: the agent may `abandon`, and the orchestrator's
`SearchPolicy` abandons it regardless on budget, deadline, depth, plateau or
domination. Either alone fails.
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from autobranch.orchestrator import (
    RESERVE_SECONDS, BranchEnv, Budget, SearchPolicy, ToolError, TreeView,
    child_branch_id, dedup_key, initial_reward, seconds_left, validate_action,
)
from autobranch.reward import RewardError, compile_reward, summarize
from autobranch.spec import EnvironmentSpec

# How long a single training call may run before we insist on re-deciding.
# Long enough to be worth the overhead, short enough that a branch cannot
# blow the whole deadline inside one tool call.
MAX_TRAIN_SECONDS = 1200
DEFAULT_TRAIN_STEPS = 20_000
MAX_ITERATIONS = 40


def _digest_for_best(b: dict[str, Any]) -> str:
    """Describe the reward that produced a sibling's *best* score.

    A branch's live `reward_config` is the next thing it intends to try, so
    pairing it with an already-measured `best_score` asserts a causal link
    that does not exist. The search agent reads these pairs and reasons from
    them -- in the SF3 run it concluded "raising damage_dealt to 0.15 scored
    0.8333" about a score measured one minute *before* that weight was set,
    and then chose its next move on that basis. Attribute to the reward
    recorded on the winning score, and say so plainly when none was recorded
    rather than substituting the current one.
    """
    scores = b.get("scores") or []
    best = None
    for s in scores:
        if not isinstance(s, dict) or not isinstance(s.get("score"), (int, float)):
            continue
        if best is None or float(s["score"]) > float(best["score"]):
            best = s
    if best is not None and best.get("reward"):
        return summarize(best["reward"])
    if best is not None:
        return "(reward at measurement not recorded)"
    return summarize(b.get("reward_config") or {}) + " (not yet measured)"


@dataclass
class BranchState:
    """Everything this branch knows about itself, rebuilt on resume.

    `branch_id` is stamped so a loaded file can say whose it is. A fork
    mounts the parent's snapshot as its own /workspace, so this file arrives
    with the inherited weights, and without the stamp a child cannot tell
    "my own state, resuming" from "my parent's state, which is not mine".
    """

    branch_id: str = ""
    reward_config: dict[str, Any] | None = None
    scores: list[float] = field(default_factory=list)
    steps_done: int = 0
    checkpoint: str | None = None
    # Where the *durable* copy of this branch's best-scoring weights lives.
    #
    # Kept apart from `checkpoint` because that field answers a different
    # question -- "where did the trainer last write?" -- and every train()
    # call overwrites it with a sandbox-local path. A branch that trained
    # past its best score therefore ended with a `/workspace` path that dies
    # with the sandbox, and consolidation, reading it, reported the winning
    # weights unreachable while a perfectly good copy sat on the Volume.
    # Only `_publish_weights` writes this, so it always names a real file.
    published_checkpoint: str | None = None
    spend_usd: float = 0.0
    tokens: int = 0
    evals_without_improvement: int = 0
    last_metrics: dict[str, Any] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    children: int = 0
    activity: str = ""
    #: Which branch wrote this state. A fork boots from a *snapshot of its
    #: parent's workspace*, so the state file it loads is the parent's -- and
    #: the per-branch accumulators in it describe the parent, not the child.
    owner: str = ""

    def best(self, objective) -> float | None:
        if not self.scores:
            return None
        return max(self.scores, key=lambda s: s * objective.sign)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "reward_config": self.reward_config,
            "scores": self.scores,
            "steps_done": self.steps_done,
            "checkpoint": self.checkpoint,
            "published_checkpoint": self.published_checkpoint,
            "spend_usd": self.spend_usd,
            "tokens": self.tokens,
            "evals_without_improvement": self.evals_without_improvement,
            "snapshots": self.snapshots,
            "children": self.children,
            "activity": self.activity,
            "owner": self.owner,
        }

    @staticmethod
    def load(path: str, *, owner: str = "") -> "BranchState":
        """Resume is the normal case for a forked branch, not an edge case.

        Most of this state is meant to be inherited -- the reward config, the
        weights pointer, the score history are exactly what a fork is for.
        Two fields are not: `spend_usd` and `tokens` are per-branch
        accumulators, and a child that inherits them reports its parent's
        bill as its own for as long as it lives.
        """
        try:
            with open(path) as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            st = BranchState()
            st.owner = owner
            return st
        st = BranchState()
        for key in st.to_dict():
            if key in raw:
                setattr(st, key, raw[key])
        if owner and st.owner and st.owner != owner:
            # Inherited from another branch: keep what a fork is for, drop
            # what belongs to the branch that paid it.
            st.spend_usd = 0.0
            st.tokens = 0
        if owner:
            st.owner = owner
        return st

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp, path)  # atomic: a torn state file reads as a lost branch


class BranchRunner:
    """Drives one branch to completion.

    Dependencies are injected rather than imported at module scope so the loop
    is testable without Modal, a GPU, or a model endpoint -- which is the only
    way its stop logic gets exercised more than once.
    """

    def __init__(
        self,
        *,
        env: BranchEnv,
        spec: EnvironmentSpec,
        agent,
        registry,
        policy: SearchPolicy,
        trainer,
        evaluator,
        forker=None,
        ledger=None,
        workspace: str = "/workspace",
        clock=time.time,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.env = env
        self.spec = spec
        self.agent = agent
        self.registry = registry
        self.policy = policy
        self.trainer = trainer
        self.evaluator = evaluator
        self.forker = forker
        self.ledger = ledger
        self.workspace = workspace
        self.clock = clock
        self.max_iterations = max_iterations
        self.state = BranchState.load(self._state_path, owner=env.branch_id)
        self._usage_mark = None
        self._claim_state(env)

    def _claim_state(self, env) -> None:
        """Decide whether the state file on disk is ours, or our parent's.

        A fork gets the parent's /workspace mounted (`forking.py` calls
        `child.mount_image(workspace, snapshot)`), so branch_state.json
        arrives alongside the inherited weights. The weights and the step
        count really are continuous -- the child goes on training the
        parent's policy -- but everything describing *which branch this is*
        belongs to the parent. The reward above all: a child that keeps its
        parent's reward retrains what the parent already measured, and the
        search silently stops searching while still paying for a GPU per
        branch. That is not hypothetical; it is what run-25ed75aa0a did
        across four branches before this existed.
        """
        if self.state.branch_id:
            inherited = self.state.branch_id != env.branch_id
        elif env.reward_config and self.state.reward_config:
            # Written before the stamp existed. The fork tool refuses a
            # reward that some branch already measured, so a child's env
            # reward never equals the parent's -- a mismatch here means we
            # booted from a parent rather than from our own last life.
            inherited = dedup_key(env.reward_config) != dedup_key(self.state.reward_config)
        else:
            inherited = False

        if inherited:
            self.state.reward_config = env.reward_config
            self.state.scores = []
            self.state.evals_without_improvement = 0
            self.state.children = 0
            self.state.snapshots = []
            self.state.spend_usd = 0.0
            self.state.tokens = 0
            self.state.activity = ""
            self.state.last_metrics = {}
        elif env.reward_config and not self.state.reward_config:
            self.state.reward_config = env.reward_config
        self.state.branch_id = env.branch_id

    # -- paths ----------------------------------------------------------

    @property
    def _state_path(self) -> str:
        return os.path.join(self.workspace, "branch_state.json")

    # -- bookkeeping ----------------------------------------------------

    def _event(self, message: str, level: str = "info") -> None:
        try:
            self.registry.append_event(self.env.run_id, level, message, self.env.branch_id)
        except Exception:  # noqa: BLE001 - telemetry must never kill a branch
            print(f"[{level}] {message}", flush=True)

    def _activity(self, text: str) -> None:
        """Publish what this sandbox is doing right now.

        Derivable from the event log, but a watcher should not have to scrape
        a log to answer "what is this sandbox doing"; and the newest event is
        not always the current activity -- a long train call emits its event
        only on completion, so during training the log's last line describes
        the previous step. This field is written BEFORE the work starts.
        """
        self.state.activity = text
        try:
            self.registry.update_branch(
                self.env.run_id, self.env.branch_id,
                activity=text, activity_since=self.clock(),
            )
        except Exception:  # noqa: BLE001
            pass

    def _persist(self, **fields: Any) -> None:
        self.state.save(self._state_path)
        payload = {
            "status": fields.pop("status", "running"),
            "reward_config": self.state.reward_config or {},
            "spend_usd": self.state.spend_usd,
            "tokens": self.state.tokens,
            "checkpoint": self.state.checkpoint,
            "published_checkpoint": self.state.published_checkpoint,
            **fields,
        }
        try:
            self.registry.update_branch(self.env.run_id, self.env.branch_id, **payload)
        except Exception as exc:  # noqa: BLE001
            # Surfaced, not swallowed: a dropped patch means the tree the user
            # is watching and the artifact we ship are both missing this.
            print(f"registry update failed ({exc}); state is still on the volume", flush=True)
            try:
                self.registry.append_event(
                    self.env.run_id, "warn",
                    f"branch state not published: {type(exc).__name__}: {exc}",
                    self.env.branch_id,
                )
            except Exception:  # noqa: BLE001
                pass

    def _tree(self) -> TreeView:
        try:
            records = self.registry.branches(self.env.run_id)
        except Exception:  # noqa: BLE001
            return TreeView()
        out = []
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            if d.get("branch_id") == self.env.branch_id:
                continue
            scores = [s.get("score") for s in (d.get("scores") or []) if s.get("score") is not None]
            out.append({
                "branch_id": d.get("branch_id"),
                "status": d.get("status"),
                "reward_config": d.get("reward_config") or {},
                "best_score": max(scores) if scores else None,
                "abandoned_reason": d.get("abandoned_reason"),
            })
        return TreeView(branches=out)

    def _budget_spent(self) -> tuple[float, int]:
        """Prefer the ledger's shard aggregate; fall back to local accounting.

        The ledger is authoritative because it sees every writer; local state
        only knows what this branch did. When the ledger is unreachable we use
        the local figure rather than pretending spend is zero.
        """
        if self.ledger is not None:
            try:
                return float(self.ledger.total_usd()), self.state.tokens
            except Exception:  # noqa: BLE001
                pass
        return self.state.spend_usd, self.state.tokens

    # -- the loop -------------------------------------------------------

    def run(self) -> dict[str, Any]:
        self._event(f"branch {self.env.branch_id} started (depth {self.env.depth})")
        self._activity("starting up")
        if not self.state.reward_config:
            self.state.reward_config = initial_reward(self.spec)
            self._event(f"baseline reward: {summarize(self.state.reward_config)}")
        self._persist(status="running")

        outcome = {"status": "running", "reason": ""}
        for iteration in range(self.max_iterations):
            stop = self._should_stop()
            if stop.stop:
                outcome = self._finish("abandoned", stop.reason)
                break

            try:
                action = self._decide()
            except ToolError as exc:
                # Our own defence-in-depth validation rejected the action. That
                # is a correction turn, not a dead branch -- the agent gets the
                # complaint and another go.
                self.agent.observe("invalid_action", {"error": str(exc)})
                self._event(f"rejected action: {exc}", "warn")
                continue
            except Exception as exc:  # noqa: BLE001
                # A model that cannot produce a parseable action after its own
                # correction turns is a dead branch, not a crashed run.
                outcome = self._finish("failed", f"agent produced no usable action: {exc}")
                break

            try:
                result = self._dispatch(action)
            except ToolError as exc:
                # Recoverable: hand the complaint back and let it try again.
                self.agent.observe(action.tool, {"error": str(exc)})
                self._event(f"rejected {action.tool}: {exc}", "warn")
                continue
            except Exception as exc:  # noqa: BLE001
                outcome = self._finish("failed", f"{action.tool} raised {type(exc).__name__}: {exc}")
                self._event(traceback.format_exc(limit=4), "error")
                break

            self.agent.observe(action.tool, result)
            if result.get("terminal"):
                outcome = self._finish(result.get("status", "succeeded"), result.get("reason", ""))
                break
        else:
            outcome = self._finish("abandoned", f"iteration cap {self.max_iterations} reached")

        return outcome

    def _should_stop(self):
        spend, tokens = self._budget_spent()
        return self.policy.should_stop_branch(
            scores=list(self.state.scores),
            depth=self.env.depth,
            spend_usd=spend,
            tokens=tokens,
            seconds_left=seconds_left(self.env.deadline_ts, now=self.clock()),
            best_sibling=self._tree().best_score(self.spec.objective),
        )

    def _sync_spend(self) -> None:
        """Record **this branch's own** spend from the ledger.

        It used to record the ledger's run-wide total. Every branch then held
        the same number, and `registry._run_spend` sums branch records when it
        cannot reach the ledger itself -- so a four-branch run reported four
        times its real cost. Harmless where the ledger is readable (the budget
        guard reads it directly, so the cap was never enforced against the
        inflated figure) and wrong everywhere else, which is every off-volume
        reader: the local UI, the artifact, any analysis afterwards.
        """
        if self.ledger is None:
            return
        try:
            mine = self.ledger.by_branch().get(self.env.branch_id, 0.0)
            self.state.spend_usd = float(mine)
            self.state.tokens = int(self.ledger.tokens())
        except Exception:  # noqa: BLE001 - keep the last known figure
            pass

    def _record_llm_usage(self) -> None:
        """Bill the agent's own tokens. They are real money: $2/$10 per Mtok.

        Read as a delta off the client's cumulative counter, so a retry or a
        correction turn is charged exactly once.
        """
        kimi = getattr(self.agent, "kimi", None)
        if self.ledger is None or kimi is None:
            return
        try:
            from autobranch.kimi import Usage, usage_minus

            now = getattr(kimi, "total_usage", None)
            if now is None:
                return
            before = self._usage_mark or Usage()
            delta = usage_minus(now, before)
            self._usage_mark = now
            if delta.prompt_tokens or delta.completion_tokens:
                self.ledger.record_llm(
                    tokens_in=delta.prompt_tokens,
                    tokens_out=delta.completion_tokens,
                    branch_id=self.env.branch_id,
                    note="agent decision",
                )
                self.ledger.commit()
        except Exception:  # noqa: BLE001 - accounting must not kill a branch
            pass

    def _decide(self):
        """Ask the agent for one action, with the tree's results in context."""
        self._activity("thinking (asking the model for its next action)")
        tree = self._tree()
        spend, tokens = self._budget_spent()
        context = self._build_context(tree, spend, tokens)
        action = self.agent.next_action(context)
        self._record_llm_usage()
        self._sync_spend()
        validate_action(action, self.spec)  # defence in depth; the agent also validates
        return action

    def _build_context(self, tree: TreeView, spend: float, tokens: int):
        """Assemble the agent's view.

        Imported lazily and shaped defensively: the agent module owns these
        types, and a mismatch here should degrade to a dict rather than kill a
        branch mid-run.
        """
        from autobranch import agent as agent_mod

        siblings = []
        for b in tree.branches:
            try:
                siblings.append(agent_mod.BranchSummary(
                    branch_id=b["branch_id"],
                    status=b.get("status") or "unknown",
                    best_objective=b.get("best_score"),
                    note=b.get("abandoned_reason") or "",
                    reward_digest=_digest_for_best(b),
                ))
            except TypeError:
                siblings.append(b)

        budget = agent_mod.Budget(
            usd=max(0.0, self.policy.budget.usd - spend),
            gpu_minutes=0.0,
            tokens=max(0, self.policy.budget.tokens - tokens),
            wall_seconds=seconds_left(self.env.deadline_ts, now=self.clock()),
        )
        return agent_mod.AgentContext(
            branch_id=self.env.branch_id,
            parent_id=self.env.parent_id,
            depth=self.env.depth,
            step=self.state.steps_done,
            reward_config=self.state.reward_config or {},
            recent_metrics=dict(self.state.last_metrics),
            objective_scores=list(self.state.scores),
            siblings=siblings,
            budget=budget,
            evals_without_improvement=self.state.evals_without_improvement,
            notes=self.env.constraints.get("text", "") if self.env.constraints else "",
        )

    # -- tools ----------------------------------------------------------

    def _dispatch(self, action) -> dict[str, Any]:
        tool = action.tool
        args = action.args or {}
        handler = {
            "propose_reward": self._tool_propose,
            "train": self._tool_train,
            "evaluate": self._tool_evaluate,
            "fork": self._tool_fork,
            "rollback": self._tool_rollback,
            "abandon": self._tool_abandon,
            "read_tree": self._tool_read_tree,
            "report": self._tool_report,
        }[tool]
        return handler(args, action)

    def _tool_propose(self, args, action) -> dict[str, Any]:
        cfg = args.get("reward") or args.get("config")
        try:
            compile_reward(cfg, self.spec)
        except RewardError as exc:
            raise ToolError(str(exc)) from None
        if dedup_key(cfg) in self._tree().tried_fingerprints():
            raise ToolError(
                "that exact reward config has already been measured by another branch; "
                "change it or read_tree to see what has been tried"
            )
        self.state.reward_config = cfg
        self._persist(rationale=getattr(action, "rationale", "") or "")
        self._event(f"reward -> {summarize(cfg)}")
        return {"ok": True, "reward": summarize(cfg)}

    def _tool_train(self, args, action) -> dict[str, Any]:
        requested = int(args.get("steps") or DEFAULT_TRAIN_STEPS)
        self._activity(f"training {requested:,} steps")
        # The trainer has no wall-clock budget of its own, so the loop caps the
        # step count instead: a branch must not spend the whole deadline inside
        # one tool call and leave nothing for evaluation.
        left = seconds_left(self.env.deadline_ts, now=self.clock())
        requested = self._cap_steps(requested, left)
        # `trainer.train(steps=...)` is a cumulative TARGET ("run until
        # steps_done >= steps"), while an agent asking to "train 30k" means
        # 30k more. Translating here keeps the agent's model natural and stops
        # a second call on a resumed workspace from being a silent no-op.
        target = self.state.steps_done + max(1, requested)
        started = self.clock()
        result = self.trainer(
            spec=self.spec,
            reward_config=self.state.reward_config,
            steps=target,
            workspace=self.workspace,
            resume=True,
        )
        elapsed = self.clock() - started
        steps = requested

        self.state.steps_done = getattr(result, "steps_done", self.state.steps_done + steps)
        self.state.checkpoint = getattr(result, "checkpoint_path", self.state.checkpoint)
        metrics = dict(getattr(result, "metrics", {}) or {})
        wall = float(getattr(result, "wall_seconds", 0.0) or elapsed or 0.0)
        if wall > 0:
            metrics["wall_seconds"] = wall
            metrics["steps_per_second"] = steps / wall
        self.state.last_metrics = metrics

        # Compute is NOT recorded here any more. `start_compute_biller` bills
        # the sandbox's wall clock continuously, and billing again per call
        # would charge the same seconds twice -- training time is already
        # inside the interval the biller measured.
        self._sync_spend()
        self._persist()
        self._event(f"trained {steps} steps in {elapsed:.0f}s (total {self.state.steps_done})")
        # Deliberately NOT returning throughput: steps are not progress.
        return {"ok": True, "steps_done": self.state.steps_done, "metrics": metrics,
                "note": "training metrics are not progress; call evaluate to learn anything"}

    def _cap_steps(self, steps: int, seconds_left_now: float) -> int:
        """Shrink a requested step count to fit the remaining clock.

        The trainer measures work in steps, not seconds, so the only lever the
        loop has is the step count. The rate is learned from this branch's own
        history rather than assumed: a GPU environment and a toy CPU one differ
        by orders of magnitude, and a fixed guess would either waste the
        deadline or refuse to train at all.
        """
        usable = max(0.0, seconds_left_now - RESERVE_SECONDS)
        if usable <= 0:
            return 0
        rate = self.state.last_metrics.get("steps_per_second")
        if not rate:
            observed = self.state.last_metrics.get("wall_seconds")
            if observed and self.state.steps_done:
                rate = self.state.steps_done / max(1e-6, float(observed))
        if not rate or rate <= 0:
            # First call on this branch: no measured rate exists yet, so there
            # is no honest way to convert the clock into steps. This used to
            # allow DEFAULT_TRAIN_STEPS * 4, which on an environment costing
            # ~0.15 s per step is over three hours -- a first call that eats
            # the whole deadline and never evaluates.
            #
            # The ceiling is now the default, not four times it, and the real
            # protection lives in the trainer, which bounds its own wall clock
            # and returns what it has. A step ceiling cannot do this job: the
            # same number is seconds in one environment and hours in another.
            return min(steps, DEFAULT_TRAIN_STEPS)
        affordable = int(float(rate) * min(usable, MAX_TRAIN_SECONDS))
        return max(1, min(steps, affordable))

    def _tool_evaluate(self, args, action) -> dict[str, Any]:
        if not self.state.checkpoint:
            raise ToolError("nothing to evaluate yet -- train before evaluating")
        self._activity(f"evaluating {self.spec.objective.name} at step {self.state.steps_done:,}")
        score = self.evaluator(
            spec=self.spec,
            checkpoint_path=self.state.checkpoint,
            budget=self.spec.objective.budget,
        )
        value = float(getattr(score, "value", score))
        detail = dict(getattr(score, "detail", {}) or {})
        command = getattr(score, "command", "")

        prev_best = self.state.best(self.spec.objective)
        self.state.scores.append(value)
        improved = prev_best is None or self.spec.objective.better(value, prev_best)
        if improved:
            self.state.evals_without_improvement = 0
            published = self._publish_weights()
            if published:
                # Point the registry at the durable copy, which is what
                # consolidation will read long after this sandbox is gone.
                # Recorded on its own field too: `checkpoint` is overwritten
                # by the next train() call, so it cannot be the thing that
                # outlives the sandbox.
                self.state.checkpoint = published
                self.state.published_checkpoint = published
                self._event(f"weights published for score {value:.4f}")
        else:
            self.state.evals_without_improvement += 1

        try:
            self.registry.append_score(
                self.env.run_id,
                self.env.branch_id,
                step=self.state.steps_done,
                score=value,
                reward=self.state.reward_config or {},
            )
        except Exception as exc:  # noqa: BLE001
            # Still non-fatal -- a registry hiccup must not destroy a branch
            # that has already done the work. But it must not be silent
            # either: a bare `pass` here hid an `append_score` signature
            # mismatch, and the only visible symptom was an empty score
            # trajectory on a branch that had plainly evaluated.
            self._event(f"WARNING: score {value:.4f} not recorded in registry: {exc}")
        self._persist(score_command=command, score_detail=detail)
        self._event(
            f"objective {self.spec.objective.name} = {value:.4f} at step {self.state.steps_done}"
        )
        return {"ok": True, "objective": value, "detail": detail, "command": command,
                "best_on_this_branch": self.state.best(self.spec.objective),
                "evals_without_improvement": self.state.evals_without_improvement}

    def _publish_weights(self) -> str | None:
        """Copy this branch's checkpoint onto the Volume.

        `/workspace` is sandbox-local: only `/data` is mounted from the
        Volume, so a checkpoint written by the trainer dies with its sandbox.
        That made the framework's actual deliverable -- the weights -- the one
        thing it did not keep. Published on every new best, so the file that
        backs the winning score is durable before anything can terminate.
        """
        src = self.state.checkpoint
        if not src or not os.path.isfile(src):
            return None
        try:
            import shutil

            from autobranch import infra

            dst_dir = os.path.join(
                infra.DATA_ROOT, "runs", self.env.run_id, "branches", self.env.branch_id
            )
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copyfile(src, dst)
            # Sandbox writes to a mounted Volume persist on their own; commit
            # raises here because a Sandbox is not a Function.
            try:
                infra.volume.commit()
            except Exception:  # noqa: BLE001
                pass
            return dst
        except Exception as exc:  # noqa: BLE001
            self._event(f"could not publish weights: {type(exc).__name__}: {exc}", "warn")
            return None

    def _tool_fork(self, args, action) -> dict[str, Any]:
        if self.forker is None:
            raise ToolError("forking is unavailable in this run configuration")
        cfg = args.get("new_reward") or args.get("reward") or args.get("config")
        try:
            compile_reward(cfg, self.spec)
        except RewardError as exc:
            raise ToolError(str(exc)) from None
        if dedup_key(cfg) in self._tree().tried_fingerprints():
            raise ToolError("a branch already measured that exact reward; mutate it further")

        tree = self._tree()
        gate = self.policy.can_fork(
            n_branches=len(tree.branches) + 1,
            depth=self.env.depth,
            n_live=sum(1 for b in tree.branches if b.get("status") == "running"),
        )
        if gate.stop:
            raise ToolError(f"fork refused: {gate.reason}")

        child_id = child_branch_id(self.env.branch_id, self.state.children)
        self._activity(f"forking {child_id}")
        try:
            info = self.forker(
                parent_env=self.env,
                child_branch_id=child_id,
                reward_config=cfg,
                note=str(args.get("note") or getattr(action, "rationale", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            # Recoverable by design: the branch keeps its own line and may
            # retry, pick a different mutation, or carry on training.
            raise ToolError(
                f"the fork could not be created ({type(exc).__name__}: {exc}); "
                "your own branch is unaffected and you may continue or try again"
            ) from None
        self.state.children += 1
        if info.get("snapshot_id"):
            self.state.snapshots.append(
                {"snapshot_id": info["snapshot_id"], "step": self.state.steps_done}
            )
        self._persist()
        self._event(f"forked {child_id}: {summarize(cfg)}")
        return {"ok": True, "child_branch_id": child_id, "snapshot_id": info.get("snapshot_id"),
                "note": "the child inherits your weights and history; it will not disturb yours"}

    def _tool_rollback(self, args, action) -> dict[str, Any]:
        target = args.get("snapshot_id")
        if not target:
            want = args.get("checkpoint")
            # Accept a step number and resolve it to the nearest snapshot taken
            # at or before that step, so an agent can say "go back to 40k".
            try:
                want_step = int(want)
            except (TypeError, ValueError):
                raise ToolError(
                    "rollback needs a 'snapshot_id', or a 'checkpoint' step number that this "
                    f"branch has snapshotted; available: {[s['step'] for s in self.state.snapshots]}"
                ) from None
            candidates = [s for s in self.state.snapshots if s["step"] <= want_step]
            if not candidates:
                raise ToolError(
                    f"no snapshot at or before step {want_step}; "
                    f"available: {[s['step'] for s in self.state.snapshots]}"
                )
            target = max(candidates, key=lambda s: s["step"])["snapshot_id"]

        if self.forker is None:
            raise ToolError("rollback is unavailable in this run configuration")
        try:
            self.forker(
                parent_env=self.env, child_branch_id=None, reward_config=None,
                note="rollback", rollback_to=target,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolError(
                f"the rollback failed ({type(exc).__name__}: {exc}); your weights are "
                "unchanged, so continue from where you are or try a different snapshot"
            ) from None
        self._event(f"rolled back to snapshot {target}", "warn")
        return {"ok": True, "snapshot_id": target,
                "note": "weights and optimizer are restored; your score history is unchanged"}

    def _tool_abandon(self, args, action) -> dict[str, Any]:
        reason = str(args.get("reason") or "").strip()
        if not reason:
            raise ToolError("abandon requires a reason -- it goes in the report")
        return {"terminal": True, "status": "abandoned", "reason": reason, "ok": True}

    def _tool_read_tree(self, args, action) -> dict[str, Any]:
        tree = self._tree()
        return {"ok": True, "tree": tree.render(self.spec.objective),
                "best_elsewhere": tree.best_score(self.spec.objective),
                "branches": tree.branches}

    def _tool_report(self, args, action) -> dict[str, Any]:
        best = self.state.best(self.spec.objective)
        if best is None:
            raise ToolError("cannot report a branch that has never been evaluated")
        try:
            run_best = self.registry.get_best(self.env.run_id)
            run_value = (run_best or {}).get("score")
            if run_value is None or self.spec.objective.better(best, float(run_value)):
                self.registry.set_best(self.env.run_id, self.env.branch_id, best)
        except Exception:  # noqa: BLE001
            pass
        return {"terminal": True, "status": "succeeded", "ok": True,
                "reason": str(args.get("summary") or "reported"), "objective": best}

    # -- exit -----------------------------------------------------------

    def _finish(self, status: str, reason: str) -> dict[str, Any]:
        best = self.state.best(self.spec.objective)
        self._activity({"succeeded": "done", "abandoned": "abandoned",
                        "failed": "failed"}.get(status, status))
        fields: dict[str, Any] = {"status": status}
        if status in ("abandoned", "failed"):
            fields["abandoned_reason"] = reason
        self._persist(**fields)
        self._event(
            f"branch {self.env.branch_id} {status}"
            + (f": {reason}" if reason else "")
            + (f" (best {best:.4f})" if best is not None else " (never evaluated)"),
            "warn" if status in ("abandoned", "failed") else "info",
        )
        # A branch that found the best score so far must publish it even if it
        # was stopped rather than choosing to report.
        if best is not None:
            try:
                run_best = self.registry.get_best(self.env.run_id) or {}
                if run_best.get("score") is None or self.spec.objective.better(
                    best, float(run_best["score"])
                ):
                    self.registry.set_best(self.env.run_id, self.env.branch_id, best)
            except Exception:  # noqa: BLE001
                pass
        return {"status": status, "reason": reason, "objective": best,
                "branch_id": self.env.branch_id}


def start_compute_biller(ledger, spec, branch_id: str, *, every: float = 60.0):
    """Bill the sandbox's wall clock continuously, in the background.

    `_tool_train` records compute when a train call *returns*, which is wrong in
    two ways that compound. A 50-minute call bills nothing until it finishes, so
    the UI shows stale spend while real money burns -- measured at $0.01 against
    a true $2.41, understated 241x. And if the sandbox is **preempted mid-call
    it bills nothing, ever**: a preempted run cost ~$4.56 and was never recorded
    at all.
    """
    import threading
    import time as _time

    def loop() -> None:
        last = _time.time()
        while True:
            _time.sleep(every)
            now = _time.time()
            seconds, last = now - last, now
            if seconds <= 0:
                continue
            try:
                ledger.record_compute(
                    seconds=seconds, gpu=spec.resources.gpu,
                    cpu=spec.resources.cpu, memory_mib=spec.resources.memory,
                    branch_id=branch_id, note="sandbox uptime",
                )
                ledger.commit()
            except Exception:  # noqa: BLE001 - never kill a branch over accounting
                pass

    thread = threading.Thread(target=loop, name="compute-biller", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    """Sandbox entrypoint: `python -m autobranch.branch_main`.

    Everything comes from the environment because a sandbox is a fresh process
    with no access to the launcher's memory, and an explicit payload is
    inspectable when a branch misbehaves.
    """
    from autobranch import infra
    from autobranch.agent import Agent
    from autobranch.kimi import Kimi
    from autobranch.ledger import Ledger
    from autobranch.objective import evaluate as objective_evaluate
    from autobranch.registry import Registry
    from autobranch.trainer import trainer_for

    env = BranchEnv.from_env()
    spec = EnvironmentSpec.parse(env.spec)
    budget = Budget.parse(env.budget)
    workspace = os.environ.get("AUTOBRANCH_WORKSPACE", "/workspace")

    registry = Registry()
    ledger = Ledger(env.run_id, root=infra.DATA_ROOT, volume=infra.volume,
                    branch_id=env.branch_id)
    kimi = Kimi(affinity_key=f"{spec.name}:{env.branch_id}")
    agent = Agent(kimi, spec, constraints=(env.constraints or {}).get("text", ""),
                  branch_id=env.branch_id)

    train_fn = trainer_for(spec)

    def trainer(**kw):
        return train_fn(**kw)

    def evaluator(**kw):
        return objective_evaluate(**kw)

    from autobranch.forking import make_forker

    # Bill continuously from here on: the agent loop can sit inside one train
    # call for an hour, and a preempted sandbox never reaches the code that
    # would have billed it.
    start_compute_biller(ledger, spec, env.branch_id)

    runner = BranchRunner(
        env=env, spec=spec, agent=agent, registry=registry,
        policy=SearchPolicy(budget, spec.objective),
        trainer=trainer, evaluator=evaluator,
        # The registry must reach the forker: it is what registers a child
        # BEFORE the child starts. Without it the child's own loop has no
        # record to patch and every update raises KeyError, so a forked branch
        # trains invisibly and never appears in the tree.
        forker=make_forker(spec=spec, budget=budget, registry=registry),
        ledger=ledger, workspace=workspace,
    )
    outcome = runner.run()
    print(json.dumps(outcome), flush=True)

    # Terminate our own sandbox. The sandbox's main process is `sleep
    # infinity` (the loop runs as an exec), so without this it stays billable
    # until its timeout -- a finished run left 15 sandboxes alive for an hour
    # each. The loop's result is already on the volume and in the registry, so
    # there is nothing left to lose by exiting.
    try:
        from autobranch.forking import self_sandbox

        self_sandbox().terminate()
    except Exception as exc:  # noqa: BLE001
        print(f"could not terminate own sandbox ({exc}); it will stop at its timeout",
              flush=True)

    return 0 if outcome.get("status") in ("succeeded", "abandoned") else 1


if __name__ == "__main__":
    sys.exit(main())
