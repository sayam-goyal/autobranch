"""What runs inside a branch sandbox.

This is the loop: the agent proposes a reward, trains, measures against the
immutable objective, and decides whether to refine, fork, roll back, or give
up. It is the only module that both talks to a model and spends GPU, so it is
also where the guards live.

Three properties are deliberate.

**The agent never scores itself.** `evaluate` calls `autobranch.objective`,
which imports no reward code; the number it returns is the only thing that
counts toward the run's result. An agent can write any reward it likes and
cannot move its own measurement.

**Two parties can stop a branch.** The agent may `abandon`; the orchestrator's
`SearchPolicy` abandons it regardless on budget, deadline, depth, plateau or
domination. Either alone fails: an optimistic model would burn the budget on
one line, and a policy alone would discard the judgement that makes this
better than a grid sweep.

**Forking mounts a snapshot, never shares a directory.** Two trainers writing
one workspace silently corrupt the artifact they are both producing, and that
failure is invisible until the weights are loaded. Mounting makes it
impossible rather than merely discouraged.
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from autobranch.orchestrator import (
    BranchEnv, Budget, SearchPolicy, ToolError, TreeView, child_branch_id,
    dedup_key, initial_reward, seconds_left, validate_action,
)
from autobranch.reward import RewardError, compile_reward, summarize
from autobranch.spec import EnvironmentSpec

# How long a single training call may run before we insist on re-deciding.
# Long enough to be worth the overhead, short enough that a branch cannot
# blow the whole deadline inside one tool call.
MAX_TRAIN_SECONDS = 1200
DEFAULT_TRAIN_STEPS = 20_000
MAX_ITERATIONS = 40


@dataclass
class BranchState:
    """Everything this branch knows about itself, rebuilt on resume."""

    reward_config: dict[str, Any] | None = None
    scores: list[float] = field(default_factory=list)
    steps_done: int = 0
    checkpoint: str | None = None
    spend_usd: float = 0.0
    tokens: int = 0
    evals_without_improvement: int = 0
    last_metrics: dict[str, Any] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    children: int = 0

    def best(self, objective) -> float | None:
        if not self.scores:
            return None
        return max(self.scores, key=lambda s: s * objective.sign)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_config": self.reward_config,
            "scores": self.scores,
            "steps_done": self.steps_done,
            "checkpoint": self.checkpoint,
            "spend_usd": self.spend_usd,
            "tokens": self.tokens,
            "evals_without_improvement": self.evals_without_improvement,
            "snapshots": self.snapshots,
            "children": self.children,
        }

    @staticmethod
    def load(path: str) -> "BranchState":
        """Resume is the normal case for a forked branch, not an edge case."""
        try:
            with open(path) as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return BranchState()
        st = BranchState()
        for key in st.to_dict():
            if key in raw:
                setattr(st, key, raw[key])
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
        self.state = BranchState.load(self._state_path)
        if env.reward_config and not self.state.reward_config:
            self.state.reward_config = env.reward_config

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

    def _persist(self, **fields: Any) -> None:
        self.state.save(self._state_path)
        payload = {
            "status": fields.pop("status", "running"),
            "reward_config": self.state.reward_config or {},
            "spend_usd": self.state.spend_usd,
            "tokens": self.state.tokens,
            "checkpoint": self.state.checkpoint,
            **fields,
        }
        try:
            self.registry.update_branch(self.env.run_id, self.env.branch_id, **payload)
        except Exception as exc:  # noqa: BLE001
            print(f"registry update failed ({exc}); state is still on the volume", flush=True)

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

    def _decide(self):
        """Ask the agent for one action, with the tree's results in context."""
        tree = self._tree()
        spend, tokens = self._budget_spent()
        context = self._build_context(tree, spend, tokens)
        action = self.agent.next_action(context)
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
                    reward_digest=summarize(b.get("reward_config") or {}),
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
        steps = int(args.get("steps") or DEFAULT_TRAIN_STEPS)
        left = seconds_left(self.env.deadline_ts, now=self.clock())
        budget_s = min(MAX_TRAIN_SECONDS, max(60.0, left - 300.0))
        started = self.clock()
        result = self.trainer(
            spec=self.spec,
            reward_config=self.state.reward_config,
            steps=steps,
            workspace=self.workspace,
            resume=True,
            max_seconds=budget_s,
        )
        elapsed = self.clock() - started

        self.state.steps_done = getattr(result, "steps_done", self.state.steps_done + steps)
        self.state.checkpoint = getattr(result, "checkpoint_path", self.state.checkpoint)
        metrics = dict(getattr(result, "metrics", {}) or {})
        self.state.last_metrics = metrics

        if self.ledger is not None:
            try:
                self.ledger.record_compute(
                    seconds=elapsed, gpu=self.spec.resources.gpu,
                    cpu=self.spec.resources.cpu, memory_mib=self.spec.resources.memory,
                    branch_id=self.env.branch_id,
                )
                self.ledger.commit()
            except Exception:  # noqa: BLE001
                pass
        self._persist()
        self._event(f"trained {steps} steps in {elapsed:.0f}s (total {self.state.steps_done})")
        # Deliberately NOT returning throughput: steps are not progress.
        return {"ok": True, "steps_done": self.state.steps_done, "metrics": metrics,
                "note": "training metrics are not progress; call evaluate to learn anything"}

    def _tool_evaluate(self, args, action) -> dict[str, Any]:
        if not self.state.checkpoint:
            raise ToolError("nothing to evaluate yet -- train before evaluating")
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
        if prev_best is None or self.spec.objective.better(value, prev_best):
            self.state.evals_without_improvement = 0
        else:
            self.state.evals_without_improvement += 1

        try:
            self.registry.append_score(
                self.env.run_id, self.env.branch_id, step=self.state.steps_done, score=value
            )
        except Exception:  # noqa: BLE001
            pass
        self._persist(score_command=command, score_detail=detail,
                      best_score=self.state.best(self.spec.objective))
        self._event(
            f"objective {self.spec.objective.name} = {value:.4f} at step {self.state.steps_done}"
        )
        return {"ok": True, "objective": value, "detail": detail, "command": command,
                "best_on_this_branch": self.state.best(self.spec.objective),
                "evals_without_improvement": self.state.evals_without_improvement}

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
        info = self.forker(
            parent_env=self.env,
            child_branch_id=child_id,
            reward_config=cfg,
            note=str(args.get("note") or getattr(action, "rationale", "") or ""),
        )
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
        self.forker(
            parent_env=self.env, child_branch_id=None, reward_config=None,
            note="rollback", rollback_to=target,
        )
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
        fields: dict[str, Any] = {"status": status, "best_score": best}
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

    runner = BranchRunner(
        env=env, spec=spec, agent=agent, registry=registry,
        policy=SearchPolicy(budget, spec.objective),
        trainer=trainer, evaluator=evaluator,
        forker=make_forker(spec=spec, budget=budget),
        ledger=ledger, workspace=workspace,
    )
    outcome = runner.run()
    print(json.dumps(outcome), flush=True)
    return 0 if outcome.get("status") in ("succeeded", "abandoned") else 1


if __name__ == "__main__":
    sys.exit(main())
