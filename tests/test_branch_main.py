"""The branch loop, driven by scripted agents.

Every dependency is injected, so the loop's decisions -- including the ones
that cost money -- are exercised here rather than first observed in a live
run. The fake agent is a list of actions; the fake trainer and evaluator are
deterministic.
"""

import json
import os

import pytest

from autobranch.branch_main import BranchRunner, BranchState
from autobranch.orchestrator import Budget, BranchEnv, SearchPolicy
from autobranch.spec import EnvironmentSpec

MANIFEST = """
name: gridworld
entrypoint: grid_env:make_env
signals:
  - {name: reached_goal, doc: reached the goal with the key}
  - {name: stepped_in_lava, doc: entered lava}
  - {name: distance_to_goal, doc: normalised distance}
objective: {name: success_rate, evaluator: grid_env:evaluate}
policy: {kind: discrete, n_actions: 5, obs_dim: 16}
"""


# ---- fakes -----------------------------------------------------------

class FakeAction:
    def __init__(self, tool, args=None, rationale=""):
        self.tool = tool
        self.args = args or {}
        self.rationale = rationale


class FakeAgent:
    """Replays a script; records what it was told."""

    def __init__(self, script):
        self.script = list(script)
        self.observed = []
        self.contexts = []

    def next_action(self, context):
        self.contexts.append(context)
        if not self.script:
            raise RuntimeError("script exhausted")
        return self.script.pop(0)

    def observe(self, tool, result):
        self.observed.append((tool, result))


class FakeRegistry:
    def __init__(self, branches=None):
        self.events = []
        self.updates = []
        self.scores = []
        self.best = None
        self._branches = branches or []

    def append_event(self, run_id, level, message, branch_id=None):
        self.events.append((level, message))

    def update_branch(self, run_id, branch_id, **fields):
        self.updates.append(fields)

    def branches(self, run_id):
        return list(self._branches)

    def append_score(self, run_id, branch_id, *, step, score):
        self.scores.append((step, score))

    def get_best(self, run_id):
        return self.best

    def set_best(self, run_id, branch_id, score):
        self.best = {"branch_id": branch_id, "score": score}


class FakeTrainResult:
    def __init__(self, steps_done, checkpoint_path, metrics=None):
        self.steps_done = steps_done
        self.checkpoint_path = checkpoint_path
        self.metrics = metrics or {"loss": 0.1, "entropy": 0.9}


class FakeScore:
    def __init__(self, value, command="fake-eval"):
        self.value = value
        self.n = 100
        self.detail = {"stochastic": value, "deterministic": value - 0.02}
        self.command = command


@pytest.fixture
def spec():
    return EnvironmentSpec.from_yaml(MANIFEST)


@pytest.fixture
def env(spec, tmp_path):
    return BranchEnv(
        run_id="run-test", branch_id="b0", parent_id=None, depth=0,
        spec=spec.to_dict(), constraints={"text": "be quick"},
        budget=Budget(usd=10, hours=1, tokens=100_000).to_dict(),
        deadline_ts=1_000_000.0,
    )


def build(spec, env, tmp_path, script, *, scores=None, registry=None,
          forker=None, clock=None, budget=None, max_iterations=20):
    scores = list(scores or [])
    trained = {"n": 0}

    def trainer(**kw):
        trained["n"] += 1
        trained["steps"] = trained.get("steps", 0) + int(kw.get("steps") or 0)
        ckpt = str(tmp_path / f"step_{trained['steps']}.pt")
        open(ckpt, "wb").write(b"w")
        return FakeTrainResult(trained["steps"], ckpt)

    def evaluator(**kw):
        return FakeScore(scores.pop(0) if scores else 0.5)

    bud = budget or Budget(usd=10, hours=1, tokens=100_000)
    runner = BranchRunner(
        env=env, spec=spec, agent=FakeAgent(script),
        registry=registry or FakeRegistry(),
        policy=SearchPolicy(bud, spec.objective),
        trainer=trainer, evaluator=evaluator, forker=forker,
        workspace=str(tmp_path), clock=clock or (lambda: 500_000.0),
        max_iterations=max_iterations,
    )
    return runner, trained


# ---- happy path ------------------------------------------------------

def test_a_branch_trains_evaluates_and_reports(spec, env, tmp_path):
    runner, trained = build(spec, env, tmp_path, [
        FakeAction("propose_reward", {"reward": {"weights": {"reached_goal": 1.0}}}),
        FakeAction("train", {"steps": 1000}),
        FakeAction("evaluate"),
        FakeAction("report", {"summary": "good enough"}),
    ], scores=[0.62])

    out = runner.run()
    assert out["status"] == "succeeded"
    assert out["objective"] == pytest.approx(0.62)
    assert trained["n"] == 1


def test_baseline_reward_is_installed_when_none_is_given(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [FakeAction("abandon", {"reason": "done looking"})])
    runner.run()
    assert runner.state.reward_config is not None
    assert runner.state.reward_config["weights"]["reached_goal"] == 1.0


def test_objective_score_is_recorded_with_its_command(spec, env, tmp_path):
    reg = FakeRegistry()
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("train", {"steps": 500}), FakeAction("evaluate"),
        FakeAction("report", {"summary": "x"}),
    ], scores=[0.4], registry=reg)
    runner.run()
    assert reg.scores == [(500, 0.4)]
    assert any(u.get("score_command") == "fake-eval" for u in reg.updates)


# ---- the rule that reward is not progress ---------------------------

def test_train_result_says_training_metrics_are_not_progress(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("train", {"steps": 100}), FakeAction("abandon", {"reason": "stop"}),
    ])
    runner.run()
    tool, result = runner.agent.observed[0]
    assert tool == "train"
    assert "not progress" in result["note"]


def test_evaluate_before_training_is_rejected_not_crashed(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("evaluate"),
        FakeAction("train", {"steps": 100}),
        FakeAction("evaluate"),
        FakeAction("report", {"summary": "ok"}),
    ], scores=[0.3])
    out = runner.run()
    assert out["status"] == "succeeded"
    # The rejection came back to the agent as feedback.
    assert any("train before evaluating" in str(r) for _, r in runner.agent.observed)


# ---- stop authority: the orchestrator overrides the agent -----------

def test_plateau_stops_the_branch_even_if_the_agent_wants_to_continue(spec, env, tmp_path):
    script = []
    for _ in range(8):
        script += [FakeAction("train", {"steps": 100}), FakeAction("evaluate")]
    runner, _ = build(spec, env, tmp_path, script, scores=[0.30, 0.30, 0.29, 0.30, 0.30, 0.30])
    out = runner.run()
    assert out["status"] == "abandoned"
    assert "no objective improvement" in out["reason"]


def test_deadline_stops_the_branch_and_reserves_time(spec, env, tmp_path):
    # 200s left: inside the reserve, so no further work is permitted.
    runner, _ = build(spec, env, tmp_path, [FakeAction("train", {"steps": 100})],
                      clock=lambda: env.deadline_ts - 200)
    out = runner.run()
    assert out["status"] == "abandoned"
    assert "deadline" in out["reason"]


def test_spend_cap_stops_the_branch(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [FakeAction("train", {"steps": 100})],
                      budget=Budget(usd=1.0, hours=1))
    runner.state.spend_usd = 1.5
    out = runner.run()
    assert out["status"] == "abandoned"
    assert "cap" in out["reason"]


def test_iteration_cap_is_a_backstop(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [FakeAction("read_tree") for _ in range(10)],
                      max_iterations=3)
    out = runner.run()
    assert out["status"] == "abandoned"
    assert "iteration cap" in out["reason"]


# ---- agent authority -------------------------------------------------

def test_agent_can_abandon_with_a_reason(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("abandon", {"reason": "shaping cannot fix a missing key signal"}),
    ])
    out = runner.run()
    assert out["status"] == "abandoned"
    assert "missing key signal" in out["reason"]


def test_abandon_without_a_reason_is_refused(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("abandon", {"reason": ""}),
        FakeAction("abandon", {"reason": "actually here is why"}),
    ])
    out = runner.run()
    assert "actually here is why" in out["reason"]


def test_report_without_an_evaluation_is_refused(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("report", {"summary": "trust me"}),
        FakeAction("abandon", {"reason": "fine"}),
    ])
    out = runner.run()
    assert out["status"] == "abandoned"
    assert any("never been evaluated" in str(r) for _, r in runner.agent.observed)


# ---- de-duplication --------------------------------------------------

def test_a_reward_another_branch_already_measured_is_refused(spec, env, tmp_path):
    reg = FakeRegistry(branches=[{
        "branch_id": "b0.9", "status": "abandoned",
        "reward_config": {"weights": {"reached_goal": 1.0}},
        "scores": [{"step": 1, "score": 0.1}],
    }])
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("propose_reward", {"reward": {"weights": {"reached_goal": 1.0}}}),
        FakeAction("abandon", {"reason": "nothing new to try"}),
    ], registry=reg)
    runner.run()
    assert any("already been measured" in str(r) for _, r in runner.agent.observed)


def test_an_undeclared_signal_is_refused_as_feedback(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("propose_reward", {"reward": {"weights": {"telepathy": 1.0}}}),
        FakeAction("abandon", {"reason": "ok"}),
    ])
    runner.run()
    assert any("undeclared signal" in str(r) for _, r in runner.agent.observed)


# ---- forking ---------------------------------------------------------

def test_fork_passes_the_child_a_mutated_reward(spec, env, tmp_path):
    calls = []

    def forker(**kw):
        calls.append(kw)
        return {"snapshot_id": "im-snap-1", "child_branch_id": kw["child_branch_id"]}

    runner, _ = build(spec, env, tmp_path, [
        FakeAction("train", {"steps": 100}),
        FakeAction("fork", {"new_reward": {"weights": {"reached_goal": 1.0,
                                                       "distance_to_goal": -0.1}},
                            "note": "try a distance penalty"}),
        FakeAction("abandon", {"reason": "handed off"}),
    ], forker=forker)
    runner.run()

    assert len(calls) == 1
    assert calls[0]["child_branch_id"] == "b0.1"
    assert calls[0]["reward_config"]["weights"]["distance_to_goal"] == -0.1
    assert runner.state.children == 1


def test_fork_is_refused_past_the_depth_limit(spec, tmp_path):
    deep = BranchEnv(
        run_id="r", branch_id="b0.1.2.3", parent_id="b0.1.2", depth=3,
        spec=spec.to_dict(), constraints={}, budget=Budget(max_depth=3).to_dict(),
        deadline_ts=1_000_000.0,
    )
    runner, _ = build(spec, deep, tmp_path, [
        FakeAction("fork", {"new_reward": {"weights": {"reached_goal": 2.0}}}),
        FakeAction("abandon", {"reason": "cannot go deeper"}),
    ], forker=lambda **kw: {"snapshot_id": "x"}, budget=Budget(max_depth=3))
    runner.run()
    assert any("fork refused" in str(r) for _, r in runner.agent.observed)


def test_rollback_resolves_a_step_number_to_a_snapshot(spec, env, tmp_path):
    calls = []
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("rollback", {"checkpoint": 1500}),
        FakeAction("abandon", {"reason": "done"}),
    ], forker=lambda **kw: calls.append(kw) or {"ok": True})
    runner.state.snapshots = [
        {"snapshot_id": "im-a", "step": 1000},
        {"snapshot_id": "im-b", "step": 2000},
    ]
    runner.run()
    assert calls[0]["rollback_to"] == "im-a"  # at or before 1500


def test_rollback_without_a_usable_target_is_explained(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("rollback", {"checkpoint": 10}),
        FakeAction("abandon", {"reason": "ok"}),
    ], forker=lambda **kw: {"ok": True})
    runner.state.snapshots = [{"snapshot_id": "im-b", "step": 2000}]
    runner.run()
    assert any("no snapshot at or before" in str(r) for _, r in runner.agent.observed)


# ---- state durability ------------------------------------------------

def test_state_survives_a_restart(spec, env, tmp_path):
    runner, _ = build(spec, env, tmp_path, [
        FakeAction("train", {"steps": 700}), FakeAction("evaluate"),
        FakeAction("abandon", {"reason": "pausing"}),
    ], scores=[0.44])
    runner.run()

    # A forked or restarted process reads the same workspace.
    revived = BranchState.load(os.path.join(str(tmp_path), "branch_state.json"))
    assert revived.steps_done == 700
    assert revived.scores == [0.44]
    assert revived.checkpoint is not None


def test_state_write_is_atomic(spec, env, tmp_path):
    """A torn state file reads as a lost branch, so the write must be atomic."""
    st = BranchState(steps_done=5, scores=[0.1])
    path = os.path.join(str(tmp_path), "nested", "branch_state.json")
    st.save(path)
    assert json.loads(open(path).read())["steps_done"] == 5
    assert not os.path.exists(path + ".tmp")


def test_a_stopped_branch_still_publishes_its_best_score(spec, env, tmp_path):
    """Being stopped does not invalidate a measurement already taken."""
    reg = FakeRegistry()
    script = []
    for _ in range(8):
        script += [FakeAction("train", {"steps": 100}), FakeAction("evaluate")]
    runner, _ = build(spec, env, tmp_path, script,
                      scores=[0.55, 0.55, 0.54, 0.55, 0.55], registry=reg)
    out = runner.run()
    assert out["status"] == "abandoned"
    assert reg.best["score"] == pytest.approx(0.55)


# ---- context handed to the agent ------------------------------------

def test_agent_context_carries_siblings_and_remaining_budget(spec, env, tmp_path):
    reg = FakeRegistry(branches=[{
        "branch_id": "b0.1", "status": "running",
        "reward_config": {"weights": {"distance_to_goal": -1.0}},
        "scores": [{"step": 1, "score": 0.77}],
    }])
    runner, _ = build(spec, env, tmp_path,
                      [FakeAction("abandon", {"reason": "just looking"})], registry=reg)
    runner.run()
    ctx = runner.agent.contexts[0]
    assert ctx.branch_id == "b0"
    assert ctx.budget.usd == pytest.approx(10.0)
    assert any(getattr(s, "branch_id", None) == "b0.1" for s in ctx.siblings)
    assert "be quick" in ctx.notes
