"""A fork must train the reward it was created to test.

A child's /workspace IS the parent's snapshot, mounted -- `forking.py` does
`child.mount_image(workspace, snapshot)`. So the parent's branch_state.json
arrives together with the parent's weights, and a fresh child boots holding
its parent's identity unless something stops it.

Observed live on run-25ed75aa0a: four branches whose agents had each proposed
a different reward were all recorded with the root's reward config, byte for
byte, including the root's `notes` string. Their own `rationale` fields
described three distinct reward edits, and the fork tool's dedup check
(`branch_main.py`: "a branch already measured that exact reward") proves the
proposals differed at fork time. The search was structurally a no-op: every
branch retrained the root's reward on a different H100.
"""

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

PARENT_REWARD = {"weights": {"reached_goal": 1.0}, "notes": "the parent's"}
PARENT_CKPT = "/data/runs/r/branches/b0/latest.pt"


class FakeAction:
    def __init__(self, tool, args=None, rationale=""):
        self.tool = tool
        self.args = args or {}
        self.rationale = rationale


class FakeAgent:
    def __init__(self, script):
        self.script = list(script)
        self.observed = []

    def next_action(self, context):
        if not self.script:
            raise RuntimeError("script exhausted")
        return self.script.pop(0)

    def observe(self, tool, result):
        self.observed.append((tool, result))


class FakeRegistry:
    def __init__(self):
        self.updates = []

    def append_event(self, *a, **k):
        pass

    def update_branch(self, run_id, branch_id, **fields):
        self.updates.append((branch_id, fields))

    def branches(self, run_id):
        return []

    def add_spend(self, *a, **k):
        pass

    def append_score(self, *a, **k):
        pass


class FakeTrainResult:
    def __init__(self, steps, ckpt):
        self.steps_done = steps
        self.checkpoint_path = ckpt
        self.metrics = {}
        self.wall_seconds = 1.0
        self.iterations = 1
        self.workspace = ""
        self.reward_fingerprint = "f" * 12


@pytest.fixture
def spec():
    return EnvironmentSpec.from_yaml(MANIFEST)


def parent_state(tmp_path, **over):
    """Write the state file a child finds in the snapshot it booted from."""
    state = BranchState(
        reward_config=dict(PARENT_REWARD),
        scores=[0.75, 0.875],
        steps_done=57_600,
        checkpoint=PARENT_CKPT,
        spend_usd=16.11,
        tokens=32_894,
        children=1,
        evals_without_improvement=2,
        snapshots=[{"snapshot_id": "im-parent", "step": 52_800}],
        activity="training 2,000 steps",
    )
    if hasattr(state, "branch_id"):
        state.branch_id = "b0"
    for k, v in over.items():
        setattr(state, k, v)
    state.save(str(tmp_path / "branch_state.json"))
    return state


def runner_for(spec, tmp_path, env):
    def trainer(**kw):
        return FakeTrainResult(1, str(tmp_path / "w.pt"))

    return BranchRunner(
        env=env, spec=spec, agent=FakeAgent([FakeAction("abandon", {"reason": "x"})]),
        registry=FakeRegistry(),
        policy=SearchPolicy(Budget(usd=10, hours=1, tokens=100_000), spec.objective),
        trainer=trainer, evaluator=lambda **kw: None, forker=None,
        workspace=str(tmp_path), clock=lambda: 500_000.0, max_iterations=2,
    )


def child_env(spec, reward_config, branch_id="b0.1", parent_id="b0", depth=1):
    return BranchEnv(
        run_id="run-test", branch_id=branch_id, parent_id=parent_id, depth=depth,
        spec=spec.to_dict(), constraints={},
        budget=Budget(usd=10, hours=1, tokens=100_000).to_dict(),
        deadline_ts=1_000_000.0, reward_config=reward_config,
    )


def test_a_fork_trains_the_reward_it_was_created_to_test(spec, tmp_path):
    parent_state(tmp_path)
    mine = {"weights": {"reached_goal": 1.0, "stepped_in_lava": -0.02},
            "notes": "reduce the lava penalty"}
    runner = runner_for(spec, tmp_path, child_env(spec, mine))

    assert runner.state.reward_config == mine, (
        "the child adopted its parent's reward, so the fork measured nothing new"
    )


def test_a_fork_does_not_inherit_its_parent_scores_or_fork_count(spec, tmp_path):
    parent_state(tmp_path)
    runner = runner_for(
        spec, tmp_path, child_env(spec, {"weights": {"reached_goal": 2.0}}))

    assert runner.state.scores == [], "the parent's scores became the child's"
    assert runner.state.children == 0, "fork numbering started mid-sequence"
    assert runner.state.evals_without_improvement == 0
    assert runner.state.snapshots == [], "the parent's snapshots became the child's"
    assert runner.state.spend_usd == 0.0, "the child was born owing the parent's spend"
    assert runner.state.tokens == 0


def test_a_fork_does_inherit_the_weights_and_the_step_count(spec, tmp_path):
    """The child resumes the parent's policy: that history IS continuous."""
    parent_state(tmp_path)
    runner = runner_for(
        spec, tmp_path, child_env(spec, {"weights": {"reached_goal": 2.0}}))

    assert runner.state.steps_done == 57_600
    assert runner.state.checkpoint == PARENT_CKPT


def test_a_resume_keeps_the_reward_the_agent_chose(spec, tmp_path):
    """Same branch restarting: a mid-run set_reward must survive."""
    chosen = {"weights": {"reached_goal": 3.0}, "notes": "what the agent picked"}
    parent_state(tmp_path, reward_config=chosen)
    # env carries the ORIGINAL launch config; the agent has since moved on.
    env = child_env(spec, {"weights": {"reached_goal": 1.0}, "notes": "launch"},
                    branch_id="b0", parent_id=None, depth=0)
    runner = runner_for(spec, tmp_path, env)

    assert runner.state.reward_config == chosen, "a restart undid the agent's reward"
    assert runner.state.scores == [0.75, 0.875], "a restart dropped real scores"
    assert runner.state.children == 1, "a restart reset the fork counter"
