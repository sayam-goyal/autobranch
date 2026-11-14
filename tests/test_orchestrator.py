"""Search policy and constraint parsing, tested without Modal.

These rules decide how the budget is spent, so they are tested as pure logic
rather than observed in a live run where a mistake costs GPU hours.
"""

import pytest

from autobranch.orchestrator import (
    Budget, Constraints, SearchPolicy, ToolError, TreeView, child_branch_id,
    dedup_key, initial_reward, seconds_left, validate_action,
)
from autobranch.spec import EnvironmentSpec

MANIFEST = """
name: gridworld
entrypoint: grid_env:make_env
signals:
  - {name: reached_goal, doc: 1 when the goal is reached with the key}
  - {name: stepped_in_lava, doc: 1 when the agent enters lava}
  - {name: distance_to_goal, doc: normalised distance}
  - {name: step_cost, doc: 1 per step}
objective: {name: success_rate, evaluator: grid_env:evaluate}
policy: {kind: discrete, n_actions: 5, obs_dim: 16}
"""


@pytest.fixture
def spec():
    return EnvironmentSpec.from_yaml(MANIFEST)


class _Action:
    def __init__(self, tool, args=None):
        self.tool = tool
        self.args = args or {}


# ---- constraints -----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("must posttrain Qwen/Qwen3-4B", "Qwen/Qwen3-4B"),
    ("please post-train from Qwen3-8B and keep it cheap", "Qwen3-8B"),
    ("finetune meta-llama/Llama-3-8B", "meta-llama/Llama-3-8B"),
])
def test_base_model_is_extracted_with_original_casing(text, expected):
    """HF repo ids are case sensitive -- a lowercased id 404s."""
    assert Constraints.parse(text).base_model == expected


def test_no_base_model_when_not_requested():
    assert Constraints.parse("just make it good").base_model is None


@pytest.mark.parametrize("text,score", [
    ("must beat 76% win rate", 0.76),
    ("get above 90 %", 0.90),
    ("at least 0.85 success", 0.85),
])
def test_target_score_is_extracted(text, score):
    assert Constraints.parse(text).target_score == pytest.approx(score)


def test_max_gpu_is_extracted():
    assert Constraints.parse("nothing bigger than an L40S please").max_gpu == "L40S"


def test_constraint_text_is_preserved_verbatim_for_the_agent():
    text = "be aggressive early, then tighten up; avoid lava at all costs"
    assert Constraints.parse(text).text == text


# ---- budget ----------------------------------------------------------

def test_budget_parse_overrides_defaults():
    b = Budget.parse({"usd": 50, "hours": 3, "max_branches": 20})
    assert (b.usd, b.hours, b.max_branches) == (50.0, 3.0, 20)
    assert b.max_depth == Budget().max_depth


@pytest.mark.parametrize("bad", [{"usd": 0}, {"hours": -1}])
def test_nonpositive_budget_is_rejected(bad):
    with pytest.raises(ValueError):
        Budget.parse(bad)


# ---- search policy ---------------------------------------------------

@pytest.fixture
def policy(spec):
    return SearchPolicy(Budget(usd=10, hours=1, max_depth=2, tokens=1000), spec.objective)


def _stop(policy, **kw):
    base = dict(scores=[], depth=0, spend_usd=0.0, tokens=0, seconds_left=10_000, best_sibling=None)
    base.update(kw)
    return policy.should_stop_branch(**base)


def test_branch_runs_when_nothing_is_wrong(policy):
    assert not _stop(policy).stop


def test_deadline_reserves_time_to_consolidate(policy):
    d = _stop(policy, seconds_left=120)
    assert d.stop and "consolidate" in d.reason


def test_spend_cap_stops_a_branch(policy):
    d = _stop(policy, spend_usd=10.0)
    assert d.stop and "cap" in d.reason


def test_token_budget_stops_a_branch(policy):
    assert _stop(policy, tokens=1000).stop


def test_depth_cap_stops_a_branch(policy):
    assert _stop(policy, depth=3).stop


def test_plateau_stops_a_branch_after_patience(policy):
    """Four evaluations with no improvement is the orchestrator's call, not the agent's."""
    d = _stop(policy, scores=[0.40, 0.40, 0.39, 0.40])
    assert d.stop and "no objective improvement" in d.reason


def test_improvement_keeps_a_branch_alive(policy):
    assert not _stop(policy, scores=[0.10, 0.20, 0.30, 0.45]).stop


def test_domination_by_a_sibling_stops_a_branch(policy):
    d = _stop(policy, scores=[0.10, 0.12], best_sibling=0.80)
    assert d.stop and "dominated" in d.reason


def test_near_sibling_is_not_dominated(policy):
    assert not _stop(policy, scores=[0.70, 0.72], best_sibling=0.80).stop


def test_minimise_objectives_invert_the_comparison():
    spec = EnvironmentSpec.parse({
        "name": "x", "entrypoint": "m:f", "signals": [{"name": "a"}],
        "objective": {"name": "loss", "evaluator": "m:e", "direction": "minimize"},
        "policy": {"kind": "discrete", "n_actions": 2},
    })
    p = SearchPolicy(Budget(), spec.objective)
    # Falling loss is improvement when minimising.
    assert not p.should_stop_branch(
        scores=[0.9, 0.7, 0.5, 0.3], depth=0, spend_usd=0, tokens=0,
        seconds_left=10_000, best_sibling=None,
    ).stop


def test_fork_refused_at_branch_cap(spec):
    p = SearchPolicy(Budget(max_branches=4), spec.objective)
    assert p.can_fork(n_branches=4, depth=0, n_live=1).stop


def test_fork_refused_at_depth_and_concurrency_limits(spec):
    p = SearchPolicy(Budget(max_depth=2, max_concurrent=2), spec.objective)
    assert p.can_fork(n_branches=0, depth=2, n_live=0).stop
    assert p.can_fork(n_branches=0, depth=0, n_live=2).stop


def test_fork_allowed_with_room(spec):
    p = SearchPolicy(Budget(), spec.objective)
    assert not p.can_fork(n_branches=1, depth=0, n_live=1).stop


# ---- action validation ----------------------------------------------

def test_unknown_tool_is_rejected(spec):
    with pytest.raises(ToolError, match="unknown tool"):
        validate_action(_Action("do_magic"), spec)


def test_propose_reward_requires_a_config(spec):
    with pytest.raises(ToolError, match="requires a 'reward' object"):
        validate_action(_Action("propose_reward", {}), spec)


def test_propose_reward_rejects_undeclared_signal(spec):
    with pytest.raises(ToolError, match="undeclared signal"):
        validate_action(_Action("propose_reward", {"reward": {"weights": {"ghost": 1.0}}}), spec)


def test_propose_reward_accepts_a_valid_config(spec):
    validate_action(_Action("propose_reward", {"reward": {"weights": {"reached_goal": 1.0}}}), spec)


def test_train_requires_positive_steps(spec):
    with pytest.raises(ToolError, match="positive integer"):
        validate_action(_Action("train", {"steps": 0}), spec)


def test_abandon_requires_a_reason(spec):
    with pytest.raises(ToolError, match="non-empty 'reason'"):
        validate_action(_Action("abandon", {"reason": "  "}), spec)


def test_rollback_requires_a_target(spec):
    with pytest.raises(ToolError, match="requires 'checkpoint'"):
        validate_action(_Action("rollback", {}), spec)


# ---- tree ------------------------------------------------------------

def test_branch_ids_encode_lineage():
    assert child_branch_id("b0", 0) == "b0.1"
    assert child_branch_id("b0.1", 2) == "b0.1.3"


def test_tree_view_reports_best_and_dedups(spec):
    tree = TreeView(branches=[
        {"branch_id": "b0", "status": "running", "reward_config": {"weights": {"reached_goal": 1.0}}, "best_score": 0.3},
        {"branch_id": "b0.1", "status": "abandoned", "reward_config": {"weights": {"step_cost": -1.0}},
         "best_score": 0.1, "abandoned_reason": "plateaued"},
    ])
    assert tree.best_score(spec.objective) == pytest.approx(0.3)
    assert dedup_key({"weights": {"reached_goal": 1.0}}) in tree.tried_fingerprints()
    rendered = tree.render(spec.objective)
    assert "b0.1" in rendered and "plateaued" in rendered


def test_initial_reward_is_plain_and_valid(spec):
    from autobranch.reward import compile_reward

    cfg = initial_reward(spec)
    compile_reward(cfg, spec)  # must be runnable as-is
    assert cfg["weights"]["reached_goal"] == 1.0
    assert cfg["weights"]["stepped_in_lava"] == -1.0
    # Deliberately no shaping: the agent should discover it, not inherit it.
    assert "shaping" not in cfg


def test_initial_reward_falls_back_when_no_heuristic_matches():
    spec = EnvironmentSpec.parse({
        "name": "x", "entrypoint": "m:f",
        "signals": [{"name": "zorb"}, {"name": "blip"}],
        "objective": {"name": "o", "evaluator": "m:e"},
        "policy": {"kind": "discrete", "n_actions": 2},
    })
    cfg = initial_reward(spec)
    assert cfg["weights"] == {"zorb": 1.0}


def test_seconds_left_floors_at_zero():
    assert seconds_left(100.0, now=200.0) == 0.0
    assert seconds_left(300.0, now=200.0) == 100.0


# ---- branch env round trip ------------------------------------------

def test_branch_env_round_trips_through_env_vars(spec):
    from autobranch.orchestrator import BranchEnv

    be = BranchEnv(
        run_id="run-abc", branch_id="b0.1", parent_id="b0", depth=1,
        spec=spec.to_dict(), constraints={"text": "go fast"}, budget=Budget().to_dict(),
        deadline_ts=1_800_000_000.0, reward_config={"weights": {"reached_goal": 1.0}},
        snapshot_id="im-123",
    )
    again = BranchEnv.from_env(be.to_env())
    assert again.branch_id == "b0.1"
    assert again.parent_id == "b0"
    assert again.depth == 1
    assert again.snapshot_id == "im-123"
    assert again.spec["name"] == "gridworld"
    assert again.reward_config == {"weights": {"reached_goal": 1.0}}
