"""Offline tests for the reward-search agent loop.

The model is a scripted `FakeKimi`, so every test here is about the contract
the agent enforces on model output: one valid action per turn, built only from
declared signals, with every rejection becoming a correction turn the model
can actually act on.
"""

import json

import pytest

from autobranch.agent import (
    ACTION_SCHEMA_HINT,
    LESSON,
    TOOLS,
    Action,
    ActionInvalid,
    Agent,
    AgentContext,
    BadArguments,
    BranchSummary,
    Budget,
    MalformedAction,
    UndeclaredSignal,
    UnknownTool,
    _estimate_tokens,
    validate_reward_config,
)
from autobranch.kimi import Reply, Usage
from autobranch.spec import EnvironmentSpec

MANIFEST = {
    "name": "street-fighter-iii",
    "entrypoint": "sf_env:make_env",
    "signals": [
        {"name": "damage_dealt", "type": "float", "range": [0, 1], "doc": "fraction dealt"},
        {"name": "damage_taken", "type": "float", "range": [0, 1], "doc": "fraction taken"},
        {"name": "round_win", "type": "int", "range": [-1, 1], "doc": "+1 won, -1 lost"},
        {"name": "health_margin", "type": "float", "range": [-1, 1], "doc": "own minus opp"},
    ],
    "objective": {
        "name": "win_rate_vs_cpu8",
        "direction": "maximize",
        "evaluator": "sf_env:evaluate",
        "budget": {"matches": 200},
    },
    "policy": {"kind": "discrete", "n_actions": 40, "obs_dim": 80},
    "trainer": "ppo",
}

GOOD_REWARD = {
    "weights": {"round_win": 1.0, "damage_dealt": 0.1},
    "shaping": [{"kind": "potential", "signal": "health_margin", "gamma": 0.99, "scale": 0.5}],
}


class FakeKimi:
    """Scripted stand-in for `Kimi`. Records the messages it was handed."""

    def __init__(self, *bodies: str) -> None:
        self.bodies = list(bodies)
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=None, temperature=0.6, json_object=False,
             affinity_key=None) -> Reply:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_tokens": max_tokens,
                "json_object": json_object,
                "affinity_key": affinity_key,
            }
        )
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return Reply(content=body, reasoning="thought", finish_reason="stop", usage=Usage())


@pytest.fixture
def spec() -> EnvironmentSpec:
    return EnvironmentSpec.parse(MANIFEST)


def act(**kw) -> str:
    return json.dumps(kw)


def agent(spec: EnvironmentSpec, *bodies: str, **kw) -> tuple[Agent, FakeKimi]:
    kimi = FakeKimi(*(bodies or (act(tool="evaluate", args={}, rationale="measure"),)))
    return Agent(kimi, spec, **kw), kimi


def ctx(**kw) -> AgentContext:
    base = {
        "branch_id": "a1",
        "parent_id": "root",
        "depth": 1,
        "step": 3,
        "reward_config": GOOD_REWARD,
        "recent_metrics": {"policy_loss": 0.21, "episode_return": 14.3},
        "objective_scores": (0.0, 0.0),
        "siblings": (
            BranchSummary("a2", status="running", best_objective=0.42, note="damage-heavy"),
            BranchSummary("a3", status="abandoned", best_objective=0.0, note="shaping blew up"),
        ),
        "budget": Budget(usd=18.0, gpu_minutes=40.0, tokens=400_000, wall_seconds=3600.0),
        "evals_without_improvement": 2,
    }
    return AgentContext(**{**base, **kw})


# ---- the system prompt --------------------------------------------------


def test_system_prompt_states_signals_objective_and_the_lesson(spec):
    a, _ = agent(spec)
    prompt = a.system_prompt
    for name in spec.signal_names:
        assert name in prompt
    assert "win_rate_vs_cpu8" in prompt
    assert "maximize" in prompt
    assert LESSON in prompt  # verbatim, including the 0.807 / 6,705 figures
    assert "0.807" in prompt and "6,705" in prompt
    # The agent must be told it has no access to the evaluator.
    assert "cannot" in prompt and "evaluator" in prompt
    assert "only measure of success" in prompt


def test_system_prompt_lists_every_tool_and_the_action_schema(spec):
    a, _ = agent(spec)
    for tool in TOOLS:
        assert tool in a.system_prompt
    assert ACTION_SCHEMA_HINT in a.system_prompt


def test_system_prompt_says_when_to_abandon(spec):
    a, _ = agent(spec)
    prompt = a.system_prompt
    assert "WHEN TO ABANDON" in prompt
    assert "has not improved over the last 3 evaluations" in prompt
    assert "budget is nearly exhausted" in prompt
    assert "best sibling" in prompt


def test_constraints_and_system_extra_are_included_but_separated(spec):
    a, _ = agent(spec, constraints="stay under $50; must beat 76%", system_extra="you are root")
    assert "stay under $50; must beat 76%" in a.system_prompt
    assert "USER CONSTRAINTS" in a.system_prompt
    assert "you are root" in a.system_prompt
    assert "do not change the objective" in a.system_prompt


def test_prompt_does_not_leak_environment_internals(spec):
    a, _ = agent(spec)
    assert "sf_env:evaluate" not in a.system_prompt
    assert "sf_env:make_env" not in a.system_prompt


# ---- context rendering --------------------------------------------------


def test_context_render_carries_budget_siblings_and_the_improvement_counter(spec):
    rendered = ctx().render()
    assert "branch: a1 (parent=root, depth=1, step=3)" in rendered
    assert "$18.00" in rendered and "40.0 gpu-minutes" in rendered
    assert "a2: objective=0.4200 status=running" in rendered
    assert "a3: objective=0.0000 status=abandoned" in rendered
    assert "evaluations since last improvement: 2" in rendered
    assert "NOT progress" in rendered  # training metrics are labelled as such


def test_context_derives_best_scores_and_budget_exhaustion():
    c = ctx(objective_scores=(0.1, 0.3, 0.2))
    assert c.best_objective == pytest.approx(0.3)
    assert c.best_sibling_objective == pytest.approx(0.42)
    assert not c.budget.nearly_exhausted
    assert Budget(usd=0.5, gpu_minutes=100, tokens=10**6, wall_seconds=10**4).nearly_exhausted
    assert Budget(usd=100, gpu_minutes=1, tokens=10**6, wall_seconds=10**4).nearly_exhausted
    assert AgentContext(branch_id="x").best_objective is None


# ---- valid actions -----------------------------------------------------


def test_valid_action_is_parsed_with_args_and_rationale(spec):
    body = act(
        tool="propose_reward",
        args={"config": GOOD_REWARD},
        rationale="round_win is the only signal aligned with the objective.",
    )
    a, kimi = agent(spec, body)
    action = a.next_action(ctx())
    assert isinstance(action, Action)
    assert action.tool == "propose_reward"
    assert action.args["config"] == GOOD_REWARD
    assert action.rationale.startswith("round_win")
    assert len(kimi.calls) == 1
    assert a.corrections == 0
    assert action.to_dict()["tool"] == "propose_reward"


def test_action_is_accepted_inside_fences_and_prose(spec):
    body = 'Reading the tree first.\n```json\n{"tool": "read_tree", "args": {}}\n```\n'
    a, _ = agent(spec, body)
    assert a.next_action(ctx()).tool == "read_tree"


def test_no_arg_tools_accept_missing_or_null_args(spec):
    a, _ = agent(spec, '{"tool": "evaluate"}')
    assert a.next_action(ctx()).args == {}
    b, _ = agent(spec, '{"tool": "evaluate", "args": null}')
    assert b.next_action(ctx()).args == {}


def test_train_steps_are_coerced_to_int(spec):
    a, _ = agent(spec, act(tool="train", args={"steps": 50000.0}, rationale="r"))
    action = a.next_action(ctx())
    assert action.args["steps"] == 50000
    assert isinstance(action.args["steps"], int)


def test_fork_reward_is_validated_like_propose_reward(spec):
    a, _ = agent(
        spec,
        act(tool="fork", args={"new_reward": GOOD_REWARD, "note": "drop damage_dealt"}, rationale="r"),
    )
    assert a.next_action(ctx()).tool == "fork"


def test_the_model_is_asked_for_json_and_pinned_to_the_branch(spec):
    a, kimi = agent(spec, branch_id="a1")
    a.next_action(ctx())
    call = kimi.calls[0]
    assert call["json_object"] is True
    assert call["affinity_key"] == "street-fighter-iii:a1"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][-1]["content"].startswith("branch: a1")


# ---- invalid actions become correction turns ---------------------------


def _correction_round(spec, bad_body: str, expected: type) -> tuple[Agent, FakeKimi]:
    """Script one bad action then a good one; assert the loop recovered."""
    good = act(tool="evaluate", args={}, rationale="measure instead")
    a, kimi = agent(spec, bad_body, good)
    action = a.next_action(ctx())
    assert action.tool == "evaluate"
    assert len(kimi.calls) == 2, "a rejected action must produce a second turn"
    assert a.corrections == 1
    retry = kimi.calls[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": bad_body}
    assert retry[-1]["role"] == "user"
    assert "Your action was rejected" in retry[-1]["content"]
    # The feedback the model gets is the same text the exception carries.
    with pytest.raises(expected) as caught:
        a.parse_action(bad_body)
    assert caught.value.feedback in retry[-1]["content"]
    return a, kimi


def test_unknown_tool_produces_a_correction_listing_the_tools(spec):
    a, kimi = _correction_round(
        spec, act(tool="shell", args={"cmd": "rm -rf /"}, rationale="faster"), UnknownTool
    )
    feedback = kimi.calls[1]["messages"][-1]["content"]
    assert "'shell' is not a tool" in feedback
    assert "propose_reward" in feedback


def test_non_json_reply_produces_a_correction(spec):
    a, kimi = _correction_round(
        spec, "I think we should train for a while longer.", MalformedAction
    )
    assert "could not find a JSON object" in kimi.calls[1]["messages"][-1]["content"]


def test_broken_json_produces_a_correction(spec):
    _correction_round(spec, '{"tool": "train", "args": {"steps": }}', MalformedAction)


def test_missing_tool_field_produces_a_correction(spec):
    _correction_round(spec, '{"args": {"steps": 100}}', MalformedAction)


def test_missing_required_arg_produces_a_correction(spec):
    a, kimi = _correction_round(spec, act(tool="train", args={}, rationale="go"), BadArguments)
    assert "missing required argument(s): steps" in kimi.calls[1]["messages"][-1]["content"]


def test_unexpected_arg_produces_a_correction(spec):
    a, kimi = _correction_round(
        spec, act(tool="train", args={"steps": 10, "lr": 3e-4}, rationale="tune"), BadArguments
    )
    assert "does not take argument(s): lr" in kimi.calls[1]["messages"][-1]["content"]


def test_bad_steps_type_produces_a_correction(spec):
    _correction_round(spec, act(tool="train", args={"steps": "lots"}, rationale="go"), BadArguments)


def test_args_not_an_object_produces_a_correction(spec):
    _correction_round(spec, act(tool="train", args=[100], rationale="go"), BadArguments)


def test_undeclared_signal_in_a_reward_produces_a_correction(spec):
    bad = act(
        tool="propose_reward",
        args={"config": {"weights": {"combo_meter": 1.0}}},
        rationale="combos win fights",
    )
    a, kimi = _correction_round(spec, bad, UndeclaredSignal)
    feedback = kimi.calls[1]["messages"][-1]["content"]
    assert "'combo_meter'" in feedback
    assert "does not report" in feedback
    assert "damage_dealt" in feedback  # the correction names what IS available


def test_the_loop_gives_up_after_three_corrections(spec):
    bad = act(tool="shell", args={}, rationale="nope")
    a, kimi = agent(spec, bad)
    with pytest.raises(UnknownTool):
        a.next_action(ctx())
    assert len(kimi.calls) == 4  # 1 attempt + 3 corrections
    assert a.corrections == 3
    # The failed exchange is retained, not discarded: a branch pruned for
    # unparseable actions has to be explainable afterwards.
    assert any(m["content"] == bad for m in a.transcript)


def test_every_invalid_class_is_an_action_invalid(spec):
    for cls in (MalformedAction, UnknownTool, BadArguments, UndeclaredSignal):
        assert issubclass(cls, ActionInvalid)


# ---- reward config validation ------------------------------------------


SIGNALS = ("damage_dealt", "damage_taken", "round_win", "health_margin")


def test_reward_validation_accepts_the_documented_shape():
    validate_reward_config(
        {
            "weights": {"round_win": 1.0, "damage_taken": -0.1},
            "shaping": [{"kind": "potential", "signal": "health_margin", "gamma": 0.99}],
            "transforms": [{"op": "clip", "signal": "damage_dealt", "lo": 0.0, "hi": 0.2}],
            "anneal": {"shaping": {"from": 1.0, "to": 0.0, "by_fraction": 0.8}},
        },
        SIGNALS,
        where="cfg",
    )


@pytest.mark.parametrize(
    "config",
    [
        {"weights": {"combo_meter": 1.0}},
        {"shaping": [{"kind": "potential", "signal": "ego"}]},
        {"transforms": [{"op": "clip", "signal": "nope", "lo": 0, "hi": 1}]},
        {"expr": "round_win + 0.5 * stun_meter"},
    ],
)
def test_reward_validation_rejects_undeclared_signals_everywhere(config):
    with pytest.raises(UndeclaredSignal):
        validate_reward_config(config, SIGNALS, where="cfg")


def test_reward_validation_allows_arithmetic_builtins_in_expr():
    validate_reward_config({"expr": "max(round_win, 0) - 0.1 * abs(damage_taken)"}, SIGNALS, where="cfg")


@pytest.mark.parametrize(
    "config",
    [
        "a string",
        {"weights": {"round_win": "one"}},
        {"weights": {"round_win": True}},
        {"weights": []},
        {"shaping": {"signal": "round_win"}},
        {"shaping": [{"kind": "potential"}]},
        {},
        {"anneal": {"shaping": {"from": 1.0}}},
    ],
)
def test_reward_validation_rejects_structural_nonsense(config):
    with pytest.raises(BadArguments):
        validate_reward_config(config, SIGNALS, where="cfg")


# ---- transcript ---------------------------------------------------------


def test_observe_appends_the_tool_result_for_the_next_turn(spec):
    a, kimi = agent(spec)
    a.next_action(ctx())
    a.observe("evaluate", {"objective": 0.0, "matches": 200})
    a.next_action(ctx())
    sent = kimi.calls[1]["messages"]
    assert any("result of evaluate" in m["content"] and '"objective": 0.0' in m["content"]
               for m in sent)


def test_observe_clips_an_enormous_result(spec):
    a, _ = agent(spec)
    a.observe("read_tree", {"branches": ["b" * 50 for _ in range(500)]})
    assert "truncated" in a.transcript[-1]["content"]
    assert len(a.transcript[-1]["content"]) < 4500


def test_transcript_is_compacted_into_a_digest_rather_than_dropped(spec):
    a, _ = agent(
        spec,
        act(tool="train", args={"steps": 100000}, rationale="first hypothesis"),
        transcript_token_budget=1500,
    )
    for i in range(12):
        a.next_action(ctx(step=i))
        a.observe("train", {"step": i, "policy_loss": 0.2, "filler": "x" * 500})

    assert a.transcript_tokens() <= 1500
    assert len(a.transcript) < 24  # 24 raw turns went in
    digest = a.transcript[0]["content"]
    assert digest.startswith("[compacted history")
    assert "nothing dropped" in digest
    # The digest keeps what the branch actually did, not just a count.
    assert "action train" in digest
    assert "first hypothesis" in digest


def test_compaction_folds_an_existing_digest_instead_of_nesting_them(spec):
    a, _ = agent(spec, transcript_token_budget=200)
    for i in range(20):
        a.observe("train", {"i": i, "filler": "y" * 200})
    digest = a.transcript[0]["content"]
    assert digest.count("[compacted history") == 1
    # A digest that overflowed its own cap says so rather than silently shrinking.
    assert "elided" in digest
    assert a.transcript_tokens() <= 200 + _estimate_tokens(a.transcript[-1]["content"])
