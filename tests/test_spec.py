"""Manifest validation is the submission gate, so it must fail loudly and early."""

import pytest

from autobranch.spec import EnvironmentSpec, SpecError, with_base_model

MINIMAL = """
name: gridworld
entrypoint: grid_env:make_env
signals:
  - {name: reached_goal, type: float, range: [0, 1], doc: 1 on the step the goal is reached}
objective:
  name: success_rate
  evaluator: grid_env:evaluate
policy: {kind: discrete, n_actions: 5, obs_dim: 16}
"""


def test_minimal_manifest_parses():
    spec = EnvironmentSpec.from_yaml(MINIMAL)
    assert spec.name == "gridworld"
    assert spec.signal_names == ("reached_goal",)
    assert spec.objective.direction == "maximize"
    assert spec.trainer == "ppo"
    assert spec.resources.gpu is None


def test_round_trips_through_dict():
    spec = EnvironmentSpec.from_yaml(MINIMAL)
    again = EnvironmentSpec.parse(spec.to_dict())
    assert again.to_dict() == spec.to_dict()


def test_to_dict_is_json_serialisable():
    """The manifest travels through a modal.Dict and a web form."""
    import json

    spec = EnvironmentSpec.from_yaml(MINIMAL)
    assert json.loads(json.dumps(spec.to_dict()))["name"] == "gridworld"


@pytest.mark.parametrize("bad_name", ["Gridworld", "grid world", "-grid", "", "a" * 70])
def test_bad_names_are_rejected(bad_name):
    with pytest.raises(SpecError, match="manifest.name"):
        EnvironmentSpec.parse(
            {"name": bad_name, "entrypoint": "m:f",
             "signals": [{"name": "s"}], "objective": {"name": "o", "evaluator": "m:e"},
             "policy": {"kind": "discrete", "n_actions": 2}}
        )


@pytest.mark.parametrize("bad", ["notacolon", "mod:", ":fn", "mod:fn:extra", "1mod:fn"])
def test_bad_entrypoints_are_rejected(bad):
    with pytest.raises(SpecError, match="entrypoint"):
        EnvironmentSpec.parse(
            {"name": "x", "entrypoint": bad, "signals": [{"name": "s"}],
             "objective": {"name": "o", "evaluator": "m:e"},
             "policy": {"kind": "discrete", "n_actions": 2}}
        )


def test_no_signals_is_rejected_with_an_explanation():
    with pytest.raises(SpecError, match="cannot be composed from nothing"):
        EnvironmentSpec.parse(
            {"name": "x", "entrypoint": "m:f", "signals": [],
             "objective": {"name": "o", "evaluator": "m:e"},
             "policy": {"kind": "discrete", "n_actions": 2}}
        )


def test_duplicate_signals_are_rejected():
    with pytest.raises(SpecError, match="duplicate signal"):
        EnvironmentSpec.parse(
            {"name": "x", "entrypoint": "m:f",
             "signals": [{"name": "a"}, {"name": "a"}],
             "objective": {"name": "o", "evaluator": "m:e"},
             "policy": {"kind": "discrete", "n_actions": 2}}
        )


def test_discrete_policy_requires_n_actions():
    with pytest.raises(SpecError, match="n_actions"):
        EnvironmentSpec.parse(
            {"name": "x", "entrypoint": "m:f", "signals": [{"name": "a"}],
             "objective": {"name": "o", "evaluator": "m:e"},
             "policy": {"kind": "discrete"}}
        )


def test_relative_volume_mount_is_rejected():
    with pytest.raises(SpecError, match="must be absolute"):
        EnvironmentSpec.parse(
            {"name": "x", "entrypoint": "m:f", "signals": [{"name": "a"}],
             "objective": {"name": "o", "evaluator": "m:e"},
             "policy": {"kind": "discrete", "n_actions": 2},
             "image": {"volumes": {"roms": "my-vol"}}}
        )


def test_objective_direction_drives_comparison():
    maximize = EnvironmentSpec.from_yaml(MINIMAL).objective
    assert maximize.better(0.9, 0.5)
    assert not maximize.better(0.5, 0.9)

    minimize = EnvironmentSpec.parse(
        {"name": "x", "entrypoint": "m:f", "signals": [{"name": "a"}],
         "objective": {"name": "loss", "evaluator": "m:e", "direction": "minimize"},
         "policy": {"kind": "discrete", "n_actions": 2}}
    ).objective
    assert minimize.better(0.1, 0.5)
    assert not minimize.better(0.5, 0.1)


def test_agent_brief_exposes_signals_and_objective_but_not_internals():
    spec = EnvironmentSpec.from_yaml(MINIMAL)
    brief = spec.agent_brief()
    assert "reached_goal" in brief
    assert "success_rate" in brief
    # The agent must not be handed the evaluator path -- it cannot be allowed
    # to reason about, or try to influence, its own measurement.
    assert "grid_env:evaluate" not in brief


def test_with_base_model_applies_a_posttrain_constraint():
    spec = EnvironmentSpec.from_yaml(MINIMAL)
    out = with_base_model(spec, "Qwen/Qwen3-4B")
    assert out.policy.base_model == "Qwen/Qwen3-4B"
    assert out.trainer == "grpo_lora"
    # The original is untouched: specs are shared across branches.
    assert spec.policy.base_model is None
    assert spec.trainer == "ppo"
    assert "Qwen/Qwen3-4B" in out.agent_brief()


def test_unknown_trainer_is_rejected():
    with pytest.raises(SpecError, match="trainer"):
        EnvironmentSpec.parse(
            {"name": "x", "entrypoint": "m:f", "signals": [{"name": "a"}],
             "objective": {"name": "o", "evaluator": "m:e"},
             "policy": {"kind": "discrete", "n_actions": 2}, "trainer": "magic"}
        )


def test_image_spec_defaults_are_sane():
    spec = EnvironmentSpec.from_yaml(MINIMAL)
    assert spec.image.base == "debian_slim"
    assert spec.image.python == "3.11"
    assert spec.image.apt == () and spec.image.pip == ()
