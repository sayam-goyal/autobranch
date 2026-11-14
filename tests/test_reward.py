"""The reward DSL is the only thing an agent writes, so it gets the strictest tests."""

import pytest

from autobranch.reward import (
    CompiledReward, RewardError, compile_reward, config_hash, diff, summarize, validate,
)
from autobranch.spec import EnvironmentSpec

MANIFEST = """
name: probe
entrypoint: probe_env:make
signals:
  - {name: hit, type: float, range: [0, 1], doc: damage dealt}
  - {name: taken, type: float, range: [0, 1], doc: damage received}
  - {name: margin, type: float, range: [-1, 1], doc: health margin}
  - {name: won, type: int, range: [-1, 1], doc: round outcome}
objective:
  name: win_rate
  evaluator: probe_env:evaluate
policy: {kind: discrete, n_actions: 4, obs_dim: 8}
"""


@pytest.fixture
def spec():
    return EnvironmentSpec.from_yaml(MANIFEST)


# ---- validation ------------------------------------------------------

def test_undeclared_signal_is_rejected(spec):
    with pytest.raises(RewardError, match="undeclared signal 'nope'"):
        validate({"weights": {"nope": 1.0}}, spec)


def test_empty_reward_is_rejected(spec):
    with pytest.raises(RewardError, match="must do something"):
        validate({}, spec)


def test_unknown_transform_op_is_rejected(spec):
    with pytest.raises(RewardError, match="op must be one of"):
        validate({"weights": {"hit": 1.0}, "transforms": [{"op": "frobnicate", "signal": "hit"}]}, spec)


def test_bad_gamma_is_rejected(spec):
    cfg = {"shaping": [{"signal": "margin", "gamma": 1.5}]}
    with pytest.raises(RewardError, match="expected \\(0, 1\\]"):
        validate(cfg, spec)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('rm -rf /')",
    "open('/etc/passwd').read()",
    "hit.__class__",
    "[x for x in range(3)]",
    "lambda: 1",
])
def test_expr_rejects_anything_that_is_not_arithmetic(spec, expr):
    with pytest.raises(RewardError):
        validate({"expr": expr}, spec)


def test_expr_accepts_arithmetic_over_declared_signals(spec):
    validate({"expr": "hit * 2 - taken + max(margin, 0)"}, spec)


def test_expr_rejects_undeclared_name(spec):
    with pytest.raises(RewardError, match="unknown name 'sneaky'"):
        validate({"expr": "hit + sneaky"}, spec)


# ---- evaluation ------------------------------------------------------

def test_weights_compose_linearly(spec):
    r = compile_reward({"weights": {"hit": 1.0, "taken": -0.5}}, spec)
    assert r({"hit": 1.0, "taken": 1.0}) == pytest.approx(0.5)


def test_vectorised_signals_keep_their_shape(spec):
    import numpy as np

    r = compile_reward({"weights": {"hit": 2.0}}, spec)
    out = r({"hit": np.array([0.0, 0.5, 1.0])})
    assert out.shape == (3,)
    assert out.tolist() == [0.0, 1.0, 2.0]


def test_transform_is_visible_to_weights(spec):
    r = compile_reward(
        {"weights": {"hit": 1.0}, "transforms": [{"op": "clip", "signal": "hit", "lo": 0.0, "hi": 0.2}]},
        spec,
    )
    assert r({"hit": 1.0}) == pytest.approx(0.2)


def test_potential_shaping_contributes_nothing_on_the_first_step(spec):
    """A first-step potential would otherwise fire a spurious full-size jolt."""
    r = compile_reward({"shaping": [{"signal": "margin", "gamma": 1.0, "scale": 1.0}]}, spec)
    assert float(r({"margin": 0.7})) == pytest.approx(0.0)
    # Second step sees the real difference.
    assert float(r({"margin": 0.9})) == pytest.approx(0.2)


def test_potential_shaping_telescopes_to_zero_on_a_round_trip(spec):
    """gamma=1 potential shaping must not change the optimal policy."""
    r = compile_reward({"shaping": [{"signal": "margin", "gamma": 1.0, "scale": 1.0}]}, spec)
    r({"margin": 0.0})
    total = sum(float(r({"margin": m})) for m in (0.3, 0.6, 0.2, 0.0))
    assert total == pytest.approx(0.0, abs=1e-9)


def test_reset_episode_state_clears_shaping_history(spec):
    r = compile_reward({"shaping": [{"signal": "margin", "gamma": 1.0}]}, spec)
    r({"margin": 0.5})
    r.reset_episode_state()
    assert float(r({"margin": 0.9})) == pytest.approx(0.0)


def test_term_totals_attribute_contributions_and_drain(spec):
    r = compile_reward({"weights": {"hit": 1.0, "taken": -1.0}}, spec)
    r({"hit": 1.0, "taken": 0.25})
    totals = r.term_totals()
    assert totals["hit"] == pytest.approx(1.0)
    assert totals["taken"] == pytest.approx(-0.25)
    assert r.term_totals() == {}  # drained


def test_anneal_scales_shaping_toward_zero(spec):
    cfg = {
        "weights": {"won": 1.0},
        "shaping": [{"signal": "margin", "gamma": 1.0, "scale": 1.0}],
        "anneal": {"shaping": {"from": 1.0, "to": 0.0, "by_fraction": 1.0}},
    }
    r = compile_reward(cfg, spec)
    r({"margin": 0.0, "won": 0}, progress=0.0)
    early = float(r({"margin": 1.0, "won": 0}, progress=0.0))
    r.reset_episode_state()
    r({"margin": 0.0, "won": 0}, progress=1.0)
    late = float(r({"margin": 1.0, "won": 0}, progress=1.0))
    assert early > 0.9
    assert late == pytest.approx(0.0)


def test_expr_evaluates_without_eval(spec):
    r = compile_reward({"expr": "max(hit - taken, 0) * 2"}, spec)
    assert float(r({"hit": 0.8, "taken": 0.3})) == pytest.approx(1.0)
    assert float(r({"hit": 0.1, "taken": 0.9})) == pytest.approx(0.0)


# ---- lineage ---------------------------------------------------------

def test_hash_is_stable_across_key_order_and_notes(spec):
    a = {"weights": {"hit": 1.0, "taken": -1.0}, "notes": "first try"}
    b = {"notes": "totally different note", "weights": {"taken": -1.0, "hit": 1.0}}
    assert config_hash(a) == config_hash(b)


def test_hash_changes_when_behaviour_changes(spec):
    a = {"weights": {"hit": 1.0}}
    b = {"weights": {"hit": 1.0001}}
    assert config_hash(a) != config_hash(b)


def test_diff_reports_the_mutation_a_fork_made():
    parent = {"weights": {"hit": 0.1, "taken": -0.1}}
    child = {"weights": {"hit": 2.0, "won": 1.0}, "shaping": [{"signal": "margin"}]}
    d = diff(parent, child)
    assert d["weights"]["hit"] == {"from": 0.1, "to": 2.0}
    assert {"signal": "won", "to": 1.0} in d["added"]
    assert {"signal": "taken", "from": -0.1} in d["removed"]
    assert "shaping" in d["other"]


def test_summarize_is_a_one_liner():
    s = summarize({"weights": {"hit": 0.1, "won": 1.0}, "shaping": [{"signal": "margin", "scale": 0.5}]})
    assert "hit=0.1" in s and "won=1" in s and "shape(margin)x0.5" in s
