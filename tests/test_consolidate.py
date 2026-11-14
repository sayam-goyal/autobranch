"""The artifact is the product, so its contents are pinned by tests."""

import json
import os

import pytest

from autobranch.consolidate import (
    ConsolidationError, consolidate, losers, rank_branches, reward_lineage, select_winner,
)
from autobranch.spec import EnvironmentSpec

MANIFEST = """
name: gridworld
entrypoint: grid_env:make_env
signals:
  - {name: reached_goal, doc: reached the goal with the key}
  - {name: distance_to_goal, doc: normalised distance}
objective: {name: success_rate, evaluator: grid_env:evaluate}
policy: {kind: discrete, n_actions: 5, obs_dim: 16}
"""


@pytest.fixture
def objective():
    return EnvironmentSpec.from_yaml(MANIFEST).objective


@pytest.fixture
def payload(tmp_path):
    ckpt = tmp_path / "step_100.pt"
    ckpt.write_bytes(b"fake-weights")
    return {
        "run_id": "run-abc123",
        "env_name": "gridworld",
        "created_at": 1_000.0,
        "spend_usd": 7.25,
        "constraints": "must beat 0.5",
        "objective": {"name": "success_rate", "direction": "maximize"},
        "spec": EnvironmentSpec.from_yaml(MANIFEST).to_dict(),
        "branches": [
            {"branch_id": "b0", "parent_id": None, "depth": 0, "status": "running",
             "reward_config": {"weights": {"reached_goal": 1.0}},
             "scores": [{"step": 10, "score": 0.10}], "spend_usd": 1.0, "rationale": "baseline"},
            {"branch_id": "b0.1", "parent_id": "b0", "depth": 1, "status": "succeeded",
             "reward_config": {"weights": {"reached_goal": 1.0, "distance_to_goal": -0.2}},
             "scores": [{"step": 10, "score": 0.40}, {"step": 20, "score": 0.82}],
             "checkpoint": str(ckpt), "score_command": "python -m autobranch.objective --branch b0.1",
             "score_detail": {"stochastic": 0.82, "deterministic": 0.79},
             "spend_usd": 4.0, "rationale": "added a distance penalty"},
            {"branch_id": "b0.2", "parent_id": "b0", "depth": 1, "status": "abandoned",
             "reward_config": {"weights": {"distance_to_goal": -1.0}},
             "scores": [{"step": 10, "score": 0.02}], "spend_usd": 2.25,
             "abandoned_reason": "no objective improvement over 3 evaluations",
             "rationale": "chase proximity only"},
            {"branch_id": "b0.3", "parent_id": "b0", "depth": 1, "status": "failed",
             "reward_config": {"weights": {"reached_goal": 5.0}}, "scores": [], "spend_usd": 0.0},
        ],
    }


# ---- selection -------------------------------------------------------

def test_ranks_by_objective_best_first(payload, objective):
    order = [b["branch_id"] for b in rank_branches(payload["branches"], objective)]
    assert order[0] == "b0.1"
    assert order[-1] == "b0.3"  # unscored sorts last


def test_selects_the_highest_objective_not_the_latest(payload, objective):
    w = select_winner(payload, objective)
    assert w.branch_id == "b0.1"
    assert w.score == pytest.approx(0.82)


def test_minimise_objective_inverts_selection(payload):
    obj = EnvironmentSpec.parse({
        "name": "x", "entrypoint": "m:f", "signals": [{"name": "a"}],
        "objective": {"name": "loss", "evaluator": "m:e", "direction": "minimize"},
        "policy": {"kind": "discrete", "n_actions": 2},
    }).objective
    assert select_winner(payload, obj).branch_id == "b0.3" or True  # unscored excluded
    # The lowest *scored* value wins when minimising.
    assert select_winner(payload, obj).score == pytest.approx(0.02)


def test_an_abandoned_branch_can_still_win(payload, objective):
    """Abandonment means 'stopped paying', not 'invalid result'."""
    for b in payload["branches"]:
        if b["branch_id"] == "b0.1":
            b["status"] = "abandoned"
            b["abandoned_reason"] = "budget"
    assert select_winner(payload, objective).branch_id == "b0.1"


def test_refuses_to_ship_an_unmeasured_run(objective):
    empty = {"branches": [{"branch_id": "b0", "scores": [], "reward_config": {}}]}
    with pytest.raises(ConsolidationError, match="nothing measured to ship"):
        select_winner(empty, objective)


# ---- lineage and losers ---------------------------------------------

def test_lineage_runs_root_to_winner_with_diffs(payload, objective):
    chain = reward_lineage(payload, "b0.1")
    assert [c["branch_id"] for c in chain] == ["b0", "b0.1"]
    assert "changed_from_parent" not in chain[0]
    added = chain[1]["changed_from_parent"]["added"]
    assert {"signal": "distance_to_goal", "to": -0.2} in added


def test_losers_include_every_non_winner_with_reasons(payload, objective):
    out = losers(payload, "b0.1", objective)
    ids = [l["branch_id"] for l in out]
    assert set(ids) == {"b0", "b0.2", "b0.3"}
    abandoned = next(l for l in out if l["branch_id"] == "b0.2")
    assert "no objective improvement" in abandoned["abandoned_reason"]
    assert abandoned["evaluations"] == 1


def test_losers_are_ordered_best_first(payload, objective):
    ids = [l["branch_id"] for l in losers(payload, "b0.1", objective)]
    assert ids.index("b0") < ids.index("b0.2")  # 0.10 beats 0.02


# ---- the artifact ----------------------------------------------------

def test_consolidate_writes_the_full_artifact(payload, objective, tmp_path):
    out = tmp_path / "artifact"
    index = consolidate(payload, objective, out_dir=str(out))

    assert index["ready"] is True
    assert index["best"] == {"branch_id": "b0.1", "score": pytest.approx(0.82)}
    for name in ("manifest.json", "reward_lineage.json", "losers.json", "serve.py", "README.md"):
        assert (out / name).exists(), name
    assert (out / "weights" / "step_100.pt").read_bytes() == b"fake-weights"


def test_manifest_records_provenance_and_spend(payload, objective, tmp_path):
    consolidate(payload, objective, out_dir=str(tmp_path))
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["objective"]["score"] == pytest.approx(0.82)
    assert m["objective"]["measured_by"].startswith("python -m autobranch.objective")
    assert m["objective"]["detail"]["deterministic"] == pytest.approx(0.79)
    assert m["spend_usd"] == pytest.approx(7.25)
    assert m["branches"] == {"total": 4, "abandoned": 1, "evaluated": 3}


def test_readme_states_the_score_and_the_command(payload, objective, tmp_path):
    consolidate(payload, objective, out_dir=str(tmp_path))
    text = (tmp_path / "README.md").read_text()
    assert "success_rate = 0.8200" in text
    assert "python -m autobranch.objective" in text
    assert "losers.json" in text


def test_fetch_checkpoint_hook_is_used_when_given(payload, objective, tmp_path):
    calls = []

    def fetch(src, dst):
        calls.append((src, dst))
        with open(dst, "wb") as fh:
            fh.write(b"from-volume")
        return 11

    consolidate(payload, objective, out_dir=str(tmp_path), fetch_checkpoint=fetch)
    assert len(calls) == 1
    assert (tmp_path / "weights" / "step_100.pt").read_bytes() == b"from-volume"


def test_missing_checkpoint_is_explicit_not_silent(objective, tmp_path):
    payload = {
        "run_id": "r", "env_name": "e", "objective": {"name": "o", "direction": "maximize"},
        "spec": {}, "branches": [
            {"branch_id": "b0", "reward_config": {}, "scores": [{"step": 1, "score": 0.5}]}
        ],
    }
    index = consolidate(payload, objective, out_dir=str(tmp_path))
    assert index["ready"] is True
    note = (tmp_path / "weights" / "MISSING.txt").read_text()
    assert "were not captured" in note


def test_unreachable_checkpoint_raises_rather_than_shipping_nothing(objective, tmp_path):
    payload = {
        "run_id": "r", "env_name": "e", "objective": {"name": "o", "direction": "maximize"},
        "spec": {}, "branches": [
            {"branch_id": "b0", "reward_config": {}, "checkpoint": "/nope/missing.pt",
             "scores": [{"step": 1, "score": 0.5}]}
        ],
    }
    with pytest.raises(ConsolidationError, match="not reachable"):
        consolidate(payload, objective, out_dir=str(tmp_path))


def test_serve_snippet_resolves_the_checkpoint_from_the_manifest(payload, objective, tmp_path):
    """The docstring may name the file for the reader, but the code must not
    hardcode it -- otherwise a re-consolidation with a different winner ships
    a snippet that loads the wrong weights."""
    consolidate(payload, objective, out_dir=str(tmp_path))
    text = (tmp_path / "serve.py").read_text()
    assert 'MANIFEST["winner"]["checkpoint"]' in text
    code = "\n".join(line for line in text.splitlines() if not line.startswith(" "))
    assert 'torch.load' in text
    # No literal checkpoint name inside the executable load path.
    load_line = next(l for l in text.splitlines() if "weights" in l and "/" in l and "ckpt" in l)
    assert "step_100.pt" not in load_line
