"""Forking, tested without Modal.

The two bugs covered here both cost a live run: publishing the runtime deleted
the package out from under the process that was running it, and a sandbox
authenticated to the wrong workspace saw an empty registry and could not fork
at all. Neither was reachable from a unit test of the happy path, so both get
a named test.
"""

import os
import pathlib
import shutil

import pytest

from autobranch.forking import ForkError, RUNTIME_DIR, ensure_runtime_on_volume, make_forker
from autobranch.orchestrator import Budget, BranchEnv
from autobranch.spec import EnvironmentSpec

MANIFEST = """
name: gridworld
entrypoint: grid_env:make_env
signals:
  - {name: reached_goal, doc: reached the goal with the key}
objective: {name: success_rate, evaluator: grid_env:evaluate}
policy: {kind: discrete, n_actions: 5, obs_dim: 16}
"""


@pytest.fixture
def spec():
    return EnvironmentSpec.from_yaml(MANIFEST)


@pytest.fixture
def env(spec):
    return BranchEnv(
        run_id="run-x", branch_id="b0", parent_id=None, depth=0,
        spec=spec.to_dict(), constraints={"text": ""}, budget=Budget().to_dict(),
        deadline_ts=1_900_000_000.0,
    )


# ---- publishing the runtime -----------------------------------------

def test_runtime_is_copied_to_the_volume(tmp_path):
    out = ensure_runtime_on_volume(data_root=str(tmp_path))
    assert out == str(tmp_path / RUNTIME_DIR)
    published = tmp_path / RUNTIME_DIR / "autobranch"
    assert (published / "branch_main.py").is_file()
    assert (published / "reward.py").is_file()
    # Caches are noise in a published runtime and can shadow a newer source.
    assert not (published / "__pycache__").exists()


def test_publishing_from_the_published_copy_is_a_no_op(tmp_path):
    """The bug this guards: a branch forking a child runs this code FROM the
    volume, so src == dst, and a naive refresh rmtree'd the live package. The
    live run died with FileNotFoundError on its own module directory."""
    ensure_runtime_on_volume(data_root=str(tmp_path))
    published = tmp_path / RUNTIME_DIR / "autobranch"
    sentinel = published / "branch_main.py"
    assert sentinel.is_file()

    # Simulate running from the published copy by pointing the module's notion
    # of its own location at the published directory.
    import autobranch.forking as fk

    original = fk.__file__
    try:
        fk.__file__ = str(published / "forking.py")
        out = ensure_runtime_on_volume(data_root=str(tmp_path))
    finally:
        fk.__file__ = original

    assert out == str(tmp_path / RUNTIME_DIR)
    assert sentinel.is_file(), "publishing deleted the runtime it was running from"


def test_republishing_replaces_rather_than_merges(tmp_path):
    """A stale module left behind by an older version is harder to diagnose
    than a slightly slower copy."""
    ensure_runtime_on_volume(data_root=str(tmp_path))
    stale = tmp_path / RUNTIME_DIR / "autobranch" / "removed_in_a_later_version.py"
    stale.write_text("# left over from an older runtime\n")
    ensure_runtime_on_volume(data_root=str(tmp_path))
    assert not stale.exists()


# ---- the forker contract --------------------------------------------

def test_fork_requires_a_child_id_and_a_reward(spec, env):
    forker = make_forker(spec=spec, budget=Budget(), launcher=lambda e: None,
                         snapshotter=lambda path: "im-fake")
    with pytest.raises(ForkError, match="child branch id and a reward config"):
        forker(parent_env=env, child_branch_id=None, reward_config={"weights": {}})
    with pytest.raises(ForkError, match="child branch id and a reward config"):
        forker(parent_env=env, child_branch_id="b0.1", reward_config=None)


def test_child_inherits_the_parents_absolute_deadline(spec, env):
    """A branch born late must not be able to extend the run."""
    seen = []
    forker = make_forker(spec=spec, budget=Budget(), launcher=seen.append,
                         snapshotter=lambda path: "im-fake")
    forker(parent_env=env, child_branch_id="b0.1",
           reward_config={"weights": {"reached_goal": 1.0}}, note="try it")
    child = seen[0]
    assert child.deadline_ts == env.deadline_ts
    assert child.depth == env.depth + 1
    assert child.parent_id == "b0"
    assert child.run_id == env.run_id


def test_child_env_carries_the_runtime_path(spec, env):
    """Without PYTHONPATH the child cannot import the framework at all: its
    image is built from the submitted manifest, which knows nothing about us."""
    seen = []
    forker = make_forker(spec=spec, budget=Budget(), launcher=seen.append,
                         snapshotter=lambda path: "im-fake")
    forker(parent_env=env, child_branch_id="b0.1",
           reward_config={"weights": {"reached_goal": 1.0}})
    environ = seen[0].to_env()
    assert "/data/lib" in environ["PYTHONPATH"]
    assert environ["AUTOBRANCH_BRANCH_ID"] == "b0.1"
    assert environ["AUTOBRANCH_PARENT_ID"] == "b0"


def test_self_sandbox_refuses_to_guess_outside_a_sandbox(monkeypatch):
    from autobranch.forking import self_sandbox

    monkeypatch.delenv("MODAL_SANDBOX_ID", raising=False)
    with pytest.raises(ForkError, match="not inside a Modal"):
        self_sandbox()
