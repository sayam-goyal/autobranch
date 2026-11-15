"""Reconciling the registry against which sandboxes are actually alive."""

import pytest

from autobranch.reconcile import LIVE_STATUSES, find_orphans
from autobranch.registry import MemoryBackend, Registry


class _Reg:
    """Just enough registry to hold branches for `find_orphans`."""

    def __init__(self, branches):
        self._branches = branches

    def branches(self, run_id):
        return self._branches


class _B:
    def __init__(self, branch_id, status, sandbox_id=None):
        self.branch_id = branch_id
        self.status = status
        self.sandbox_id = sandbox_id


def test_a_running_branch_whose_sandbox_is_gone_is_an_orphan():
    """The failure this module exists for.

    A branch writes `running` when it starts and updates that record from
    inside its own sandbox. Preemption, an OOM kill or a lost worker never runs
    Python again, so the terminal status is never written and the registry
    reports a healthy branch forever. An SF3 run preempted 37 minutes in went
    unnoticed for eleven hours exactly this way: the monitor polled the
    registry, and the registry could not know.
    """
    reg = _Reg([
        _B("b0", "running", "sb-alive"),
        _B("b0.1", "running", "sb-dead"),
        _B("b0.2", "abandoned", "sb-dead-too"),
        _B("b0.3", "succeeded", "sb-also-dead"),
    ])
    orphans = find_orphans(reg, "run-x", live={"sb-alive"})
    ids = [o["branch_id"] for o in orphans]

    assert ids == ["b0.1"], orphans
    assert orphans[0]["verdict"] == "sandbox is gone"


def test_terminal_branches_are_never_orphans():
    """A dead sandbox is *consistent* with abandoned or succeeded.

    Only a status claiming something still executes can be contradicted by a
    missing sandbox, which is why the check is against LIVE_STATUSES rather
    than against "not finished".
    """
    reg = _Reg([_B(f"b{i}", s, "sb-dead") for i, s in enumerate(
        ("abandoned", "succeeded", "failed", "stopped"))])
    assert find_orphans(reg, "run-x", live=set()) == []
    assert set(LIVE_STATUSES) == {"running", "pending"}


def test_a_branch_with_no_sandbox_yet_is_reported_but_not_actionable():
    """There are a few seconds between registering a branch and recording its
    sandbox id. Treating that window as death would abandon live branches, so
    it is reported with a different verdict and `--apply` skips it."""
    reg = _Reg([_B("b0", "pending", None)])
    orphans = find_orphans(reg, "run-x", live=set())

    assert len(orphans) == 1
    assert orphans[0]["sandbox_id"] is None
    assert orphans[0]["verdict"] == "no sandbox recorded yet"


def test_reconcile_is_read_only_without_apply(monkeypatch):
    """A reconciler that guesses wrong and abandons a live branch is worse than
    one that only reports, so writing is opt-in."""
    from autobranch import reconcile as mod

    from autobranch.registry import BranchRecord

    registry = Registry(backend=MemoryBackend())
    registry.create_run(env_name="e", objective={"name": "o", "direction": "maximize"},
                        run_id="run-x")
    registry.put_branch("run-x", BranchRecord(
        branch_id="b0", parent_id=None, depth=0, status="running",
        reward_config={}, sandbox_id="sb-dead",
    ))

    monkeypatch.setattr(mod, "live_sandbox_ids", lambda app_name="autobranch": set())
    monkeypatch.setattr("autobranch.registry.Registry", lambda *a, **k: registry)

    report = mod.reconcile("run-x", apply=False)
    assert report["runs"]["run-x"]["orphans"][0]["branch_id"] == "b0"
    assert "applied" not in report["runs"]["run-x"]
    assert registry.get_branch("run-x", "b0").status == "running", "must not write"
