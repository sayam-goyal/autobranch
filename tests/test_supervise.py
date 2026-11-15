"""Restarting a run whose branches have died."""

import time

import pytest

from autobranch.supervise import GRACE_S, LIVE_STATUSES, TERMINAL_RUN_STATES, check


class _Run:
    def __init__(self, state="running", created_at=None, hours=4.0):
        self.state = state
        self.created_at = created_at if created_at is not None else time.time()
        self.budget = {"hours": hours}
        self.constraints = ""
        self.spec = {}


class _B:
    def __init__(self, branch_id, status, sandbox_id=None, created_at=None):
        self.branch_id = branch_id
        self.status = status
        self.sandbox_id = sandbox_id
        self.created_at = created_at if created_at is not None else time.time()
        self.reward_config = {}


def _wire(monkeypatch, run, branches, live):
    """Point `check` at fixed state instead of Modal and the real registry."""
    from autobranch import reconcile as rec

    class _Reg:
        def get_run(self, run_id):
            return run

        def branches(self, run_id):
            return branches

    monkeypatch.setattr("autobranch.registry.Registry", lambda *a, **k: _Reg())
    monkeypatch.setattr(rec, "live_sandbox_ids", lambda app_name="autobranch": live)


def test_a_dead_branch_with_deadline_left_asks_for_a_restart(monkeypatch):
    """The eleven-hour failure, as a decision.

    The branch claims `running`, its sandbox is gone, and the clock has not run
    out. Reporting that is not enough -- the run needs starting again, which is
    only safe because the trainer now resumes from the Volume.
    """
    _wire(monkeypatch, _Run(), [_B("b0", "running", "sb-gone")], live=set())
    status = check("run-x")

    assert status["alive"] == []
    assert status["should_restart"] is True
    assert status["should_finalize"] is False


def test_a_live_branch_is_left_alone(monkeypatch):
    _wire(monkeypatch, _Run(), [_B("b0", "running", "sb-up")], live={"sb-up"})
    status = check("run-x")

    assert status["alive"] == ["b0"]
    assert status["should_restart"] is False


def test_a_branch_too_young_to_have_a_sandbox_id_is_not_restarted(monkeypatch):
    """There are seconds between registering a branch and recording its sandbox.

    Restarting inside that window would launch a second root for a run that
    already has a healthy one -- two agents writing the same branch record,
    which is worse than the outage being fixed.
    """
    _wire(monkeypatch, _Run(), [_B("b0", "pending", None)], live=set())
    status = check("run-x")

    assert status["should_restart"] is False
    assert "too young" in status["verdict"]


def test_an_old_branch_with_no_sandbox_is_not_treated_as_young(monkeypatch):
    """The grace window has to expire, or a branch that never started is never
    noticed."""
    old = time.time() - (GRACE_S + 60)
    _wire(monkeypatch, _Run(), [_B("b0", "pending", None, created_at=old)], live=set())
    status = check("run-x")

    assert "too young" not in status["verdict"]


def test_a_dead_run_past_its_deadline_is_finalized_not_restarted(monkeypatch):
    """Restarting past the deadline would spend money the budget already
    refused."""
    run = _Run(created_at=time.time() - 5 * 3600, hours=4.0)
    _wire(monkeypatch, run, [_B("b0", "running", "sb-gone")], live=set())
    status = check("run-x")

    assert status["seconds_left"] < 0
    assert status["should_restart"] is False
    assert status["should_finalize"] is True


@pytest.mark.parametrize("state", TERMINAL_RUN_STATES)
def test_a_terminal_run_is_never_restarted(monkeypatch, state):
    """succeeded, failed and stopped all mean "do not touch"."""
    _wire(monkeypatch, _Run(state=state), [_B("b0", "running", "sb-gone")], live=set())
    status = check("run-x")

    assert status["should_restart"] is False
    assert status["should_finalize"] is False
    assert set(LIVE_STATUSES).isdisjoint(TERMINAL_RUN_STATES)
