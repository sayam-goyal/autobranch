"""A fork must not inherit its parent's bill, and the ledger is the total.

A forked branch boots from a snapshot of its parent's workspace, so the state
file it loads is the parent's. Most of that is the point of forking -- reward
config, weights pointer, score history. `spend_usd` and `tokens` are not:
they are per-branch accumulators, and a child that keeps them reports its
parent's spend as its own for as long as it lives.

Observed on run-25ed75aa0a: three children each published ~$16 while the
ledger attributed ~$1.30 to them. `registry._run_spend` then took
`max(ledger_total, sum_of_records)`, so the inflated sum won and a run that
had truly spent $22 reported $63 to the UI, the artifact, and every analysis.
The budget guard reads the ledger directly and was never fooled.
"""

import json

from autobranch.branch_main import BranchState


def _write(tmp_path, **fields):
    st = BranchState(**fields)
    p = tmp_path / "state.json"
    p.write_text(json.dumps(st.to_dict()))
    return str(p)


def test_fork_does_not_inherit_spend_or_tokens(tmp_path):
    path = _write(tmp_path, owner="b0", spend_usd=16.12, tokens=98_000,
                  steps_done=52_800, scores=[0.875],
                  reward_config={"terms": ["damage"]})

    child = BranchState.load(path, owner="b0.1")

    assert child.spend_usd == 0.0, "the parent paid that, not the child"
    assert child.tokens == 0
    # Everything a fork exists to inherit still comes across.
    assert child.scores == [0.875]
    assert child.steps_done == 52_800
    assert child.reward_config == {"terms": ["damage"]}
    assert child.owner == "b0.1"


def test_a_branch_resuming_its_own_state_keeps_its_spend(tmp_path):
    """Restart after preemption is not a fork and must not lose the bill."""
    path = _write(tmp_path, owner="b0", spend_usd=16.12, tokens=98_000)

    same = BranchState.load(path, owner="b0")

    assert same.spend_usd == 16.12
    assert same.tokens == 98_000


def test_state_without_an_owner_is_left_alone(tmp_path):
    """State written before `owner` existed must not be silently zeroed."""
    raw = BranchState(spend_usd=9.5, tokens=1234).to_dict()
    raw.pop("owner")
    p = tmp_path / "state.json"
    p.write_text(json.dumps(raw))

    st = BranchState.load(str(p), owner="b0")
    assert st.spend_usd == 9.5
    assert st.owner == "b0"


def test_owner_round_trips(tmp_path):
    path = _write(tmp_path, owner="b0.1.1")
    assert BranchState.load(path).owner == "b0.1.1"


def test_missing_file_still_records_the_owner(tmp_path):
    st = BranchState.load(str(tmp_path / "absent.json"), owner="b0.2")
    assert st.owner == "b0.2"
    assert st.spend_usd == 0.0


class _Rec:
    def __init__(self, usd):
        self.spend_usd = usd


def test_ledger_with_entries_is_the_total_not_the_max(monkeypatch, tmp_path):
    """The bug: stale records summed higher than the ledger and won."""
    from autobranch import registry as R

    class FakeEntry:
        def __init__(self, usd):
            self.usd = usd

    class FakeLedger:
        def __init__(self, *a, **k):
            pass

        def entries(self, *a, **k):
            return [FakeEntry(1.30), FakeEntry(1.50), FakeEntry(18.20)]

    import autobranch.ledger as L

    monkeypatch.setattr(L, "Ledger", FakeLedger)
    reg = R.Registry.__new__(R.Registry)
    stale = [_Rec(16.12), _Rec(16.14), _Rec(16.20), _Rec(14.37)]

    total = reg._run_spend("run-x", stale)

    assert round(total, 2) == 21.00, (
        "the ledger saw $21.00; the stale records summed to $62.83 and must lose"
    )


def test_empty_ledger_falls_back_to_records(monkeypatch):
    """An unreachable or unwritten ledger means the records know more."""
    from autobranch import registry as R

    class FakeLedger:
        def __init__(self, *a, **k):
            pass

        def entries(self, *a, **k):
            return []

    import autobranch.ledger as L

    monkeypatch.setattr(L, "Ledger", FakeLedger)
    reg = R.Registry.__new__(R.Registry)

    assert reg._run_spend("run-x", [_Rec(4.0), _Rec(6.0)]) == 10.0
