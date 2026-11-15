"""Registry tests, against an in-memory backend.

The concurrency tests here are the reason this module exists in the shape it
does. In the predecessor project, six concurrent writers sharing one state
blob ended with 29 entries that all belonged to a single writer and no error
anywhere. These tests run genuinely concurrent writers and assert that
nothing goes missing.
"""

import threading

import pytest

from autobranch.registry import (
    STATUSES,
    BranchRecord,
    MemoryBackend,
    Registry,
    RoutingBackend,
    RunRecord,
)

OBJECTIVE = {"name": "win_rate_vs_cpu8", "direction": "maximize"}


@pytest.fixture
def backend() -> MemoryBackend:
    return MemoryBackend()


@pytest.fixture
def reg(backend) -> Registry:
    return Registry(backend=backend, writer_id="w-test")


def _run(reg: Registry, **kwargs) -> RunRecord:
    return reg.create_run(
        env_name="street-fighter-iii",
        objective=OBJECTIVE,
        spec={"name": "street-fighter-iii"},
        constraints="stay under $50",
        budget={"usd": 50.0},
        **kwargs,
    )


def _branch(reg: Registry, run_id: str, branch_id: str, parent_id=None, **kwargs) -> BranchRecord:
    rec = BranchRecord(
        branch_id=branch_id,
        parent_id=parent_id,
        depth=branch_id.count("."),
        **kwargs,
    )
    reg.put_branch(run_id, rec)
    return rec


# ---- basics -------------------------------------------------------------


def test_create_and_get_run_roundtrips(reg):
    run = _run(reg)
    fetched = reg.get_run(run.run_id)

    assert fetched is not None
    assert fetched.env_name == "street-fighter-iii"
    assert fetched.objective["direction"] == "maximize"
    assert fetched.budget == {"usd": 50.0}
    assert reg.get_run("nope") is None


def test_run_ids_are_sortable_by_time(backend):
    ticks = iter([1_700_000_000.0, 1_700_000_900.0])
    reg = Registry(backend=backend, writer_id="w", clock=lambda: next(ticks))
    first = _run(reg).run_id
    second = _run(reg).run_id
    assert first < second


def test_list_runs_is_newest_first_and_needs_no_index(backend):
    tick = [10.0]

    def clock() -> float:
        tick[0] += 10.0
        return tick[0]

    reg = Registry(backend=backend, writer_id="w", clock=clock)
    old = _run(reg, run_id="r-old")
    new = _run(reg, run_id="r-new")
    assert new.created_at > old.created_at

    runs = reg.list_runs()

    assert [r.run_id for r in runs] == [new.run_id, old.run_id]
    # No "index" key exists to drift out of sync with the actual runs.
    assert not any("index" in key for key in backend.keys())


def test_branch_status_is_validated(reg):
    run = _run(reg)
    with pytest.raises(ValueError, match="unknown status"):
        reg.put_branch(run.run_id, BranchRecord(branch_id="b0", status="probably-fine"))
    _branch(reg, run.run_id, "b0")
    with pytest.raises(ValueError, match="unknown status"):
        reg.update_branch(run.run_id, "b0", status="winning")
    assert set(STATUSES) == {
        "pending", "running", "evaluating", "abandoned", "succeeded", "failed",
    }


def test_branches_are_returned_in_tree_order(reg):
    run = _run(reg)
    for bid in ("b0.10", "b0.2", "b0", "b0.1.1", "b0.1"):
        _branch(reg, run.run_id, bid, parent_id=bid.rpartition(".")[0] or None)

    assert [b.branch_id for b in reg.branches(run.run_id)] == [
        "b0", "b0.1", "b0.1.1", "b0.2", "b0.10",
    ]


# ---- the layout that makes concurrency safe -----------------------------


def test_each_branch_and_writer_gets_its_own_key(backend, reg):
    run = _run(reg)
    _branch(reg, run.run_id, "b0")
    _branch(reg, run.run_id, "b0.1", parent_id="b0")
    reg.append_event(run.run_id, "info", "hello", branch_id="b0")

    keys = sorted(backend.keys())
    assert f"{run.run_id}/branch/b0" in keys
    assert f"{run.run_id}/branch/b0.1" in keys
    assert f"{run.run_id}/event/b0:w-test" in keys
    # Nothing holds more than one branch, so nothing can clobber a neighbour.
    for key in keys:
        value = backend.get(key)
        if "/branch/" in key:
            assert isinstance(value, dict) and "branch_id" in value


def test_concurrent_branch_updates_do_not_clobber_each_other(backend):
    """Six writers, each hammering its own branch, interleaved on purpose."""
    reg = Registry(backend=backend, writer_id="setup")
    run = _run(reg)
    writers = [f"b0.{i}" for i in range(1, 7)]
    for bid in writers:
        _branch(reg, run.run_id, bid, parent_id="b0")

    rounds = 20
    start = threading.Barrier(len(writers))
    errors: list[BaseException] = []

    def work(branch_id: str) -> None:
        own = Registry(backend=backend, writer_id=branch_id)
        try:
            start.wait(timeout=5)
            for step in range(rounds):
                own.append_score(run.run_id, branch_id, step, step / 100.0)
                own.add_spend(run.run_id, branch_id, 0.01)
                own.append_event(run.run_id, "info", f"{branch_id} step {step}", branch_id)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(b,)) for b in writers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert errors == []
    for bid in writers:
        rec = reg.get_branch(run.run_id, bid)
        assert len(rec.scores) == rounds, f"{bid} lost score updates"
        assert rec.spend_usd == pytest.approx(rounds * 0.01)

    # Every event from every writer survived -- this is the exact assertion
    # the predecessor project would have failed (29 of 120 kept, all from one
    # writer).
    events = reg.events(run.run_id)
    assert len(events) == len(writers) * rounds
    per_writer = {b: sum(1 for e in events if e["branch_id"] == b) for b in writers}
    assert set(per_writer.values()) == {rounds}

    # And the run total is derived from the branch keys, not a shared counter.
    assert reg.get_run(run.run_id).spend_usd == pytest.approx(len(writers) * rounds * 0.01)


def test_event_shards_merge_in_timestamp_order(backend):
    """Shards are per-writer; the merge is what produces a single log."""
    clock_a = iter([1.0, 3.0, 5.0])
    clock_b = iter([2.0, 4.0, 6.0])
    setup = Registry(backend=backend, writer_id="s", clock=lambda: 0.5)
    run = _run(setup)
    a = Registry(backend=backend, writer_id="wa", clock=lambda: next(clock_a))
    b = Registry(backend=backend, writer_id="wb", clock=lambda: next(clock_b))

    # Written in a different order than their timestamps imply.
    a.append_event(run.run_id, "info", "a1", "b0.1")
    a.append_event(run.run_id, "info", "a3", "b0.1")
    b.append_event(run.run_id, "info", "b2", "b0.2")
    a.append_event(run.run_id, "info", "a5", "b0.1")
    b.append_event(run.run_id, "info", "b4", "b0.2")
    b.append_event(run.run_id, "info", "b6", "b0.2")

    merged = setup.events(run.run_id)
    assert [e["message"] for e in merged] == ["a1", "b2", "a3", "b4", "a5", "b6"]
    assert [e["ts"] for e in merged] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert [e["message"] for e in setup.events(run.run_id, since=3.0)] == ["b4", "a5", "b6"]


def test_events_within_one_clock_tick_keep_their_order(backend):
    reg = Registry(backend=backend, writer_id="w", clock=lambda: 7.0)
    run = _run(reg)
    for i in range(4):
        reg.append_event(run.run_id, "info", f"m{i}", "b0")

    messages = [e["message"] for e in reg.events(run.run_id)]
    assert messages == ["m0", "m1", "m2", "m3"]


def test_routing_backend_splits_runs_from_branches():
    runs, branches = MemoryBackend(), MemoryBackend()
    reg = Registry(backend=RoutingBackend(runs, branches), writer_id="w")
    run = _run(reg)
    _branch(reg, run.run_id, "b0")
    reg.append_event(run.run_id, "info", "hi", "b0")

    assert list(runs.keys()) == [f"run/{run.run_id}"]
    assert sorted(branches.keys()) == [
        f"{run.run_id}/branch/b0",
        f"{run.run_id}/event/b0:w",
    ]
    assert reg.api_payload(run.run_id)["branches"][0]["branch_id"] == "b0"


# ---- best ---------------------------------------------------------------


def test_set_best_never_regresses(reg):
    run = _run(reg)
    assert reg.set_best(run.run_id, "b0.1", 0.5)["score"] == 0.5
    assert reg.set_best(run.run_id, "b0.2", 0.9)["branch_id"] == "b0.2"
    # A late straggler reporting a worse score must not demote the winner.
    assert reg.set_best(run.run_id, "b0.3", 0.1)["branch_id"] == "b0.2"


def test_set_best_respects_a_minimize_objective(backend):
    reg = Registry(backend=backend, writer_id="w")
    run = reg.create_run(
        env_name="toy", objective={"name": "loss", "direction": "minimize"}, spec={}
    )
    reg.set_best(run.run_id, "b0.1", 0.5)
    assert reg.set_best(run.run_id, "b0.2", 0.9)["branch_id"] == "b0.1"
    assert reg.set_best(run.run_id, "b0.3", 0.2)["branch_id"] == "b0.3"


# ---- the API contract ---------------------------------------------------


def test_api_payload_matches_the_frontend_contract_exactly(backend):
    reg = Registry(backend=backend, writer_id="w", clock=lambda: 1000.0)
    run = _run(reg, run_id="r-1")
    reg.update_run(run.run_id, state="running")
    _branch(
        reg,
        run.run_id,
        "b0",
        status="running",
        reward_config={"weights": {"round_win": 1.0}},
        scores=[{"step": 100, "score": 0.42}],
        spend_usd=1.25,
        tokens=5000,
        rationale="start from the obvious terminal reward",
    )
    _branch(
        reg,
        run.run_id,
        "b0.1",
        parent_id="b0",
        status="abandoned",
        abandoned_reason="no improvement over 3 evaluations",
    )
    reg.append_event(run.run_id, "warn", "b0.1 abandoned", "b0.1")
    reg.set_best(run.run_id, "b0", 0.42)

    payload = reg.api_payload(run.run_id)

    assert set(payload) == {
        "run_id", "env_name", "objective", "state", "created_at", "elapsed_s",
        "spend_usd", "budget", "best", "branches", "events",
    }
    assert set(payload["objective"]) == {"name", "direction"}
    assert set(payload["best"]) == {"branch_id", "score"}
    assert payload["best"] == {"branch_id": "b0", "score": 0.42}
    assert set(payload["branches"][0]) == {
        "branch_id", "parent_id", "depth", "status", "reward_config", "scores",
        "spend_usd", "tokens", "rationale", "abandoned_reason",
        # A branch is a sandbox, and the tree view's first question is what it
        # is doing; these travel with the node so the client never scrapes the
        # event log to answer it.
        "activity", "activity_since", "sandbox_id", "best_score", "checkpoint",
        # Provenance: the artifact states how its headline number was measured.
        "score_command", "score_detail",
    }
    assert set(payload["events"][0]) == {"ts", "level", "branch_id", "message"}
    assert payload["run_id"] == "r-1"
    assert payload["state"] == "running"
    assert payload["spend_usd"] == pytest.approx(1.25)
    assert [b["branch_id"] for b in payload["branches"]] == ["b0", "b0.1"]
    assert payload["branches"][1]["abandoned_reason"] == "no improvement over 3 evaluations"


def test_api_payload_best_is_none_before_any_score(reg):
    run = _run(reg)
    assert reg.api_payload(run.run_id)["best"] is None


def test_api_payload_is_json_serialisable(reg):
    import json

    run = _run(reg)
    _branch(reg, run.run_id, "b0", status="succeeded")
    reg.append_event(run.run_id, "info", "done", "b0")
    reg.set_best(run.run_id, "b0", 1.0)

    assert json.loads(json.dumps(reg.api_payload(run.run_id)))["run_id"] == run.run_id


def test_api_payload_raises_for_a_missing_run(reg):
    with pytest.raises(KeyError):
        reg.api_payload("r-nope")


def test_elapsed_freezes_when_a_run_finishes(backend):
    now = [100.0]
    reg = Registry(backend=backend, writer_id="w", clock=lambda: now[0])
    run = _run(reg)
    now[0] = 160.0
    assert reg.get_run(run.run_id).elapsed_s == pytest.approx(60.0)

    reg.update_run(run.run_id, state="succeeded")
    now[0] = 9999.0
    assert reg.get_run(run.run_id).elapsed_s == pytest.approx(60.0)


def test_update_branch_rejects_unknown_fields(reg):
    run = _run(reg)
    _branch(reg, run.run_id, "b0")
    with pytest.raises(AttributeError):
        reg.update_branch(run.run_id, "b0", reward="nope")
    with pytest.raises(KeyError):
        reg.update_branch(run.run_id, "b9", status="running")


# ---- a finished run must actually finish -----------------------------

def _done(reg, run_id, bid, status, *, scores=None, parent=None, spend=0.0):
    reg.put_branch(run_id, BranchRecord(
        branch_id=bid, parent_id=parent, depth=0 if parent is None else 1,
        status=status, scores=scores or [], spend_usd=spend,
    ))


def test_a_run_whose_branches_are_all_done_reports_terminal(reg):
    """The bug this guards: a finished run sat at `running` with its elapsed
    clock ticking forever, because nothing owns "the run is over" -- each
    branch only knows about itself."""
    run = _run(reg, run_id="r-fin")
    reg.update_run("r-fin", state="running")
    _done(reg, "r-fin", "b0", "succeeded", scores=[{"step": 10, "score": 0.9}])
    _done(reg, "r-fin", "b0.1", "abandoned", parent="b0")

    got = reg.get_run("r-fin")
    assert got.state == "succeeded"

    # And it stays frozen: a later read must not keep advancing elapsed.
    first = got.elapsed_s
    again = reg.get_run("r-fin")
    assert again.state == "succeeded"
    assert again.elapsed_s == pytest.approx(first)


def test_a_run_still_working_stays_running(reg):
    _run(reg, run_id="r-busy")
    reg.update_run("r-busy", state="running")
    _done(reg, "r-busy", "b0", "succeeded", scores=[{"step": 1, "score": 0.5}])
    _done(reg, "r-busy", "b0.1", "running", parent="b0")
    assert reg.get_run("r-busy").state == "running"


def test_a_run_where_nothing_was_measured_is_a_failure(reg):
    """A tree where every branch died without scoring is a failed run, not a
    successful one that happens to have no result."""
    _run(reg, run_id="r-dud")
    reg.update_run("r-dud", state="running")
    _done(reg, "r-dud", "b0", "failed")
    _done(reg, "r-dud", "b0.1", "abandoned", parent="b0")
    assert reg.get_run("r-dud").state == "failed"


def test_run_spend_falls_back_to_branch_records_off_volume(reg):
    _run(reg, run_id="r-cost")
    _done(reg, "r-cost", "b0", "succeeded",
          scores=[{"step": 1, "score": 0.1}], spend=0.25)
    _done(reg, "r-cost", "b0.1", "abandoned", parent="b0", spend=0.10)
    assert reg.get_run("r-cost").spend_usd == pytest.approx(0.35)


def test_a_run_with_no_branches_yet_is_not_declared_finished(reg):
    _run(reg, run_id="r-new")
    reg.update_run("r-new", state="running")
    assert reg.get_run("r-new").state == "running"


# ---- serialisation must cover every field ---------------------------

def test_branch_record_to_dict_covers_every_field():
    """The bug this guards: five fields were added to the dataclass and not to
    to_dict(), so they were set in memory, dropped on write, and read back as
    defaults. The visible symptoms were an artifact shipped with
    weights/MISSING.txt and a tree view with no activity -- neither of which
    points at serialisation. A field-coverage assertion does."""
    from dataclasses import fields

    declared = {f.name for f in fields(BranchRecord)}
    serialised = set(BranchRecord(branch_id="b0").to_dict())
    assert declared == serialised, (
        f"to_dict() drops {sorted(declared - serialised)} and invents "
        f"{sorted(serialised - declared)}"
    )


def test_branch_record_round_trips_every_field():
    rec = BranchRecord(
        branch_id="b0.1", parent_id="b0", depth=1, status="succeeded",
        reward_config={"weights": {"reached_goal": 1.0}},
        scores=[{"step": 10, "score": 0.5}], spend_usd=1.25, tokens=4096,
        rationale="why", abandoned_reason=None, snapshot_id="im-1",
        sandbox_id="sb-1", created_at=1.0, updated_at=2.0,
        activity="training 50,000 steps", activity_since=3.0,
        checkpoint="/data/runs/r/branches/b0.1/step_100.pt",
        score_command="modal run ...", score_detail={"deterministic": 0.48},
    )
    back = BranchRecord.from_dict(rec.to_dict())
    assert back == rec


def test_run_record_to_dict_covers_every_field():
    from dataclasses import fields

    declared = {f.name for f in fields(RunRecord)}
    serialised = set(RunRecord(run_id="r").to_dict())
    assert declared == serialised, (
        f"to_dict() drops {sorted(declared - serialised)}"
    )
