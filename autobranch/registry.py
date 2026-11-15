"""Run and branch state: the source of truth for the tree and the web API.

The shape of this module is dictated by one failure from the project this
framework generalizes: a single shared state file under six concurrent
writers ended up holding 29 entries that all belonged to *one* writer. The
other five writers' work was not corrupted, it was absent, and nothing
reported an error. Read-modify-write on a shared blob loses data silently.

So there is no shared blob here. The layout is:

    run/<run_id>                      run header, written by the orchestrator
    run/<run_id>/best                 best-so-far, its own key
    <run_id>/branch/<branch_id>       one key per branch, one writer per key
    <run_id>/event/<writer>           one append-only shard per writer

Every write touches exactly one key, and every key has exactly one writer, so
two branches updating at the same time cannot overwrite each other no matter
how their calls interleave. Anything that looks like a total -- run spend,
the event log -- is *derived at read* by scanning keys, never maintained as a
counter that two writers would have to share.

Everything stored is JSON-serialisable, because it travels through a
`modal.Dict` and out of an HTTP endpoint unchanged.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

STATUSES = ("pending", "running", "evaluating", "abandoned", "succeeded", "failed")
TERMINAL_STATUSES = ("abandoned", "succeeded", "failed")
RUN_STATES = ("pending", "running", "succeeded", "failed", "stopped")

# A per-process token. Two processes writing events "as branch b0.1" would
# otherwise share an event shard and reintroduce exactly the bug this module
# exists to prevent.
PROCESS_TOKEN = uuid.uuid4().hex[:8]


# ---- backends -----------------------------------------------------------


class Backend(Protocol):
    """Single-key key/value access. Deliberately no update-in-place."""

    def get(self, key: str, default: Any = None) -> Any: ...
    def put(self, key: str, value: Any) -> None: ...
    def keys(self) -> Iterable[str]: ...
    def delete(self, key: str) -> None: ...


class MemoryBackend:
    """In-process backend for tests and dry runs.

    The lock is not decoration: tests drive this from several threads to prove
    that concurrent branch writes survive, and a backend that corrupted its
    own storage would make that test prove nothing.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class DictBackend:
    """`modal.Dict` adapter.

    `modal.Dict.put` is a server-side single-key write, which is what makes
    the per-key layout above safe across processes.
    """

    def __init__(self, dct: Any) -> None:
        self._dct = dct

    def get(self, key: str, default: Any = None) -> Any:
        return self._dct.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self._dct.put(key, value)

    def keys(self) -> list[str]:
        return [k for k in self._dct.keys()]

    def delete(self, key: str) -> None:
        self._dct.pop(key, None)


class RoutingBackend:
    """Two stores behind one interface, split by key shape.

    Branch records and event shards are written constantly; run headers are
    written a handful of times per run. Keeping them in separate Dicts means
    scanning for runs does not have to page through thousands of branch keys.
    """

    def __init__(self, runs: Backend, branches: Backend) -> None:
        self._runs = runs
        self._branches = branches

    def _pick(self, key: str) -> Backend:
        return self._branches if ("/branch/" in key or "/event/" in key) else self._runs

    def get(self, key: str, default: Any = None) -> Any:
        return self._pick(key).get(key, default)

    def put(self, key: str, value: Any) -> None:
        self._pick(key).put(key, value)

    def keys(self) -> list[str]:
        return [*self._runs.keys(), *self._branches.keys()]

    def delete(self, key: str) -> None:
        self._pick(key).delete(key)


def default_backend() -> Backend:
    """The real thing: `autobranch-runs` + `autobranch-branches`."""
    from . import infra  # lazy: importing modal costs seconds, tests skip it

    return RoutingBackend(DictBackend(infra.runs_dict()), DictBackend(infra.branches_dict()))


# ---- records ------------------------------------------------------------


@dataclass
class BranchRecord:
    """One node of the search tree, as the frontend sees it.

    `scores` is a list of `{step, score}` rather than a single number because
    the pruner needs the trajectory: "failed to improve over K evaluations" is
    not answerable from a scalar. Scores are *objective* scores -- reward is
    never ranked.
    """

    branch_id: str
    parent_id: str | None = None
    depth: int = 0
    status: str = "pending"
    reward_config: dict[str, Any] = field(default_factory=dict)
    scores: list[dict[str, Any]] = field(default_factory=list)
    spend_usd: float = 0.0
    tokens: int = 0
    rationale: str = ""
    abandoned_reason: str | None = None
    snapshot_id: str | None = None
    sandbox_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    # What this sandbox is doing right now, written by the branch loop BEFORE
    # it starts the work. Not derivable from the event log: a long train call
    # emits its event only on completion, so while training the log's last
    # line still describes the previous step.
    activity: str = ""
    activity_since: float = 0.0
    # The checkpoint the branch would ship, and the provenance of its latest
    # objective score. Carried here so consolidation never has to guess.
    checkpoint: str | None = None
    score_command: str = ""
    score_detail: dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> float | None:
        if not self.scores:
            return None
        return max((float(s.get("score", 0.0)) for s in self.scores), default=None)

    @property
    def latest_score(self) -> float | None:
        if not self.scores:
            return None
        return float(self.scores[-1].get("score", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "status": self.status,
            "reward_config": dict(self.reward_config),
            "scores": [dict(s) for s in self.scores],
            "spend_usd": float(self.spend_usd),
            "tokens": int(self.tokens),
            "rationale": self.rationale,
            "abandoned_reason": self.abandoned_reason,
            "snapshot_id": self.snapshot_id,
            "sandbox_id": self.sandbox_id,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "BranchRecord":
        return BranchRecord(
            branch_id=raw["branch_id"],
            parent_id=raw.get("parent_id"),
            depth=int(raw.get("depth", 0)),
            status=raw.get("status", "pending"),
            reward_config=dict(raw.get("reward_config") or {}),
            scores=[dict(s) for s in (raw.get("scores") or [])],
            spend_usd=float(raw.get("spend_usd", 0.0)),
            tokens=int(raw.get("tokens", 0)),
            rationale=raw.get("rationale", "") or "",
            abandoned_reason=raw.get("abandoned_reason"),
            snapshot_id=raw.get("snapshot_id"),
            sandbox_id=raw.get("sandbox_id"),
            created_at=float(raw.get("created_at", 0.0)),
            updated_at=float(raw.get("updated_at", 0.0)),
        )


@dataclass
class RunRecord:
    """The run header.

    `spend_usd` and `elapsed_s` are *derived* fields: whatever is stored is
    ignored on read and recomputed from the branch keys and the clock. A
    stored running total would need every branch to read-modify-write one
    key, which is the failure this module is built to avoid.
    """

    run_id: str
    env_name: str = ""
    objective: dict[str, Any] = field(default_factory=dict)
    state: str = "pending"
    constraints: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    elapsed_s: float = 0.0
    spend_usd: float = 0.0
    best: dict[str, Any] | None = None
    spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "env_name": self.env_name,
            "objective": dict(self.objective),
            "state": self.state,
            "constraints": self.constraints,
            "budget": dict(self.budget),
            "created_at": float(self.created_at),
            "elapsed_s": float(self.elapsed_s),
            "spend_usd": float(self.spend_usd),
            "best": dict(self.best) if self.best else None,
            "spec": dict(self.spec),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "RunRecord":
        return RunRecord(
            run_id=raw["run_id"],
            env_name=raw.get("env_name", "") or "",
            objective=dict(raw.get("objective") or {}),
            state=raw.get("state", "pending"),
            constraints=raw.get("constraints", "") or "",
            budget=dict(raw.get("budget") or {}),
            created_at=float(raw.get("created_at", 0.0)),
            elapsed_s=float(raw.get("elapsed_s", 0.0)),
            spend_usd=float(raw.get("spend_usd", 0.0)),
            best=dict(raw["best"]) if raw.get("best") else None,
            spec=dict(raw.get("spec") or {}),
        )


def _objective_dict(objective: Any) -> dict[str, Any]:
    """Accept a `spec.Objective`, a dict, or a bare name."""
    if objective is None:
        return {"name": "", "direction": "maximize"}
    if isinstance(objective, str):
        return {"name": objective, "direction": "maximize"}
    if isinstance(objective, dict):
        out = dict(objective)
        out.setdefault("name", "")
        out.setdefault("direction", "maximize")
        return out
    return {
        "name": getattr(objective, "name", ""),
        "direction": getattr(objective, "direction", "maximize"),
        "evaluator": getattr(objective, "evaluator", None),
        "budget": dict(getattr(objective, "budget", {}) or {}),
    }


# ---- registry -----------------------------------------------------------


class Registry:
    """Reads and writes run state. One instance per writer.

    `backend` is an injection seam: tests pass a `MemoryBackend`, production
    gets `RoutingBackend` over the two named `modal.Dict`s. `writer_id`
    identifies this instance's event shard and must differ per concurrent
    writer -- which it does by default, via `PROCESS_TOKEN`.
    """

    def __init__(self, *, backend: Backend | None = None, clock: Any = time.time,
                 writer_id: str | None = None) -> None:
        self.backend = backend if backend is not None else default_backend()
        self._now = clock
        self.writer_id = writer_id or PROCESS_TOKEN
        self._event_lock = threading.Lock()

    # -- keys ------------------------------------------------------------

    @staticmethod
    def run_key(run_id: str) -> str:
        return f"run/{run_id}"

    @staticmethod
    def best_key(run_id: str) -> str:
        return f"run/{run_id}/best"

    @staticmethod
    def branch_key(run_id: str, branch_id: str) -> str:
        return f"{run_id}/branch/{branch_id}"

    def event_key(self, run_id: str, branch_id: str | None) -> str:
        """One shard per (branch, process). Never shared, therefore never lossy."""
        who = branch_id or "orchestrator"
        return f"{run_id}/event/{who}:{self.writer_id}"

    # -- runs ------------------------------------------------------------

    def create_run(
        self,
        *,
        env_name: str,
        objective: Any,
        spec: dict[str, Any] | None = None,
        run_id: str | None = None,
        constraints: str = "",
        budget: dict[str, Any] | None = None,
        state: str = "pending",
    ) -> RunRecord:
        now = self._now()
        if run_id is None:
            # Sortable by construction: lexicographic order is time order.
            run_id = f"r{int(now)}-{uuid.uuid4().hex[:6]}"
        rec = RunRecord(
            run_id=run_id,
            env_name=env_name,
            objective=_objective_dict(objective),
            state=state,
            constraints=constraints,
            budget=dict(budget or {}),
            created_at=now,
            elapsed_s=0.0,
            spend_usd=0.0,
            best=None,
            spec=dict(spec or {}),
        )
        self.backend.put(self.run_key(run_id), rec.to_dict())
        return rec

    def get_run(self, run_id: str) -> RunRecord | None:
        raw = self.backend.get(self.run_key(run_id))
        if not raw:
            return None
        rec = RunRecord.from_dict(raw)
        return self._hydrate_run(rec)

    def _hydrate_run(self, rec: RunRecord, *, keys: list[str] | None = None) -> RunRecord:
        """Fill the derived fields. See `RunRecord` for why they are derived."""
        rec.best = self.backend.get(self.best_key(rec.run_id)) or None
        rec.spend_usd = round(
            sum(b.spend_usd for b in self.branches(rec.run_id, keys=keys)), 6
        )
        if rec.state in ("pending", "running"):
            rec.elapsed_s = max(0.0, self._now() - rec.created_at)
        return rec

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        """Newest first. Enumerated by key scan -- there is no index blob to
        fall out of sync with reality."""
        keys = list(self.backend.keys())
        run_keys = [k for k in keys if k.startswith("run/") and k.count("/") == 1]
        runs: list[RunRecord] = []
        for key in run_keys:
            raw = self.backend.get(key)
            if raw:
                runs.append(self._hydrate_run(RunRecord.from_dict(raw), keys=keys))
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    def update_run(self, run_id: str, **fields: Any) -> RunRecord:
        """Update the run header. Orchestrator-only, so single-writer by design."""
        raw = self.backend.get(self.run_key(run_id))
        if not raw:
            raise KeyError(run_id)
        rec = RunRecord.from_dict(raw)
        if "state" in fields and fields["state"] not in RUN_STATES:
            raise ValueError(f"unknown run state {fields['state']!r}; expected one of {RUN_STATES}")
        if "objective" in fields:
            fields["objective"] = _objective_dict(fields["objective"])
        for key, value in fields.items():
            if not hasattr(rec, key):
                raise AttributeError(f"RunRecord has no field {key!r}")
            setattr(rec, key, value)
        if rec.state not in ("pending", "running"):
            # Freeze the clock on a finished run, so the UI stops counting.
            rec.elapsed_s = max(rec.elapsed_s, self._now() - rec.created_at)
        self.backend.put(self.run_key(run_id), rec.to_dict())
        return self._hydrate_run(rec)

    # -- branches --------------------------------------------------------

    def put_branch(self, run_id: str, rec: BranchRecord) -> None:
        if rec.status not in STATUSES:
            raise ValueError(f"unknown status {rec.status!r}; expected one of {STATUSES}")
        now = self._now()
        if not rec.created_at:
            rec.created_at = now
        rec.updated_at = now
        self.backend.put(self.branch_key(run_id, rec.branch_id), rec.to_dict())

    def get_branch(self, run_id: str, branch_id: str) -> BranchRecord | None:
        raw = self.backend.get(self.branch_key(run_id, branch_id))
        return BranchRecord.from_dict(raw) if raw else None

    def branches(self, run_id: str, *, keys: list[str] | None = None) -> list[BranchRecord]:
        """Every branch of a run, in tree order (siblings numerically adjacent)."""
        from .branch import sort_key

        prefix = f"{run_id}/branch/"
        all_keys = keys if keys is not None else list(self.backend.keys())
        out: list[BranchRecord] = []
        for key in all_keys:
            if not key.startswith(prefix):
                continue
            raw = self.backend.get(key)
            if raw:
                out.append(BranchRecord.from_dict(raw))

        def _key(rec: BranchRecord) -> tuple[Any, ...]:
            try:
                return (0, sort_key(rec.branch_id))
            except Exception:
                return (1, rec.branch_id)

        out.sort(key=_key)
        return out

    def update_branch(self, run_id: str, branch_id: str, **fields: Any) -> BranchRecord:
        """Patch one branch.

        This reads and writes a single key that belongs to a single branch, so
        two branches updating concurrently never touch the same key. Two
        writers for the *same* branch would still race -- but a branch has one
        agent by construction, and the orchestrator patches only branches
        whose agent has exited.
        """
        key = self.branch_key(run_id, branch_id)
        raw = self.backend.get(key)
        if not raw:
            raise KeyError(f"{run_id}/{branch_id}")
        rec = BranchRecord.from_dict(raw)
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"unknown status {fields['status']!r}; expected one of {STATUSES}")
        for name, value in fields.items():
            if not hasattr(rec, name):
                raise AttributeError(f"BranchRecord has no field {name!r}")
            setattr(rec, name, value)
        rec.updated_at = self._now()
        self.backend.put(key, rec.to_dict())
        return rec

    def append_score(self, run_id: str, branch_id: str, step: int, score: float) -> BranchRecord:
        """Append one objective evaluation to a branch's trajectory."""
        rec = self.get_branch(run_id, branch_id)
        if rec is None:
            raise KeyError(f"{run_id}/{branch_id}")
        scores = [*rec.scores, {"step": int(step), "score": float(score)}]
        return self.update_branch(run_id, branch_id, scores=scores)

    def add_spend(self, run_id: str, branch_id: str, usd: float, tokens: int = 0) -> BranchRecord:
        """Accumulate a branch's own spend on its own key.

        The run total is summed from these at read; see `_hydrate_run`.
        """
        rec = self.get_branch(run_id, branch_id)
        if rec is None:
            raise KeyError(f"{run_id}/{branch_id}")
        return self.update_branch(
            run_id,
            branch_id,
            spend_usd=round(rec.spend_usd + float(usd), 6),
            tokens=rec.tokens + int(tokens),
        )

    # -- events ----------------------------------------------------------

    def append_event(self, run_id: str, level: str, message: str,
                     branch_id: str | None = None) -> dict[str, Any]:
        """Append to *this writer's* event shard.

        The read-modify-write below is safe precisely because the key is
        private to this writer. Aggregation happens in `events()`.
        """
        key = self.event_key(run_id, branch_id)
        with self._event_lock:
            shard = list(self.backend.get(key) or [])
            ts = self._now()
            if shard and ts <= float(shard[-1]["ts"]):
                # Two events inside one clock tick must still order correctly
                # within the shard; nudging ts keeps the merge stable.
                ts = float(shard[-1]["ts"]) + 1e-6
            event = {"ts": ts, "level": level, "branch_id": branch_id, "message": message}
            shard.append(event)
            self.backend.put(key, shard)
        return event

    def events(self, run_id: str, since: float = 0.0) -> list[dict[str, Any]]:
        """Merge every shard, oldest first.

        Ordering across shards is by timestamp only. Clocks on separate
        sandboxes are not perfectly aligned, so this is a best-effort
        interleaving of independent logs, not a causal order.
        """
        prefix = f"{run_id}/event/"
        merged: list[dict[str, Any]] = []
        for key in self.backend.keys():
            if not key.startswith(prefix):
                continue
            for event in self.backend.get(key) or []:
                if float(event.get("ts", 0.0)) > since:
                    merged.append(dict(event))
        merged.sort(key=lambda e: (float(e.get("ts", 0.0)), e.get("branch_id") or ""))
        return merged

    # -- best ------------------------------------------------------------

    def set_best(self, run_id: str, branch_id: str, score: float) -> dict[str, Any]:
        """Record the best branch so far, respecting the objective's direction.

        Monotone: a worse score never displaces a better one, so a late
        straggler reporting a bad result cannot demote the winner. Two writers
        improving at the same instant can still race, and the loser of that
        race is simply re-recorded by the next improvement.
        """
        run = self.backend.get(self.run_key(run_id)) or {}
        direction = (run.get("objective") or {}).get("direction", "maximize")
        sign = 1.0 if direction != "minimize" else -1.0
        current = self.backend.get(self.best_key(run_id))
        score = float(score)
        if current and float(current.get("score", 0.0)) * sign >= score * sign:
            return dict(current)
        best = {"branch_id": branch_id, "score": score}
        self.backend.put(self.best_key(run_id), best)
        return best

    def get_best(self, run_id: str) -> dict[str, Any] | None:
        return self.backend.get(self.best_key(run_id)) or None

    # -- api -------------------------------------------------------------

    def api_payload(self, run_id: str) -> dict[str, Any]:
        """Exactly the document the frontend consumes. No extra keys.

        The contract is pinned by `tests/test_registry.py`: the frontend reads
        these names directly, so adding a field here is a frontend change.
        """
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return {
            "run_id": run.run_id,
            "env_name": run.env_name,
            "objective": {
                "name": run.objective.get("name", ""),
                "direction": run.objective.get("direction", "maximize"),
            },
            "state": run.state,
            "created_at": run.created_at,
            "elapsed_s": run.elapsed_s,
            "spend_usd": run.spend_usd,
            "budget": dict(run.budget),
            "best": (
                {"branch_id": run.best["branch_id"], "score": run.best["score"]}
                if run.best
                else None
            ),
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "parent_id": b.parent_id,
                    "depth": b.depth,
                    "status": b.status,
                    "reward_config": b.reward_config,
                    "scores": b.scores,
                    "spend_usd": b.spend_usd,
                    "tokens": b.tokens,
                    "rationale": b.rationale,
                    "abandoned_reason": b.abandoned_reason,
                    # A watcher's first question is "what is this sandbox
                    # doing", so it ships with the node rather than having to
                    # be scraped out of the event log.
                    "activity": b.activity,
                    "activity_since": b.activity_since,
                    "sandbox_id": b.sandbox_id,
                    "best_score": b.best_score,
                    "checkpoint": b.checkpoint,
                }
                for b in self.branches(run_id)
            ],
            "events": [
                {
                    "ts": e["ts"],
                    "level": e.get("level", "info"),
                    "branch_id": e.get("branch_id"),
                    "message": e.get("message", ""),
                }
                for e in self.events(run_id)
            ],
        }
