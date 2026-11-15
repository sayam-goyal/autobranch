"""Keep a run alive: detect a dead branch and restart it.

Written because a run was preempted 37 minutes in and nobody noticed for
**eleven hours**. `reconcile.py` is the diagnosis half -- it asks Modal which
sandboxes are alive and corrects the registry. This is the treatment half: when
the root branch is gone and the run still has deadline left, start it again.

Three things make the restart worth doing rather than just alarming:

* **The trainer resumes.** `grpo_full` mirrors `progress.json` and its HF export
  to the Volume, so a new sandbox continues from the last export rather than
  from zero. Before that fix a restart would have silently started over, which
  is worse than not restarting.
* **The run keeps its identity.** `launch_root` takes an existing `run_id` and
  `deadline_ts`, so a restart is the same run with a new sandbox -- not a new
  run that fragments the history and the artifact.
* **The reward survives.** The branch's recorded `reward_config` is passed back
  in, so a restart does not discard whatever the agent had settled on and
  reopen the search at the baseline.

Deliberately narrow. It restarts the **root** of a run whose branches are all
dead, and it does nothing else: it will not resurrect one child of a live tree
(the orchestrator's own fork logic owns that), it will not extend a deadline,
and it will not exceed `--max-restarts`, because a branch that dies immediately
and repeatedly is a bug to read rather than a thing to keep relaunching.

    python -m autobranch.supervise --run-id run-xxxx            # one check
    python -m autobranch.supervise --run-id run-xxxx --loop 60   # supervise
"""

import argparse
import json
import time
from typing import Any

TERMINAL_RUN_STATES = ("succeeded", "failed", "stopped")
LIVE_STATUSES = ("running", "pending")

#: A branch registered less than this long ago may not have recorded a sandbox
#: id yet. Restarting inside that window would duplicate a healthy branch.
GRACE_S = 180.0


def _budget(raw: dict[str, Any]):
    from autobranch.orchestrator import Budget

    fields = {f for f in Budget.__dataclass_fields__}
    return Budget(**{k: v for k, v in (raw or {}).items() if k in fields})


def check(run_id: str) -> dict[str, Any]:
    """What is alive, what the registry believes, and what is left of the clock."""
    from autobranch.reconcile import live_sandbox_ids
    from autobranch.registry import Registry

    registry = Registry()
    run = registry.get_run(run_id)
    if run is None:
        return {"run_id": run_id, "verdict": "no such run"}

    live = live_sandbox_ids()
    branches = list(registry.branches(run_id))
    claiming = [b for b in branches if getattr(b, "status", "") in LIVE_STATUSES]
    alive = [b for b in claiming if getattr(b, "sandbox_id", None) in live]
    young = [b for b in claiming
             if not getattr(b, "sandbox_id", None)
             and (time.time() - float(getattr(b, "created_at", 0) or 0)) < GRACE_S]

    deadline = float(getattr(run, "created_at", 0) or 0) + \
        float((getattr(run, "budget", {}) or {}).get("hours", 0)) * 3600.0
    left = deadline - time.time()

    state = getattr(run, "state", "?")
    if state in TERMINAL_RUN_STATES:
        verdict = f"run is {state}; nothing to supervise"
    elif alive:
        verdict = f"{len(alive)} branch(es) alive"
    elif young:
        verdict = f"{len(young)} branch(es) too young to judge (< {GRACE_S:.0f}s)"
    elif left <= 0:
        verdict = "all branches dead and the deadline has passed"
    else:
        verdict = "all branches dead with deadline remaining -- RESTART"

    return {
        "run_id": run_id,
        "state": state,
        "branches": len(branches),
        "claiming_to_run": [b.branch_id for b in claiming],
        "alive": [b.branch_id for b in alive],
        "seconds_left": round(left, 1),
        "verdict": verdict,
        "should_restart": verdict.endswith("RESTART"),
        "should_finalize": verdict.startswith("all branches dead and the deadline"),
    }


def restart(run_id: str) -> dict[str, Any]:
    """Relaunch the root of a dead run, in place, resuming from the Volume.

    **Runs inside Modal only.** `launch_root` calls
    `ensure_runtime_on_volume`, which publishes the framework to `/data`, and
    that path is a read-only mount outside a container -- calling this from a
    laptop fails with `OSError: Read-only file system: '/data'`, which reads
    like a permissions problem and is a location problem. Use
    `modal run app.py::cli_restart --run-id ...`, or the deployed supervisor.
    """
    from autobranch.forking import launch_root
    from autobranch.registry import Registry
    from autobranch.spec import EnvironmentSpec

    registry = Registry()
    run = registry.get_run(run_id)
    if run is None:
        return {"ok": False, "reason": "no such run"}

    spec = EnvironmentSpec.parse(getattr(run, "spec", None) or {})
    budget = _budget(getattr(run, "budget", {}) or {})
    deadline = float(getattr(run, "created_at", 0) or 0) + float(budget.hours) * 3600.0

    # Carry the branch's own reward forward. Restarting at the baseline would
    # throw away whatever the agent had settled on and reopen the search.
    reward = None
    for b in registry.branches(run_id):
        if b.branch_id == "b0":
            reward = getattr(b, "reward_config", None) or None
            break

    constraints = {"text": getattr(run, "constraints", "") or ""}
    registry.append_event(
        run_id, level="warn", branch_id="b0",
        message=("supervisor: root branch was dead with deadline remaining; "
                 "relaunching in place and resuming from the Volume"),
    )
    out = launch_root(
        spec=spec, constraints=constraints, budget=budget, run_id=run_id,
        registry=registry, deadline_ts=deadline, reward_config=reward,
    )
    registry.update_run(run_id, state="running")
    return {"ok": True, **out}


def supervise(run_id: str, *, loop: float = 0.0, max_restarts: int = 4,
              heartbeat_every: int = 20) -> int:
    """Check once, or keep checking and restarting until the deadline.

    Prints only on a restart, a failure, or a terminal state -- plus a sparse
    heartbeat. The heartbeat matters: a supervisor that is silent when healthy
    is indistinguishable from a supervisor that has been killed, and being
    unable to tell those apart is precisely what cost eleven hours.
    """
    restarts = 0
    checks = 0
    while True:
        checks += 1
        try:
            status = check(run_id)
        except Exception as exc:  # noqa: BLE001 - a supervisor must not die on a blip
            print(f"[{time.strftime('%H:%M:%S')}] check failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if not loop:
                return 2
            time.sleep(loop)
            continue

        stamp = time.strftime("%H:%M:%S")
        if status.get("should_restart"):
            if restarts >= max_restarts:
                print(f"[{stamp}] DEAD and out of restarts ({restarts}/{max_restarts}) "
                      f"-- a branch dying this often is a bug to read, not to relaunch",
                      flush=True)
                return 1
            print(f"[{stamp}] DEAD: {status['verdict']} "
                  f"({status['seconds_left']:.0f}s left) -- restarting", flush=True)
            try:
                out = restart(run_id)
                restarts += 1
                print(f"[{stamp}] RESTARTED {run_id} in {out.get('sandbox_id')} "
                      f"(restart {restarts}/{max_restarts})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[{stamp}] RESTART FAILED: {type(exc).__name__}: {exc}", flush=True)
        elif status.get("should_finalize"):
            print(f"[{stamp}] deadline passed with everything dead; reconciling", flush=True)
            try:
                from autobranch.reconcile import reconcile

                reconcile(run_id, apply=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[{stamp}] reconcile failed: {type(exc).__name__}: {exc}", flush=True)
            return 0
        elif status.get("state") in TERMINAL_RUN_STATES:
            print(f"[{stamp}] {status['verdict']}", flush=True)
            return 0

        if not loop:
            print(json.dumps(status, indent=2))
            return 0

        if heartbeat_every and checks % heartbeat_every == 0:
            print(f"[{stamp}] alive: {status['verdict']}, "
                  f"{status['seconds_left'] / 3600:.1f}h left, "
                  f"{restarts} restart(s) so far", flush=True)
        time.sleep(loop)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--loop", type=float, default=0.0,
                        help="seconds between checks; 0 checks once and exits")
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("--heartbeat-every", type=int, default=20,
                        help="print a liveness line every N checks (0 disables)")
    args = parser.parse_args()
    return supervise(args.run_id, loop=args.loop, max_restarts=args.max_restarts,
                     heartbeat_every=args.heartbeat_every)


if __name__ == "__main__":
    raise SystemExit(main())
