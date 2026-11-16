"""Make the registry agree with which sandboxes are actually alive.

This exists because of a specific, repeated failure. A branch records
`status="running"` when it starts and updates that record from inside its own
sandbox. If the sandbox dies in a way that does not run Python -- Modal
preemption, an OOM kill, a lost worker -- nothing ever writes the terminal
status, and the registry reports a healthy running branch forever.

That is not a cosmetic problem. It is the reason an SF3 run that was preempted
37 minutes in went unnoticed for **eleven hours**: the monitor polled the
registry, the registry said `running`, and the registry had no way to know.
Polling real state rather than trusting a record is README lesson 8, and
the registry is a record.

So: ask Modal which sandboxes are alive, and mark the branches whose sandboxes
are not. Read-only unless `apply=True`, because a reconciler that guesses
wrong and abandons a live branch is worse than one that reports.

    python -m autobranch.reconcile                      # report every run
    python -m autobranch.reconcile --run-id run-xxxx    # one run
    python -m autobranch.reconcile --apply              # write the corrections
"""

import argparse
import json
from typing import Any

#: Branch statuses that claim something is still executing. Only these can be
#: contradicted by a dead sandbox; `abandoned` and `succeeded` are terminal and
#: a dead sandbox is consistent with them.
LIVE_STATUSES = ("running", "pending")


def live_sandbox_ids(app_name: str = "autobranch") -> set[str]:
    """Object ids of sandboxes Modal still considers running."""
    import modal

    try:
        app = modal.App.lookup(app_name, create_if_missing=False)
    except Exception:  # noqa: BLE001 - no app means nothing is alive
        return set()
    out: set[str] = set()
    for sb in modal.Sandbox.list(app_id=app.app_id):
        try:
            if sb.poll() is None:
                out.add(sb.object_id)
        except Exception:  # noqa: BLE001 - treat an unreadable sandbox as dead
            continue
    return out


def find_orphans(registry, run_id: str, live: set[str]) -> list[dict[str, Any]]:
    """Branches claiming to run whose sandbox is gone.

    A branch with no `sandbox_id` at all is reported separately rather than
    assumed dead: it may simply not have reached the point of recording one,
    which is a few seconds after it is registered.
    """
    out: list[dict[str, Any]] = []
    for b in registry.branches(run_id):
        status = getattr(b, "status", "")
        if status not in LIVE_STATUSES:
            continue
        sandbox = getattr(b, "sandbox_id", None)
        if not sandbox:
            out.append({"branch_id": b.branch_id, "status": status,
                        "sandbox_id": None, "verdict": "no sandbox recorded yet"})
            continue
        if sandbox not in live:
            out.append({"branch_id": b.branch_id, "status": status,
                        "sandbox_id": sandbox, "verdict": "sandbox is gone"})
    return out


def reconcile(run_id: str | None = None, *, apply: bool = False,
              app_name: str = "autobranch") -> dict[str, Any]:
    """Report, and optionally correct, branches whose sandbox has died."""
    from autobranch.registry import Registry

    registry = Registry()
    live = live_sandbox_ids(app_name)
    run_ids = [run_id] if run_id else [r.run_id for r in registry.list_runs()]

    report: dict[str, Any] = {"live_sandboxes": len(live), "runs": {}}
    for rid in run_ids:
        run = registry.get_run(rid)
        if run is None:
            continue
        orphans = find_orphans(registry, rid, live)
        actionable = [o for o in orphans if o["sandbox_id"]]
        if not orphans and getattr(run, "state", "") not in LIVE_STATUSES:
            continue
        report["runs"][rid] = {
            "state": getattr(run, "state", "?"),
            "orphans": orphans,
        }
        if not (apply and actionable):
            continue

        reason = (
            "sandbox is gone and never wrote a terminal status -- preemption, an "
            "OOM kill or a lost worker. Reconciled from Modal's sandbox list, "
            "not inferred from the record."
        )
        for o in actionable:
            registry.update_branch(rid, o["branch_id"], status="failed",
                                   abandoned_reason=reason, activity=None)
        # Only the run's own state follows, and only if nothing is left alive.
        remaining = [b for b in registry.branches(rid)
                     if getattr(b, "status", "") in LIVE_STATUSES]
        if not remaining:
            registry.update_run(rid, state="failed")
            registry.append_event(
                rid, level="warn", branch_id=None,
                message=(f"reconciled: {len(actionable)} branch(es) had dead sandboxes; "
                         f"run marked failed"),
            )
        report["runs"][rid]["applied"] = [o["branch_id"] for o in actionable]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--app", default="autobranch")
    parser.add_argument("--apply", action="store_true",
                        help="write the corrections; omit to report only")
    args = parser.parse_args()

    report = reconcile(args.run_id or None, apply=args.apply, app_name=args.app)
    print(json.dumps(report, indent=2))
    # Exit non-zero when something is wrong, so a monitor can use this directly
    # instead of parsing the output.
    return 1 if any(r["orphans"] for r in report["runs"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
