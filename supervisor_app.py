"""The supervisor, running on Modal instead of on someone's laptop.

`python -m autobranch.supervise --loop 60` works, and dies with the terminal
that started it. That is not a theoretical problem: a run was preempted and sat
dead for eleven hours, and the second time the laptop that was watching it
closed. A supervisor that depends on a laptop being open is a supervisor that
is absent exactly when nobody is looking.

So it runs here, on a schedule, server-side:

    modal deploy supervisor_app.py

Every two minutes it asks Modal which sandboxes are alive, and if a supervised
run's branches are all gone while its deadline has not passed, it relaunches
the root in place -- same run id, same deadline, same reward config, resuming
from the HF export on the Volume.

Which runs to supervise lives in a `modal.Dict` rather than in this file, so
adding or removing one does not need a redeploy:

    python supervisor_app.py watch run-xxxx
    python supervisor_app.py unwatch run-xxxx
    python supervisor_app.py status
"""

import sys

import modal

APP_NAME = "autobranch-supervisor"

#: Runs under supervision, and each one's restart count. A Dict rather than a
#: constant so the watch list survives a redeploy and can be edited live.
WATCHLIST_NAME = "autobranch-supervisor-watchlist"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("modal==1.6.4", "pyyaml==6.0.2", "numpy>=1.26")
    .add_local_python_source("autobranch", copy=True)
)

#: Hard stop on relaunching one run. A branch that dies immediately and
#: repeatedly is a bug to read, not a thing to keep paying for -- and this one
#: runs unattended, so the ceiling matters more here than it does locally.
MAX_RESTARTS = 6


@app.function(
    image=image,
    schedule=modal.Period(minutes=2),
    timeout=600,
    secrets=[
        modal.Secret.from_name("autobranch-modal-auth"),
        modal.Secret.from_name("autobranch-kimi-auth"),
    ],
)
def tick() -> dict:
    """One supervision pass over every watched run."""
    import json

    from autobranch.supervise import check, restart

    watchlist = modal.Dict.from_name(WATCHLIST_NAME, create_if_missing=True)
    report: dict = {}

    for run_id in list(watchlist.keys()):
        entry = watchlist.get(run_id) or {}
        if not isinstance(entry, dict):
            entry = {"restarts": 0}
        try:
            status = check(run_id)
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop the rest
            report[run_id] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        if status.get("should_restart"):
            if entry.get("restarts", 0) >= MAX_RESTARTS:
                report[run_id] = {"verdict": "out of restarts", **status}
                continue
            try:
                out = restart(run_id)
                entry["restarts"] = entry.get("restarts", 0) + 1
                entry["last_restart_sandbox"] = out.get("sandbox_id")
                watchlist[run_id] = entry
                report[run_id] = {"restarted": out.get("sandbox_id"),
                                  "count": entry["restarts"]}
            except Exception as exc:  # noqa: BLE001
                report[run_id] = {"restart_failed": f"{type(exc).__name__}: {exc}"}
        else:
            report[run_id] = {"verdict": status.get("verdict"),
                              "alive": status.get("alive"),
                              "hours_left": round(status.get("seconds_left", 0) / 3600, 2)}
            # Stop watching a run that has finished: leaving it on the list
            # means polling Modal forever for something that will never change.
            if status.get("state") in ("succeeded", "failed", "stopped"):
                watchlist.pop(run_id, None)
                report[run_id]["unwatched"] = "run reached a terminal state"

    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(action: str = "status", run_id: str = ""):
    """Edit the watch list, or read it."""
    watchlist = modal.Dict.from_name(WATCHLIST_NAME, create_if_missing=True)

    if action == "watch":
        if not run_id:
            print("usage: modal run supervisor_app.py --action watch --run-id run-xxxx")
            return
        watchlist[run_id] = {"restarts": 0}
        print(f"watching {run_id}")
    elif action == "unwatch":
        watchlist.pop(run_id, None)
        print(f"no longer watching {run_id}")
    elif action == "tick":
        print(tick.remote())
    else:
        entries = {k: watchlist.get(k) for k in watchlist.keys()}
        print(f"{len(entries)} run(s) supervised:")
        for k, v in entries.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    # Convenience for the shell: `python supervisor_app.py watch run-xxxx`.
    args = sys.argv[1:]
    action = args[0] if args else "status"
    rid = args[1] if len(args) > 1 else ""
    wl = modal.Dict.from_name(WATCHLIST_NAME, create_if_missing=True)
    if action == "watch" and rid:
        wl[rid] = {"restarts": 0}
        print(f"watching {rid}")
    elif action == "unwatch" and rid:
        wl.pop(rid, None)
        print(f"no longer watching {rid}")
    else:
        print({k: wl.get(k) for k in wl.keys()})
