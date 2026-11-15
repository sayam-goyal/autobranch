"""Watch the SF3 reward search, and read back what it recorded.

    python examples/sf3/watch.py --run-id run-xxxx            # one snapshot
    python examples/sf3/watch.py --run-id run-xxxx --loop 120 # every 2 minutes
    python examples/sf3/watch.py --run-id run-xxxx --metrics  # per-branch curves

Reads the registry Dicts and the Volume directly from the client, so it needs
no container and cannot perturb the thing it is measuring. That matters more
than convenience here: the predecessor project lost monitors repeatedly and
learned to poll real state rather than trust a completion event, because every
silent failure it hit was caught by comparing counters that disagreed.

Two sources, deliberately:

* the **registry** holds the tree -- which branches exist, their reward
  configs, their objective scores, who abandoned and why;
* the **Volume** holds `runs/<run>/<branch>/metrics.jsonl`, written once per
  training iteration and surviving the sandbox that wrote it.

The second is what makes post-hoc analysis possible at all, and the reason the
trainer writes there instead of to `/workspace`.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DATA_VOLUME = os.environ.get("AUTOBRANCH_SF3_VOLUME", "autobranch-data")


def _fmt_reward(config: dict) -> str:
    weights = (config or {}).get("weights") or {}
    terms = " ".join(f"{k}={v:g}" for k, v in sorted(weights.items()))
    shaping = (config or {}).get("shaping") or []
    if shaping:
        terms += " +shaping[" + ",".join(
            f"{s.get('kind')}:{s.get('signal')}" for s in shaping
        ) + "]"
    if (config or {}).get("anneal"):
        terms += " +anneal"
    return terms or "(none)"


def snapshot(run_id: str) -> dict:
    from autobranch.registry import Registry

    registry = Registry()
    run = registry.get_run(run_id)
    branches = registry.branches(run_id)
    return {"run": run, "branches": branches}


def show(run_id: str) -> None:
    from autobranch.consolidate import rank_branches  # noqa: F401 - parity with the artifact

    data = snapshot(run_id)
    run = data["run"]
    branches = data["branches"]
    if run is None:
        print(f"no run {run_id}")
        return

    elapsed = getattr(run, "elapsed_s", None)
    print(f"{run_id}  {getattr(run, 'state', '?')}  "
          f"best={getattr(run, 'best_score', None)}  "
          f"spend=${getattr(run, 'spend_usd', 0.0):.2f}  "
          f"elapsed={(elapsed or 0) / 3600:.2f}h  "
          f"branches={len(branches)}")

    for b in sorted(branches, key=lambda x: str(getattr(x, "branch_id", ""))):
        bid = getattr(b, "branch_id", "?")
        status = getattr(b, "status", "?")
        best = getattr(b, "best_score", None)
        steps = getattr(b, "steps_done", 0)
        spend = getattr(b, "spend_usd", 0.0)
        mark = {"succeeded": "*", "running": ">", "abandoned": "x"}.get(status, "?")
        print(f"  {mark} {bid:<12} d{getattr(b, 'depth', 0)} {status:<10} "
              f"best={best if best is None else round(best, 4)!s:<8} "
              f"steps={steps:<9} ${spend:.2f}")
        print(f"      reward: {_fmt_reward(getattr(b, 'reward_config', {}) or {})}")
        reason = getattr(b, "abandoned_reason", None)
        if reason:
            print(f"      stopped: {reason}")
        activity = getattr(b, "activity", None)
        if activity and status == "running":
            print(f"      doing: {activity}")


def metrics(run_id: str, *, tail: int = 6) -> None:
    """Per-branch training curves, read off the Volume."""
    import modal

    volume = modal.Volume.from_name(DATA_VOLUME)
    prefix = f"runs/{run_id}"
    try:
        entries = list(volume.listdir(prefix, recursive=False))
    except Exception as exc:  # noqa: BLE001 - a run with no iterations yet
        print(f"no metrics on the volume yet at {prefix}: {type(exc).__name__}: {exc}")
        return

    for entry in sorted(entries, key=lambda e: e.path):
        branch = entry.path.rstrip("/").rsplit("/", 1)[-1]
        path = f"{prefix}/{branch}/metrics.jsonl"
        try:
            raw = b"".join(volume.read_file(path)).decode()
        except Exception:  # noqa: BLE001 - branch exists but has not logged yet
            continue
        lines = [json.loads(l) for l in raw.splitlines() if l.strip()]
        if not lines:
            continue
        print(f"\n{branch}: {len(lines)} iterations")
        print(f"  {'iter':>5} {'step':>8} {'ret_mean':>9} {'ratio':>6} {'clip':>6} "
              f"{'grad':>8} {'ent':>6} {'inval':>6} {'rounds':>8}")
        for line in lines[-tail:]:
            print(f"  {line.get('iteration', 0):>5} {line.get('step', 0):>8} "
                  f"{line.get('return_mean', 0):>9.4f} "
                  f"{line.get('ratio_mean', 0):>6.3f} "
                  f"{line.get('clip_frac', 0):>6.3f} "
                  f"{line.get('grad_norm', 0):>8.1f} "
                  f"{line.get('entropy', 0):>6.3f} "
                  f"{line.get('invalid_move_rate', 0):>6.3f} "
                  f"{line.get('rounds_won', 0)}/{line.get('rounds_lost', 0):<5}")
        # The invariant that caught right-padding. Printed every time because a
        # wrong gradient produces a complete, plausible training curve.
        drift = max(abs(l.get("ratio_mean", 1.0) - 1.0) for l in lines)
        if drift > 0.05:
            print(f"  !! max first-epoch ratio drift {drift:.3f} -- collection and update "
                  f"are not scoring the same thing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--loop", type=float, default=0.0, help="seconds between snapshots")
    parser.add_argument("--metrics", action="store_true", help="also read the volume curves")
    parser.add_argument("--tail", type=int, default=6)
    args = parser.parse_args()

    while True:
        print("=" * 78)
        print(time.strftime("%H:%M:%S %Z"))
        try:
            show(args.run_id)
            if args.metrics:
                metrics(args.run_id, tail=args.tail)
        except Exception as exc:  # noqa: BLE001 - a monitor must not die on a blip
            print(f"snapshot failed: {type(exc).__name__}: {exc}")
        if not args.loop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
