"""Snapshot each new export of a run that is training under the old code.

The retention fix lands in the trainer, but the live run started before it and
would need a restart to pick it up -- and a restart costs more than the thing
it protects. So retention runs from outside instead: poll progress.json, and
whenever the step advances, copy the export aside before the next save
overwrites it.

Also marks a checkpoint `keep.marker` once it has been measured, so the
retention window can never prune weights a result refers to.

    python watch_and_snapshot.py run-25ed75aa0a b0
"""

import json
import sys
import time

import modal

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "run-25ed75aa0a"
BRANCH = sys.argv[2] if len(sys.argv) > 2 else "b0"
POLL_S = 180

vol = modal.Volume.from_name("autobranch-data")
snapshot = modal.Function.from_name("sf3-ckpt-snapshot", "snapshot")


def read_json(path, default=None):
    try:
        return json.loads(b"".join(vol.read_file(path)).decode())
    except Exception:  # noqa: BLE001 - a transient volume read must not exit
        return default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    root = f"runs/{RUN_ID}/{BRANCH}"
    seen = set()
    # Anything already retained counts as seen, so a restart of this watcher
    # does not re-copy 16 GiB per existing checkpoint.
    try:
        for e in vol.listdir(f"{root}/checkpoints", recursive=False):
            name = e.path.rstrip("/").split("/")[-1]
            if name.startswith("step-"):
                seen.add(int(name.split("-")[-1]))
    except Exception:  # noqa: BLE001 - no checkpoints dir yet is normal
        pass
    log(f"already retained: {sorted(seen) or 'none'}")

    while True:
        progress = read_json(f"{root}/progress.json", {}) or {}
        step = progress.get("step")

        if isinstance(step, int) and step not in seen:
            log(f"new export at step {step:,} -- snapshotting")
            try:
                out = snapshot.remote(run_id=RUN_ID, branch_id=BRANCH)
                if out.get("ok"):
                    seen.add(out.get("step", step))
                    log(f"  retained {out.get('dest')} "
                        f"({out.get('gib', 'already present')} GiB)")
                else:
                    log(f"  FAILED: {out}")
            except Exception as exc:  # noqa: BLE001 - keep watching
                log(f"  snapshot error: {type(exc).__name__}: {exc}")

        # Mark every measured step so pruning can never take it. The eval file
        # is the record of what has actually been scored.
        for rec in read_json(f"{root}/independent_eval.json", []) or []:
            s = rec.get("step")
            if not isinstance(s, int) or not rec.get("ok"):
                continue
            # Only mark a checkpoint that actually holds weights. Marking a
            # step whose export is already gone would mint a directory
            # containing nothing but a marker -- something that reads like a
            # retained checkpoint and would make retain_checkpoint() skip the
            # real copy if training ever revisited that step. Steps measured
            # before retention existed are recorded as losses, not as
            # checkpoints; see LOST_CHECKPOINTS.json.
            ckpt = f"{root}/checkpoints/step-{s}"
            try:
                b"".join(vol.read_file(f"{ckpt}/model.safetensors.index.json"))
            except Exception:  # noqa: BLE001 - no weights, nothing to protect
                continue

            marker = f"{ckpt}/keep.marker"
            try:
                b"".join(vol.read_file(marker))
                continue  # already marked
            except Exception:  # noqa: BLE001 - absent is the normal case
                pass
            try:
                with vol.batch_upload(force=True) as up:
                    up.put_file(
                        io_bytes(f"{rec.get('objective', 'win_rate')}="
                                 f"{rec.get('value')} n={rec.get('n')} "
                                 f"seed={rec.get('seed')}\n"),
                        marker,
                    )
                log(f"  marked step {s:,} keep (measured {rec.get('value')})")
            except Exception as exc:  # noqa: BLE001
                log(f"  could not mark step {s}: {type(exc).__name__}: {exc}")

        time.sleep(POLL_S)


def io_bytes(text: str):
    import io

    return io.BytesIO(text.encode())


if __name__ == "__main__":
    main()
