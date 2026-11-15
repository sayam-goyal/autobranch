"""Probe: does one SF3 emulator lane boot in a Modal Sandbox and step?

Run locally -- it creates the sandbox remotely:

    python probe_sf3_env.py

This is the second of the two unknowns the SF3 plan rests on. The first was
whether an 8.77B VL model takes a full-parameter update on one H100
(`probe_full_ft.py`: yes, 49.5 GiB of 79.2, 2.93 s per step). This one asks
whether a lane is cheap enough to be a training environment at all, which is a
question about *boot* more than about stepping: the emulator has to walk the
attract screens, character select and super-art select before the first frame
of a match exists, and if that takes minutes then a reward search that resets
lanes constantly is dead on arrival regardless of how fast the GPU is.

What it reports, in the order it matters: boot seconds, whether the action
tables on both sides agree, seconds per step, the frame's shape, and whether
the declared signals actually arrive.
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "sf3"))

import sf3_env  # noqa: E402


def main(steps: int = 12, difficulty: int = 8) -> int:
    print(f"booting one lane (difficulty={difficulty})...", flush=True)
    lane = sf3_env.EmulatorLane(difficulty=difficulty, seed=0)
    t0 = time.time()
    try:
        lane.start()
    except Exception as exc:
        print(f"FAILED to boot: {type(exc).__name__}: {exc}")
        if lane.sandbox is not None:
            try:
                print("--- sandbox stdout ---")
                print(lane.sandbox.stdout.read()[-4000:])
                print("--- sandbox stderr ---")
                print(lane.sandbox.stderr.read()[-4000:])
            except Exception:  # noqa: BLE001
                pass
        lane.terminate()
        return 1
    print(f"booted in {lane.boot_s}s (wall {time.time() - t0:.1f}s) at {lane.url}")

    try:
        payload = lane.reset()
        frame = sf3_env._decode_frame(payload["frame_png_b64"])
        print(f"reset: frame {frame.shape} dtype={frame.dtype} state={payload['state']}")

        missing = set(sf3_env.SIGNAL_NAMES) - set(payload["signals"])
        if missing:
            print(f"WARNING: reset is missing declared signals: {sorted(missing)}")

        durations = []
        totals = {name: 0.0 for name in sf3_env.SIGNAL_NAMES}
        for i in range(steps):
            # Cycle a few genuinely different actions: Wait, Move Closer, a
            # poke, and a Hadouken, so "it stepped" is not just the no-op path.
            action = [0, 1, 15, 32, 3, 18][i % 6]
            t = time.time()
            out = lane.step(action)
            durations.append(time.time() - t)
            for name in sf3_env.SIGNAL_NAMES:
                totals[name] += float(out["signals"].get(name, 0.0))
            if out["terminated"]:
                print(f"terminated after {i + 1} steps")
                break

        print(f"\nstep seconds: mean {statistics.fmean(durations):.2f} "
              f"min {min(durations):.2f} max {max(durations):.2f} (n={len(durations)})")
        print("signal totals over the probe:")
        for name, total in totals.items():
            print(f"  {name:<18} {total:.4f}")
        frames_total = totals["decision_frames"]
        print(f"\ngame frames consumed: {frames_total:.0f} "
              f"({frames_total / 60:.1f}s of game time in "
              f"{sum(durations):.1f}s of wall clock)")
        print("\nPROBE OK")
        return 0
    finally:
        lane.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
