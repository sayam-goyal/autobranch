"""The SF3 emulator, behind a request/response HTTP API.

This runs inside a Modal Sandbox that holds MAME and the ROM. It exists to
invert the control flow of the predecessor project's harness, which had the
emulator *call out* to a policy server for each action. That shape is fine when
the emulator drives the match, but autobranch's environment contract is
`reset()` / `step(action)` -- the trainer drives. Keeping the old direction
would have meant the trainer could not be the thing that decides when to step,
so the whole reward search would have had to be rebuilt around the emulator's
loop instead.

So: one process, one match, two endpoints.

    POST /reset            -> first observation
    POST /step {action}    -> next observation, signals, terminated
    GET  /health           -> ready and alive

**Signals, never rewards.** Every number this returns is a fact about the
transition -- damage dealt, damage taken, whether the round ended and who won,
whether the macro action was rejected as unavailable, how many frames it
consumed. It has no idea what any of them are worth. That is the framework's
spine: an agent composes a reward from these and cannot touch the measurement
that judges it.

One match per process on purpose. The adapter supports exactly one reset
(`LocalSfiiiAdapter` raises on a second), and a fresh emulator per match is
also the only way to be sure a match is not inheriting state from the last
one -- which matters most for the evaluator, whose whole value is being
trustworthy.
"""

import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sf_core.actions import ACTION_NAMES, N_ACTIONS, action_frames  # noqa: E402
from sf_core.env import EnvironmentConfig, create_environment  # noqa: E402

HEALTH_MAX = 160.0
SUPER_BAR_MAX = 128.0
STUN_BAR_MAX = 72.0
TIMER_MAX = 100.0

PORT = int(os.environ.get("SF3_PORT", "8000"))
ROMS_PATH = os.environ.get("SF3_ROMS", "/rom")
DIFFICULTY = int(os.environ.get("SF3_DIFFICULTY", "8"))
SEED = int(os.environ.get("SF3_SEED", "0"))
#: Matches the predecessor harness. A match that has not ended by here is
#: hung, and a hung emulator holding a sandbox is the expensive failure.
MAX_STEPS = int(os.environ.get("SF3_MAX_STEPS", "2000"))


def _clamp_health(value: float) -> float:
    """Health reads below zero for a frame or two around a KO."""
    return max(0.0, min(HEALTH_MAX, float(value)))


def _scalar(value) -> float:
    if hasattr(value, "reshape"):
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _frame_png(frame) -> str:
    """The frame as a base64 PNG, which is what a VL processor wants.

    PNG rather than raw: the frame crosses an HTTP boundary every step, and
    SF3's flat pixel art compresses by roughly an order of magnitude. Lossless
    because the policy is reading a HUD -- health bars, meters and a timer --
    and JPEG ringing on a 384x224 frame lands exactly on those thin elements.
    """
    from PIL import Image

    import numpy as np

    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False, compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Match:
    """One emulator, one match, stepped on request."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.env = None
        self.closed = False
        self.steps = 0
        self.side = 0
        self.prev = {}
        self.wins = [0, 0]
        self.error = None
        self.booted_at = None

    # ---- lifecycle -----------------------------------------------------

    def boot(self) -> None:
        config = EnvironmentConfig(
            characters=("Ryu", "Ryu"),
            outfits=(1, 1),
            super_arts=(1, 1),
            step_ratio=1,
            roms_path=ROMS_PATH,
            vs_cpu=True,
            cpu_difficulty=DIFFICULTY,
            lock_cpu_opponent=False,
        )
        t0 = time.time()
        self.env = create_environment(config)
        observation, _ = self.env.reset()
        self.booted_at = round(time.time() - t0, 1)
        self._remember(observation)
        return observation

    def close(self) -> None:
        if self.env is not None and not self.closed:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001 - teardown must not mask a result
                pass
        self.closed = True

    # ---- observation ---------------------------------------------------

    def _remember(self, observation) -> None:
        p1, p2 = observation["P1"], observation["P2"]
        self.side = int(_scalar(p1["side"]))
        self.prev = {
            "h1": _clamp_health(_scalar(p1["health"])),
            "h2": _clamp_health(_scalar(p2["health"])),
            "super1": _scalar(p1["super_bar"]),
            "stun2": _scalar(p2["stun_bar"]),
            "wins1": _scalar(p1.get("wins", 0)),
            "wins2": _scalar(p2.get("wins", 0)),
        }

    def _payload(self, observation, *, signals, terminated) -> dict:
        p1, p2 = observation["P1"], observation["P2"]
        return {
            "frame_png_b64": _frame_png(observation["frame"]),
            "signals": signals,
            "terminated": bool(terminated),
            "steps": self.steps,
            # State the prompt needs, as plain numbers. Not signals: the
            # reward may not be composed from these, they are what the policy
            # is allowed to *see*.
            "state": {
                "health": round(_clamp_health(_scalar(p1["health"])) / HEALTH_MAX, 4),
                "opp_health": round(_clamp_health(_scalar(p2["health"])) / HEALTH_MAX, 4),
                "super": round(_scalar(p1["super_bar"]) / SUPER_BAR_MAX, 4),
                "opp_stun": round(_scalar(p2["stun_bar"]) / STUN_BAR_MAX, 4),
                "timer": round(_scalar(observation.get("timer", 0)) / TIMER_MAX, 4),
                "side": self.side,
                "wins": [int(self.prev["wins1"]), int(self.prev["wins2"])],
            },
        }

    # ---- stepping ------------------------------------------------------

    def step(self, action: int) -> dict:
        if not (0 <= action < N_ACTIONS):
            raise ValueError(f"action {action} outside [0, {N_ACTIONS})")

        frames = action_frames(action, self.side)
        # An action the character cannot perform right now consumes no frames.
        # That is a fact worth reporting -- Hinh pays for it with an explicit
        # penalty term -- so it becomes a signal rather than being silently
        # swallowed or silently retried.
        invalid = 1.0 if not frames else 0.0
        if invalid:
            frames = action_frames(0, self.side)  # Wait, so the match advances

        observation, _net_damage, terminated, _trunc, info = self.env.step_frames(
            [{"agent_0": mask} for mask in frames]
        )
        self.steps += 1
        elapsed = int(info.get("frames", len(frames)))

        p1, p2 = observation["P1"], observation["P2"]
        h1 = _clamp_health(_scalar(p1["health"]))
        h2 = _clamp_health(_scalar(p2["health"]))
        wins1 = _scalar(p1.get("wins", 0))
        wins2 = _scalar(p2.get("wins", 0))

        # Damage is a *difference*, so it is computed here where both ends of
        # the difference are known, rather than being left to the trainer to
        # reconstruct from a pair of states it might sample out of order.
        dealt = max(0.0, self.prev["h2"] - h2)
        taken = max(0.0, self.prev["h1"] - h1)
        round_won = 1.0 if wins1 > self.prev["wins1"] else 0.0
        round_lost = 1.0 if wins2 > self.prev["wins2"] else 0.0

        signals = {
            "damage_dealt": round(dealt / HEALTH_MAX, 6),
            "damage_taken": round(taken / HEALTH_MAX, 6),
            "round_won": round_won,
            "round_lost": round_lost,
            "invalid_move": invalid,
            "decision_frames": float(elapsed),
            "super_gain": round(
                max(0.0, _scalar(p1["super_bar"]) - self.prev["super1"]) / SUPER_BAR_MAX, 6
            ),
            "opp_stun_gain": round(
                max(0.0, _scalar(p2["stun_bar"]) - self.prev["stun2"]) / STUN_BAR_MAX, 6
            ),
            "health_frac": round(h1 / HEALTH_MAX, 6),
            "opp_health_frac": round(h2 / HEALTH_MAX, 6),
        }

        self._remember(observation)
        self.wins = [int(wins1), int(wins2)]
        if self.steps >= MAX_STEPS:
            terminated = True
        return self._payload(observation, signals=signals, terminated=terminated)


MATCH = Match()
FIRST = {"observation": None}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: A003 - silence per-request logging
        pass

    def _send(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._send(200, {
                "ready": MATCH.env is not None,
                "closed": MATCH.closed,
                "steps": MATCH.steps,
                "boot_s": MATCH.booted_at,
                "n_actions": N_ACTIONS,
                "actions": list(ACTION_NAMES),
                "difficulty": DIFFICULTY,
                "error": MATCH.error,
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "body is not JSON"})
            return

        try:
            if self.path == "/reset":
                with MATCH.lock:
                    observation = FIRST["observation"]
                    if observation is None:
                        raise RuntimeError("emulator did not boot")
                    self._send(200, MATCH._payload(
                        observation,
                        signals={k: 0.0 for k in (
                            "damage_dealt", "damage_taken", "round_won", "round_lost",
                            "invalid_move", "decision_frames", "super_gain",
                            "opp_stun_gain",
                        )} | {
                            "health_frac": MATCH.prev["h1"] / HEALTH_MAX,
                            "opp_health_frac": MATCH.prev["h2"] / HEALTH_MAX,
                        },
                        terminated=False,
                    ))
            elif self.path == "/step":
                with MATCH.lock:
                    self._send(200, MATCH.step(int(body["action"])))
            elif self.path == "/close":
                MATCH.close()
                self._send(200, {"closed": True, "wins": MATCH.wins})
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 - the client needs the reason
            MATCH.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self._send(500, {"error": MATCH.error})


def main() -> None:
    import random

    random.seed(SEED)
    print(f"sf3 emulator: booting (difficulty={DIFFICULTY}, seed={SEED})", flush=True)
    try:
        FIRST["observation"] = MATCH.boot()
        print(f"sf3 emulator: booted in {MATCH.booted_at}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        MATCH.error = f"boot failed: {type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"sf3 emulator: {MATCH.error}", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"sf3 emulator: serving on :{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
