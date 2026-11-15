"""Street Fighter III 3rd Strike as an autobranch environment: Ryu vs the CPU.

The emulator cannot run beside the policy. It needs MAME, the ROM, and a CPU;
the policy needs an 80 GB GPU and a 16 GB model. So an environment lane here is
a **Modal Sandbox running `emulator_server.py`**, and this module is the client
that drives it. `step()` is an HTTP round trip.

That has a consequence worth stating plainly, because it shapes every number
this environment produces: lanes are expensive and slow, not free. A lane costs
a sandbox and a ~22 s boot through the attract screens, character select and
super-art select, and a step costs a request plus the frames the macro action
consumes (~0.25 s measured). Matches therefore run concurrently, and a lane
that fails to boot is dropped rather than retried in place.

**One lane is one match.** `LocalSfiiiAdapter` raises on a second reset, so a
lane cannot be reused -- and reusing one is worse than it sounds: `/reset`
would hand back the cached first observation while the emulator sat hundreds of
decisions deep in a finished match, so the frames and the signals would
disagree with each other and with the game, silently. Every `reset()` therefore
replaces its lanes.

**The observation is a frame plus a short state line.** Qwen3-VL-8B-Instruct is
a vision-language model and Hinh fed it frames, so this does too: the
observation is the PNG the emulator drew, and the numeric state (health, meters,
timer) rides alongside as text because a model reading a health bar off pixels
is doing avoidable work. The policy sees both; the reward may be composed from
neither, only from declared signals.

**Nothing here knows what a reward is.** `step()` returns facts. The objective
below reads match outcomes only -- who won -- and never sees a reward config,
so `tests/test_objective_independence.py`'s import-graph walk holds.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

import numpy as np

from autobranch.env_api import StepResult

#: Ryu's macro-action menu, as the vendored table defines it. Imported rather
#: than restated: the manifest's `policy.n_actions` and the policy's logit mask
#: must agree with the emulator's table exactly, and two copies of a 40-entry
#: list is how that stops being true.
from sf_core.actions import ACTION_NAMES, N_ACTIONS  # noqa: E402

CHARACTER = "Ryu"
SUPER_ART = 1
DEFAULT_DIFFICULTY = int(os.environ.get("SF3_DIFFICULTY", "8"))

#: Where the emulator image and ROM come from. The ROM volume is the one the
#: predecessor project already verified by sha256; this never ships a ROM.
ROM_VOLUME = os.environ.get("SF3_ROM_VOLUME", "sf3-rom")
EMULATOR_APP = os.environ.get("SF3_EMU_APP", "autobranch-sf3-emulator")

BOOT_TIMEOUT_S = float(os.environ.get("SF3_BOOT_TIMEOUT_S", "420"))
STEP_TIMEOUT_S = float(os.environ.get("SF3_STEP_TIMEOUT_S", "120"))

SIGNAL_NAMES = (
    "damage_dealt",
    "damage_taken",
    "round_won",
    "round_lost",
    "invalid_move",
    "decision_frames",
    "super_gain",
    "opp_stun_gain",
    "health_frac",
    "opp_health_frac",
)

PROMPT = (
    "You are {character} in Street Fighter III 3rd Strike, round {round}.\n"
    "health {health:.2f} opponent {opp_health:.2f} super {super:.2f} "
    "opponent_stun {opp_stun:.2f} timer {timer:.2f} facing {facing}\n"
    "recent: {recent}\n"
    "Choose exactly one move."
)


def _post(url: str, body: dict, *, timeout: float) -> dict:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def _get(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def _decode_frame(frame_b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(frame_b64)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# one emulator sandbox
# ---------------------------------------------------------------------------


def emulator_image():
    """The image the emulator sandbox runs.

    Separate from the trainer's image on purpose, and this is the reason the
    bundle manifest's single `image` is not enough: the trainer needs torch and
    an 8B model, the emulator needs MAME and 2 CPUs. Putting MAME in the
    trainer image would add nothing to the GPU container and putting torch in
    the emulator image would make every lane pay a multi-GB pull.

    The package list and the three environment variables are copied from the
    predecessor project's working image rather than derived. That is deliberate:
    the first attempt here installed the SDL runtime libraries, which looks like
    the right answer and fails with `libGL.so.1: cannot open shared object
    file`, because MAME only avoids needing a GL context at all once
    `SDL_VIDEODRIVER=dummy` is set. Headless emulator images are a well-known
    tar pit and this one was already measured against the real ROM.
    """
    import modal

    bundle = os.path.dirname(os.path.abspath(__file__))
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libturbojpeg-dev")
        .env({
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            "XDG_RUNTIME_DIR": "/tmp",
        })
        .uv_pip_install("MAMEToolkit==1.1.0", "numpy==2.2.6", "pillow==11.3.0")
        .add_local_dir(bundle, "/bundle", copy=True)
    )


class EmulatorLane:
    """A sandbox running one match, addressed over a tunnel."""

    def __init__(self, *, difficulty: int, seed: int, image=None, app=None) -> None:
        self.difficulty = int(difficulty)
        self.seed = int(seed)
        self.sandbox = None
        self.url = None
        self.boot_s = None
        self._image = image
        self._app = app

    def start(self) -> None:
        import modal

        app = self._app or modal.App.lookup(EMULATOR_APP, create_if_missing=True)
        image = self._image or emulator_image()
        self.sandbox = modal.Sandbox.create(
            "python",
            "/bundle/emulator_server.py",
            image=image,
            app=app,
            cpu=2,
            memory=4096,
            # A branch killed outright never runs `close()`, so this timeout is
            # what reaps the lane. One match is ~8 minutes of wall clock and a
            # collection ~2.5, so 20 minutes is generous for the work and caps
            # what an orphan can cost. An hour, as this was, left nine lanes
            # billing after the first branch was replaced.
            timeout=20 * 60,
            encrypted_ports=[8000],
            volumes={"/rom": modal.Volume.from_name(ROM_VOLUME)},
            env={
                "SF3_PORT": "8000",
                "SF3_ROMS": "/rom",
                "SF3_DIFFICULTY": str(self.difficulty),
                "SF3_SEED": str(self.seed),
                "PYTHONPATH": "/bundle",
                "PYTHONUNBUFFERED": "1",
            },
        )
        tunnel = self.sandbox.tunnels()[8000]
        self.url = f"https://{tunnel.host}"

        # Boot is slow -- attract screens, character select, super-art select --
        # and polling /health is the only honest readiness signal. A fixed sleep
        # would either waste a minute per lane or start stepping a menu.
        deadline = time.time() + BOOT_TIMEOUT_S
        t0 = time.time()
        last = None
        while time.time() < deadline:
            try:
                health = _get(f"{self.url}/health", timeout=20)
                last = health
                if health.get("ready"):
                    self.boot_s = round(time.time() - t0, 1)
                    if int(health.get("n_actions", 0)) != N_ACTIONS:
                        raise RuntimeError(
                            f"emulator reports {health.get('n_actions')} actions but this "
                            f"bundle declares {N_ACTIONS}; the action tables have diverged"
                        )
                    return
                if health.get("error"):
                    raise RuntimeError(f"emulator failed to boot: {health['error']}")
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            if self.sandbox.poll() is not None:
                raise RuntimeError(
                    f"emulator sandbox exited during boot (code {self.sandbox.poll()})"
                )
            time.sleep(3)
        raise TimeoutError(f"emulator did not become ready in {BOOT_TIMEOUT_S}s (last={last})")

    def reset(self) -> dict:
        return _post(f"{self.url}/reset", {}, timeout=STEP_TIMEOUT_S)

    def step(self, action: int) -> dict:
        return _post(f"{self.url}/step", {"action": int(action)}, timeout=STEP_TIMEOUT_S)

    def terminate(self) -> None:
        if self.sandbox is not None:
            try:
                self.sandbox.terminate()
            except Exception:  # noqa: BLE001 - never mask a result with teardown
                pass
            self.sandbox = None


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------


class Sf3Env:
    """`n_envs` concurrent matches, each a sandbox, behind the step contract."""

    def __init__(self, *, n_envs: int = 1, difficulty: int = DEFAULT_DIFFICULTY,
                 seed: int = 0, recent: int = 8) -> None:
        self.n_envs = int(n_envs)
        self.difficulty = int(difficulty)
        self.seed = int(seed)
        self.recent_len = int(recent)
        self.action_names = ACTION_NAMES
        self.n_actions = N_ACTIONS
        self.lanes: list[EmulatorLane] = []
        self._frames: list[np.ndarray] = []
        self._state: list[dict] = []
        self._recent: list[list[str]] = []
        self._rounds: list[int] = []
        self._image = None
        self._app = None

    # ---- lifecycle -----------------------------------------------------

    def _ensure_lanes(self, *, fresh: bool = False) -> None:
        """Boot `n_envs` lanes, replacing any that exist when `fresh`.

        Lanes are **not** reusable across matches, and this is the emulator's
        constraint rather than a choice: `LocalSfiiiAdapter` raises on a second
        reset ("create a new environment"), so one sandbox is one match. Reusing
        a lane would have `/reset` hand back the cached first observation while
        the emulator sat 600 decisions deep in a finished match -- the frames
        and the signals would disagree with each other and with the game, and
        nothing would raise.

        The cost is a boot per collection, ~22 s measured, paid concurrently
        across lanes against a ~150 s collection. That is the right trade: the
        alternative is training on observations that do not describe the state
        the actions were taken in.
        """
        if self.lanes and not fresh:
            return
        for lane in self.lanes:
            lane.terminate()
        self.lanes = []
        import modal

        self._app = modal.App.lookup(EMULATOR_APP, create_if_missing=True)
        self._image = emulator_image()
        # Boot concurrently: a lane takes tens of seconds and booting eight of
        # them in series would dominate a short training run.
        from concurrent.futures import ThreadPoolExecutor

        lanes = [
            EmulatorLane(difficulty=self.difficulty, seed=self.seed + i,
                         image=self._image, app=self._app)
            for i in range(self.n_envs)
        ]
        with ThreadPoolExecutor(max_workers=max(1, self.n_envs)) as pool:
            errors = list(pool.map(self._start_one, lanes))
        alive = [lane for lane, err in zip(lanes, errors, strict=True) if err is None]
        if not alive:
            raise RuntimeError(f"no emulator lane booted: {errors[0]}")
        self.lanes = alive
        self.n_envs = len(alive)

    @staticmethod
    def _start_one(lane: EmulatorLane):
        try:
            lane.start()
            return None
        except Exception as exc:  # noqa: BLE001 - reported, not raised, so one
            lane.terminate()      # bad lane does not kill the whole collection
            return exc

    def close(self) -> None:
        for lane in self.lanes:
            lane.terminate()
        self.lanes = []

    # ---- the contract --------------------------------------------------

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        # Every reset is a new match, so every reset is a new set of lanes.
        self._ensure_lanes(fresh=True)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(self.lanes)) as pool:
            payloads = list(pool.map(lambda lane: lane.reset(), self.lanes))
        self._frames = [_decode_frame(p["frame_png_b64"]) for p in payloads]
        self._state = [p["state"] for p in payloads]
        self._recent = [[] for _ in payloads]
        self._rounds = [1 for _ in payloads]
        return self._obs(), {}

    def step(self, action) -> StepResult:
        actions = np.atleast_1d(np.asarray(action, dtype=np.int64)).reshape(-1)
        if actions.size != len(self.lanes):
            raise ValueError(f"expected {len(self.lanes)} actions, got {actions.size}")
        from concurrent.futures import ThreadPoolExecutor

        def one(pair):
            lane, act = pair
            try:
                return lane.step(int(act)), None
            except Exception as exc:  # noqa: BLE001
                return None, exc

        with ThreadPoolExecutor(max_workers=len(self.lanes)) as pool:
            results = list(pool.map(one, zip(self.lanes, actions, strict=True)))

        signals = {name: np.zeros(len(self.lanes), dtype=np.float64) for name in SIGNAL_NAMES}
        terminated = np.zeros(len(self.lanes), dtype=bool)
        truncated = np.zeros(len(self.lanes), dtype=bool)
        for i, (payload, exc) in enumerate(results):
            if exc is not None:
                # A dead lane is truncated, not terminated: the match did not
                # end, we lost the ability to observe it. Conflating the two
                # would teach the policy that crashing is an outcome.
                truncated[i] = True
                continue
            for name in SIGNAL_NAMES:
                signals[name][i] = float(payload["signals"].get(name, 0.0))
            terminated[i] = bool(payload["terminated"])
            self._frames[i] = _decode_frame(payload["frame_png_b64"])
            self._state[i] = payload["state"]
            self._recent[i] = (self._recent[i] + [ACTION_NAMES[int(actions[i])]])[-self.recent_len:]
            if payload["signals"].get("round_won") or payload["signals"].get("round_lost"):
                self._rounds[i] += 1

        return StepResult(
            obs=self._obs(),
            signals=signals,
            terminated=terminated,
            truncated=truncated,
            info={"action_names": [ACTION_NAMES[int(a)] for a in actions]},
        )

    # ---- observation ---------------------------------------------------

    def _obs(self) -> dict:
        """Frames plus the text the prompt needs.

        A dict rather than an array because this observation is genuinely two
        things. Flattening a 384x224x3 frame into the numeric vector the other
        environments use would hand a vision model its input as 258,048 comma
        separated integers, which is the single most wasteful thing this
        project could do with a context window.
        """
        return {
            "frames": list(self._frames),
            "text": [
                PROMPT.format(
                    character=CHARACTER,
                    round=self._rounds[i],
                    health=state["health"],
                    opp_health=state["opp_health"],
                    super=state["super"],
                    opp_stun=state["opp_stun"],
                    timer=state["timer"],
                    facing="right" if state["side"] == 0 else "left",
                    recent=", ".join(self._recent[i]) or "none",
                )
                for i, state in enumerate(self._state)
            ],
        }


def make_env(*, n_envs: int = 1, seed: int = 0, difficulty: int = DEFAULT_DIFFICULTY,
             **_ignored: Any) -> Sf3Env:
    return Sf3Env(n_envs=n_envs, difficulty=difficulty, seed=seed)


# ---------------------------------------------------------------------------
# the objective -- match outcomes only
# ---------------------------------------------------------------------------


def evaluate(
    policy,
    *,
    seeds: Sequence[int] = (12345,),
    budget: dict[str, Any] | None = None,
    deterministic: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """Win rate against the CPU, from match outcomes and nothing else.

    This is the only number that counts and the only one an agent may not
    touch. It reads who won each match -- never damage, never a reward term,
    never a training metric -- so there is no intermediate quantity here for a
    reward search to inflate instead of winning.

    The difficulty is fixed by the caller rather than sampled, unlike training:
    an opponent distribution that moves with the learner makes two checkpoints
    unmeasurable against each other, and the predecessor project established
    that the opponent draw inside arcade mode is already endogenous to the
    policy, which is as much unpinned variance as this metric can carry.
    """
    budget = dict(budget or {})
    n_matches = int(budget.get("matches", budget.get("episodes", 20)))
    difficulty = int(budget.get("difficulty", DEFAULT_DIFFICULTY))
    concurrency = int(budget.get("concurrency", min(8, n_matches)))

    wins = 0
    played = 0
    rounds_won = 0
    rounds_lost = 0
    invalid = 0
    decisions = 0
    failures: list[str] = []

    seed_bank = list(seeds) or [12345]
    env = Sf3Env(n_envs=concurrency, difficulty=difficulty, seed=seed_bank[0])
    try:
        remaining = n_matches
        batch = 0
        while remaining > 0:
            seed = seed_bank[batch % len(seed_bank)] + batch * 1009
            env.close()
            env = Sf3Env(n_envs=min(concurrency, remaining), difficulty=difficulty, seed=seed)
            obs, _ = env.reset()
            if hasattr(policy, "reseed"):
                policy.reseed(seed)
            if hasattr(policy, "deterministic"):
                policy.deterministic = bool(deterministic)

            lanes = len(env.lanes)
            live = np.ones(lanes, dtype=bool)
            tally = np.zeros((lanes, 2), dtype=np.int64)  # rounds won, lost
            steps = 0
            while live.any() and steps < 2000:
                action = policy(obs)
                result = env.step(np.atleast_1d(np.asarray(action)).reshape(-1))
                obs = result.obs
                steps += 1
                decisions += int(live.sum())
                invalid += int(np.asarray(result.signals["invalid_move"])[live].sum())
                tally[:, 0] += np.asarray(result.signals["round_won"], dtype=np.int64)
                tally[:, 1] += np.asarray(result.signals["round_lost"], dtype=np.int64)
                live &= ~result.done

            for i in range(lanes):
                played += 1
                rounds_won += int(tally[i, 0])
                rounds_lost += int(tally[i, 1])
                # A match is won by taking more rounds than the CPU. Ties and
                # unfinished matches count as not-won rather than being
                # dropped: dropping them would let a policy that hangs the
                # clock look better than one that loses honestly.
                if tally[i, 0] > tally[i, 1]:
                    wins += 1
            remaining -= lanes
            batch += 1
    finally:
        env.close()

    win_rate = wins / played if played else 0.0
    return {
        "value": win_rate,
        "win_rate": win_rate,
        "n": played,
        "detail": {
            "matches": played,
            "wins": wins,
            "difficulty": difficulty,
            "rounds_won": rounds_won,
            "rounds_lost": rounds_lost,
            "invalid_move_rate": (invalid / decisions) if decisions else 0.0,
            "decisions": decisions,
            "mode": "deterministic" if deterministic else "stochastic",
            "character": CHARACTER,
            "super_art": SUPER_ART,
            "failures": failures[:5],
        },
    }
