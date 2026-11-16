"""Gridworld: the reference environment, and the P1 gate.

A locked-door gridworld, vectorised over lanes with NumPy. It exists to be the
smallest environment where **reward design genuinely decides the outcome**, so
the agent tree has a real search to do rather than a formality to pass.

The geometry is chosen, not arbitrary:

* The goal is a **locked door**. Without the key it behaves like a wall, so
  `distance_to_goal` bottoms out at 1 and never reaches 0.
* The key sits **outside the bounding box of start and goal**. Every path that
  decreases `distance_to_goal` monotonically stays inside that box, so a
  reward built only from `distance_to_goal` walks straight to the door, parks
  next to it, and scores exactly zero forever. That is the plateau the search
  has to climb out of, and it is why this env discriminates between reward
  configs instead of rewarding any of them.
* **Lava kills**, and a slip probability means a policy that hugs lava dies
  sometimes. Avoiding it is a real constraint, not decoration.

The whole file reports **signals only**. There is no reward here and no way to
add one: `step` returns `StepResult`, which has no reward field. The objective
at the bottom reads episode outcomes (did this episode end on the goal) and
nothing else -- it never sees a reward, a weight, or a training step, which is
what `tests/test_objective_independence.py` enforces.

Layout for the default 8x8 (`.` free, `S` start, `K` key, `G` locked goal,
`X` lava):

    S . . . . . . .
    . . . . . . . .
    . . . . . . . .
    X X X X X . . .
    . . . . . . . .
    . . . . . G . .
    X X X X X . . .
    . . . . . . . K
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from autobranch.env_api import INFO_FINAL_OBS, StepResult

#: 0 up, 1 down, 2 left, 3 right. Movement only; the key is picked up by
#: stepping on it, so there is no "interact" action to get wrong.
N_ACTIONS = 4
_DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)

SIGNAL_NAMES = (
    "reached_goal",
    "stepped_in_lava",
    "picked_up_key",
    "distance_to_goal",
    "distance_to_key",
    "step_cost",
    "revisited_tile",
)


@dataclass(frozen=True)
class GridLayout:
    """Where everything is. Fixed by default so runs are comparable."""

    size: int = 8
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (5, 5)
    key: tuple[int, int] = (7, 7)
    lava: tuple[tuple[int, int], ...] = (
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
        (6, 0), (6, 1), (6, 2), (6, 3), (6, 4),
    )

    def validate(self) -> None:
        n = self.size
        for name, (r, c) in (("start", self.start), ("goal", self.goal), ("key", self.key)):
            if not (0 <= r < n and 0 <= c < n):
                raise ValueError(f"{name} {(r, c)} is outside a {n}x{n} grid")
        specials = {self.start, self.goal, self.key}
        if len(specials) != 3:
            raise ValueError("start, goal and key must be three distinct tiles")
        if specials & set(self.lava):
            raise ValueError("start, goal and key must not be lava")
        # The property the whole environment rests on: the key must lie outside
        # the start-goal bounding box, or a distance-to-goal reward would pick
        # it up by accident and the env would stop discriminating.
        r_lo, r_hi = sorted((self.start[0], self.goal[0]))
        c_lo, c_hi = sorted((self.start[1], self.goal[1]))
        if r_lo <= self.key[0] <= r_hi and c_lo <= self.key[1] <= c_hi:
            raise ValueError(
                f"key {self.key} lies inside the start-goal bounding box "
                f"rows {r_lo}..{r_hi} cols {c_lo}..{c_hi}; a monotone path to the goal would "
                "collect it for free and a distance-only reward would solve the task"
            )


class GridWorld:
    """Vectorised locked-door gridworld.

    Lanes auto-reset: a finished lane restarts inside the same `step`, and the
    pre-reset observation is published under `info["final_obs"]` so the trainer
    can bootstrap time-limited episodes correctly.
    """

    def __init__(
        self,
        *,
        n_envs: int = 256,
        max_steps: int = 100,
        slip_prob: float = 0.05,
        layout: GridLayout | None = None,
    ):
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if not 0.0 <= slip_prob < 1.0:
            raise ValueError(f"slip_prob must be in [0, 1), got {slip_prob}")
        self.layout = layout or GridLayout()
        self.layout.validate()
        self._n_envs = int(n_envs)
        self.max_steps = int(max_steps)
        self.slip_prob = float(slip_prob)

        n = self.layout.size
        self.n = n
        self._lava = np.zeros((n, n), dtype=bool)
        for r, c in self.layout.lava:
            self._lava[r, c] = True
        self._max_dist = float(2 * (n - 1))

        self._rng = np.random.default_rng(0)
        self._pos = np.zeros((self._n_envs, 2), dtype=np.int64)
        self._has_key = np.zeros(self._n_envs, dtype=bool)
        self._t = np.zeros(self._n_envs, dtype=np.int64)
        self._visited = np.zeros((self._n_envs, n * n), dtype=bool)

    # -- contract -------------------------------------------------------

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def obs_dim(self) -> int:
        """`n*n` position one-hot plus 17 scalar features. Declared in the manifest."""
        return self.n * self.n + 17

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._reset_lanes(np.ones(self._n_envs, dtype=bool))
        return self._observe()

    def step(self, action: Any) -> StepResult:
        act = np.asarray(action).reshape(self._n_envs).astype(np.int64)
        if np.any((act < 0) | (act >= N_ACTIONS)):
            bad = np.unique(act[(act < 0) | (act >= N_ACTIONS)])
            raise ValueError(f"actions {bad} outside 0..{N_ACTIONS - 1}")

        if self.slip_prob > 0.0:
            # A slip replaces the action with a different one, so the hazard is
            # real: a policy that walks along the lava edge dies sometimes.
            slip = self._rng.random(self._n_envs) < self.slip_prob
            if slip.any():
                offset = self._rng.integers(1, N_ACTIONS, size=int(slip.sum()))
                act = act.copy()
                act[slip] = (act[slip] + offset) % N_ACTIONS

        target = self._pos + _DELTAS[act]
        np.clip(target, 0, self.n - 1, out=target)  # the border is a wall: stay put

        on_key = (
            (target[:, 0] == self.layout.key[0])
            & (target[:, 1] == self.layout.key[1])
            & ~self._has_key
        )
        on_goal_tile = (target[:, 0] == self.layout.goal[0]) & (target[:, 1] == self.layout.goal[1])
        locked = on_goal_tile & ~self._has_key
        target = np.where(locked[:, None], self._pos, target)  # the door does not open

        self._pos = target
        self._has_key |= on_key
        self._t += 1

        flat = self._pos[:, 0] * self.n + self._pos[:, 1]
        revisited = self._visited[np.arange(self._n_envs), flat]
        self._visited[np.arange(self._n_envs), flat] = True

        in_lava = self._lava[self._pos[:, 0], self._pos[:, 1]]
        reached_goal = on_goal_tile & self._has_key & ~locked

        signals = {
            "reached_goal": reached_goal.astype(np.float32),
            "stepped_in_lava": in_lava.astype(np.float32),
            "picked_up_key": on_key.astype(np.float32),
            "distance_to_goal": self._distance_to_goal(),
            "distance_to_key": self._distance_to_key(),
            "step_cost": np.ones(self._n_envs, dtype=np.float32),
            "revisited_tile": revisited.astype(np.float32),
        }

        terminated = reached_goal | in_lava
        truncated = (~terminated) & (self._t >= self.max_steps)
        done = terminated | truncated

        final_obs = self._observe()
        if done.any():
            self._reset_lanes(done)
        obs = self._observe()
        # `final_obs` carries the pre-reset observation for finished lanes; the
        # entries for live lanes are simply the current observation.
        return StepResult(
            obs=obs,
            signals=signals,
            terminated=terminated,
            truncated=truncated,
            info={INFO_FINAL_OBS: final_obs, "has_key": self._has_key.copy()},
        )

    def close(self) -> None:
        """Nothing to release; present because the contract requires it."""

    # -- internals ------------------------------------------------------

    def _reset_lanes(self, mask: np.ndarray) -> None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        self._pos[idx] = np.asarray(self.layout.start, dtype=np.int64)
        self._has_key[idx] = False
        self._t[idx] = 0
        self._visited[idx] = False
        self._visited[idx, self.layout.start[0] * self.n + self.layout.start[1]] = True

    def _distance_to_goal(self) -> np.ndarray:
        goal = np.asarray(self.layout.goal, dtype=np.int64)
        d = np.abs(self._pos - goal).sum(axis=1)
        return (d / self._max_dist).astype(np.float32)

    def _distance_to_key(self) -> np.ndarray:
        """Zero once the key is held, so the signal means 'distance to what I still need'."""
        key = np.asarray(self.layout.key, dtype=np.int64)
        d = np.abs(self._pos - key).sum(axis=1)
        out = (d / self._max_dist).astype(np.float32)
        out[self._has_key] = 0.0
        return out

    def _observe(self) -> np.ndarray:
        n, n_envs = self.n, self._n_envs
        flat = self._pos[:, 0] * n + self._pos[:, 1]
        one_hot = np.zeros((n_envs, n * n), dtype=np.float32)
        one_hot[np.arange(n_envs), flat] = 1.0

        goal = np.asarray(self.layout.goal, dtype=np.float32)
        key = np.asarray(self.layout.key, dtype=np.float32)
        pos = self._pos.astype(np.float32)
        scale = float(n - 1)

        rel_goal = (goal - pos) / scale
        rel_key = (key - pos) / scale
        rel_key[self._has_key] = 0.0

        features = [
            one_hot,
            self._has_key.astype(np.float32)[:, None],
            rel_key,
            rel_goal,
            self._distance_to_key()[:, None],
            self._distance_to_goal()[:, None],
            self._neighbourhood(),
            (1.0 - self._t / self.max_steps).astype(np.float32)[:, None],
        ]
        return np.concatenate(features, axis=1).astype(np.float32)

    def _neighbourhood(self) -> np.ndarray:
        """3x3 'this tile would hurt or block me' patch, border included.

        Without it the policy has to memorise the lava map from the position
        one-hot alone, which it can -- but a local hazard view is what a real
        environment would expose, and it keeps the task learnable at larger
        grid sizes.
        """
        n, n_envs = self.n, self._n_envs
        padded = np.ones((n + 2, n + 2), dtype=np.float32)  # outside the grid = blocked
        padded[1:-1, 1:-1] = self._lava.astype(np.float32)
        rows = self._pos[:, 0][:, None] + np.arange(3)[None, :]
        cols = self._pos[:, 1][:, None] + np.arange(3)[None, :]
        patch = padded[rows[:, :, None], cols[:, None, :]]
        return patch.reshape(n_envs, 9)


def make_env(**kwargs: Any) -> GridWorld:
    """Manifest entrypoint (`grid_env:make_env`)."""
    return GridWorld(**kwargs)


# ---------------------------------------------------------------------------
# the objective: success_rate, and nothing a reward can reach
# ---------------------------------------------------------------------------


def evaluate(
    policy,
    *,
    seeds: Sequence[int] = (12345,),
    budget: dict[str, Any] | None = None,
    deterministic: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """Fraction of episodes that end on the goal -- which requires the key.

    Reads episode *outcomes* only: whether the episode terminated on the goal,
    in lava, or on the clock. It is handed a callable policy and has no access
    to weights, rewards, or training state, so there is nothing here for a
    reward search to exploit. Coupling is structural rather than checked: the
    goal cannot be entered without the key, so "reached the goal" already
    means "reached the goal with the key" -- `key_rate` is reported separately
    as diagnostics.

    Episodes are split evenly across the caller's fixed seed bank, so the
    number is comparable between branches.
    """
    budget = dict(budget or {})
    episodes = max(1, int(budget.get("episodes", 256)))
    lanes = max(1, int(budget.get("lanes", 32)))
    max_steps = int(budget.get("max_steps", 100))
    slip_prob = float(budget.get("slip_prob", 0.05))
    seeds = tuple(int(s) for s in (seeds or (12345,)))
    per_seed = max(1, episodes // len(seeds))

    outcomes = {"goal": 0, "lava": 0, "timeout": 0}
    keys = 0
    lengths: list[int] = []
    counted = 0

    for seed in seeds:
        # At least 2 lanes: a size-1 leading dimension has tripped contiguity
        # checks in native libraries.
        env = GridWorld(
            n_envs=max(2, min(lanes, per_seed)), max_steps=max_steps, slip_prob=slip_prob
        )
        obs = env.reset(seed=seed)
        if hasattr(policy, "reseed"):
            policy.reseed(seed)
        got_key = np.zeros(env.n_envs, dtype=bool)
        steps = np.zeros(env.n_envs, dtype=np.int64)
        taken = 0
        # Hard cap: enough steps for every lane to finish its share of
        # episodes, plus slack. A policy that never terminates must still
        # produce a score rather than hang the evaluation.
        budget_steps = (per_seed // env.n_envs + 2) * (max_steps + 1)
        for _ in range(budget_steps):
            if taken >= per_seed:
                break
            result = env.step(policy(obs))
            obs = result.obs
            got_key |= np.asarray(result.signals["picked_up_key"], dtype=bool)
            steps += 1
            done = np.asarray(result.terminated, dtype=bool) | np.asarray(
                result.truncated, dtype=bool
            )
            for lane in np.flatnonzero(done):
                if taken >= per_seed:
                    break
                if result.signals["reached_goal"][lane]:
                    outcomes["goal"] += 1
                elif result.signals["stepped_in_lava"][lane]:
                    outcomes["lava"] += 1
                else:
                    outcomes["timeout"] += 1
                keys += int(got_key[lane])
                lengths.append(int(steps[lane]))
                taken += 1
                counted += 1
            got_key[done] = False
            steps[done] = 0
        env.close()

    n = max(1, counted)
    return {
        "value": outcomes["goal"] / n,
        "n": counted,
        "detail": {
            "success_rate": outcomes["goal"] / n,
            "lava_rate": outcomes["lava"] / n,
            "timeout_rate": outcomes["timeout"] / n,
            "key_rate": keys / n,
            "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "episodes_per_seed": per_seed,
            "seeds": list(seeds),
            "deterministic": bool(deterministic),
        },
    }
