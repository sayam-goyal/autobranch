"""A collection that truncates every match must say so.

`shaping_only` fires only when *no* round resolves anywhere, so a collection
in which every single match is cut off by `max_decisions_per_match` reads as
healthy as long as a few rounds happen to land. The SF3 reproduction run spent
11 iterations in exactly that state -- 4,800 decisions per iteration, which is
8 lanes x 600 exactly, so not one match in 88 ever finished -- and the metrics
said nothing. Training saw match prefixes while the objective scored whole
matches, which is how 73 wins in 76 resolved rounds sat beside a 0.542 win
rate.
"""

import numpy as np
import pytest

from autobranch.posttrain_full import FullHyper, collect_group


class StubEnv:
    """`lanes` concurrent matches that end after `ends_after` steps.

    `ends_after=None` means the lane never terminates, which is what a real
    match longer than the decision cap looks like from here.
    """

    def __init__(self, lanes=4, ends_after=None, round_at=None, how="terminated"):
        self.lanes = lanes
        self.ends_after = ends_after
        self.round_at = round_at
        self.how = how
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return {"text": ["obs"] * self.lanes,
                "image": [np.zeros((4, 4, 3), np.uint8)] * self.lanes}, {}

    def step(self, actions):
        from autobranch.env_api import StepResult

        self.t += 1
        done = np.zeros(self.lanes, bool)
        if self.ends_after is not None and self.t >= self.ends_after:
            done[:] = True
        won = np.zeros(self.lanes, np.float64)
        if self.round_at is not None and self.t == self.round_at:
            won[:] = 1.0
        ended = done if self.how == "terminated" else np.zeros(self.lanes, bool)
        cut = done if self.how == "truncated" else np.zeros(self.lanes, bool)
        return StepResult(
            obs={"text": ["obs"] * self.lanes,
                 "image": [np.zeros((4, 4, 3), np.uint8)] * self.lanes},
            terminated=ended,
            truncated=cut,
            signals={"round_won": won,
                     "round_lost": np.zeros(self.lanes),
                     "invalid_move": np.zeros(self.lanes),
                     "damage_dealt": np.zeros(self.lanes),
                     "damage_taken": np.zeros(self.lanes),
                     "decision_frames": np.ones(self.lanes)},
        )


class StubPolicy:
    def __init__(self, n_actions=4):
        self.n_actions = n_actions

    def logits(self, obs):
        n = len(obs["text"])
        return (np.zeros((n, self.n_actions), np.float64),
                list(obs["text"]), list(obs["image"]))


def _collect(env, cap):
    hyper = FullHyper(max_decisions_per_match=cap)
    return collect_group(
        env, StubPolicy(), lambda signals, progress=0.0: np.zeros(env.lanes),
        hyper, group=0, rng=np.random.default_rng(0), progress=0.0,
    )


def test_every_match_truncated_is_reported():
    env = StubEnv(lanes=4, ends_after=None, round_at=3)
    _, totals = _collect(env, cap=10)

    assert totals["lanes"] == 4
    assert totals["lanes_done"] == 0, "no lane terminated; cap cut them all off"
    # Rounds did resolve, so the shaping-only signal stays silent -- which is
    # precisely why a separate completion metric is needed.
    assert totals["rounds_won"] == 4
    assert totals["decisions"] == 4 * 10


def test_completed_matches_are_counted():
    env = StubEnv(lanes=4, ends_after=5, round_at=2)
    _, totals = _collect(env, cap=10)

    assert totals["lanes_done"] == 4
    assert totals["decisions"] == 4 * 5, "loop stops once every lane is done"


@pytest.mark.parametrize("cap,ends,expect_done", [(10, None, 0), (10, 4, 4), (3, 5, 0)])
def test_completion_tracks_the_cap_not_the_intent(cap, ends, expect_done):
    env = StubEnv(lanes=4, ends_after=ends)
    _, totals = _collect(env, cap=cap)
    assert totals["lanes_done"] == expect_done


def test_env_truncation_also_counts_as_ended():
    """`done` is terminated-or-truncated, so a lane the env cuts short is not
    a cap truncation and must not be reported as one."""
    env = StubEnv(lanes=4, ends_after=4, how="truncated")
    _, totals = _collect(env, cap=10)
    assert totals["lanes_done"] == 4
