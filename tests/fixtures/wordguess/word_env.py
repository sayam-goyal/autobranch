"""Wordguess: a tiny text environment where an LM policy is the natural fit.

A hidden four-letter word is revealed one letter at a time. Each action names
one letter of a ten-letter alphabet, so the whole action space is a short list
of symbols -- which is exactly the shape a language-model policy can be
restricted to (see `autobranch.posttrain`: one token per action, logits masked
to the action vocabulary, so an illegal move cannot be emitted and there is
never an "invalid move" penalty to tune).

It is deliberately built so **reward design decides the outcome**, the same
property that makes the gridworld bundle worth training on:

* `letter_correct` fires whenever the guessed letter occurs in the word --
  **including on a repeat of a letter already revealed**. It reports the guess,
  not the progress. A reward built from it alone pays full price forever for
  re-guessing one good letter, and such a policy never solves a word.
* `letter_new` is the honest progress signal: it only fires when a guess
  reveals a position that was hidden. A reward that uses it (or that penalises
  `repeat_guess`) climbs out of that trap.

So two reward configs that both look sensible in a diff produce a solve rate
of 1.0 and of 0.0, and the objective -- which reads outcomes only -- is what
tells them apart. `tests/test_posttrain.py` asserts exactly that inversion.

Everything here reports **signals**, never a reward: `step` returns
`StepResult`, which has no reward field. The objective at the bottom looks at
episode outcomes (did this episode reveal the whole word, how many guesses did
it take) and nothing else. This module imports no reward code, no trainer and
no post-training code, so the measurement path cannot reach any of them.

Observation layout (26 floats, all in [0, 1]), which is also the only thing a
policy is allowed to see:

    [0 : 10]    guessed[k]        1 once letter k has been guessed
    [10 : 20]   present[k]        1 once letter k has been guessed AND it is
                                  in the word (so it is knowledge, not a peek)
    [20 : 24]   revealed[pos]     1 once position `pos` is revealed
    [24]        revealed_fraction fraction of positions revealed
    [25]        budget_left       fraction of the step budget remaining
"""

import copy
from typing import Any, Sequence

import numpy as np

from autobranch.env_api import INFO_FINAL_OBS, StepResult

#: The restricted alphabet. Ten letters, one action each; every word in the
#: bank is spelled from these and nothing else.
ALPHABET = "aceilnorst"

#: One action per letter. `policy.n_actions` in the manifest must match, and
#: `posttrain.build_action_vocab` turns this count into one token per action.
N_ACTIONS = len(ALPHABET)

WORD_LENGTH = 4

#: Every word is four *distinct* letters over `ALPHABET`, so solving always
#: needs four different correct guesses -- a policy that latches onto one good
#: letter cannot stumble into a win.
WORD_BANK = (
    "lane", "cost", "race", "tile", "coal", "neat", "salt", "cane",
    "lace", "rice", "nice", "iron", "star", "slot", "lion", "corn",
)

SIGNAL_NAMES = (
    "letter_correct",
    "letter_new",
    "letter_wrong",
    "repeat_guess",
    "word_solved",
    "guess_cost",
    "revealed_fraction",
)

#: Slices into the observation vector. Exposed so evaluators, demo policies
#: and tests can read an observation without re-deriving the layout from the
#: docstring above -- a layout that is documented in two places drifts.
OBS_GUESSED = slice(0, N_ACTIONS)
OBS_PRESENT = slice(N_ACTIONS, 2 * N_ACTIONS)
OBS_REVEALED = slice(2 * N_ACTIONS, 2 * N_ACTIONS + WORD_LENGTH)
OBS_REVEALED_FRACTION = 2 * N_ACTIONS + WORD_LENGTH
OBS_BUDGET_LEFT = OBS_REVEALED_FRACTION + 1
OBS_DIM = OBS_BUDGET_LEFT + 1


def encode_words(words: Sequence[str]) -> np.ndarray:
    """Encode words as letter indices, failing loudly on a bad one.

    The default bank is checked at import rather than at first reset: a word
    with a letter outside the alphabet, or with a repeated letter, silently
    changes the difficulty of every episode it appears in. A caller-supplied
    bank goes through the same gate for the same reason.
    """
    index = {ch: i for i, ch in enumerate(ALPHABET)}
    rows = []
    if not words:
        raise ValueError("the word bank is empty")
    for word in words:
        if len(word) != WORD_LENGTH:
            raise ValueError(f"word {word!r} is not {WORD_LENGTH} letters long")
        if len(set(word)) != WORD_LENGTH:
            raise ValueError(f"word {word!r} repeats a letter; the bank must be distinct-letter")
        missing = sorted(set(word) - set(ALPHABET))
        if missing:
            raise ValueError(f"word {word!r} uses letters {missing} outside ALPHABET {ALPHABET!r}")
        rows.append([index[ch] for ch in word])
    return np.asarray(rows, dtype=np.int64)


#: `[len(WORD_BANK), WORD_LENGTH]` letter indices.
BANK = encode_words(WORD_BANK)


class WordGuess:
    """Vectorised letter-guessing environment.

    Lanes auto-reset: a finished lane draws a new word inside the same `step`,
    and the pre-reset observation is published under `info["final_obs"]` so a
    trainer can bootstrap time-limited episodes correctly.
    """

    def __init__(self, *, n_envs: int = 16, max_steps: int = 12, word_bank: Sequence[str] | None = None):
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if max_steps < WORD_LENGTH:
            raise ValueError(
                f"max_steps must be at least {WORD_LENGTH} (the number of distinct letters a "
                f"word needs), got {max_steps}"
            )
        self._n_envs = int(n_envs)
        self.max_steps = int(max_steps)
        if word_bank is None:
            self._bank = BANK
            self.words = tuple(WORD_BANK)
        else:
            self.words = tuple(str(w) for w in word_bank)
            self._bank = encode_words(self.words)

        self._rng = np.random.default_rng(0)
        self._word = np.zeros(self._n_envs, dtype=np.int64)
        self._letters = np.zeros((self._n_envs, WORD_LENGTH), dtype=np.int64)
        self._in_word = np.zeros((self._n_envs, N_ACTIONS), dtype=bool)
        self._guessed = np.zeros((self._n_envs, N_ACTIONS), dtype=bool)
        self._revealed = np.zeros((self._n_envs, WORD_LENGTH), dtype=bool)
        self._t = np.zeros(self._n_envs, dtype=np.int64)

    # -- contract -------------------------------------------------------

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def obs_dim(self) -> int:
        """Declared in the manifest as `policy.obs_dim`."""
        return OBS_DIM

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

        lanes = np.arange(self._n_envs)
        repeat = self._guessed[lanes, act]
        in_word = self._in_word[lanes, act]

        # Positions this guess uncovers. Empty for a repeat (already revealed)
        # and for a wrong letter, which is what makes `letter_new` progress
        # rather than activity.
        hit = (self._letters == act[:, None]) & ~self._revealed
        letter_new = hit.any(axis=1)
        self._revealed |= hit
        self._guessed[lanes, act] = True
        self._t += 1

        solved = self._revealed.all(axis=1)
        revealed_fraction = self._revealed.mean(axis=1)

        signals = {
            "letter_correct": in_word.astype(np.float32),
            "letter_new": letter_new.astype(np.float32),
            "letter_wrong": (~in_word).astype(np.float32),
            "repeat_guess": repeat.astype(np.float32),
            "word_solved": solved.astype(np.float32),
            "guess_cost": np.ones(self._n_envs, dtype=np.float32),
            "revealed_fraction": revealed_fraction.astype(np.float32),
        }

        terminated = solved
        truncated = (~terminated) & (self._t >= self.max_steps)
        done = terminated | truncated

        final_obs = self._observe()
        if done.any():
            self._reset_lanes(done)
        return StepResult(
            obs=self._observe(),
            signals=signals,
            terminated=terminated,
            truncated=truncated,
            info={INFO_FINAL_OBS: final_obs},
        )

    def close(self) -> None:
        """Nothing to release; present because the contract requires it."""

    # -- internals ------------------------------------------------------

    def _reset_lanes(self, mask: np.ndarray) -> None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        choice = self._rng.integers(0, self._bank.shape[0], size=idx.size)
        self._word[idx] = choice
        self._letters[idx] = self._bank[choice]
        self._in_word[idx] = False
        rows = np.repeat(idx, WORD_LENGTH)
        self._in_word[rows, self._letters[idx].reshape(-1)] = True
        self._guessed[idx] = False
        self._revealed[idx] = False
        self._t[idx] = 0

    def _observe(self) -> np.ndarray:
        features = [
            self._guessed.astype(np.float32),
            (self._guessed & self._in_word).astype(np.float32),
            self._revealed.astype(np.float32),
            self._revealed.mean(axis=1, dtype=np.float32)[:, None],
            (1.0 - self._t / self.max_steps).astype(np.float32)[:, None],
        ]
        return np.concatenate(features, axis=1).astype(np.float32)

    # -- read-only introspection ---------------------------------------

    def hidden_words(self) -> tuple[str, ...]:
        """The current word per lane. For tests and debugging only.

        Not reachable from a policy, a reward or the objective: all three are
        handed observations, signals and outcomes respectively.
        """
        return tuple("".join(ALPHABET[i] for i in row) for row in self._letters)


def make_env(**kwargs: Any) -> WordGuess:
    """Manifest entrypoint (`word_env:make_env`)."""
    return WordGuess(**kwargs)


# ---------------------------------------------------------------------------
# observation decoding, shared by policies and the evaluator
# ---------------------------------------------------------------------------


def decode_obs(obs: Any) -> dict[str, Any]:
    """Turn one observation vector into the facts it encodes.

    Exists so a demo policy, the evaluator and the tests all read the layout
    from one place. It takes an observation -- never environment state -- so
    anything built on it is a legal policy.
    """
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size != OBS_DIM:
        raise ValueError(f"expected an observation of {OBS_DIM} floats, got {arr.size}")
    return {
        "guessed": arr[OBS_GUESSED] > 0.5,
        "present": arr[OBS_PRESENT] > 0.5,
        "revealed": arr[OBS_REVEALED] > 0.5,
        "revealed_fraction": float(arr[OBS_REVEALED_FRACTION]),
        "budget_left": float(arr[OBS_BUDGET_LEFT]),
    }


def render_obs(obs: Any) -> str:
    """Human-readable one-liner, e.g. `revealed 2/4 | tried a,c | present a`.

    Diagnostics only. The prompt a language-model policy actually sees is
    built by `autobranch.posttrain.serialize_observation`, which is env
    agnostic on purpose -- a per-env prompt format would be one more thing to
    keep in sync between trainer and evaluator.
    """
    f = decode_obs(obs)
    tried = ",".join(ALPHABET[i] for i in np.flatnonzero(f["guessed"])) or "-"
    present = ",".join(ALPHABET[i] for i in np.flatnonzero(f["present"])) or "-"
    revealed = int(f["revealed"].sum())
    return (
        f"revealed {revealed}/{WORD_LENGTH} | tried {tried} | present {present} "
        f"| budget {f['budget_left']:.2f}"
    )


def sweep_policy(obs: Any) -> Any:
    """Guess the lowest-index letter not yet tried. Solves every word.

    Shipped with the bundle because an environment whose objective nobody has
    ever seen reach 1.0 is an environment that might just be broken. This is
    the control: a distinct-letter sweep needs at most ten guesses, the step
    budget is twelve, so `solve_rate` is exactly 1.0 and any lower number from
    a trained policy is the policy's fault, not the task's.
    """
    arr = np.asarray(obs, dtype=np.float64)
    single = arr.ndim == 1
    rows = arr.reshape(1, -1) if single else arr
    guessed = rows[:, OBS_GUESSED] > 0.5
    # `argmin` over a boolean row gives the first False, i.e. the first untried
    # letter; an all-tried row cannot occur because ten guesses reveal every
    # word, but argmin would return 0 and still be legal if it did.
    actions = np.argmin(guessed, axis=1).astype(np.int64)
    return actions[0] if single else actions


def sticky_policy(obs: Any) -> Any:
    """Re-guess a letter already known to be in the word; otherwise sweep.

    This is not a strawman: it is the behaviour `letter_correct` pays for.
    That signal fires on every guess of an in-word letter, repeats included, so
    a reward weighting it alone is maximised by exactly this policy -- which
    never solves a word. Keeping it next to `sweep_policy` makes the trap
    inspectable without training anything.
    """
    arr = np.asarray(obs, dtype=np.float64)
    single = arr.ndim == 1
    rows = arr.reshape(1, -1) if single else arr
    guessed = rows[:, OBS_GUESSED] > 0.5
    present = rows[:, OBS_PRESENT] > 0.5
    sweep = np.argmin(guessed, axis=1)
    stick = np.argmax(present, axis=1)
    actions = np.where(present.any(axis=1), stick, sweep).astype(np.int64)
    return actions[0] if single else actions


def clone(env: WordGuess) -> WordGuess:
    """A deep copy, for one-step lookahead over a reward config.

    The state is plain NumPy, so a copy is cheap and exact. Used by tests that
    ask "what does this reward actually pay for" without needing a trained
    policy to answer.
    """
    return copy.deepcopy(env)


# ---------------------------------------------------------------------------
# the objective: solve_rate, and nothing a reward can reach
# ---------------------------------------------------------------------------


def evaluate(
    policy,
    *,
    seeds: Sequence[int] = (12345,),
    budget: dict[str, Any] | None = None,
    deterministic: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """Fraction of episodes in which the whole word gets revealed in budget.

    Reads episode outcomes only: solved or not, and how many guesses it took.
    It is handed a callable `policy(obs) -> action` and has no access to
    weights, rewards or training state, so there is nothing here for a reward
    search to exploit.

    Episodes are split evenly across the caller's fixed seed bank, so two
    branches are scored on identical hidden words.
    """
    budget = dict(budget or {})
    episodes = max(1, int(budget.get("episodes", 128)))
    lanes = max(1, int(budget.get("lanes", 16)))
    max_steps = int(budget.get("max_steps", 12))
    seeds = tuple(int(s) for s in (seeds or (12345,)))
    per_seed = max(1, episodes // len(seeds))

    solved = 0
    counted = 0
    guesses: list[int] = []
    repeats = 0
    total_guesses = 0
    revealed_at_end: list[float] = []

    for seed in seeds:
        # At least 2 lanes: a size-1 leading dimension has tripped contiguity
        # checks in native libraries.
        env = WordGuess(n_envs=max(2, min(lanes, per_seed)), max_steps=max_steps)
        obs = env.reset(seed=seed)
        if hasattr(policy, "reseed"):
            policy.reseed(seed)
        steps = np.zeros(env.n_envs, dtype=np.int64)
        taken = 0
        # Hard cap: enough steps for every lane to finish its share of
        # episodes, plus slack. A policy that never solves anything must still
        # produce a score rather than hang the evaluation.
        budget_steps = (per_seed // env.n_envs + 2) * (max_steps + 1)
        for _ in range(budget_steps):
            if taken >= per_seed:
                break
            result = env.step(policy(obs))
            obs = result.obs
            steps += 1
            total_guesses += env.n_envs
            repeats += int(np.asarray(result.signals["repeat_guess"]).sum())
            done = np.asarray(result.terminated, dtype=bool) | np.asarray(
                result.truncated, dtype=bool
            )
            for lane in np.flatnonzero(done):
                if taken >= per_seed:
                    break
                won = bool(result.signals["word_solved"][lane])
                solved += int(won)
                guesses.append(int(steps[lane]))
                revealed_at_end.append(float(result.signals["revealed_fraction"][lane]))
                taken += 1
                counted += 1
            steps[done] = 0
        env.close()

    n = max(1, counted)
    return {
        "value": solved / n,
        "n": counted,
        "detail": {
            "solve_rate": solved / n,
            "timeout_rate": (counted - solved) / n,
            "mean_guesses": float(np.mean(guesses)) if guesses else 0.0,
            "mean_revealed_fraction": float(np.mean(revealed_at_end)) if revealed_at_end else 0.0,
            "repeat_guess_rate": repeats / max(1, total_guesses),
            "episodes_per_seed": per_seed,
            "seeds": list(seeds),
            "deterministic": bool(deterministic),
        },
    }
