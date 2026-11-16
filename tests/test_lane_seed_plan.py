"""The seed bank has to reach the lanes, not just exist.

SEED_BANK_SIZE is 8 because "a single layout/seed rewards memorisation". The
SF3 evaluator took one base seed per *batch* and gave lane i `base + i`, so a
config whose concurrency equals its match count ran one batch and used one of
the eight seeds. That is the SF3 default (matches 24, concurrency 24), so
every evaluation in the reproduction run rested on a single seed family -- and
two families disagreed by 33 points on identical weights.

These tests fail on that behaviour and pass on the fix.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "sf3"))

import pytest

from autobranch.objective import SEED_BANK_SIZE, seed_bank


def plan():
    from sf3_env import lane_seed_plan

    return lane_seed_plan


def test_one_batch_still_uses_the_whole_bank():
    """The regression that mattered: concurrency >= matches must not collapse
    the bank to a single seed."""
    bank = seed_bank(12345, SEED_BANK_SIZE)
    batches = plan()(24, 24, bank)

    assert len(batches) == 1, "24 matches at 24 lanes is one batch, as before"
    assert len(set(batches[0])) == 24, "no match may reuse another's seed"

    # A lane seed is `family + offset` with a small offset, so the family it
    # came from is the bank entry it sits just above.
    offsets = {f: [s - f for s in batches[0] if 0 <= s - f <= 2] for f in bank}
    reached = {f for f, got in offsets.items() if got}
    assert len(reached) == len(bank), (
        f"only {len(reached)} of {len(bank)} seed families reached the lanes"
    )


def test_matches_are_spread_evenly_across_families():
    bank = seed_bank(12345, 8)
    batch = plan()(24, 24, bank)[0]
    per_family = {f: sum(1 for s in batch if 0 <= s - f < 3) for f in bank}
    assert set(per_family.values()) == {3}, per_family


def test_batches_when_concurrency_is_smaller():
    bank = seed_bank(12345, 8)
    batches = plan()(72, 24, bank)
    assert [len(b) for b in batches] == [24, 24, 24]
    flat = [s for b in batches for s in b]
    assert len(set(flat)) == 72, "seeds must stay unique across batches too"


def test_fewer_matches_than_bank_uses_distinct_families():
    bank = seed_bank(12345, 8)
    batch = plan()(3, 8, bank)[0]
    assert len(batch) == 3
    assert len(set(batch)) == 3
    # Three different families, not three offsets of one.
    assert sum(1 for f in bank if f in batch) == 3


def test_single_seed_bank_degrades_to_consecutive_seeds():
    """eval_sf3.py passes one seed deliberately; that must still work."""
    batch = plan()(4, 4, [777])[0]
    assert batch == [777, 778, 779, 780]


def test_zero_matches_is_an_empty_plan():
    assert plan()(0, 24, seed_bank(1, 8)) == []


@pytest.mark.parametrize("matches,concurrency", [(24, 24), (72, 24), (5, 2), (1, 8)])
def test_plan_always_covers_exactly_the_requested_matches(matches, concurrency):
    batches = plan()(matches, concurrency, seed_bank(99, 8))
    flat = [s for b in batches for s in b]
    assert len(flat) == matches
    assert len(set(flat)) == matches
    assert all(len(b) <= concurrency for b in batches)
