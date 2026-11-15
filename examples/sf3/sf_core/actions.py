"""Discrete macro-action table for Ryu.

Each action is a *name* from the vendored, upstream-verified move tables
(sf.vendor.moves). At play time the name resolves — via the same
``parse_move`` the upstream project uses — to a side-corrected sequence of
per-frame button bitmasks, which the worker plays through
``env.step_frames`` in one MAME round trip. Special-move motions (QCF,
DP, QCB, double-QCF) are therefore frame-exact; there is no reaction
delay, which is the structural edge over the upstream LLM policy.

The table's order is part of every checkpoint's contract: tests pin its
fingerprint so a saved policy can never be silently re-indexed.
"""

from __future__ import annotations

from sf_core.moves import parse_move

CHARACTER = "Ryu"

# Short macros are padded with no-input frames up to this length, which both
# bounds the decision rate (~15 Hz at 60 fps) and keeps MAME pipe round
# trips per decision low. Attack normals have longer recovery than this, so
# the padding is behaviourally free.
MIN_MACRO_FRAMES = 4

ACTION_NAMES: tuple[str, ...] = (
    # -- movement / defense ------------------------------------------------
    "Wait",
    "Move Closer",
    "Move Away",
    "Crouch",
    "Guard High",
    "Guard Low",
    "Jump Up",
    "Jump Closer",
    "Jump Away",
    "Dash Forward",
    "Dash Back",
    "Parry High",
    "Parry Low",
    "Recover From Stun",
    # -- standing normals --------------------------------------------------
    "Low Punch",
    "Medium Punch",
    "High Punch",
    "Low Kick",
    "Medium Kick",
    "High Kick",
    # -- crouching normals -------------------------------------------------
    "Crouching LK",
    "Crouching MK",
    "Crouching HK",
    "Crouching HP",
    # -- command normals / universal ----------------------------------------
    "Sakotsu Wari",      # f+MP overhead
    "Overhead",          # MP+MK universal overhead
    "Throw",
    "Target HP HK",      # target combo
    # -- jumping attacks ----------------------------------------------------
    "Jump Forward HK",
    "Jump Up HP",
    # -- specials ------------------------------------------------------------
    "Hadouken (MP)",
    "Hadouken (HP)",
    "Shoryuken (LP)",
    "Shoryuken (HP)",
    "Tatsumaki Senpukyaku (MK)",
    "Joudan Sokutou Geri (MK)",
    "EX Hadouken",
    "EX Shoryuken",
    "EX Tatsumaki Senpukyaku",
    # -- super art 1 (matches upstream demo: SA1 Shinkuu-Hadouken) ----------
    "1 Shinkuu-Hadouken",
)

N_ACTIONS = len(ACTION_NAMES)


def action_frames(action_id: int, side: int) -> list[int]:
    """Frame-bitmask sequence for an action, corrected for the fighter's side.

    ``side`` is the raw emulator byte (0 or 1), with the same semantics the
    upstream worker uses when resolving LLM move names.
    """
    name = ACTION_NAMES[action_id]
    sequence = parse_move(CHARACTER, name, side)
    if sequence is None:  # pragma: no cover - table is validated by tests
        raise KeyError(f"Action {action_id} ({name!r}) did not resolve")
    frames = list(sequence)
    if len(frames) < MIN_MACRO_FRAMES:
        frames.extend([0] * (MIN_MACRO_FRAMES - len(frames)))
    return frames


def table_fingerprint() -> str:
    """Stable digest of (names, resolved sequences) for checkpoint safety."""
    import hashlib

    payload = repr(
        [
            (name, parse_move(CHARACTER, name, 0), parse_move(CHARACTER, name, 1))
            for name in ACTION_NAMES
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
