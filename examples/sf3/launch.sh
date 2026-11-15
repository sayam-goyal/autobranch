#!/usr/bin/env bash
# Launch the SF3 reward search. Absolute values only -- no remembered numbers.
#
#   bash examples/sf3/launch.sh
#
# Everything below is a deliberate choice with a reason, because a restarted
# session must be able to re-issue this identically rather than re-derive it.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- budget ---------------------------------------------------------------
# One H100 per branch (probe_full_ft: 49.5 GiB of 79.2, so no model
# parallelism) at ~$3.95/hr, plus ~8 emulator lanes per branch at 2 CPUs each.
# That is roughly $5/hr per branch. Eight branches never run at once -- the
# orchestrator's max_concurrent governs -- so $220 buys well over the window
# below and the cap exists to stop a runaway, not to shape the search.
USD="${SF3_USD:-220}"

# 5.5 hours of training inside an 8-hour envelope that started at 21:05 PST.
# The remainder is not slack: the objective is 40 matches per evaluation at
# ~0.25 s per decision, and an unmeasured policy is worth nothing, so the
# evaluation window is reserved first and the training window gets the rest.
HOURS="${SF3_HOURS:-5.5}"

MAX_BRANCHES="${SF3_MAX_BRANCHES:-8}"

# ---- constraints, in the words the agent reads ----------------------------
# The reference reward is stated as context, not as an instruction. Opening a
# search by defending someone else's local optimum is how the predecessor
# project spent six configurations at 0.000.
read -r -d '' CONSTRAINTS <<'EOF' || true
Win as Ryu against the arcade CPU at difficulty 8, measured by
win_rate_vs_cpu8 over real kickoffs. That measurement is the only thing that
counts: never trade independent win rate for training return. The predecessor
run on this exact character and difficulty saw a policy peak and then collapse
on the objective while its training return rose twentyfold, so a rising return
is not evidence.

Reference point for context, not a target to defend. modal-labs/sf3 hand-tuned
this reward as: round outcome +/-1.0, damage dealt and taken at 0.1 each, and
an invalid-move penalty of 20/160 = 0.125, with the outcome discounted on a
3600-frame half-life. In your DSL that is weights {round_won: 1.0,
round_lost: -1.0, damage_dealt: 0.1, damage_taken: -0.1, invalid_move: -0.125}.
It is worth measuring early as one candidate among several.

Two things known from measurement on this environment that should shape where
you look:

- Sparse outcome-only reward is weak here. A round takes roughly 600 decisions,
  so round_won arrives once per several hundred steps and carries almost no
  gradient on its own.
- Dense shaping has a window with edges on both sides. On a comparable setup,
  shaping that was too strong produced a policy that peaked at 0.599 and
  collapsed to 0.353, while removing shaping entirely never left 0.355. The
  configuration that won used moderate shaping and a small contact reward.

invalid_move fires when the chosen macro action is unavailable in the current
state -- an EX move without meter, a super without a full bar -- and the game
advances on a Wait instead. It is a real cost, but actions are masked tokens
here, so no action is ever unparseable; a large penalty on it may be weight
spent on a problem this interface does not have.

decision_frames is how many game frames the action consumed. Returns are
already discounted by it, so it is available as a signal but you do not need it
to express a time preference.

must posttrain Qwen/Qwen3-VL-8B-Instruct
nothing bigger than an H100
EOF

echo "=== SF3 reward search ==="
echo "  bundle        examples/sf3"
echo "  usd cap       \$${USD}"
echo "  hours         ${HOURS}"
echo "  max branches  ${MAX_BRANCHES}"
echo "  launched at   $(date '+%Y-%m-%d %H:%M:%S %Z') (epoch $(date +%s))"
echo

exec modal run app.py::cli_submit \
  --bundle examples/sf3 \
  --constraints "${CONSTRAINTS}" \
  --usd "${USD}" \
  --hours "${HOURS}" \
  --max-branches "${MAX_BRANCHES}"
