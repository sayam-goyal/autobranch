"""Turn the SF3 run's recorded data into an analysis.

    python examples/sf3/analyze.py --run-id run-4342abbaad
    python examples/sf3/analyze.py --run-id run-4342abbaad --out report.md

Reads two sources and joins them, which is the whole point: the registry knows
*what each branch tried and what it scored*, and the Volume knows *what
happened inside each branch while it trained*. Neither alone answers the
question this run exists to ask.

The question is not "did it train". It is whether a searched reward beats the
hand-tuned one on the objective, and that can only be answered from the
objective -- so every table here is anchored on `win_rate_vs_cpu8` and the
training curves appear next to it as explanation, never as evidence. The
predecessor project watched a policy peak at 0.599 and collapse to 0.353 while
its training return rose twentyfold; a report that leads with return would have
called that a success.

Numbers are printed with the n they came from. A 24-match evaluation has a
standard error of about 10 points at p=0.5, so two branches inside 10 points of
each other are not ranked by this data and the report says so rather than
implying an order.
"""

import argparse
import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DATA_VOLUME = os.environ.get("AUTOBRANCH_SF3_VOLUME", "autobranch-data")

#: Hinh's constants as the DSL expresses them. The comparison is against this.
HINH_REWARD = {
    "round_won": 1.0,
    "round_lost": -1.0,
    "damage_dealt": 0.1,
    "damage_taken": -0.1,
    "invalid_move": -0.125,
}


def _se(p: float, n: int) -> float:
    """Standard error of a proportion. Zero n is honest, not an error."""
    if n <= 0:
        return float("nan")
    return math.sqrt(max(0.0, p * (1.0 - p)) / n)


def _weights(branch) -> dict:
    return dict((getattr(branch, "reward_config", {}) or {}).get("weights") or {})


def _is_hinh(branch) -> bool:
    w = _weights(branch)
    if set(w) != set(HINH_REWARD):
        return False
    return all(abs(float(w[k]) - v) < 1e-9 for k, v in HINH_REWARD.items())


def _fmt_reward(config: dict) -> str:
    weights = (config or {}).get("weights") or {}
    out = " ".join(f"{k}={v:g}" for k, v in sorted(weights.items()))
    for shaping in (config or {}).get("shaping") or []:
        out += f" +{shaping.get('kind')}({shaping.get('signal')})"
    if (config or {}).get("anneal"):
        out += " +anneal"
    return out or "(none)"


def load_metrics(run_id: str) -> dict[str, list[dict]]:
    """Per-branch iteration logs off the Volume."""
    import modal

    volume = modal.Volume.from_name(DATA_VOLUME)
    out: dict[str, list[dict]] = {}
    prefix = f"runs/{run_id}"
    try:
        entries = list(volume.listdir(prefix, recursive=False))
    except Exception:  # noqa: BLE001 - nothing logged yet is a real answer
        return out
    for entry in entries:
        branch = entry.path.rstrip("/").rsplit("/", 1)[-1]
        try:
            raw = b"".join(volume.read_file(f"{prefix}/{branch}/metrics.jsonl")).decode()
        except Exception:  # noqa: BLE001
            continue
        lines = [json.loads(l) for l in raw.splitlines() if l.strip()]
        if lines:
            out[branch] = lines
    return out


def report(run_id: str) -> str:
    from autobranch.registry import Registry

    registry = Registry()
    run = registry.get_run(run_id)
    branches = registry.branches(run_id)
    metrics = load_metrics(run_id)

    L: list[str] = []
    add = L.append

    if run is None:
        return f"# {run_id}\n\nNo such run in the registry.\n"

    best = getattr(run, "best_score", None)
    add(f"# SF3 reward search — {run_id}")
    add("")
    add(f"**{getattr(run, 'objective_name', 'objective')} "
        f"= {best if best is None else round(best, 4)}** "
        f"({getattr(run, 'state', '?')}), "
        f"{len(branches)} branches, "
        f"${getattr(run, 'spend_usd', 0.0):.2f}, "
        f"{(getattr(run, 'elapsed_s', 0) or 0) / 3600:.2f} h")
    add("")
    add("Qwen3-VL-8B-Instruct, full-parameter (`grpo_full`), Ryu vs the arcade CPU "
        "at difficulty 8. Reproduces modal-labs/sf3's model, inputs and fine-tune "
        "scope; the reward is searched rather than hand-tuned.")
    add("")

    # ---- the table that answers the question --------------------------
    add("## Branches, by objective")
    add("")
    add("| branch | parent | status | best | n | ±1 SE | steps | $ | reward |")
    add("|---|---|---|---:|---:|---:|---:|---:|---|")
    scored = []
    for b in sorted(branches, key=lambda x: str(getattr(x, "branch_id", ""))):
        bid = str(getattr(b, "branch_id", "?"))
        score = getattr(b, "best_score", None)
        detail = getattr(b, "score_detail", {}) or {}
        n = int(detail.get("n") or detail.get("matches") or 0)
        se = _se(float(score), n) if (score is not None and n) else float("nan")
        tag = " **(Hinh)**" if _is_hinh(b) else ""
        add(f"| `{bid}`{tag} | {getattr(b, 'parent_id', None) or '—'} "
            f"| {getattr(b, 'status', '?')} "
            f"| {'—' if score is None else f'{score:.4f}'} "
            f"| {n or '—'} "
            f"| {'—' if se != se else f'{se:.3f}'} "
            f"| {getattr(b, 'steps_done', 0):,} "
            f"| {getattr(b, 'spend_usd', 0.0):.2f} "
            f"| `{_fmt_reward(getattr(b, 'reward_config', {}) or {})}` |")
        if score is not None:
            scored.append((bid, float(score), n, _is_hinh(b)))
    add("")

    # ---- the comparison ------------------------------------------------
    add("## Searched reward vs the hand-tuned one")
    add("")
    hinh = [s for s in scored if s[3]]
    others = [s for s in scored if not s[3]]
    # "Never ran it" and "ran it but has not scored it yet" are different
    # states and the report must not blur them: the first says the comparison
    # is impossible, the second says it is pending. The table above tags the
    # configuration whether or not it has a score, so a single message for both
    # would contradict it.
    ran_hinh = [b for b in branches if _is_hinh(b)]
    if not hinh and ran_hinh:
        pending = ", ".join(f"`{getattr(b, 'branch_id', '?')}`" for b in ran_hinh)
        add(f"Hinh's configuration is running on {pending} but has not been "
            f"evaluated yet, so the comparison is pending rather than absent. "
            f"Nothing here should be read as a result until it has a score and "
            f"an n.")
    elif not hinh:
        add("No branch ran Hinh's exact configuration, so there is no "
            "within-run baseline. The comparison would have to be against the "
            "predecessor project's measured 55.0% for that reward (n=100), which "
            "is a different codebase and a different policy class — weaker "
            "evidence, and it should be labelled as such.")
    elif not others:
        add("Only Hinh's configuration was scored, so the search has not yet "
            "produced a comparison. Whatever the number is, it says nothing "
            "about whether searching helps.")
    else:
        hb, hv, hn, _ = max(hinh, key=lambda s: s[1])
        ob, ov, on, _ = max(others, key=lambda s: s[1])
        gap = ov - hv
        pooled = math.sqrt(_se(hv, hn) ** 2 + _se(ov, on) ** 2)
        add(f"- Hinh's reward: **{hv:.4f}** on `{hb}` (n={hn})")
        add(f"- best searched: **{ov:.4f}** on `{ob}` (n={on})")
        add(f"- difference: **{gap:+.4f}**, pooled SE {pooled:.3f}")
        add("")
        if pooled == pooled and abs(gap) < 2 * pooled:
            add(f"That difference is inside two standard errors, so **this run does "
                f"not separate them**. Reporting it as a win either way would be "
                f"reading noise. The honest statement is the interval, and the fix "
                f"is n, not rhetoric.")
        else:
            better = "the searched reward" if gap > 0 else "the hand-tuned reward"
            add(f"That clears two standard errors, so {better} is ahead on this "
                f"evidence — at this difficulty, on this character, at these "
                f"match counts.")
    add("")

    # ---- what happened inside each branch ------------------------------
    add("## Training curves (explanation, not evidence)")
    add("")
    if not metrics:
        add("No per-iteration metrics on the Volume. Either no branch completed an "
            "iteration, or the trainer failed to write to "
            f"`runs/{run_id}/<branch>/metrics.jsonl` — which would also mean the HF "
            "exports are missing, so check that before trusting any score above.")
    for branch, lines in sorted(metrics.items()):
        last = lines[-1]
        drift = max(abs(l.get("ratio_mean", 1.0) - 1.0) for l in lines)
        degenerate = max(l.get("zero_variance_group_fraction", 0.0) for l in lines)
        ent_first = lines[0].get("entropy", float("nan"))
        ent_last = last.get("entropy", float("nan"))
        add(f"### `{branch}` — {len(lines)} iterations, {last.get('step', 0):,} decisions")
        add("")
        add(f"- return: {lines[0].get('return_mean', 0):+.4f} → "
            f"{last.get('return_mean', 0):+.4f}")
        add(f"- entropy: {ent_first:.3f} → {ent_last:.3f} nats "
            f"(of {math.log(40):.3f} available)")
        add(f"- invalid-move rate: {lines[0].get('invalid_move_rate', 0):.3f} → "
            f"{last.get('invalid_move_rate', 0):.3f}")
        add(f"- rounds won/lost in the last collection: "
            f"{last.get('rounds_won', 0)}/{last.get('rounds_lost', 0)}")
        add("")
        # The two readings that invalidate everything else.
        if drift > 0.05:
            add(f"- **max first-epoch ratio drift {drift:.3f}** — above 0.05, so "
                f"collection and the update were not scoring the same thing and "
                f"the gradients here are not trustworthy. Every other number in "
                f"this section describes a run that was not learning what it "
                f"appears to have learned.")
        else:
            add(f"- first-epoch ratio drift stayed at {drift:.3f} (< 0.05): "
                f"collection and update agree, so the gradient means what it says.")
        if degenerate > 0.5:
            add(f"- **zero-variance groups peaked at {degenerate:.2f}** — a group "
                f"whose members all score identically has no advantage for any "
                f"member. At this level the branch was not learning slowly, it was "
                f"not learning.")
        if ent_last == ent_last and ent_last < 0.3:
            add(f"- entropy ended at {ent_last:.3f} nats, which is close to a "
                f"single action. The predecessor policy won 0 of 100 at its mode "
                f"while competent while sampling, so a collapsed policy can still "
                f"score well stochastically and be worthless deterministically.")
        add("")

    # ---- provenance ----------------------------------------------------
    add("## Where this came from")
    add("")
    add(f"- registry: `modal.Dict`, run `{run_id}`")
    add(f"- metrics: `{DATA_VOLUME}:runs/{run_id}/<branch>/metrics.jsonl`")
    add(f"- weights: `{DATA_VOLUME}:runs/{run_id}/<branch>/hf/` "
        "(full HF checkpoints, loadable without this framework)")
    add("- probe measurements that set the run's shape: `MEASUREMENTS.md`")
    add("- the two discarded runs and why: `RUNBOOK.md`")
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    text = report(args.run_id)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out} ({len(text)} bytes)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
