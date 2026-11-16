"""Turning a finished search into something you can run.

The deliverable is weights plus everything needed to use them, plus an honest
account of how they were found. The artifact carries four things: the winning
checkpoint, a manifest with the objective score and its provenance command,
the reward lineage from root to winner, and every abandoned branch with its
reason.

The losers file is the result, not an apology. A search that reports only its
winner cannot be told apart from a lucky guess.

Ranking is always by objective, never by training reward.
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable

ARTIFACT_DIRNAME = "artifacts"


class ConsolidationError(RuntimeError):
    """Raised when there is nothing defensible to ship."""

@dataclass
class Winner:
    branch_id: str
    score: float
    reward_config: dict[str, Any]
    checkpoint: str | None
    command: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    # The training step that produced `score`. Needed because a full
    # fine-tune's tensors live in a per-step directory on the Volume, and the
    # branch's live `hf/` export is the *latest* step, not the best-scoring
    # one. In run-25ed75aa0a those differed badly: the winner scored 0.736 at
    # n=72 on step 72,000 and 0.375 on step 86,400, so an artifact pointing
    # at `hf/` advertised one number and handed over weights worth half of it.
    step: int | None = None


def rank_branches(branches: list[dict[str, Any]], objective) -> list[dict[str, Any]]:
    """Best first, by objective score. Unscored branches sort last.

    An abandoned branch is still eligible: abandonment means "this line stopped
    paying", not "this result is invalid". If the best score in the whole tree
    came from a branch we stopped early, that is the number we ship, and the
    lineage will say so.
    """
    def key(b: dict[str, Any]):
        score = _best_score(b)
        return (0 if score is not None else 1, -(score or 0.0) * objective.sign)

    return sorted(branches, key=key)


def _best_score(branch: dict[str, Any]) -> float | None:
    if branch.get("best_score") is not None:
        return float(branch["best_score"])
    scores = [s.get("score") for s in (branch.get("scores") or []) if s.get("score") is not None]
    return max(scores) if scores else None


def select_winner(payload: dict[str, Any], objective) -> Winner:
    """Pick what to ship, or refuse to ship.

    Refusing matters: a run where nothing was ever evaluated has no defensible
    output, and emitting an unscored checkpoint would imply a measurement that
    never happened.
    """
    branches = payload.get("branches") or []
    ranked = rank_branches(branches, objective)
    scored = [b for b in ranked if _best_score(b) is not None]
    if not scored:
        raise ConsolidationError(
            "no branch produced an objective score; there is nothing measured to ship"
        )
    top = scored[0]
    best = _best_score(top)
    assert best is not None
    return Winner(
        branch_id=top["branch_id"],
        score=best,
        reward_config=_reward_for_best(top),
        # `published_checkpoint` first: it is the durable copy of the weights
        # that earned `best`, whereas `checkpoint` follows the trainer and
        # reverts to a sandbox-local path whenever a branch keeps training
        # past its best score -- which is the common case, and left this
        # function reporting the winning weights unreachable while a good
        # copy sat on the Volume.
        checkpoint=top.get("published_checkpoint") or top.get("checkpoint"),
        command=top.get("score_command", ""),
        detail=top.get("score_detail") or {},
        step=_step_of_best(top),
    )


def _independent_summary(payload: dict[str, Any], winner: Winner) -> list[dict[str, Any]]:
    """Independent evaluations of the winning branch, largest n first.

    Injected into the payload by whoever has the Volume mounted, so this
    module stays free of I/O. Absent is normal and renders as absent -- the
    artifact must not imply an independent check that nobody ran.
    """
    for b in payload.get("branches") or []:
        if b.get("branch_id") != winner.branch_id:
            continue
        rows = []
        for e in b.get("independent_evals") or []:
            if not isinstance(e, dict):
                continue
            detail = e.get("detail") or {}
            rows.append({
                "step": e.get("step"),
                "export_dir": e.get("export_dir"),
                "n": e.get("n") or detail.get("matches"),
                "win_rate": e.get("value", e.get("win_rate")),
                "wins": detail.get("wins"),
                "rounds_won": detail.get("rounds_won"),
                "rounds_lost": detail.get("rounds_lost"),
                "seed_bank": e.get("seed_bank"),
                "mode": e.get("mode"),
            })
        rows.sort(key=lambda r: (r.get("n") or 0), reverse=True)
        return rows
    return []


def _step_of_best(branch: dict[str, Any]) -> int | None:
    """The step whose evaluation produced the branch's best score."""
    best, best_score = None, None
    for s in branch.get("scores") or []:
        v = s.get("score")
        if v is None:
            continue
        if best_score is None or float(v) > float(best_score):
            best, best_score = s, float(v)
    if best is None or best.get("step") is None:
        return None
    return int(best["step"])


def _reward_for_best(branch: dict[str, Any]) -> dict[str, Any]:
    """The reward that produced the winning score, not the branch's latest.

    A branch's `reward_config` is the next thing it intended to try. Shipping
    it as the provenance of an earlier score mislabels the artifact: the SF3
    run's winner was measured on the reference weights while its config had
    already moved on.
    """
    best, best_score = None, None
    for s in branch.get("scores") or []:
        v = s.get("score")
        if v is None:
            continue
        if best_score is None or float(v) > float(best_score):
            best, best_score = s, float(v)
    if best is not None and best.get("reward"):
        return dict(best["reward"])
    return dict(branch.get("reward_config") or {})


def reward_lineage(payload: dict[str, Any], winner_branch_id: str) -> list[dict[str, Any]]:
    """Root to winner, with the diff each fork introduced.

    Showing the mutation rather than two blobs of JSON is the difference
    between a lineage someone reads and one they skip.
    """
    from autobranch.reward import diff, summarize

    by_id = {b["branch_id"]: b for b in (payload.get("branches") or [])}
    chain: list[dict[str, Any]] = []
    cur = by_id.get(winner_branch_id)
    while cur is not None:
        chain.append(cur)
        parent_id = cur.get("parent_id")
        cur = by_id.get(parent_id) if parent_id else None
    chain.reverse()

    out: list[dict[str, Any]] = []
    for i, b in enumerate(chain):
        entry = {
            "branch_id": b["branch_id"],
            "depth": b.get("depth", i),
            "summary": summarize(b.get("reward_config") or {}),
            "reward_config": b.get("reward_config") or {},
            "best_score": _best_score(b),
            "status": b.get("status"),
            "rationale": b.get("rationale", ""),
        }
        if i > 0:
            entry["changed_from_parent"] = diff(
                chain[i - 1].get("reward_config") or {}, b.get("reward_config") or {}
            )
        out.append(entry)
    return out


def losers(payload: dict[str, Any], winner_branch_id: str, objective) -> list[dict[str, Any]]:
    """Every branch that is not the winner, worst-understood first.

    Ordered by score so a reader sees the near-misses next to the winner and
    the dead ends at the bottom -- the shape of the search, not just its tip.
    """
    from autobranch.reward import summarize

    out = []
    for b in rank_branches(payload.get("branches") or [], objective):
        if b["branch_id"] == winner_branch_id:
            continue
        out.append({
            "branch_id": b["branch_id"],
            "parent_id": b.get("parent_id"),
            "status": b.get("status"),
            "summary": summarize(b.get("reward_config") or {}),
            "reward_config": b.get("reward_config") or {},
            "best_score": _best_score(b),
            "evaluations": len(b.get("scores") or []),
            "spend_usd": b.get("spend_usd", 0.0),
            "abandoned_reason": b.get("abandoned_reason"),
            "rationale": b.get("rationale", ""),
        })
    return out


def build_manifest(payload: dict[str, Any], winner: Winner) -> dict[str, Any]:
    objective = payload.get("objective") or {}
    spec = payload.get("spec") or {}
    return {
        "run_id": payload.get("run_id"),
        "environment": payload.get("env_name"),
        "created_at": payload.get("created_at"),
        "finished_at": time.time(),
        "objective": {
            "name": objective.get("name"),
            "direction": objective.get("direction", "maximize"),
            "score": winner.score,
            "measured_by": winner.command or "(command not recorded)",
            "measured_at_step": winner.step,
            "detail": winner.detail,
            # The search's own score is measured at whatever n a branch could
            # afford. Where a larger independent evaluation exists it belongs
            # beside it, not buried: in run-25ed75aa0a the search read 0.9583
            # at n=24 on a checkpoint an n=72 run put at 0.7361.
            "independent": _independent_summary(payload, winner),
        },
        "winner": {
            "branch_id": winner.branch_id,
            "reward_config": winner.reward_config,
            "step": winner.step,
            "checkpoint": os.path.basename(winner.checkpoint) if winner.checkpoint else None,
        },
        "policy": (spec.get("policy") or {}),
        "trainer": spec.get("trainer"),
        "base_model": (spec.get("policy") or {}).get("base_model"),
        "spend_usd": payload.get("spend_usd", 0.0),
        "constraints": payload.get("constraints", ""),
        "branches": {
            "total": len(payload.get("branches") or []),
            "abandoned": sum(
                1 for b in (payload.get("branches") or []) if b.get("status") == "abandoned"
            ),
            "evaluated": sum(
                1 for b in (payload.get("branches") or []) if (b.get("scores") or [])
            ),
        },
    }


RUN_SNIPPET = '''"""Run the policy this search produced.

    python serve.py            # one episode, prints the objective-relevant outcome

The checkpoint is `weights/{checkpoint}`. The reward that produced it is in
manifest.json for the record -- it is NOT needed at inference time, which is
the point of keeping reward and objective separate.
"""

import json
import pathlib

import torch

HERE = pathlib.Path(__file__).parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())


def load_policy():
    ckpt = HERE / "weights" / MANIFEST["winner"]["checkpoint"]
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    return payload


if __name__ == "__main__":
    payload = load_policy()
    print("objective:", MANIFEST["objective"]["name"], "=", MANIFEST["objective"]["score"])
    print("measured by:", MANIFEST["objective"]["measured_by"])
    print("checkpoint keys:", sorted(payload)[:10])
'''


def artifact_readme(manifest: dict[str, Any]) -> str:
    obj = manifest["objective"]
    win = manifest["winner"]
    branches = manifest["branches"]
    lines = [
        f"# {manifest['environment']} - {manifest['run_id']}",
        "",
        f"**{obj['name']} = {obj['score']:.4f}** ({obj['direction']}), produced by branch "
        f"`{win['branch_id']}`.",
        "",
        f"Measured by: `{obj['measured_by']}`",
        "",
    ]
    independent = obj.get("independent") or []
    if independent:
        lines += [
            "## Independent evaluation",
            "",
            "The score above is the search's own, measured at whatever match count",
            "a branch could afford. These were measured separately, after the run,",
            "at larger n. Where they disagree, **trust these**: a small evaluation",
            "is not a noisier estimate of a large one, it is a narrower draw, and a",
            "search that selects on it selects the optimistic end of its own noise.",
            "",
            "| step | n | win rate | wins | rounds W/L | mode |",
            "|---:|---:|---:|---:|---:|---|",
        ]
        for r in independent:
            wr = r.get("win_rate")
            rounds = (
                f"{r.get('rounds_won')}/{r.get('rounds_lost')}"
                if r.get("rounds_won") is not None else "-"
            )
            lines.append(
                f"| {r.get('step') if r.get('step') is not None else '-':,} "
                f"| {r.get('n') or '-'} "
                f"| {f'{float(wr):.4f}' if wr is not None else '-'} "
                f"| {r.get('wins') if r.get('wins') is not None else '-'} "
                f"| {rounds} | {r.get('mode') or '-'} |"
            )
        lines.append("")
    lines += [
        "## What is here",
        "",
        "| file | what it is |",
        "|---|---|",
        "| `weights/` | the winning checkpoint |",
        "| `manifest.json` | score, provenance, policy shape, spend |",
        "| `reward_lineage.json` | root to winner, every mutation and its score |",
        "| `losers.json` | every other branch and why it stopped |",
        "| `serve.py` | minimal inference entrypoint |",
        "",
        "## The search",
        "",
        f"{branches['total']} branches, {branches['evaluated']} evaluated, "
        f"{branches['abandoned']} abandoned. Total spend ${manifest['spend_usd']:.2f}.",
        "",
        "The winning reward config:",
        "",
        "```json",
        json.dumps(win["reward_config"], indent=2),
        "```",
        "",
        "`losers.json` is part of the result, not an appendix: a search that "
        "reports only its winner cannot be told apart from a lucky guess.",
    ]
    if manifest.get("base_model"):
        lines += ["", f"Post-trained from `{manifest['base_model']}`."]
    return "\n".join(lines)


def weights_note(payload: dict[str, Any], winner: Winner, ckpt_name: str | None) -> str:
    """Say where the tensors actually are, per trainer.

    The one thing a person downloading this needs and cannot infer: whether
    `weights/*.pt` contains the model or a record pointing at it.
    """
    run_id = payload.get("run_id", "<run>")
    trainer = (payload.get("spec") or {}).get("trainer", "unknown")
    branch = winner.branch_id
    lines = [
        "# Where the weights are",
        "",
        f"Run `{run_id}`, winning branch `{branch}`, trainer `{trainer}`.",
        "",
    ]
    if trainer == "grpo_full":
        # `hf/` is the branch's LATEST export, overwritten every save, so it
        # is only the scored weights when the branch stopped improving at its
        # final step. Name the best-scoring step's directory whenever the
        # trajectory records one. run-25ed75aa0a is why: its winner measured
        # 0.736 at n=72 on step 72,000 and 0.375 on step 86,400 -- the policy
        # had collapsed into never losing a round and rarely winning one --
        # and `hf/` held the latter while the artifact advertised the former.
        if winner.step is not None:
            export = f"runs/{run_id}/{branch}/checkpoints/step-{winner.step}/"
            which = (
                f"This is the export for **step {winner.step:,}**, the step whose "
                f"evaluation produced the {winner.score:.4f} above. Do not substitute "
                f"`runs/{run_id}/{branch}/hf/`: that is whatever step the branch "
                f"wrote last, which is not generally the step that scored."
            )
        else:
            export = f"runs/{run_id}/{branch}/hf/"
            which = (
                "The winning score recorded no step, so this is the branch's last "
                "export rather than a step known to match the score above. Treat the "
                "two as unverified against each other until you re-measure."
            )
        lines += [
            f"`{ckpt_name or 'latest.pt'}` in this directory is the **record**, not the",
            "model. It carries the architecture, the action vocabulary, the prompt and",
            "the reward fingerprint, and its `policy` field is deliberately empty.",
            "",
            "The trained tensors are a full Hugging Face export of roughly 16 GiB, on",
            "the `autobranch-data` Volume:",
            "",
            "```",
            export,
            "```",
            "",
            which,
            "",
            "Fetch it with:",
            "",
            "```bash",
            f"modal volume get autobranch-data {export.rstrip('/')} ./hf",
            "```",
            "",
            "It is not copied in here because duplicating 16 GiB onto the same Volume",
            "buys nothing, and because an HF export is loadable by anything that reads",
            "`transformers` -- which a tensor dict inside a pickle is not. Load it with",
            "`autobranch.lm_policy.load_full_policy(<this checkpoint>)`, which resolves",
            "the export beside the record, or directly with `from_pretrained` on the",
            "directory above.",
        ]
    elif trainer == "grpo_lora":
        lines += [
            f"`{ckpt_name or 'latest.pt'}` carries the **adapter tensors inline**, so it",
            "is standalone: `autobranch.lm_policy.load_policy` rebuilds the adapter from",
            "this file alone and does not need the `adapter/` directory the trainer wrote.",
            "",
            "It does need the **base model** named in `arch.base_model`, which is",
            "downloaded on demand to the Volume under `models/`.",
        ]
    else:
        lines += [
            f"`{ckpt_name or 'latest.pt'}` carries the **weights inline**. It loads in",
            "plain torch with no framework and nothing else is required:",
            "",
            "```python",
            "import torch",
            f"payload = torch.load('weights/{ckpt_name or 'latest.pt'}', map_location='cpu',",
            "                     weights_only=False)",
            "payload['arch'], payload['policy'].keys()",
            "```",
        ]
    return "\n".join(lines) + "\n"


def consolidate(
    payload: dict[str, Any],
    objective,
    *,
    out_dir: str,
    fetch_checkpoint: Callable[[str, str], int] | None = None,
) -> dict[str, Any]:
    """Write the artifact tree and return an index of what was written.

    `fetch_checkpoint(src, dst) -> bytes_written` is injected so this function
    stays testable without Modal and works whether the checkpoint lives on a
    Volume, in a snapshot, or on the local disk.
    """
    winner = select_winner(payload, objective)
    manifest = build_manifest(payload, winner)

    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    files: list[dict[str, Any]] = []

    def _write(name: str, text: str) -> None:
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        files.append({"name": name, "bytes": os.path.getsize(path)})

    ckpt_name = None
    if winner.checkpoint:
        ckpt_name = os.path.basename(winner.checkpoint)
        dst = os.path.join(out_dir, "weights", ckpt_name)
        if fetch_checkpoint is not None:
            n = fetch_checkpoint(winner.checkpoint, dst)
        elif os.path.exists(winner.checkpoint):
            shutil.copyfile(winner.checkpoint, dst)
            n = os.path.getsize(dst)
        else:
            raise ConsolidationError(
                f"winning checkpoint {winner.checkpoint!r} is not reachable and no "
                "fetch_checkpoint was provided"
            )
        files.append({"name": f"weights/{ckpt_name}", "bytes": n})
    else:
        # Be explicit rather than shipping a silently weightless artifact.
        _write(
            "weights/MISSING.txt",
            "The winning branch recorded no checkpoint path. The score in "
            "manifest.json is real but the weights were not captured.\n",
        )

    # A full fine-tune's tensors are not in its checkpoint record. `grpo_full`
    # writes `"policy": {}` on purpose -- 16 GiB of tensors inside a pickle
    # would double the artifact and could not be served by anything but this
    # framework -- and keeps them in an HF export on the Volume. So an artifact
    # that copied only the record would be well-formed, look complete, and
    # contain no weights. That is the same failure as shipping MISSING.txt,
    # except it does not announce itself.
    #
    # Written unconditionally because it is always true and never wrong: for
    # `ppo` and `grpo_lora` the tensors really are inline, and saying so costs
    # a paragraph.
    _write("weights/WEIGHTS.md", weights_note(payload, winner, ckpt_name))

    _write("manifest.json", json.dumps(manifest, indent=2))
    _write("reward_lineage.json", json.dumps(reward_lineage(payload, winner.branch_id), indent=2))
    _write("losers.json", json.dumps(losers(payload, winner.branch_id, objective), indent=2))
    _write("serve.py", RUN_SNIPPET.replace("{checkpoint}", ckpt_name or "MISSING.txt"))
    _write("README.md", artifact_readme(manifest))

    return {
        "ready": True,
        "best": {"branch_id": winner.branch_id, "score": winner.score},
        "run_command": "python serve.py",
        "files": files,
        "manifest": manifest,
    }
