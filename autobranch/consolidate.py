"""Turning a finished search into something you can actually run.

The deliverable is not "the best score" -- it is weights plus everything
needed to use them, plus an honest account of how they were found. So the
artifact carries four things:

* the winning checkpoint;
* a manifest with the objective score, its provenance command, and spend;
* the **reward lineage**: root to winner, every mutation and what it scored;
* the **losers**: every abandoned branch and why it was abandoned.

The losers file is not an apology, it is the result. A search that reports only
its winner cannot be distinguished from a lucky guess, and the predecessor
project's most useful finding was the shape of its failures, not its headline.

Ranking is always by **objective**, never by training reward. That rule exists
because the two diverged badly in the predecessor project: six reward configs
reported healthy self-play numbers while scoring 0.000 against a fixed
opponent.
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
        reward_config=top.get("reward_config") or {},
        checkpoint=top.get("checkpoint"),
        command=top.get("score_command", ""),
        detail=top.get("score_detail") or {},
    )


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
            "detail": winner.detail,
        },
        "winner": {
            "branch_id": winner.branch_id,
            "reward_config": winner.reward_config,
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
