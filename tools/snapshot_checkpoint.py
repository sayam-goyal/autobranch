"""Copy a branch's live HF export aside before the trainer overwrites it.

The trainer writes every export to one path, runs/<run>/<branch>/hf, so each
save destroys the previous one. That already cost this project the step-28,800
weights -- the only checkpoint that ever measured 0.750 -- and there is nothing
left to re-evaluate. This preserves the current export under a step-stamped
path so the number we quote stays attached to weights that still exist.

Read-only with respect to everything the live run touches: it reads hf/ and
writes to a sibling directory the trainer never opens.

    modal run tools/snapshot_checkpoint.py --run-id run-25ed75aa0a
"""

import modal

app = modal.App("sf3-ckpt-snapshot")

data = modal.Volume.from_name("autobranch-data", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/data": data}, timeout=60 * 45,
              cpu=4, memory=8192)
def snapshot(run_id: str, branch_id: str = "b0") -> dict:
    import json
    import pathlib
    import shutil

    root = pathlib.Path(f"/data/runs/{run_id}/{branch_id}")
    src = root / "hf"
    progress = json.loads((root / "progress.json").read_text())
    step = progress.get("step")
    iteration = progress.get("iteration")

    dest = root / "checkpoints" / f"step-{step}"
    if dest.is_dir() and (dest / "model.safetensors.index.json").is_file():
        return {"ok": True, "already": True, "dest": str(dest), "step": step}

    # Stage under a temp name so an interrupted copy can never be mistaken for
    # a complete checkpoint by whatever reads this directory later.
    staging = root / "checkpoints" / f".staging-step-{step}"
    if staging.is_dir():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.is_file())
    staging.mkdir(parents=True)
    total = 0
    for p in files:
        shutil.copy2(p, staging / p.name)
        total += p.stat().st_size

    # The index names every shard the weights need; if one is missing the
    # snapshot is useless and should not be published under a real name.
    index = json.loads((staging / "model.safetensors.index.json").read_text())
    needed = set(index["weight_map"].values())
    missing = sorted(n for n in needed if not (staging / n).is_file())
    if missing:
        shutil.rmtree(staging)
        return {"ok": False, "error": f"incomplete shards: {missing}"}

    staging.rename(dest)
    (dest / "SNAPSHOT.json").write_text(json.dumps({
        "run_id": run_id, "branch_id": branch_id,
        "step": step, "iteration": iteration,
        "source": str(src), "files": len(files),
        "bytes": total,
    }, indent=2))

    data.commit()
    return {"ok": True, "dest": str(dest), "step": step,
            "iteration": iteration, "files": len(files),
            "gib": round(total / 2**30, 2)}


@app.local_entrypoint()
def main(run_id: str, branch_id: str = "b0"):
    import json
    print(json.dumps(snapshot.remote(run_id=run_id, branch_id=branch_id),
                     indent=2))
