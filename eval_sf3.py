"""Score an SF3 checkpoint against the objective, independently of the agent.

    modal run eval_sf3.py --run-id run-25ed75aa0a --matches 24

Pass --export-dir to score a retained checkpoint instead of the live export,
which is the only way to measure a policy training has already moved past:

    modal run eval_sf3.py --run-id run-25ed75aa0a \
        --export-dir checkpoints/step-52800 --matches 24

The branch agent decides when to evaluate. That is the design -- the search
owns its own budget -- but it means a run can spend its whole deadline training
and produce no number, and an unmeasured policy is worth nothing. The
predecessor project put it exactly right: a measured 70% is a result, an
unmeasured 77% is not.

This reads the HF export the trainer wrote to the Volume and runs the manifest's
objective against it. It imports no reward code and takes no training state --
it is the same `objective.evaluate` the branch would call, pointed at the same
weights -- so the number is comparable to anything the agent produces later.
"""

import modal

APP_NAME = "autobranch-sf3-eval"

app = modal.App(APP_NAME)

data = modal.Volume.from_name("autobranch-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "bitsandbytes==0.48.1",
        "pillow==11.3.0",
        "numpy==2.2.6",
        "modal==1.6.4",
        "pyyaml==6.0.2",
    )
    .env({
        "HF_HOME": "/data/hf",
        "TOKENIZERS_PARALLELISM": "false",
        "AUTOBRANCH_DATA_ROOT": "/data",
        "SF3_ROM_VOLUME": "sf3-rom",
    })
    .add_local_python_source("autobranch", copy=True)
    .add_local_dir("examples/sf3", "/bundle", copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    volumes={"/data": data},
    timeout=60 * 150,
    secrets=[modal.Secret.from_name("autobranch-modal-auth")],
)
def evaluate(run_id: str, branch_id: str = "b0", matches: int = 24,
             difficulty: int = 8, mode: str = "stochastic",
             export_dir: str = "hf", concurrency: int = 0,
             seed: int = 12345) -> dict:
    import json
    import pathlib
    import sys
    import time

    sys.path.insert(0, "/bundle")

    import torch
    import yaml

    from autobranch import lm_policy
    from autobranch.env_api import import_target
    from autobranch.spec import EnvironmentSpec

    spec = EnvironmentSpec.parse(
        yaml.safe_load(pathlib.Path("/bundle/autobranch.yaml").read_text())
    )
    branch_root = pathlib.Path(f"/data/runs/{run_id}/{branch_id}")
    export = branch_root / export_dir
    if not (export / "config.json").is_file():
        return {"ok": False, "reason": f"no HF export at {export}"}

    # A retained checkpoint carries its step in its name (checkpoints/step-N),
    # and SNAPSHOT.json confirms it when one is present.
    scored_step = None
    if export.name.startswith("step-"):
        try:
            scored_step = int(export.name.split("-")[-1])
        except ValueError:
            scored_step = None
    snap = export / "SNAPSHOT.json"
    if snap.is_file():
        try:
            scored_step = json.loads(snap.read_text()).get("step", scored_step)
        except Exception:  # noqa: BLE001 - a bad marker must not block scoring
            pass

    progress = {}
    state = branch_root / "progress.json"
    if state.is_file():
        progress = json.loads(state.read_text())

    # Build the policy straight from the export. `load_full_policy` wants a
    # checkpoint record; here the export is the whole story, so the arch record
    # is reconstructed from the manifest and the vocabulary rebuilt the same
    # way training built it -- same tokenizer, same label pool, same order.
    import transformers

    processor = lm_policy.prepare_vision_processor(
        transformers.AutoProcessor.from_pretrained(str(export))
    )
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration",
                        transformers.AutoModelForImageTextToText)
    model = model_cls.from_pretrained(str(export), dtype=torch.bfloat16)
    model.eval().to("cuda")

    vocab = lm_policy.build_action_vocab(
        processor.tokenizer, int(spec.policy.n_actions or 0),
        model_name=str(spec.policy.base_model),
    )

    env_factory = import_target(spec.entrypoint, search_path="/bundle")
    probe = env_factory(n_envs=1, seed=0, difficulty=difficulty)
    action_names = tuple(getattr(probe, "action_names", ()) or ())
    probe.close()

    policy = lm_policy.VisionActionPolicy(
        model, processor, vocab, device="cuda",
        temperature=1.0, action_names=action_names,
    )
    policy.deterministic = (mode == "deterministic")
    policy.reseed(seed)

    evaluator = import_target(spec.objective.evaluator, search_path="/bundle")
    budget = dict(spec.objective.budget)
    # Lanes are a shared, finite resource: this app runs alongside whatever
    # the live branch is doing, and `concurrency = matches` meant asking for
    # one lane per match -- fine at 24, a stampede at 200 that would contend
    # with training for the same emulator pool. Sample size and parallelism
    # are now separate knobs; 0 keeps the old one-lane-per-match default.
    lanes = concurrency if concurrency > 0 else matches
    budget.update({"matches": matches, "difficulty": difficulty,
                   "concurrency": max(1, min(lanes, matches))})

    t0 = time.time()
    # The seed is a parameter because the evaluator derives every match seed
    # from it (seed + batch * 1009). Two evaluations of the same weights at the
    # same seed therefore replay the *same* matches and produce the same number
    # -- a replication, not a second sample. Varying it is the only way to take
    # independent samples of one checkpoint and pool them into a tighter
    # interval, which is what n=24 (+/-17 points) needs to become a usable
    # claim.
    raw = evaluator(policy, seeds=(seed,), budget=budget,
                    deterministic=policy.deterministic)
    out = {
        "ok": True,
        "run_id": run_id,
        "branch_id": branch_id,
        # Attribute the score to the weights that produced it. progress.json
        # tracks the LIVE export and races ahead of any retained checkpoint,
        # so trusting it would file a step-28,800 measurement under whatever
        # step training happened to reach while the match was running -- and a
        # win rate pinned to the wrong weights is worse than no win rate.
        # A step-stamped directory names its own step; only the live export
        # has to ask progress.json.
        "step": scored_step if scored_step is not None else progress.get("step"),
        "iteration": None if scored_step is not None else progress.get("iteration"),
        "export_dir": export_dir,
        "export_is_live": export_dir == "hf",
        "n": matches,
        "difficulty": difficulty,
        "concurrency": budget["concurrency"],
        "seed": seed,
        "mode": mode,
        "wall_s": round(time.time() - t0, 1),
        **(raw if isinstance(raw, dict) else {"value": float(raw)}),
    }
    print(json.dumps(out, indent=2, default=str))

    # Keep it beside the weights it describes.
    dest = pathlib.Path(f"/data/runs/{run_id}/{branch_id}/independent_eval.json")
    try:
        prior = json.loads(dest.read_text()) if dest.is_file() else []
    except Exception:  # noqa: BLE001
        prior = []
    prior.append(out)
    dest.write_text(json.dumps(prior, indent=2, default=str))
    return out


@app.local_entrypoint()
def main(run_id: str, branch_id: str = "b0", matches: int = 24,
         difficulty: int = 8, mode: str = "stochastic",
         export_dir: str = "hf", concurrency: int = 0,
         seed: int = 12345):
    import json

    print(json.dumps(
        evaluate.remote(run_id=run_id, branch_id=branch_id, matches=matches,
                        difficulty=difficulty, mode=mode,
                        export_dir=export_dir, concurrency=concurrency,
                        seed=seed),
        indent=2, default=str))
