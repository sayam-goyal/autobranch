"""Score an SF3 checkpoint against the objective, independently of the agent.

    modal run eval_sf3.py --run-id run-25ed75aa0a --matches 24

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
    timeout=60 * 90,
    secrets=[modal.Secret.from_name("autobranch-modal-auth")],
)
def evaluate(run_id: str, branch_id: str = "b0", matches: int = 24,
             difficulty: int = 8, mode: str = "stochastic",
             concurrency: int = 0, seeds: str = "") -> dict:
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
    export = pathlib.Path(f"/data/runs/{run_id}/{branch_id}/hf")
    if not (export / "config.json").is_file():
        return {"ok": False, "reason": f"no HF export at {export}"}

    progress = {}
    state = pathlib.Path(f"/data/runs/{run_id}/{branch_id}/progress.json")
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
    policy.reseed(12345)

    evaluator = import_target(spec.objective.evaluator, search_path="/bundle")
    budget = dict(spec.objective.budget)
    # Concurrency is not just a throughput knob: the evaluator runs one batch
    # per `concurrency` matches and derives that batch's seed from the batch
    # index, so concurrency == matches collapses the whole sample onto a
    # single seed. Raising `matches` with it pinned would then buy more
    # matches from one emulator draw rather than a wider sample, and
    # re-running would reproduce the same matches instead of adding
    # independent ones. Default to the 24 lanes this hardware is known to
    # sustain and let the batch loop vary the seed.
    lanes = int(concurrency) if concurrency else min(24, matches)
    budget.update({"matches": matches, "difficulty": difficulty,
                   "concurrency": max(1, lanes)})

    # A bank of unrelated base seeds, not one. The evaluator offsets a single
    # base seed by the batch index, so one seed makes every batch a near
    # neighbour of the same emulator draw; a bank makes each batch an
    # independent family. The n=24 scores this run searched on came from one
    # family and did not replicate at larger n, which is the whole reason to
    # widen the draw rather than just add matches to it.
    bank = tuple(int(s) for s in str(seeds).split(",") if s.strip()) if seeds else (
        12345, 4242, 777, 20260115, 31337, 8675309, 1618033, 2718281,
    )
    t0 = time.time()
    raw = evaluator(policy, seeds=bank, budget=budget,
                    deterministic=policy.deterministic)
    out = {
        "ok": True,
        "run_id": run_id,
        "branch_id": branch_id,
        "step": progress.get("step"),
        "iteration": progress.get("iteration"),
        "mode": mode,
        # Provenance, so the number is readable without reconstructing the
        # command that produced it. `matches` and `seed_bank` are the two
        # things that decide how much the number is worth.
        "export_dir": str(export),
        "matches_requested": int(matches),
        "concurrency": int(budget["concurrency"]),
        "seed_bank": list(bank),
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
         difficulty: int = 8, mode: str = "stochastic", concurrency: int = 0,
         seeds: str = ""):
    import json

    print(json.dumps(
        evaluate.remote(run_id=run_id, branch_id=branch_id, matches=matches,
                        difficulty=difficulty, mode=mode, concurrency=concurrency,
                        seeds=seeds),
        indent=2, default=str))
