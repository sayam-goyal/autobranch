"""Probe: does `grpo_full` complete one real iteration on SF3?

    modal run probes/probe_sf3_train.py

The two primitives are already measured -- the model takes a full-parameter
step on one H100 (49.5 GiB, 2.93 s) and an emulator lane boots in 27 s and
steps in 0.25 s. This probe asks the question neither of those answers: does
the *loop* close? Collect decisions from real lanes, score them with a real
reward config, standardise within the group, push a clipped surrogate through
8.77B parameters, and write a checkpoint an evaluator can load.

It runs one tiny iteration, because every failure mode worth finding here is a
first-iteration failure: a shape mismatch between the processor's batch and the
action mask, an action vocabulary the tokenizer cannot build, a lane that boots
but cannot be driven from inside a container rather than from a laptop, an
export that writes but does not load. Spending four hours of H100 to discover
any of those is the expensive version of this probe.

Deliberately its own app. The volumes are the ones the real run will use, so a
success here also leaves the 16 GiB of base weights cached for it.
"""

import modal

APP_NAME = "autobranch-sf3-trainprobe"

app = modal.App(APP_NAME)

data = modal.Volume.from_name("autobranch-sf3-data", create_if_missing=True)

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
    timeout=60 * 60,
    # The trainer creates emulator sandboxes, which needs workspace credentials
    # exactly as a forking branch does.
    secrets=[modal.Secret.from_name("autobranch-modal-auth")],
)
def probe(lanes: int = 2, decisions: int = 24, subsample: int = 16) -> dict:
    import json
    import pathlib
    import sys
    import time

    sys.path.insert(0, "/bundle")

    import yaml

    from autobranch import posttrain_full
    from autobranch.spec import EnvironmentSpec

    report: dict = {}
    spec = EnvironmentSpec.parse(yaml.safe_load(pathlib.Path("/bundle/autobranch.yaml").read_text()))
    report["env"] = spec.name
    report["trainer"] = spec.trainer
    report["base_model"] = spec.policy.base_model
    report["n_actions"] = spec.policy.n_actions

    # Hinh's reward, expressed in the DSL. Used here rather than something
    # arbitrary so the probe also proves his configuration is reachable: round
    # outcome at 1.0, damage at 0.1 either way, invalid move at 20/160.
    config = {
        "weights": {
            "round_won": 1.0,
            "round_lost": -1.0,
            "damage_dealt": 0.1,
            "damage_taken": -0.1,
            "invalid_move": -0.125,
        },
        "notes": "modal-labs/sf3 constants, as the DSL expresses them",
    }
    report["reward"] = config

    workspace = "/data/probe-ws"
    t0 = time.time()
    result = posttrain_full.train(
        spec,
        config,
        steps=int(lanes * decisions),
        workspace=workspace,
        resume=False,
        seed=0,
        hyper={
            "group_size": lanes,
            "groups_per_iteration": 1,
            "max_decisions_per_match": decisions,
            "train_subsample": subsample,
            "micro_batch_size": 4,
            "durable_root": "/data/probe-durable",
        },
        search_path="/bundle",
    )
    report["train_s"] = round(time.time() - t0, 1)
    report["steps"] = result.steps_done
    report["checkpoint"] = result.checkpoint_path
    report["iterations"] = result.iterations
    report["wall_s"] = result.wall_seconds
    report["metrics"] = result.metrics

    # The export has to load back, or the artifact is decoration.
    export = pathlib.Path("/data/probe-durable") / posttrain_full.HF_EXPORT_DIRNAME
    report["hf_export_files"] = sorted(p.name for p in export.glob("*"))[:12]
    report["hf_export_ok"] = (export / "config.json").is_file()

    from autobranch import objective

    policy = objective.load_checkpoint_policy(result.checkpoint_path, device="cuda")
    report["loaded_policy"] = type(policy).__name__
    report["policy_callable"] = callable(policy)
    report["has_reseed"] = hasattr(policy, "reseed")

    print(json.dumps(report, indent=2, default=str))
    return report


@app.local_entrypoint()
def main(lanes: int = 2, decisions: int = 24, subsample: int = 16):
    import json

    print(json.dumps(probe.remote(lanes=lanes, decisions=decisions, subsample=subsample),
                     indent=2, default=str))
