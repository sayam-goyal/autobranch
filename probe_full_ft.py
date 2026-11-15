"""Probe: can Qwen3-VL-8B-Instruct take a FULL-parameter update on Modal?

Every expensive mistake in this project came from designing around a primitive
that was never measured, so this measures the one that governs the whole SF3
plan before a line is built on top of it.

The question is not "does 8B fit" -- inference obviously fits. It is whether
params + grads + optimizer state + *image-token activations* fit while all 8B
parameters are trainable, because Hinh full fine-tuned (slime/Megatron, 8 GPUs,
iter_*_hf checkpoints, no LoRA anywhere) and matching that is the requirement.

The arithmetic says 8-bit Adam is the difference between fitting and not:

    bf16 params              16 GB
    bf16 grads               16 GB
    Adam fp32 m,v            64 GB   -> 96 GB, over one 80 GB H100
    Adam 8-bit m,v           16 GB   -> 48 GB, fits sharded on 2

but arithmetic is not a measurement: a VL prompt carries hundreds of image
tokens, and activation memory under those is exactly what the estimate omits.
So this runs a real forward+backward+step on a real frame-sized image and
reports peak bytes per GPU.

    modal run probe_full_ft.py

Deliberately its own app and its own volumes -- nothing here may touch the
sf3-rl apps or the rocket-league volumes.
"""

import modal

APP_NAME = "autobranch-sf3-ftprobe"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("autobranch-sf3-hf", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0",
        # Qwen3-VL ships a video-capable processor, and AutoVideoProcessor
        # imports torchvision even for a single still frame.
        "torchvision==0.23.0",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "bitsandbytes==0.48.1",
        "pillow==11.3.0",
        "numpy>=1.26",
    )
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(
    image=image,
    gpu="H100:2",
    cpu=8,
    memory=64 * 1024,
    volumes={"/hf": hf_cache},
    timeout=60 * 45,
    # No HF secret: Qwen3-VL-8B-Instruct is public, and this workspace has no
    # huggingface-secret. A gated model would need one; this one does not.
)
def probe(seq_images: int = 1) -> dict:
    """One real optimizer step on all 8B parameters, with an image in the prompt."""
    import json
    import os
    import time

    import torch

    report: dict = {"model": MODEL_ID, "gpus": torch.cuda.device_count()}
    report["gpu_name"] = torch.cuda.get_device_name(0)
    report["gpu_total_gib"] = round(
        torch.cuda.get_device_properties(0).total_memory / 2**30, 1
    )

    from transformers import AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    report["processor_s"] = round(time.time() - t0, 1)

    # Load the real architecture. device_map=None: we want it on one device so
    # the peak we measure is the honest single-shard figure, not a silently
    # offloaded one.
    t0 = time.time()
    try:
        from transformers import Qwen3VLForConditionalGeneration as Model
        report["model_class"] = "Qwen3VLForConditionalGeneration"
    except ImportError:
        from transformers import AutoModelForImageTextToText as Model
        report["model_class"] = "AutoModelForImageTextToText"

    model = Model.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    report["load_s"] = round(time.time() - t0, 1)
    n_params = sum(p.numel() for p in model.parameters())
    report["params_b"] = round(n_params / 1e9, 2)
    report["params_bf16_gib"] = round(n_params * 2 / 2**30, 1)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to("cuda:0")
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    report["trainable_b"] = round(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9, 2
    )

    # 8-bit Adam is the thing being tested: fp32 moments would need 64 GiB.
    import bitsandbytes as bnb

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-6)
    report["optimizer"] = "bnb.optim.AdamW8bit"

    # A real SF3-sized frame. 384x224 is the emulator's native resolution.
    from PIL import Image

    frame = Image.new("RGB", (384, 224), (40, 60, 90))
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image"} for _ in range(seq_images)]
            + [{"type": "text", "text": "You are Ryu. Choose one move: Jab Punch"}]
        ),
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[text], images=[frame] * seq_images, return_tensors="pt", padding=True
    ).to("cuda:0")
    report["prompt_tokens"] = int(batch["input_ids"].shape[1])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model(**batch, labels=batch["input_ids"])
    loss = out.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    report["step_s"] = round(time.time() - t0, 2)
    report["loss"] = float(loss.detach())
    report["peak_gib_cuda0"] = round(torch.cuda.max_memory_allocated(0) / 2**30, 1)
    report["reserved_gib_cuda0"] = round(torch.cuda.max_memory_reserved(0) / 2**30, 1)
    report["fits_on_one_80gb"] = report["reserved_gib_cuda0"] < 78.0

    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(seq_images: int = 1):
    import json

    print(json.dumps(probe.remote(seq_images=seq_images), indent=2))
