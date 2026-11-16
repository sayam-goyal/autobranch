"""Probe: with gradient checkpointing on, which parameters actually get gradients?

    modal run probes/probe_gradflow.py

The live SF3 branch logs this on every run:

    torch/utils/checkpoint.py:85: UserWarning: None of the inputs have
    requires_grad=True. Gradients will be None

Reentrant checkpointing (`use_reentrant=True`, still the default) only tracks
the *tensor inputs* of a checkpointed block, not the parameters used inside it.
If no input tensor requires grad, the block's backward is never invoked and
every parameter inside it gets **no gradient** -- silently. The language stack
escapes this because its input comes from an embedding whose weight requires
grad; a vision tower fed `pixel_values` does not.

grad_norm is 65-166 on the live run, so *something* is training. That is exactly
why this needs measuring rather than reasoning: a partially-trained model
produces a perfectly healthy-looking curve, and the part that cannot learn here
is the part that looks at the game.

This counts parameters with and without gradients, split by tower, under both
checkpointing modes.
"""

import modal

app = modal.App("autobranch-gradflow-probe")

data = modal.Volume.from_name("autobranch-sf3-hf", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.1",
        "accelerate==1.10.1", "bitsandbytes==0.48.1", "pillow==11.3.0",
        "numpy>=1.26",
    )
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _measure(use_reentrant: bool) -> dict:
    """One forward+backward; report which parameters received a gradient."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration as Model
    except ImportError:
        from transformers import AutoModelForImageTextToText as Model

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Model.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    model.config.use_cache = False
    model.to("cuda").train()
    for p in model.parameters():
        p.requires_grad_(True)

    frame = Image.new("RGB", (384, 224), (40, 60, 90))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "You are Ryu. Choose one move."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[frame], return_tensors="pt").to("cuda")

    out = model(**batch)
    # A loss over the action logits, as the trainer's surrogate is.
    loss = out.logits[:, -1, :].float().log_softmax(-1)[:, :40].mean()
    loss.backward()

    def bucket(name: str) -> str:
        low = name.lower()
        if "visual" in low or "vision" in low or "image" in low:
            return "vision"
        return "language"

    stats: dict = {}
    for name, p in model.named_parameters():
        b = bucket(name)
        s = stats.setdefault(b, {"with_grad": 0, "no_grad": 0,
                                 "params_with_grad": 0, "params_no_grad": 0})
        if p.grad is None:
            s["no_grad"] += 1
            s["params_no_grad"] += p.numel()
        else:
            s["with_grad"] += 1
            s["params_with_grad"] += p.numel()

    for b, s in stats.items():
        total = s["params_with_grad"] + s["params_no_grad"]
        s["pct_params_trained"] = round(100.0 * s["params_with_grad"] / max(1, total), 2)
        s["params_with_grad_b"] = round(s["params_with_grad"] / 1e9, 3)
        s["params_no_grad_b"] = round(s["params_no_grad"] / 1e9, 3)
    del model
    torch.cuda.empty_cache()
    return stats


@app.function(image=image, gpu="H100", cpu=8, memory=65536,
              volumes={"/hf": data}, timeout=60 * 40)
def probe() -> dict:
    import json

    report = {}
    for reentrant in (True, False):
        key = "use_reentrant=True (the default, what the run used)" if reentrant \
            else "use_reentrant=False"
        report[key] = _measure(reentrant)
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(probe.remote(), indent=2))
