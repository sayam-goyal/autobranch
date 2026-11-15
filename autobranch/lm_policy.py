"""Running a post-trained policy: everything the scorer needs, and no reward.

Split out of `posttrain.py` for one structural reason. `objective.py` must be
able to load and run a `grpo_lora` checkpoint, and `objective.py` may not
import reward code -- `tests/test_objective_independence.py` walks the import
graph and fails if it ever does, because an agent that can reach its own
measurement can make itself look good without getting good. `posttrain.py`
imports `autobranch.reward` (it has to; composing the reward is its job), so
`objective.py` importing `posttrain` would have torn that wall down.

Dynamic import would have "solved" it by hiding the edge from the graph
walker. That is strictly worse than the problem: the wall would still be
breached at runtime, just invisibly. So the inference half lives here instead,
and this module imports nothing that can compose a reward -- which a test
asserts directly rather than leaving to review.

What that means in practice: this module knows how to turn an observation into
a prompt, restrict logits to the action vocabulary, sample an action, and
rebuild all of that from a checkpoint record plus the base weights on the
Volume. It does not know what a reward is, what a group is, or what an
advantage is. `posttrain.py` imports and re-exports every name here, so its
public API is unchanged and `grpo_lora` callers never need to know about the
split.

The heavy dependencies -- torch, transformers, peft, huggingface_hub -- are
not in the framework's base image, so each is imported inside a function and a
missing one raises an error naming the exact manifest key to fix.
"""

import importlib
import json
import os
import shutil
import string
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# -------------------------------------------------------------------------
# dependencies, named so a missing one is actionable
# -------------------------------------------------------------------------

#: What a bundle has to declare to use this trainer, in the order they are
#: needed. The error message quotes these verbatim, so they are the answer to
#: "what do I add to the manifest", not an approximation of it.
REQUIRED_PIP = (
    ("torch", "torch>=2.2"),
    ("transformers", "transformers>=4.44"),
    ("peft", "peft>=0.11"),
    ("huggingface_hub", "huggingface_hub>=0.23"),
)


class MissingDependencyError(RuntimeError):
    """A heavy dependency this trainer needs is absent from the image."""


class PostTrainError(RuntimeError):
    """A post-training run cannot proceed (never raised for a merely bad reward)."""


def _import_or_explain(module: str, requirement: str):
    """Import `module`, or explain precisely how to make it importable.

    The message names `image.pip` and the pin to add because the alternative
    -- a bare `ModuleNotFoundError: No module named 'peft'` eight minutes into
    a sandbox launch -- tells the person reading the branch log nothing about
    where to fix it.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        wanted = ", ".join(req for _, req in REQUIRED_PIP)
        raise MissingDependencyError(
            f"trainer 'grpo_lora' needs the Python package {module!r} ({requirement}), which is "
            f"not installed in this image. Declare it in the bundle manifest under image.pip:\n\n"
            f"image:\n  pip: [{wanted}]\n\n"
            f"The framework's base image ships only pyyaml, modal and numpy, so a bundle that "
            f"post-trains a base model has to ask for the rest. (import failed: {exc})"
        ) from exc


def require_dependencies() -> dict[str, Any]:
    """Import every heavy dependency up front, or fail naming the manifest key.

    Called before the environment is loaded or a single byte is downloaded: a
    run that is going to die of a missing package should die in the first
    second, not after a 9 GB snapshot.
    """
    return {name: _import_or_explain(name, req) for name, req in REQUIRED_PIP}

# -------------------------------------------------------------------------
# base model weights on the Volume
# -------------------------------------------------------------------------

#: Mirrors `infra.DATA_ROOT`. Duplicated deliberately: this module runs inside
#: training containers and importing `infra` would drag in `modal` for the
#: sake of one string -- the same trade `ledger.py` makes.
DEFAULT_DATA_ROOT = "/data"


MODELS_DIRNAME = "models"


#: Written *inside* a model directory before it is moved into place. Its
#: presence is the only thing that makes a directory usable, which is what
#: makes an interrupted download safe.
DOWNLOAD_MARKER = ".autobranch-download-complete"


def escape_repo_id(repo_id: str) -> str:
    """`Qwen/Qwen3-4B` -> `Qwen--Qwen3-4B`, safely.

    Slashes become `--` (the convention the Hugging Face cache uses) and
    anything else outside `[A-Za-z0-9._-]` becomes `_`, so a repo id can never
    walk out of the models directory. A leading dot is rewritten too: a
    directory starting with `.` would be mistaken for one of our own
    scratch directories.
    """
    text = str(repo_id).strip().strip("/")
    if not text:
        raise ValueError("base model repo id is empty")
    safe = text.replace("/", "--")
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in safe)
    # No `..` survives anywhere in the name. A single component containing one
    # is harmless today, but "harmless as long as nobody splits this string
    # later" is not a property worth relying on for a path built from a value
    # an agent chose.
    while ".." in safe:
        safe = safe.replace("..", "__")
    while safe.startswith("."):
        safe = "_" + safe[1:]
    if safe in ("", ".", ".."):
        raise ValueError(f"base model repo id {repo_id!r} does not escape to a usable directory name")
    return safe


def default_data_root() -> str:
    """Where the `autobranch-data` Volume is mounted, locally overridable.

    `AUTOBRANCH_DATA_ROOT` is the same switch `server.py` and `run_local.py`
    use, so a local run keeps its downloads out of `/data`.
    """
    return os.environ.get("AUTOBRANCH_DATA_ROOT", DEFAULT_DATA_ROOT)


def model_dir_for(repo_id: str, *, data_root: str | os.PathLike | None = None) -> Path:
    """The one path a given base model lives at, per data root."""
    root = Path(str(data_root) if data_root is not None else default_data_root())
    return root / MODELS_DIRNAME / escape_repo_id(repo_id)


def is_model_ready(path: str | os.PathLike) -> bool:
    """True only if the completion marker is present.

    Directory existence is not the test. The marker is written inside the
    staging directory *before* the move, so a directory that exists without
    one was never finished by us.
    """
    return (Path(path) / DOWNLOAD_MARKER).is_file()


def _hf_token() -> str | None:
    """`HF_TOKEN` from the environment, if a gated repo needs one."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _default_downloader(**kwargs: Any) -> Any:
    """`huggingface_hub.snapshot_download`, imported only when actually used."""
    hub = _import_or_explain("huggingface_hub", "huggingface_hub>=0.23")
    return hub.snapshot_download(**kwargs)


def _best_effort_commit() -> None:
    """Push the Volume so sibling containers see the weights.

    Best effort on purpose: a Sandbox is not a Function, so `volume.commit()`
    raises there even though sandbox writes to a mounted Volume persist on
    their own (verified in `ledger.py` the hard way). Raising would make the
    caller think a download that is sitting safely on the Volume was lost.
    """
    try:
        from autobranch import infra

        infra.volume.commit()
    except Exception:  # noqa: BLE001 - see above; also covers "modal not installed"
        pass


def ensure_base_model(
    repo_id: str,
    *,
    data_root: str | os.PathLike | None = None,
    revision: str | None = None,
    downloader: Any = None,
) -> str:
    """Download `repo_id` into the data Volume once; return the local path.

    The guarantees, in the order they matter:

    1. **A half-finished download is never usable.** Files land in a scratch
       directory; the completion marker is written inside it; only then is the
       directory moved into place with `os.replace`. There is no window in
       which the final path exists without a marker.
    2. **Two branches racing on the same repo cannot corrupt it.** Each uses
       its own scratch directory. The loser's `os.replace` fails because the
       destination is a non-empty directory, it sees the winner's marker, and
       discards its own copy.
    3. **A directory without a marker is refused, not deleted.** We only ever
       create the final path atomically, so an unmarked one is someone else's
       -- deleting a 9 GB directory on a guess is not this function's call.

    `downloader` is an injection seam for tests (it receives `repo_id`,
    `local_dir`, `revision` and `token`); production uses
    `huggingface_hub.snapshot_download`.
    """
    dest = model_dir_for(repo_id, data_root=data_root)
    if is_model_ready(dest):
        return str(dest)
    if dest.exists() and any(dest.iterdir()):
        raise PostTrainError(
            f"{dest} exists but has no {DOWNLOAD_MARKER} marker, so it is not a model this "
            "framework finished downloading. Remove it by hand if it is junk; a trainer will not "
            "delete weights it cannot account for."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".staging-{dest.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    fetch = downloader if downloader is not None else _default_downloader
    try:
        fetch(
            repo_id=str(repo_id),
            local_dir=str(staging),
            revision=revision,
            token=_hf_token(),
        )
        files = sorted(p.name for p in staging.iterdir())
        if not files:
            raise PostTrainError(
                f"downloading {repo_id!r} produced no files in {staging}; refusing to mark it complete"
            )
        payload = {
            "repo_id": str(repo_id),
            "revision": revision,
            "completed_at": time.time(),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()),
        }
        (staging / DOWNLOAD_MARKER).write_text(json.dumps(payload, indent=2, sort_keys=True))
        try:
            os.replace(staging, dest)
        except OSError:
            # Someone else got there first. Their copy is marked, so it is as
            # good as ours; ours is the one that goes in the bin.
            if is_model_ready(dest):
                shutil.rmtree(staging, ignore_errors=True)
                return str(dest)
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if data_root is None or str(data_root) == default_data_root():
        # Only the real Volume mount is worth a commit. A test or a local run
        # pointing somewhere else should not reach for `modal` at all.
        _best_effort_commit()
    return str(dest)

# -------------------------------------------------------------------------
# the action vocabulary -- the reason an illegal action cannot happen
# -------------------------------------------------------------------------

#: One character per action. Digits and letters only: on every tokenizer we
#: care about these are single tokens, which is the whole requirement -- an
#: action label that tokenises to two pieces cannot be sampled in one step.
LABEL_POOL = string.digits + string.ascii_uppercase + string.ascii_lowercase


@dataclass(frozen=True)
class ActionVocab:
    """A bijection between action indices and single tokens.

    `labels[i]` is what action `i` looks like in the prompt and
    `token_ids[i]` is the one token the model must emit for it. Sampling
    happens over these `len(labels)` logits and nothing else, so every sampled
    index is a legal action by construction -- there is no decode step that
    can fail and therefore no invalid-action penalty anywhere in this
    framework.
    """

    labels: tuple[str, ...]
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.token_ids):
            raise ValueError("labels and token_ids must be the same length")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError(
                f"action token ids {list(self.token_ids)} contain duplicates; two actions sharing "
                "a token would make the policy unable to distinguish them"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict[str, Any]:
        return {"labels": list(self.labels), "token_ids": [int(i) for i in self.token_ids]}

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ActionVocab":
        return ActionVocab(
            labels=tuple(str(x) for x in raw["labels"]),
            token_ids=tuple(int(x) for x in raw["token_ids"]),
        )


def action_labels(n_actions: int) -> tuple[str, ...]:
    """One single-character label per action, stable across runs."""
    n = int(n_actions)
    if n < 2:
        raise ValueError(f"need at least 2 actions, got {n}")
    if n > len(LABEL_POOL):
        raise ValueError(
            f"{n} actions exceeds the {len(LABEL_POOL)} single-character labels available; a "
            "multi-character label does not tokenise to one token and so cannot be sampled in "
            "one step"
        )
    return tuple(LABEL_POOL[:n])


def build_action_vocab(tokenizer: Any, n_actions: int, *, model_name: str = "") -> ActionVocab:
    """Map each action index to exactly one token id of `tokenizer`.

    A label is accepted only if it encodes to a single token. The leading-space
    variant is tried as well because many BPE vocabularies hold ` A` as one
    token and `A` as one token with a different id -- either is fine, two
    tokens is not. Anything unmappable raises here, at startup, rather than
    producing a policy that cannot express half its action space.

    `tokenizer` only has to provide `encode(text, add_special_tokens=False)`,
    which is what lets this be tested without `transformers` installed.
    """
    labels = action_labels(n_actions)
    chosen_labels: list[str] = []
    chosen_ids: list[int] = []
    for label in labels:
        token_id = None
        used = label
        for candidate in (label, " " + label):
            ids = list(tokenizer.encode(candidate, add_special_tokens=False))
            if len(ids) == 1 and int(ids[0]) not in chosen_ids:
                token_id, used = int(ids[0]), candidate
                break
        if token_id is None:
            raise PostTrainError(
                f"tokenizer for {model_name or 'this base model'} has no single unused token for "
                f"action label {label!r}; the action vocabulary cannot be built, so this base "
                "model cannot drive a discrete policy of this size"
            )
        chosen_labels.append(used)
        chosen_ids.append(token_id)
    return ActionVocab(labels=tuple(chosen_labels), token_ids=tuple(chosen_ids))


def _to_numpy(values: Any) -> np.ndarray:
    """Accept a torch tensor or anything array-like, without importing torch."""
    if hasattr(values, "detach"):
        values = values.detach().to("cpu").float().numpy()
    return np.asarray(values, dtype=np.float64)


def restrict_logits(logits: Any, vocab: ActionVocab) -> np.ndarray:
    """Keep only the action vocabulary's columns of a logits row (or batch).

    This single line is the masking: after it, the distribution is over
    actions, the sampled index *is* the action, and no token outside the
    vocabulary can be produced. Everything else in this module assumes it has
    happened.

    The gather runs before the conversion to NumPy -- on a torch tensor that
    keeps it on the model's device, and a full vocabulary row is ~150k floats
    per lane that nobody needs a copy of.
    """
    return _to_numpy(logits[..., list(vocab.token_ids)])


def log_softmax(values: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    arr = np.asarray(values, dtype=np.float64) / float(temperature)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def sample_action_indices(
    action_logits: Any,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    greedy: bool = False,
) -> np.ndarray:
    """Sample one action index per row of already-restricted logits.

    Returns indices in `[0, n_actions)` -- the restriction happened in
    `restrict_logits`, so this cannot return anything else even if the raw
    vocabulary's largest logit sits on some unrelated token. Sampling uses an
    explicit `Generator` rather than a global so a collection is reproducible
    from `(seed, iteration)` alone.
    """
    logits = np.atleast_2d(np.asarray(action_logits, dtype=np.float64))
    if greedy:
        return logits.argmax(axis=-1).astype(np.int64)
    probs = np.exp(log_softmax(logits, temperature=temperature))
    cdf = np.cumsum(probs, axis=-1)
    draws = rng.random(size=(logits.shape[0], 1))
    idx = (draws > cdf).sum(axis=-1)
    # A float cumsum can land a hair under 1.0; clamping keeps the result a
    # legal index instead of one past the end.
    return np.clip(idx, 0, logits.shape[-1] - 1).astype(np.int64)

# -------------------------------------------------------------------------
# observations as prompts -- one function, on purpose
# -------------------------------------------------------------------------

#: Bumped whenever `serialize_observation` changes shape. It travels in the
#: checkpoint so a policy is always rebuilt behind the prompt it was trained
#: behind -- a silently reworded prompt is a silently different policy.
PROMPT_VERSION = 1


#: The prompt, in one place. Stored in the checkpoint's `arch` so an evaluator
#: can reproduce it without importing this module.
PROMPT_TEMPLATE = "env: {env}\nstate[{n}]: {values}\nactions: {labels}\naction:"


def serialize_observation(
    obs: Any,
    *,
    labels: Sequence[str],
    env_name: str = "env",
    precision: int = 2,
) -> str:
    """Render one observation vector as the prompt the policy sees.

    Kept as a single function with a version constant because it is the
    policy's entire view of the world: if the trainer and an evaluator disagree
    about the prompt by one space, the evaluated policy is not the trained one
    and the score is quietly wrong. `PROMPT_VERSION` and this template travel
    inside the checkpoint for exactly that reason.

    The format is env agnostic -- numbers and the legal labels, nothing about
    what the numbers mean. That is deliberate: a per-environment prompt would
    be a second place where environment semantics live, and the framework's
    whole bet is that the environment's semantics stay inside the environment.

    Rounding is explicit so the string is stable across float32 and float64
    observations of the same episode; `+ 0.0` normalises `-0.00` to `0.00`.
    """
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    p = int(precision)
    values = " ".join(f"{round(float(v), p) + 0.0:.{p}f}" for v in arr)
    return PROMPT_TEMPLATE.format(
        env=env_name, n=arr.size, values=values, labels=" ".join(str(l).strip() for l in labels)
    )

# -------------------------------------------------------------------------
# torch plumbing, all behind lazy imports
# -------------------------------------------------------------------------

def _resolve_device(torch: Any, requested: str) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype(torch: Any, requested: str, device: str) -> Any:
    """bf16 on a GPU, fp32 on a CPU, unless the manifest insists otherwise.

    CPU bf16 matmuls are emulated and slower than fp32, and fp32 on a GPU
    wastes memory that LoRA post-training would rather spend on batch size --
    so "auto" means two different things and says so.
    """
    if requested in (None, "", "auto"):
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    dtype = getattr(torch, str(requested), None)
    if dtype is None:
        raise PostTrainError(f"unknown dtype {requested!r}; expected e.g. float32, bfloat16")
    return dtype


def _prepare_tokenizer(tokenizer: Any) -> None:
    """Left padding, and a pad token if the model shipped without one.

    Left padding is not cosmetic: the policy reads the logits at the *last*
    position, and with right padding that position is a pad token for every
    row shorter than the longest one in the batch -- the actions would be
    sampled from padding.
    """
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise PostTrainError(
                "this tokenizer has neither a pad token nor an eos token, so a batch of prompts "
                "of different lengths cannot be padded; pick a base model whose tokenizer has one"
            )
        tokenizer.pad_token = tokenizer.eos_token


def _tokenize(tokenizer: Any, prompts: Sequence[str], *, device: str, max_tokens: int) -> dict:
    batch = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_tokens),
    )
    return {k: v.to(device) for k, v in batch.items()}


def _last_token_logits(
    model: Any, tokenizer: Any, prompts: Sequence[str], *, device: str, max_tokens: int
) -> Any:
    inputs = _tokenize(tokenizer, prompts, device=device, max_tokens=max_tokens)
    return model(**inputs).logits[:, -1, :]

# -------------------------------------------------------------------------
# the policy an evaluator runs
# -------------------------------------------------------------------------

CHECKPOINT_FORMAT = 1


ADAPTER_DIRNAME = "adapter"


class LoraPolicy:
    """A post-training checkpoint turned into `obs -> action`, and nothing else.

    The same duck type `objective.CheckpointPolicy` presents -- callable,
    `reseed`, `deterministic` -- so an environment's evaluator does not care
    which trainer produced the thing it is scoring. It carries no reward, no
    optimiser and no training state: the only thing an evaluator can do with
    one is run it.
    """

    def __init__(self, arch: dict[str, Any], *, model: Any, tokenizer: Any, device: str = "cpu"):
        self.arch = dict(arch)
        self.vocab = ActionVocab.from_dict(arch["action_vocab"])
        self.prompt = dict(arch.get("prompt") or {})
        self.env_name = str(self.prompt.get("env_name", "env"))
        self.precision = int(self.prompt.get("precision", 2))
        self.temperature = float(arch.get("temperature", 1.0))
        self.device = device
        self._model = model
        self._tokenizer = tokenizer
        self.deterministic = True
        self._rng = np.random.default_rng(0)
        if int(self.prompt.get("version", PROMPT_VERSION)) != PROMPT_VERSION:
            raise PostTrainError(
                f"checkpoint was trained with prompt version {self.prompt.get('version')} but this "
                f"code serialises version {PROMPT_VERSION}; scoring it would measure a different "
                "policy than the one that was trained"
            )

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    def prompts_for(self, obs: Any) -> list[str]:
        arr = np.asarray(obs, dtype=np.float64)
        rows = arr.reshape(1, -1) if arr.ndim == 1 else arr
        return [
            serialize_observation(
                row, labels=self.vocab.labels, env_name=self.env_name, precision=self.precision
            )
            for row in rows
        ]

    def __call__(self, obs: Any) -> Any:
        torch = _import_or_explain("torch", "torch>=2.2")
        arr = np.asarray(obs, dtype=np.float64)
        single = arr.ndim == 1
        prompts = self.prompts_for(arr)
        with torch.no_grad():
            logits = _last_token_logits(
                self._model,
                self._tokenizer,
                prompts,
                device=self.device,
                max_tokens=int(self.prompt.get("max_tokens", 1024)),
            )
        action_logits = restrict_logits(logits, self.vocab)
        actions = sample_action_indices(
            action_logits, self._rng, temperature=self.temperature, greedy=self.deterministic
        )
        return int(actions[0]) if single else actions


def _adapter_from_record(peft: Any, base: Any, arch: dict[str, Any], tensors: dict[str, Any]) -> Any:
    """Rebuild the LoRA adapter from the checkpoint alone, no directory needed.

    This is what makes a consolidated artifact usable. `consolidate` ships one
    file -- the winning checkpoint -- so the `adapter/` directory the trainer
    wrote stays behind in a sandbox that no longer exists. Requiring that
    directory would mean the framework's deliverable could be scored during the
    run and never again, which is the same class of bug as checkpoints dying in
    `/workspace`.

    It works because the record is complete: `arch.lora` carries the exact
    config the trainer used and `payload["policy"]` carries the trained
    tensors. The config is rebuilt with the same keyword shape `train` passes to
    `LoraConfig`, so the module layout matches and the keys land.

    The failure this guards is specific and silent. LoRA initialises its `B`
    matrices to zero, so an adapter whose weights do not load is not a broken
    model -- it is exactly the base model, which runs fine and scores like an
    untrained policy. A reward search reading that number concludes the reward
    was bad. So a load that applies no tensors raises here instead.
    """
    if not tensors:
        raise PostTrainError(
            "checkpoint carries no adapter tensors under 'policy', and no adapter directory was "
            "found either; there is nothing to load the trained parameters from"
        )
    lora = dict(arch.get("lora") or {})
    kwargs: dict[str, Any] = {
        "r": int(lora.get("r", 16)),
        "lora_alpha": int(lora.get("alpha", 32)),
        "lora_dropout": float(lora.get("dropout", 0.0)),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if lora.get("target_modules"):
        kwargs["target_modules"] = list(lora["target_modules"])
    model = peft.get_peft_model(base, peft.LoraConfig(**kwargs))

    # Resolved by name rather than assumed: it is public API in every peft we
    # support, but a bare AttributeError eight minutes into a scoring pass
    # would say nothing about which dependency moved.
    apply_state = getattr(peft, "set_peft_model_state_dict", None)
    if apply_state is None:
        raise PostTrainError(
            "this peft build has no set_peft_model_state_dict, so adapter tensors cannot be "
            "loaded from a checkpoint record; pin peft>=0.11 in the bundle's image.pip"
        )
    result = apply_state(model, dict(tensors))
    unexpected = list(getattr(result, "unexpected_keys", None) or [])
    # Every key being unexpected means none were applied -- see the docstring:
    # that serves the bare base model under the trained policy's score.
    if unexpected and len(unexpected) >= len(tensors):
        raise PostTrainError(
            f"none of the {len(tensors)} adapter tensors in the checkpoint matched the rebuilt "
            f"LoRA layout (unexpected keys: {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}). "
            "Serving this would run the unmodified base model while reporting the trained "
            "policy's score, so it is refused."
        )
    return model


def load_policy(
    checkpoint_path: str | os.PathLike,
    *,
    device: str = "cpu",
    data_root: str | os.PathLike | None = None,
) -> LoraPolicy:
    """Rebuild `obs -> action` from a checkpoint and the Volume, nothing else.

    Everything it needs is in the record: the base model repo id (resolved
    against the Volume, or re-downloaded if this container has never seen it),
    the LoRA weights, the action vocabulary and the prompt. No reward config,
    no hyperparameters, no manifest.

    The adapter is taken from its directory when that directory is still there
    -- the normal case mid-run, and the cheaper path -- and rebuilt from the
    checkpoint record when it is not, which is the normal case for a downloaded
    artifact. Either way the policy is the same one; only the source of the
    tensors differs.
    """
    torch = _import_or_explain("torch", "torch>=2.2")
    transformers = _import_or_explain("transformers", "transformers>=4.44")
    peft = _import_or_explain("peft", "peft>=0.11")

    path = Path(checkpoint_path)
    if not path.exists():
        raise PostTrainError(f"no checkpoint at {path}")
    # weights_only=False because our own checkpoints carry RNG state tuples;
    # these files are written by the trainer in the same workspace.
    payload = torch.load(path, map_location=device, weights_only=False)
    arch = (payload or {}).get("arch") or {}
    if arch.get("kind") != "lm_lora_discrete":
        raise PostTrainError(
            f"{path} is not a grpo_lora checkpoint (arch.kind={arch.get('kind')!r}); "
            "use autobranch.objective.load_checkpoint_policy for PPO checkpoints"
        )

    base_path = arch.get("base_model_path")
    if not base_path or not is_model_ready(base_path):
        base_path = ensure_base_model(arch["base_model"], data_root=data_root)
    adapter = payload.get("adapter_path") or str(
        path.parent.parent / arch.get("adapter", ADAPTER_DIRNAME)
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_path)
    _prepare_tokenizer(tokenizer)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=_torch_dtype(torch, arch.get("dtype", "float32"), device)
    )
    if Path(adapter, "adapter_config.json").is_file():
        model = peft.PeftModel.from_pretrained(base, adapter)
    else:
        model = _adapter_from_record(peft, base, arch, (payload or {}).get("policy") or {})
    model.eval().to(device)
    return LoraPolicy(arch, model=model, tokenizer=tokenizer, device=device)


# -------------------------------------------------------------------------
# the vision policy an evaluator runs
# -------------------------------------------------------------------------

class VisionActionPolicy:
    """`obs -> action` for a VL base model, over a fixed action vocabulary.

    Presents the duck type `objective.load_checkpoint_policy` hands an
    evaluator -- callable, `reseed`, `deterministic` -- so the evaluator never
    learns which trainer produced it.
    """

    def __init__(self, model, processor, vocab: ActionVocab, *, device: str = "cuda",
                 temperature: float = 1.0, action_names: Sequence[str] = ()) -> None:
        self.model = model
        self.processor = processor
        self.vocab = vocab
        self.device = device
        self.temperature = float(temperature)
        self.action_names = tuple(action_names)
        self.deterministic = False
        self._rng = np.random.default_rng(0)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    def prompts(self, obs: dict) -> tuple[list[str], list[Any]]:
        """Build one chat prompt per lane, with the frame as an image part."""
        frames = list(obs["frames"])
        texts = list(obs["text"])
        from PIL import Image

        images = [
            f if hasattr(f, "size") else Image.fromarray(np.asarray(f, dtype=np.uint8))
            for f in frames
        ]
        built = []
        for text in texts:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }]
            built.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return built, images

    def logits(self, obs: dict):
        """Action logits for every lane: one forward, last position, masked."""
        torch = _import_or_explain("torch", "torch>=2.2")
        texts, images = self.prompts(obs)
        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(self.device)
        out = self.model(**batch)
        return restrict_logits(out.logits[:, -1, :], self.vocab), texts, images

    def __call__(self, obs: dict):
        torch = _import_or_explain("torch", "torch>=2.2")
        with torch.no_grad():
            action_logits, _texts, _images = self.logits(obs)
        return sample_action_indices(
            action_logits, self._rng,
            temperature=self.temperature, greedy=self.deterministic,
        )


def load_full_policy(
    checkpoint_path: str | os.PathLike,
    *,
    device: str = "cuda",
) -> VisionActionPolicy:
    """Rebuild a full-parameter vision policy from its checkpoint record.

    `grpo_full` writes two things side by side: a small `.pt` holding the
    self-describing `arch` record, and an `hf/` directory holding the actual
    weights in Hugging Face format. The weights are not inlined in the `.pt`
    on purpose -- 16 GiB of tensors inside a pickle would double the checkpoint
    and could not be served by anything except this framework, whereas the HF
    export is the same shape the reproduction target published and serves.

    So this reads the record, finds the export (beside the checkpoint first,
    then the absolute path recorded at save time, because an artifact gets
    moved and a path baked in during training is the first thing to go stale),
    and rebuilds `obs -> action` with the action vocabulary the record carries.
    No reward config, no optimizer, no manifest.
    """
    torch = _import_or_explain("torch", "torch>=2.2")
    transformers = _import_or_explain("transformers", "transformers>=4.57")
    _import_or_explain("PIL", "pillow>=10")

    path = Path(checkpoint_path)
    if not path.exists():
        raise PostTrainError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = (payload or {}).get("arch") or {}
    if arch.get("kind") != "lm_full_discrete":
        raise PostTrainError(
            f"{path} is not a grpo_full checkpoint (arch.kind={arch.get('kind')!r})"
        )

    candidates = [
        path.parent.parent / str(arch.get("hf_export") or "hf"),
        path.parent / str(arch.get("hf_export") or "hf"),
        Path(str(payload.get("hf_export_path") or "")),
    ]
    export = next((c for c in candidates if c and (c / "config.json").is_file()), None)
    if export is None:
        raise PostTrainError(
            f"checkpoint {path} records an HF export at {arch.get('hf_export')!r} but no "
            f"config.json was found at any of {[str(c) for c in candidates if c]}; the "
            "record and the weights have been separated"
        )

    processor = transformers.AutoProcessor.from_pretrained(str(export))
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = transformers.AutoModelForImageTextToText
    dtype = getattr(torch, str(arch.get("dtype", "bfloat16")), torch.bfloat16)
    model = model_cls.from_pretrained(str(export), dtype=dtype)
    model.eval().to(device)

    return VisionActionPolicy(
        model,
        processor,
        ActionVocab.from_dict(arch["action_vocab"]),
        device=device,
        temperature=float(arch.get("temperature", 1.0)),
        action_names=tuple((arch.get("prompt") or {}).get("labels") or ()),
    )
