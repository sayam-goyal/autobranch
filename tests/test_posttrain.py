"""Post-training tests: weights, masking, advantages, resume, and wordguess.

Everything in this file is CPU-only, touches no network, downloads no model,
and passes with `torch`, `transformers` and `peft` all absent -- which is not
a convenience, it is the property being tested. `autobranch/posttrain.py` is
imported by `trainer.trainer_for` on a machine that may have none of those
packages, and a module that cannot be imported there cannot produce the error
message telling the operator what to put in `image.pip`.

So the heavy parts are tested through their seams rather than through a GPU:
a fake downloader for the Volume logic, a fake tokenizer for the action
vocabulary, plain NumPy for the masking and the group-relative arithmetic.
Anything that genuinely needs torch is marked `slow` or skipped when the
import is unavailable, and the one end-to-end GRPO run additionally requires
`AUTOBRANCH_TEST_BASE_MODEL` so no test run ever pulls weights by surprise.
"""

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autobranch import posttrain
from autobranch.env_api import blocking, load_env, validate_env
from autobranch.posttrain import (
    ActionVocab,
    GRPOHyper,
    MissingDependencyError,
    PostTrainError,
    PostTrainWorkspace,
    action_labels,
    build_action_vocab,
    checkpoint_payload,
    ensure_base_model,
    escape_repo_id,
    group_advantages,
    is_model_ready,
    lora_arch,
    model_dir_for,
    record_lineage,
    restore_progress,
    restrict_logits,
    sample_action_indices,
    serialize_observation,
    zero_variance_mask,
)
from autobranch.reward import compile_reward
from autobranch.spec import EnvironmentSpec, with_base_model

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "examples" / "wordguess"
MANIFEST = BUNDLE / "autobranch.yaml"

BASE_MODEL = "Qwen/Qwen3-4B"

HEAVY = ("torch", "transformers", "peft")


def _installed(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def spec() -> EnvironmentSpec:
    return EnvironmentSpec.from_yaml(MANIFEST.read_text())


@pytest.fixture(scope="module")
def posttrain_spec(spec: EnvironmentSpec) -> EnvironmentSpec:
    """The manifest as a `must posttrain X` constraint leaves it."""
    return with_base_model(spec, BASE_MODEL)


@pytest.fixture(scope="module")
def word_env_module():
    sys.path.insert(0, str(BUNDLE))
    try:
        import word_env

        return word_env
    finally:
        sys.path.remove(str(BUNDLE))


# ---------------------------------------------------------------------------
# honest about dependencies
# ---------------------------------------------------------------------------


IMPORT_PROBE = """
import sys

BLOCKED = {"torch", "transformers", "peft", "huggingface_hub", "accelerate"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked for this test: {name}")
        return None


sys.meta_path.insert(0, Blocker())
import autobranch.posttrain as p

print(p.PROMPT_VERSION, len(p.LABEL_POOL), p.escape_repo_id("Qwen/Qwen3-4B"))
"""


def test_the_module_imports_with_none_of_the_heavy_dependencies() -> None:
    """The error message is useless if producing it requires the missing package.

    Run in a subprocess with the imports blocked at the meta-path level, which
    is stricter than "they happen to be absent here": it fails even on a
    machine where torch is installed.
    """
    out = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.split() == ["1", "62", "Qwen--Qwen3-4B"], out.stdout


def test_a_missing_dependency_names_the_manifest_key_and_the_pin() -> None:
    with pytest.raises(MissingDependencyError) as exc:
        posttrain._import_or_explain("autobranch_absent_probe_pkg", "peft>=0.11")
    message = str(exc.value)
    assert "image.pip" in message, message
    assert "peft>=0.11" in message and "torch>=2.2" in message, message
    assert "transformers>=4.44" in message, message
    assert "grpo_lora" in message


@pytest.mark.skipif(
    all(_installed(m) for m in HEAVY),
    reason="every heavy dependency is installed, so there is no missing one to report",
)
def test_train_refuses_before_loading_an_env_or_touching_the_network(
    posttrain_spec: EnvironmentSpec, tmp_path
) -> None:
    """The dependency check runs first, so nothing expensive happens first."""
    with pytest.raises(MissingDependencyError) as exc:
        posttrain.train(
            posttrain_spec,
            {"weights": {"word_solved": 1.0}},
            steps=1,
            workspace=tmp_path / "ws",
            search_path=BUNDLE,
        )
    assert "image.pip" in str(exc.value)


def test_a_spec_without_a_base_model_is_refused_with_the_constraint_that_fixes_it(
    spec: EnvironmentSpec, tmp_path
) -> None:
    with pytest.raises(PostTrainError) as exc:
        posttrain.train(
            spec, {"weights": {"word_solved": 1.0}}, steps=1, workspace=tmp_path / "ws"
        )
    assert "must posttrain" in str(exc.value)
    assert "policy.base_model" in str(exc.value)


# ---------------------------------------------------------------------------
# weights onto the Volume, exactly once
# ---------------------------------------------------------------------------


class FakeDownloader:
    """Stands in for `snapshot_download`, recording how it was called."""

    def __init__(self, *, files=("config.json", "model.safetensors"), body="weights"):
        self.calls: list[dict] = []
        self.files = files
        self.body = body

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        target = Path(kwargs["local_dir"])
        for name in self.files:
            (target / name).write_text(self.body)
        return str(target)


def test_model_dir_is_one_escaped_path_under_the_volumes_models_directory(tmp_path) -> None:
    path = model_dir_for(BASE_MODEL, data_root=tmp_path)
    assert path == tmp_path / "models" / "Qwen--Qwen3-4B"


def test_the_data_root_defaults_to_the_volume_mount(monkeypatch) -> None:
    monkeypatch.delenv("AUTOBRANCH_DATA_ROOT", raising=False)
    assert str(model_dir_for(BASE_MODEL)).startswith("/data/models/")
    monkeypatch.setenv("AUTOBRANCH_DATA_ROOT", "/tmp/elsewhere")
    assert str(model_dir_for(BASE_MODEL)).startswith("/tmp/elsewhere/models/")


@pytest.mark.parametrize(
    "repo_id", ["../../etc/passwd", "a/../../b", ".hidden/model", "weird name/v1"]
)
def test_an_escaped_repo_id_cannot_leave_the_models_directory(repo_id: str, tmp_path) -> None:
    escaped = escape_repo_id(repo_id)
    assert "/" not in escaped
    assert not escaped.startswith(".")
    assert ".." not in escaped
    path = model_dir_for(repo_id, data_root=tmp_path)
    assert path.parent == tmp_path / "models"
    assert path.resolve().parent == (tmp_path / "models").resolve()


def test_a_model_is_downloaded_once_and_then_reused(tmp_path) -> None:
    fake = FakeDownloader()
    first = ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=fake)
    second = ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=fake)
    assert first == second == str(tmp_path / "models" / "Qwen--Qwen3-4B")
    assert len(fake.calls) == 1, "the second branch re-downloaded the base model"
    assert is_model_ready(first)
    assert (Path(first) / "model.safetensors").read_text() == "weights"


def test_the_marker_records_what_was_downloaded(tmp_path) -> None:
    path = ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=FakeDownloader())
    marker = json.loads((Path(path) / posttrain.DOWNLOAD_MARKER).read_text())
    assert marker["repo_id"] == BASE_MODEL
    assert marker["files"] == 2
    assert marker["bytes"] > 0


def test_the_download_lands_in_a_scratch_directory_not_the_final_one(tmp_path) -> None:
    """There must be no moment at which the final path exists unfinished."""
    dest = model_dir_for(BASE_MODEL, data_root=tmp_path)
    seen = {}

    def downloader(**kwargs):
        seen["local_dir"] = kwargs["local_dir"]
        seen["dest_exists_mid_download"] = dest.exists()
        Path(kwargs["local_dir"], "config.json").write_text("{}")

    ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=downloader)
    assert seen["local_dir"] != str(dest)
    assert seen["dest_exists_mid_download"] is False
    assert is_model_ready(dest)


def test_a_failed_download_leaves_nothing_usable_and_nothing_behind(tmp_path) -> None:
    def broken(**kwargs):
        Path(kwargs["local_dir"], "config.json").write_text("half")
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=broken)
    dest = model_dir_for(BASE_MODEL, data_root=tmp_path)
    assert not dest.exists()
    assert not is_model_ready(dest)
    # No scratch directory left to be mistaken for a model later.
    assert list((tmp_path / "models").iterdir()) == []
    # And a retry still works.
    ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=FakeDownloader())
    assert is_model_ready(dest)


def test_an_empty_download_is_refused_rather_than_marked_complete(tmp_path) -> None:
    with pytest.raises(PostTrainError, match="no files"):
        ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=lambda **kw: None)
    assert not model_dir_for(BASE_MODEL, data_root=tmp_path).exists()


def test_an_unmarked_directory_is_refused_rather_than_deleted(tmp_path) -> None:
    dest = model_dir_for(BASE_MODEL, data_root=tmp_path)
    dest.mkdir(parents=True)
    (dest / "something.bin").write_text("not ours")
    fake = FakeDownloader()
    with pytest.raises(PostTrainError, match=posttrain.DOWNLOAD_MARKER):
        ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=fake)
    assert fake.calls == [], "it should refuse before downloading anything"
    assert (dest / "something.bin").exists(), "a trainer must not delete weights it cannot account for"


def test_a_branch_that_loses_a_race_keeps_the_winners_weights(tmp_path) -> None:
    """Two branches on one repo id: the loser discards its copy, not the winner's."""
    dest = model_dir_for(BASE_MODEL, data_root=tmp_path)

    def downloader(**kwargs):
        # While "we" download, another branch finishes and moves its own copy
        # into place -- complete, marker and all.
        dest.mkdir(parents=True)
        (dest / "model.safetensors").write_text("winner")
        (dest / posttrain.DOWNLOAD_MARKER).write_text("{}")
        Path(kwargs["local_dir"], "model.safetensors").write_text("loser")

    path = ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=downloader)
    assert path == str(dest)
    assert (dest / "model.safetensors").read_text() == "winner"
    staging = [p.name for p in (tmp_path / "models").iterdir() if p.name.startswith(".staging")]
    assert staging == [], f"the loser left scratch directories behind: {staging}"


def test_the_hf_token_is_forwarded_when_the_environment_has_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    fake = FakeDownloader()
    ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=fake)
    assert fake.calls[0]["token"] == "hf_secret"
    assert fake.calls[0]["repo_id"] == BASE_MODEL


def test_no_token_is_invented_when_the_environment_has_none(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    fake = FakeDownloader()
    ensure_base_model(BASE_MODEL, data_root=tmp_path, downloader=fake)
    assert fake.calls[0]["token"] is None


# ---------------------------------------------------------------------------
# the action vocabulary: an illegal action must be unrepresentable
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Encodes a fixed table, so the vocabulary logic needs no transformers.

    `"2"` is deliberately two tokens and `" 2"` one, which is the real
    situation on BPE vocabularies and the reason `build_action_vocab` tries the
    space-prefixed variant.
    """

    def __init__(self, table: dict[str, list[int]] | None = None):
        self.table = table or {}
        self.padding_side = "right"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False, "action labels must not gain special tokens"
        if text in self.table:
            return list(self.table[text])
        raise AssertionError(f"unexpected encode({text!r}) in this test")


def simple_tokenizer(n: int = 10) -> FakeTokenizer:
    labels = action_labels(n)
    table: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        table[label] = [100 + i]
        table[" " + label] = [200 + i]
    return FakeTokenizer(table)


def test_each_action_maps_to_exactly_one_unique_token() -> None:
    vocab = build_action_vocab(simple_tokenizer(10), 10)
    assert len(vocab) == 10
    assert vocab.labels == action_labels(10)
    assert vocab.token_ids == tuple(range(100, 110))
    assert len(set(vocab.token_ids)) == 10


def test_a_label_that_only_exists_space_prefixed_is_still_usable() -> None:
    table = {"0": [10], "1": [11, 12], " 1": [13], "2": [14], "3": [15]}
    vocab = build_action_vocab(FakeTokenizer(table), 4)
    assert vocab.labels == ("0", " 1", "2", "3")
    assert vocab.token_ids == (10, 13, 14, 15)


def test_an_unmappable_label_fails_at_startup_not_mid_run() -> None:
    table = {"0": [10], "1": [11, 12], " 1": [13, 14], "2": [15]}
    with pytest.raises(PostTrainError, match="single unused token"):
        build_action_vocab(FakeTokenizer(table), 3, model_name="fake/model")


def test_two_actions_may_not_share_a_token() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        ActionVocab(labels=("0", "1"), token_ids=(7, 7))


def test_more_actions_than_single_character_labels_is_refused() -> None:
    with pytest.raises(ValueError, match="single-character labels"):
        action_labels(len(posttrain.LABEL_POOL) + 1)


def test_restriction_keeps_exactly_the_action_columns() -> None:
    vocab = ActionVocab(labels=("0", "1", "2"), token_ids=(5, 7, 9))
    logits = np.tile(np.arange(32.0), (4, 1))
    restricted = restrict_logits(logits, vocab)
    assert restricted.shape == (4, 3)
    assert np.array_equal(restricted[0], [5.0, 7.0, 9.0])


def test_sampling_only_ever_returns_a_legal_action_index() -> None:
    """The point of masking: no draw can name a token outside the vocabulary.

    The illegal columns carry the largest logits in the whole row, so an
    unmasked policy would pick one almost every time.
    """
    vocab = ActionVocab(labels=("0", "1", "2"), token_ids=(5, 7, 9))
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(256, 32))
    illegal = [c for c in range(32) if c not in vocab.token_ids]
    logits[:, illegal] += 50.0

    actions = sample_action_indices(restrict_logits(logits, vocab), rng)
    assert actions.shape == (256,)
    assert actions.dtype == np.int64
    assert set(np.unique(actions)).issubset({0, 1, 2})
    # And all three are reachable, so this is not passing by collapsing.
    assert len(set(np.unique(actions))) == 3


def test_greedy_sampling_takes_the_best_legal_action_not_the_best_token() -> None:
    vocab = ActionVocab(labels=("0", "1", "2"), token_ids=(5, 7, 9))
    logits = np.zeros((1, 32))
    logits[0, 3] = 99.0  # an illegal token with a huge logit
    logits[0, 7] = 1.0  # the best legal one, action index 1
    action = sample_action_indices(restrict_logits(logits, vocab), np.random.default_rng(0), greedy=True)
    assert action.tolist() == [1]


def test_temperature_zero_is_refused_rather_than_dividing_by_it() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GRPOHyper.parse({"temperature": 0.0})


@pytest.mark.skipif(not _installed("torch"), reason="needs torch for the tensor path")
def test_restriction_accepts_a_torch_tensor_without_a_torch_import_here() -> None:
    """`restrict_logits` is duck-typed, which is what keeps this module importable."""
    import torch

    vocab = ActionVocab(labels=("0", "1"), token_ids=(2, 4))
    restricted = restrict_logits(torch.arange(8.0).reshape(1, 8), vocab)
    assert isinstance(restricted, np.ndarray)
    assert np.array_equal(restricted[0], [2.0, 4.0])


# ---------------------------------------------------------------------------
# group-relative advantages, including the group that teaches nothing
# ---------------------------------------------------------------------------


def test_group_advantages_are_the_group_relative_arithmetic_and_nothing_else() -> None:
    returns = np.array([[0.0, 1.0], [-1.0, 1.0], [3.0, 5.0]])
    adv, stats = group_advantages(returns, eps=0.0)
    # Every group is two samples either side of its own mean, so every
    # standardised advantage is exactly -1 and +1 regardless of scale: that
    # scale invariance is the point of dividing by the group's own std.
    for row in adv:
        assert row == pytest.approx([-1.0, 1.0])
    assert stats.n_groups == 3 and stats.group_size == 2
    assert stats.return_mean == pytest.approx(1.5)
    assert stats.zero_variance_groups == 0


def test_a_zero_variance_group_gets_zero_advantage_and_is_counted() -> None:
    """A group where every sample scores the same contains no preference.

    It must not blow up (that is what `eps` is for) and it must not be
    invisible: a run can look busy while every group is degenerate, which is
    the difference between learning slowly and not learning.
    """
    returns = np.array([[1.0, 1.0], [0.0, 4.0]])
    adv, stats = group_advantages(returns, eps=0.0)  # even with no epsilon to hide behind
    assert adv[0] == pytest.approx([0.0, 0.0])
    assert np.all(np.isfinite(adv))
    assert stats.zero_variance_groups == 1
    assert stats.zero_variance_fraction == pytest.approx(0.5)
    assert np.array_equal(zero_variance_mask(returns), [True, False])


def test_every_group_degenerate_is_reported_as_such() -> None:
    adv, stats = group_advantages(np.full((4, 4), 2.5))
    assert not adv.any()
    assert stats.zero_variance_fraction == 1.0
    assert stats.within_group_std_mean == 0.0


def test_a_group_of_one_is_refused_because_it_has_no_baseline() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        group_advantages(np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="group_size"):
        GRPOHyper.parse({"group_size": 1})


def test_returns_must_be_a_group_matrix() -> None:
    with pytest.raises(ValueError, match="n_groups, group_size"):
        group_advantages(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# prompt serialisation
# ---------------------------------------------------------------------------


def test_prompt_serialisation_is_deterministic_and_dtype_stable() -> None:
    labels = action_labels(4)
    obs64 = np.array([0.125, -1.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    obs32 = obs64.astype(np.float32)
    first = serialize_observation(obs64, labels=labels, env_name="wordguess")
    assert first == serialize_observation(obs64, labels=labels, env_name="wordguess")
    assert first == serialize_observation(obs32, labels=labels, env_name="wordguess")
    assert first != serialize_observation(obs64 + 0.5, labels=labels, env_name="wordguess")


def test_the_prompt_carries_the_state_and_the_legal_labels_only() -> None:
    labels = action_labels(3)
    prompt = serialize_observation([0.0, 1.0], labels=labels, env_name="wordguess")
    assert prompt == "env: wordguess\nstate[2]: 0.00 1.00\nactions: 0 1 2\naction:"


def test_a_negative_zero_does_not_render_as_minus_zero() -> None:
    """Two observations that are equal must not produce two different prompts."""
    prompt = serialize_observation([-1e-9, 1e-9], labels=action_labels(2), env_name="e")
    assert "-0.00" not in prompt
    assert prompt == serialize_observation([0.0, 0.0], labels=action_labels(2), env_name="e")


def test_label_whitespace_is_stripped_so_the_prompt_is_the_same_either_way() -> None:
    bare = serialize_observation([1.0], labels=("0", "1"), env_name="e")
    spaced = serialize_observation([1.0], labels=(" 0", " 1"), env_name="e")
    assert bare == spaced


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def test_a_fresh_workspace_starts_at_zero(tmp_path) -> None:
    ws = PostTrainWorkspace(tmp_path / "ws")
    progress = restore_progress(ws)
    assert (progress.steps_done, progress.iteration, progress.resumed) == (0, 0, False)
    assert ws.checkpoints.is_dir()
    assert not ws.has_adapter()


def test_resume_continues_the_step_count_from_state_json(tmp_path) -> None:
    """A child mounting its parent's workspace must not restart the count.

    `steps` is a cumulative target, so a resumed run that read 0 here would
    train the whole budget again and the branch loop's bookkeeping would be
    wrong by exactly one budget.
    """
    ws = PostTrainWorkspace(tmp_path / "ws")
    ws.write_state({
        "steps_done": 1024,
        "iterations": 7,
        "wall_seconds": 12.5,
        "reward_lineage": [{"fingerprint": "aaa", "at_step": 0, "hash": "aaa"}],
    })
    progress = restore_progress(ws, resume=True)
    assert progress.steps_done == 1024
    assert progress.iteration == 7
    assert progress.wall_seconds == pytest.approx(12.5)
    assert progress.resumed is True
    assert len(progress.reward_lineage) == 1


def test_a_fork_records_its_new_reward_as_lineage_rather_than_failing(tmp_path) -> None:
    ws = PostTrainWorkspace(tmp_path / "ws")
    ws.write_state({"steps_done": 500, "iterations": 3,
                    "reward_lineage": [{"fingerprint": "parent", "at_step": 0, "hash": "p"}]})
    progress = restore_progress(ws)
    record_lineage(progress, {"weights": {"word_solved": 1.0}}, "child")
    assert [entry["fingerprint"] for entry in progress.reward_lineage] == ["parent", "child"]
    assert progress.reward_lineage[-1]["at_step"] == 500
    # Calling again with the same reward does not grow the lineage.
    record_lineage(progress, {"weights": {"word_solved": 1.0}}, "child")
    assert len(progress.reward_lineage) == 2


def test_a_from_scratch_run_rotates_the_old_metrics_stream(tmp_path) -> None:
    ws = PostTrainWorkspace(tmp_path / "ws")
    ws.append_metrics({"step": 10})
    progress = restore_progress(ws, resume=False)
    assert progress.steps_done == 0
    assert not ws.metrics.exists()
    archived = sorted(p.name for p in ws.root.glob("metrics.*.jsonl"))
    assert len(archived) == 1, archived


def test_a_torn_state_file_is_no_state_rather_than_a_crash(tmp_path) -> None:
    ws = PostTrainWorkspace(tmp_path / "ws")
    ws.state.write_text('{"steps_done": 10')  # killed mid-write
    assert ws.read_state() == {}
    assert restore_progress(ws).steps_done == 0


def test_metrics_round_trip_through_the_workspace(tmp_path) -> None:
    ws = PostTrainWorkspace(tmp_path / "ws")
    ws.append_metrics({"step": 1, "loss": 0.5})
    ws.append_metrics({"step": 2, "loss": 0.25})
    lines = ws.read_metrics()
    assert [l["step"] for l in lines] == [1, 2]
    summary = posttrain.summarize_metrics(lines)
    assert summary["step"] == 2
    assert summary["tail_mean"]["loss"] == pytest.approx(0.375)


# ---------------------------------------------------------------------------
# the checkpoint record
# ---------------------------------------------------------------------------


def test_the_checkpoint_describes_the_policy_it_holds(posttrain_spec, tmp_path) -> None:
    """Self-describing, the same way a PPO checkpoint carries its `arch`.

    A LoRA policy is the base model plus an adapter plus an action vocabulary
    plus a prompt. Any one of those missing from the record and whoever
    rebuilds it has rebuilt a different policy.
    """
    hyp = GRPOHyper()
    vocab = build_action_vocab(simple_tokenizer(10), 10)
    arch = lora_arch(
        posttrain_spec, hyp,
        base_model=BASE_MODEL,
        base_model_path=f"/data/models/{escape_repo_id(BASE_MODEL)}",
        vocab=vocab,
        dtype="float32",
    )
    payload = checkpoint_payload(
        arch=arch,
        adapter_state={"base_model.layers.0.lora_A.weight": [0.0]},
        adapter_path=str(tmp_path / "ws" / "adapter"),
        step=256, iteration=4, hyper=hyp, spec=posttrain_spec,
        reward_fingerprint="abc123", wall_seconds=1.5,
    )

    assert payload["trainer"] == "grpo_lora"
    assert payload["arch"]["kind"] == "lm_lora_discrete"
    assert payload["arch"]["base_model"] == BASE_MODEL
    assert payload["arch"]["base_model_path"].endswith("Qwen--Qwen3-4B")
    assert payload["arch"]["adapter"] == "adapter"
    assert payload["adapter_path"].endswith("adapter")
    assert payload["arch"]["n_actions"] == 10 == posttrain_spec.policy.n_actions
    assert payload["arch"]["obs_dim"] == posttrain_spec.policy.obs_dim
    assert payload["arch"]["prompt"]["version"] == posttrain.PROMPT_VERSION
    assert payload["arch"]["prompt"]["template"] == posttrain.PROMPT_TEMPLATE
    assert ActionVocab.from_dict(payload["arch"]["action_vocab"]) == vocab
    assert payload["step"] == 256 and payload["iteration"] == 4
    assert payload["env"]["objective"] == posttrain_spec.objective.name
    assert payload["reward_fingerprint"] == "abc123"
    # The record is the trained parameters, not a pointer to them.
    assert payload["policy"] == {"base_model.layers.0.lora_A.weight": [0.0]}


def test_a_continuous_policy_is_refused_with_a_reason(posttrain_spec) -> None:
    import dataclasses

    policy = dataclasses.replace(posttrain_spec.policy, kind="continuous")
    continuous = dataclasses.replace(posttrain_spec, policy=policy)
    with pytest.raises(PostTrainError, match="discrete action vocabulary"):
        lora_arch(
            continuous, GRPOHyper(), base_model=BASE_MODEL, base_model_path="/data/x",
            vocab=build_action_vocab(simple_tokenizer(10), 10), dtype="float32",
        )


def test_hyper_rejects_unknown_keys_and_round_trips(posttrain_spec) -> None:
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        GRPOHyper.parse({"n_steps": 32})
    hyp = GRPOHyper.parse({"group_size": 8, "epochs": 3, "minibatches": 2,
                           "lora_target_modules": ["q_proj", "v_proj"]})
    assert hyp.group_size == 8
    assert hyp.lora_target_modules == ("q_proj", "v_proj")
    assert hyp.grad_steps_per_collection == 6
    as_dict = hyp.to_dict()
    assert as_dict["grad_steps_per_collection"] == 6
    assert json.loads(json.dumps(as_dict))["lora_target_modules"] == ["q_proj", "v_proj"]


# ---------------------------------------------------------------------------
# dispatch: the gap this module closes
# ---------------------------------------------------------------------------


def test_trainer_for_no_longer_raises_for_grpo_lora_in_the_source() -> None:
    """Read the dispatch without importing it, so this holds with no torch."""
    tree = ast.parse((REPO / "autobranch" / "trainer.py").read_text())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "trainer_for"
    )
    source = ast.unparse(fn)
    assert "grpo_lora" in source
    assert "posttrain" in source
    assert "NotImplementedError" not in source


@pytest.mark.skipif(not _installed("torch"), reason="autobranch.trainer imports torch")
def test_trainer_for_returns_the_posttrain_trainer(spec, posttrain_spec) -> None:
    from autobranch.trainer import train as ppo_train, trainer_for

    assert trainer_for(spec) is ppo_train
    assert posttrain_spec.trainer == "grpo_lora"
    assert posttrain_spec.policy.base_model == BASE_MODEL
    chosen = trainer_for(posttrain_spec)
    assert callable(chosen)
    assert chosen is posttrain.train


@pytest.mark.skipif(not _installed("torch"), reason="autobranch.trainer imports torch")
def test_both_trainers_take_the_same_call_the_branch_loop_makes() -> None:
    """`branch_main` calls whichever it got with the same keywords.

    So the signatures are compared directly rather than trusted: a drifting
    keyword here fails at the first `train` tool call inside a sandbox, which
    is the most expensive place to discover it.
    """
    from autobranch.trainer import train as ppo_train

    ppo = inspect.signature(ppo_train)
    grpo = inspect.signature(posttrain.train)
    assert list(ppo.parameters) == list(grpo.parameters)
    for name, param in ppo.parameters.items():
        assert grpo.parameters[name].kind == param.kind, name
    # The call `branch_main` actually makes, all keywords.
    for name in ("spec", "reward_config", "steps", "workspace", "resume"):
        assert name in grpo.parameters


# ---------------------------------------------------------------------------
# the wordguess bundle
# ---------------------------------------------------------------------------


CHEAP = {"episodes": 32, "lanes": 8, "max_steps": 12}

#: What `letter_correct` pays for: it fires on repeats, so this reward is
#: maximised by re-guessing one known-good letter forever.
NAIVE_REWARD = {"weights": {"letter_correct": 1.0}}

#: Progress rather than activity.
SENSIBLE_REWARD = {
    "weights": {
        "word_solved": 1.0,
        "letter_new": 0.3,
        "repeat_guess": -0.2,
        "guess_cost": -0.01,
    }
}


def test_the_manifest_describes_the_environment_it_ships_with(spec, word_env_module) -> None:
    assert spec.name == "wordguess"
    assert spec.policy.kind == "discrete"
    assert spec.policy.n_actions == word_env_module.N_ACTIONS == len(word_env_module.ALPHABET)
    assert spec.policy.obs_dim == word_env_module.OBS_DIM
    assert spec.objective.name == "solve_rate"
    assert set(spec.signal_names) == set(word_env_module.SIGNAL_NAMES)
    assert spec.resources.gpu is None, "the bundle must stay CPU-runnable on its own"
    # The trainer this bundle demonstrates is not in the base image, so the
    # manifest has to ask for its dependencies.
    pins = " ".join(spec.image.pip)
    for package in ("torch", "transformers", "peft", "huggingface_hub"):
        assert package in pins, f"{package} is missing from image.pip"


def test_a_must_posttrain_constraint_is_the_whole_switch(spec) -> None:
    assert spec.trainer == "ppo" and spec.policy.base_model is None
    flipped = with_base_model(spec, BASE_MODEL)
    assert flipped.trainer == "grpo_lora"
    assert flipped.policy.base_model == BASE_MODEL
    # And the original is untouched: a constraint must not mutate the manifest.
    assert spec.trainer == "ppo"


def test_validate_env_reports_no_problems(spec) -> None:
    env = load_env(spec.entrypoint, search_path=BUNDLE, n_envs=4)
    try:
        problems = validate_env(env, spec, steps=6, seed=3)
    finally:
        env.close()
    assert blocking(problems) == [], problems


def test_identical_seeds_give_identical_words(word_env_module) -> None:
    a = word_env_module.WordGuess(n_envs=6)
    b = word_env_module.WordGuess(n_envs=6)
    a.reset(seed=11)
    b.reset(seed=11)
    assert a.hidden_words() == b.hidden_words()
    b.reset(seed=12)
    assert a.hidden_words() != b.hidden_words()


def test_letter_correct_fires_on_a_repeat_which_is_the_trap(word_env_module) -> None:
    """The documented smell, asserted: this is why reward design matters here."""
    env = word_env_module.WordGuess(n_envs=2, max_steps=12, word_bank=["lane"])
    env.reset(seed=0)
    first = env.step(np.zeros(2, dtype=np.int64))  # 'a', which is in "lane"
    again = env.step(np.zeros(2, dtype=np.int64))  # the same letter

    assert first.signals["letter_correct"].tolist() == [1.0, 1.0]
    assert first.signals["letter_new"].tolist() == [1.0, 1.0]
    assert again.signals["letter_correct"].tolist() == [1.0, 1.0], "the trap is gone"
    assert again.signals["letter_new"].tolist() == [0.0, 0.0]
    assert again.signals["repeat_guess"].tolist() == [1.0, 1.0]
    assert again.signals["revealed_fraction"].tolist() == [0.25, 0.25]


def test_solving_ends_the_episode_and_the_lane_restarts(word_env_module) -> None:
    env = word_env_module.WordGuess(n_envs=2, max_steps=12, word_bank=["lane"])
    env.reset(seed=0)
    letters = [word_env_module.ALPHABET.index(ch) for ch in "lane"]
    for letter in letters[:-1]:
        result = env.step(np.full(2, letter, dtype=np.int64))
        assert not result.terminated.any()
    result = env.step(np.full(2, letters[-1], dtype=np.int64))
    assert result.terminated.all()
    assert result.signals["word_solved"].tolist() == [1.0, 1.0]
    assert result.signals["revealed_fraction"].tolist() == [1.0, 1.0]
    # Auto-reset: the post-step observation is a fresh episode, the pre-reset
    # one is published for bootstrapping.
    assert np.asarray(result.obs)[:, word_env_module.OBS_REVEALED].sum() == 0.0
    assert np.asarray(result.info["final_obs"])[:, word_env_module.OBS_REVEALED].sum() == 8.0


def test_episodes_truncate_at_the_guess_budget(word_env_module) -> None:
    env = word_env_module.WordGuess(n_envs=2, max_steps=5, word_bank=["lane"])
    env.reset(seed=0)
    wrong = word_env_module.ALPHABET.index("o")  # not in "lane"
    for _ in range(4):
        result = env.step(np.full(2, wrong, dtype=np.int64))
        assert not result.truncated.any()
    result = env.step(np.full(2, wrong, dtype=np.int64))
    assert result.truncated.all()
    assert not result.terminated.any()


def test_illegal_actions_are_refused_by_the_environment(word_env_module) -> None:
    env = word_env_module.WordGuess(n_envs=2)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="outside 0"):
        env.step(np.full(2, word_env_module.N_ACTIONS, dtype=np.int64))


def test_a_bad_word_bank_is_refused_at_construction(word_env_module) -> None:
    with pytest.raises(ValueError, match="outside ALPHABET"):
        word_env_module.WordGuess(n_envs=2, word_bank=["lamb"])
    with pytest.raises(ValueError, match="repeats a letter"):
        word_env_module.WordGuess(n_envs=2, word_bank=["sees"])
    with pytest.raises(ValueError, match="max_steps"):
        word_env_module.WordGuess(n_envs=2, max_steps=2)


def test_the_objective_separates_a_solver_from_a_sticky_policy(spec, word_env_module) -> None:
    sweep = word_env_module.evaluate(
        word_env_module.sweep_policy, seeds=(1, 2, 3, 4), budget=CHEAP
    )
    sticky = word_env_module.evaluate(
        word_env_module.sticky_policy, seeds=(1, 2, 3, 4), budget=CHEAP
    )
    assert sweep["value"] == 1.0, sweep["detail"]
    assert sticky["value"] == 0.0, sticky["detail"]
    assert sweep["n"] >= 32 and sticky["n"] >= 32
    assert spec.objective.better(sweep["value"], sticky["value"])
    assert sticky["detail"]["repeat_guess_rate"] > sweep["detail"]["repeat_guess_rate"]


def test_the_objective_is_reproducible_for_the_same_policy_and_seeds(word_env_module) -> None:
    a = word_env_module.evaluate(word_env_module.sweep_policy, seeds=(7, 8), budget=CHEAP)
    b = word_env_module.evaluate(word_env_module.sweep_policy, seeds=(7, 8), budget=CHEAP)
    assert a == b


def mean_step_reward(word_env_module, spec, reward_config, policy, *, steps: int = 24) -> float:
    """Composed reward per environment step for a scripted policy.

    This is the quantity a trainer maximises, computed with the real compiler
    so the comparison below is between rewards as the framework composes them
    rather than as a test re-implements them.
    """
    compiled = compile_reward(reward_config, spec)
    env = word_env_module.WordGuess(n_envs=8, max_steps=12)
    obs = env.reset(seed=5)
    compiled.reset_episode_state()
    total = 0.0
    for _ in range(steps):
        result = env.step(policy(obs))
        obs = result.obs
        total += float(np.sum(compiled(result.signals)))
    env.close()
    return total / (8 * steps)


def test_a_naive_reward_prefers_exactly_the_policy_the_objective_rejects(
    spec, word_env_module
) -> None:
    """The reason this bundle is worth searching over.

    `letter_correct` looks like the obvious signal to pay for and it ranks the
    sticky policy -- which solves nothing -- *above* the solver. The objective
    ranks them the other way round. That inversion is the whole argument for
    measuring branches with an immutable evaluator instead of with their own
    training reward.
    """
    sweep, sticky = word_env_module.sweep_policy, word_env_module.sticky_policy
    naive = {
        "sweep": mean_step_reward(word_env_module, spec, NAIVE_REWARD, sweep),
        "sticky": mean_step_reward(word_env_module, spec, NAIVE_REWARD, sticky),
    }
    sensible = {
        "sweep": mean_step_reward(word_env_module, spec, SENSIBLE_REWARD, sweep),
        "sticky": mean_step_reward(word_env_module, spec, SENSIBLE_REWARD, sticky),
    }
    assert naive["sticky"] > naive["sweep"], naive
    assert sensible["sweep"] > sensible["sticky"], sensible

    objective = {
        name: word_env_module.evaluate(policy, seeds=(1, 2), budget=CHEAP)["value"]
        for name, policy in (("sweep", sweep), ("sticky", sticky))
    }
    assert objective["sweep"] > objective["sticky"]


def test_the_measurement_path_cannot_reach_reward_or_training_code() -> None:
    """The bundle's evaluator must not be able to see how rewards are made.

    `tests/test_objective_independence.py` walks the import graph for the
    gridworld bundle; this is the same wall for this one, plus the trainer
    introduced here -- `posttrain` imports the reward compiler, so an evaluator
    that imported `posttrain` would quietly put reward code back on the
    measurement path.
    """
    forbidden = ("autobranch.reward", "autobranch.trainer", "autobranch.posttrain")
    code = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "sys.path.insert(0, %r);"
        "import word_env;"
        "print([m for m in %r if m in sys.modules])" % (str(BUNDLE), str(REPO), forbidden)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), check=True
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_the_prompt_a_wordguess_policy_sees_is_short_and_legible(spec, word_env_module) -> None:
    """A sanity check on the thing the LM is actually asked to read."""
    env = word_env_module.WordGuess(n_envs=2, word_bank=["lane"])
    obs = env.reset(seed=0)
    vocab = build_action_vocab(simple_tokenizer(spec.policy.n_actions), spec.policy.n_actions)
    prompt = serialize_observation(
        obs[0], labels=vocab.labels, env_name=spec.name, precision=2
    )
    assert prompt.startswith("env: wordguess\n")
    assert f"state[{spec.policy.obs_dim}]" in prompt
    assert prompt.endswith("action:")
    assert len(prompt) < 400, prompt
    assert word_env_module.render_obs(obs[0]).startswith("revealed 0/4")


# ---------------------------------------------------------------------------
# the training loop, driven by a stub language model
# ---------------------------------------------------------------------------
#
# `train` is the one part of this module that cannot be reached without torch,
# transformers and peft, and the machine this was written on has only torch.
# So the two libraries are replaced by stubs that implement exactly the surface
# `train` uses -- a tokenizer, a causal LM with a switchable "adapter", and
# four peft functions. That is enough to run masked sampling, group returns, a
# real gradient step through a real optimiser, a real checkpoint and a real
# resume. It proves the plumbing, not the learning; `AUTOBRANCH_TEST_BASE_MODEL`
# below is for the learning.


def _stub_libraries(monkeypatch, n_actions: int, vocab_size: int = 64):
    """Install stub `transformers` and `peft` modules for the duration of a test."""
    import types
    from contextlib import contextmanager

    import torch
    import torch.nn as nn

    labels = action_labels(n_actions)
    label_ids = {label: 10 + i for i, label in enumerate(labels)}

    class StubTokenizer:
        def __init__(self):
            self.padding_side = "right"
            self.pad_token = None
            self.eos_token = "<eos>"

        def encode(self, text, add_special_tokens=False):
            if text in label_ids:
                return [label_ids[text]]
            return [2, 3]  # not a single token: forces the bare-label branch

        def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True,
                     max_length=None):
            width = 6
            ids = torch.zeros((len(prompts), width), dtype=torch.long)
            for row, prompt in enumerate(prompts):
                # Left padding, like the real tokenizer is configured to use:
                # the informative ids sit at the end, where the policy reads.
                digest = [ord(c) % (vocab_size - 1) for c in prompt[-width:]]
                ids[row] = torch.as_tensor(digest[-width:], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class StubLM(nn.Module):
        """A tiny causal LM whose only trainable parameter is the "adapter"."""

        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.embed = nn.Embedding(vocab_size, 8)
            self.head = nn.Linear(8, vocab_size)
            self.adapter = nn.Parameter(torch.zeros(vocab_size))
            self.embed.requires_grad_(False)
            self.head.requires_grad_(False)
            self._adapter_on = True

        def forward(self, input_ids=None, attention_mask=None, **_ignored):
            logits = self.head(self.embed(input_ids))
            if self._adapter_on:
                logits = logits + self.adapter
            return types.SimpleNamespace(logits=logits)

        @contextmanager
        def disable_adapter(self):
            self._adapter_on = False
            try:
                yield self
            finally:
                self._adapter_on = True

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            Path(path, "adapter_config.json").write_text('{"peft_type": "LORA"}')
            torch.save({"adapter": self.adapter.detach().cpu()}, Path(path, "adapter.pt"))

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: StubTokenizer()
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: StubLM()
    )

    def from_pretrained(base, path, is_trainable=False, **_kwargs):
        loaded = torch.load(Path(path, "adapter.pt"), weights_only=False)
        with torch.no_grad():
            base.adapter.copy_(loaded["adapter"])
        return base

    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    peft.get_peft_model = lambda base, config: base
    peft.get_peft_model_state_dict = lambda model: {"adapter": model.adapter.detach().cpu()}

    def set_peft_model_state_dict(model, state_dict, *_a, **_k):
        """Apply saved adapter tensors, as the real peft does for a rebuild."""
        with torch.no_grad():
            model.adapter.copy_(torch.as_tensor(state_dict["adapter"]))
        return types.SimpleNamespace(unexpected_keys=[], missing_keys=[])

    peft.set_peft_model_state_dict = set_peft_model_state_dict
    peft.PeftModel = types.SimpleNamespace(from_pretrained=from_pretrained)

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    return transformers, peft


def _fake_volume_with_model(tmp_path, monkeypatch, repo_id: str) -> Path:
    """A data root that already holds `repo_id`, so no download is attempted."""
    monkeypatch.setenv("AUTOBRANCH_DATA_ROOT", str(tmp_path / "data"))
    local = model_dir_for(repo_id)
    local.mkdir(parents=True)
    (local / posttrain.DOWNLOAD_MARKER).write_text("{}")
    (local / "config.json").write_text("{}")
    return local


STUB_HYPER = {
    "group_size": 3,
    "episode_steps": 4,
    "epochs": 1,
    "minibatches": 2,
    "micro_batch": 4,
    "lr": 0.05,
    "device": "cpu",
    "dtype": "float32",
    "env_kwargs": {"n_envs": 2},
}


@pytest.mark.slow
@pytest.mark.skipif(not _installed("torch"), reason="the gradient step needs real torch")
def test_a_collection_trains_resumes_and_logs_without_an_illegal_action(
    spec, tmp_path, monkeypatch
) -> None:
    """The loop end to end: masked sampling, group returns, gradient, checkpoint.

    That it completes at all is the legality proof: `WordGuess.step` raises on
    an out-of-range action, so a single leaked token would fail the test
    rather than quietly costing a penalty.
    """
    _stub_libraries(monkeypatch, spec.policy.n_actions)
    _fake_volume_with_model(tmp_path, monkeypatch, "stub/model")
    posttrain_spec = with_base_model(spec, "stub/model")
    ws_path = tmp_path / "ws"

    result = posttrain.train(
        posttrain_spec, SENSIBLE_REWARD, steps=12, workspace=ws_path,
        resume=False, seed=0, hyper=STUB_HYPER, search_path=BUNDLE,
    )

    assert result.steps_done >= 12
    assert result.iterations >= 1
    assert result.reward_fingerprint
    assert Path(result.checkpoint_path).exists()
    ws = PostTrainWorkspace(ws_path)
    assert ws.has_adapter()

    state = ws.read_state()
    assert state["steps_done"] == result.steps_done
    assert state["base_model"] == "stub/model"
    assert state["rng"]["sampling_seed"] == [0, result.iterations]
    assert state["reward_lineage"][-1]["fingerprint"] == result.reward_fingerprint

    line = ws.read_metrics()[-1]
    for key in ("step", "loss", "policy_loss", "kl_to_reference", "group_return_mean",
                "group_return_std", "zero_variance_group_fraction", "within_group_std_mean",
                "reward_terms", "signal_step_means", "grad_steps"):
        assert key in line, key
    assert line["group_size"] == 3
    assert 0.0 <= line["zero_variance_group_fraction"] <= 1.0
    assert set(line["reward_terms"]) <= set(SENSIBLE_REWARD["weights"])
    assert "tokens_per_second" not in line, "throughput is never progress"
    assert "tokens_per_second" not in line["diagnostics"]
    assert line["diagnostics"]["n_actions"] == spec.policy.n_actions

    # The adapter moved: a run that reported a loss and changed nothing is the
    # failure this is here to catch.
    import torch

    trained = torch.load(result.checkpoint_path, weights_only=False)
    assert trained["arch"]["base_model"] == "stub/model"
    assert float(trained["policy"]["adapter"].abs().sum()) > 0.0

    # Resuming at an already-reached target is a no-op, and resuming at a
    # higher one continues rather than restarting.
    same = posttrain.train(
        posttrain_spec, SENSIBLE_REWARD, steps=result.steps_done, workspace=ws_path,
        resume=True, seed=0, hyper=STUB_HYPER, search_path=BUNDLE,
    )
    assert (same.steps_done, same.iterations) == (result.steps_done, result.iterations)
    further = posttrain.train(
        posttrain_spec, SENSIBLE_REWARD, steps=result.steps_done + 12, workspace=ws_path,
        resume=True, seed=0, hyper=STUB_HYPER, search_path=BUNDLE,
    )
    assert further.steps_done > result.steps_done
    assert further.iterations > result.iterations
    steps = [l["step"] for l in ws.read_metrics()]
    assert steps == sorted(steps), f"the metrics stream went backwards: {steps}"


@pytest.mark.slow
@pytest.mark.skipif(not _installed("torch"), reason="loading a checkpoint needs real torch")
def test_a_checkpoint_rebuilds_a_policy_that_only_emits_legal_actions(
    spec, tmp_path, monkeypatch
) -> None:
    """`load_policy` needs the record and the Volume -- no reward, no manifest."""
    _stub_libraries(monkeypatch, spec.policy.n_actions)
    _fake_volume_with_model(tmp_path, monkeypatch, "stub/model")
    posttrain_spec = with_base_model(spec, "stub/model")
    result = posttrain.train(
        posttrain_spec, SENSIBLE_REWARD, steps=8, workspace=tmp_path / "ws",
        resume=False, seed=1, hyper=STUB_HYPER, search_path=BUNDLE,
    )

    policy = posttrain.load_policy(result.checkpoint_path)
    env = load_env(posttrain_spec.entrypoint, search_path=BUNDLE, n_envs=4)
    try:
        obs = env.reset(seed=0)
        for _ in range(6):
            actions = np.asarray(policy(obs))
            assert actions.shape == (4,)
            assert set(np.unique(actions)).issubset(set(range(spec.policy.n_actions)))
            obs = env.step(actions).obs  # raises if an action is out of range
        policy.deterministic = False
        policy.reseed(3)
        assert set(np.unique(np.asarray(policy(obs)))).issubset(
            set(range(spec.policy.n_actions))
        )
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.skipif(not _installed("torch"), reason="needs real torch")
def test_a_checkpoint_from_a_different_base_model_is_refused(spec, tmp_path, monkeypatch) -> None:
    """Resuming someone else's adapter over the wrong base model is a silent wrong answer."""
    _stub_libraries(monkeypatch, spec.policy.n_actions)
    _fake_volume_with_model(tmp_path, monkeypatch, "stub/model")
    _fake_volume_with_model(tmp_path, monkeypatch, "other/model")
    ws_path = tmp_path / "ws"
    posttrain.train(
        with_base_model(spec, "stub/model"), SENSIBLE_REWARD, steps=6, workspace=ws_path,
        resume=False, seed=0, hyper=STUB_HYPER, search_path=BUNDLE,
    )
    with pytest.raises(PostTrainError, match="fresh workspace"):
        posttrain.train(
            with_base_model(spec, "other/model"), SENSIBLE_REWARD, steps=12,
            workspace=ws_path, resume=True, seed=0, hyper=STUB_HYPER, search_path=BUNDLE,
        )


@pytest.mark.slow
@pytest.mark.skipif(not _installed("torch"), reason="objective.py imports torch")
def test_the_bundle_scores_through_the_standard_objective_path(spec, tmp_path) -> None:
    """An untrained policy must be scoreable, and must score poorly.

    The bundle ships `trainer: ppo` so it is trainable and scoreable with no
    base model at all, and this is the honest baseline every later number
    should be read against: random guessing solves roughly a fifth of words
    inside the budget, and the achievable score is 1.0 (`sweep_policy`).
    """
    from autobranch.objective import evaluate
    from autobranch.trainer import init_checkpoint

    checkpoint = init_checkpoint(
        spec, tmp_path / "untrained.pt", seed=0, hyper={"hidden": (32, 32)}
    )
    score = evaluate(
        spec, checkpoint, search_path=BUNDLE, budget=CHEAP, deterministic=False
    )
    assert 0.05 < score.value < 0.5, f"random guessing should be weak but not zero: {score.value}"
    # An argmax policy on an untrained network is a constant, and a constant
    # guess can never reveal four distinct letters.
    assert score.detail["modes"]["deterministic"]["value"] == 0.0
    assert score.n == 32


@pytest.mark.slow
@pytest.mark.skipif(not _installed("torch"), reason="objective.py imports torch")
def test_a_lora_checkpoint_fails_loudly_in_the_mlp_loader(posttrain_spec, tmp_path) -> None:
    """The known limitation, pinned so it cannot become a silent wrong number.

    `objective.load_checkpoint_policy` rebuilds a dense MLP from `arch` and
    loads `policy` with `strict=True`. A LoRA record does not satisfy that, and
    the important property is that it *raises* rather than scoring a randomly
    initialised network that happens to have the right shape. `load_policy`
    here is the loader for this kind of checkpoint.
    """
    import torch

    from autobranch.objective import load_checkpoint_policy

    vocab = build_action_vocab(simple_tokenizer(posttrain_spec.policy.n_actions),
                               posttrain_spec.policy.n_actions)
    payload = checkpoint_payload(
        arch=lora_arch(
            posttrain_spec, GRPOHyper(), base_model=BASE_MODEL,
            base_model_path="/data/models/Qwen--Qwen3-4B", vocab=vocab, dtype="float32",
        ),
        adapter_state={"base_model.layers.0.lora_A.weight": torch.zeros(4, 4)},
        adapter_path=str(tmp_path / "adapter"),
        step=1, iteration=1, hyper=GRPOHyper(), spec=posttrain_spec,
        reward_fingerprint="x", wall_seconds=0.0,
    )
    path = tmp_path / "latest.pt"
    torch.save(payload, path)
    with pytest.raises(Exception) as exc:
        load_checkpoint_policy(path)
    assert "lora" in str(exc.value).lower() or "unexpected key" in str(exc.value).lower(), exc.value


# ---------------------------------------------------------------------------
# the real thing, opt-in
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not all(_installed(m) for m in HEAVY) or not os.environ.get("AUTOBRANCH_TEST_BASE_MODEL"),
    reason="needs torch/transformers/peft and AUTOBRANCH_TEST_BASE_MODEL (a tiny local model)",
)
def test_a_real_grpo_collection_trains_and_resumes(spec, tmp_path) -> None:
    """One end-to-end collection against a real (tiny) base model.

    Opt-in via an environment variable pointing at a small model, because a
    test that downloads a 4B checkpoint is not a test anyone will run. What it
    proves is the plumbing: masked sampling, group returns, a gradient step, a
    loadable checkpoint, and a resume that continues the step count.
    """
    base = os.environ["AUTOBRANCH_TEST_BASE_MODEL"]
    posttrain_spec = with_base_model(spec, base)
    hyper = {
        "group_size": 2, "episode_steps": 4, "epochs": 1, "minibatches": 1,
        "micro_batch": 4, "env_kwargs": {"n_envs": 2}, "device": "cpu",
        "dtype": "float32", "lora_r": 4, "lora_alpha": 8,
    }
    result = posttrain.train(
        posttrain_spec, SENSIBLE_REWARD, steps=8, workspace=tmp_path / "ws",
        resume=False, seed=0, hyper=hyper, search_path=BUNDLE,
    )
    assert result.steps_done >= 8
    assert Path(result.checkpoint_path).exists()
    ws = PostTrainWorkspace(tmp_path / "ws")
    assert ws.has_adapter()
    line = ws.read_metrics()[-1]
    for key in ("loss", "kl_to_reference", "group_return_mean", "group_return_std",
                "zero_variance_group_fraction", "reward_terms"):
        assert key in line, key
    assert "tokens_per_second" not in line

    policy = posttrain.load_policy(result.checkpoint_path)
    env = load_env(posttrain_spec.entrypoint, search_path=BUNDLE, n_envs=2)
    try:
        actions = policy(env.reset(seed=0))
        assert set(np.unique(actions)).issubset(set(range(spec.policy.n_actions)))
    finally:
        env.close()

    again = posttrain.train(
        posttrain_spec, SENSIBLE_REWARD, steps=result.steps_done, workspace=tmp_path / "ws",
        resume=True, seed=0, hyper=hyper, search_path=BUNDLE,
    )
    assert again.steps_done == result.steps_done, "a reached target must be a no-op"
    assert again.iterations == result.iterations


# ---------------------------------------------------------------------------
# the measurement path: objective.py loading what grpo_lora wrote
# ---------------------------------------------------------------------------


def _lora_checkpoint(tmp_path, n_actions=4, *, adapter_dir=None, tensors=None):
    """A minimal `lm_lora_discrete` checkpoint on disk, torch only."""
    import torch

    arch = {
        "kind": "lm_lora_discrete",
        "obs_dim": 3,
        "n_actions": n_actions,
        "act_dim": 1,
        "base_model": "stub/base",
        "base_model_path": "",
        "adapter": posttrain.ADAPTER_DIRNAME,
        "action_vocab": {"labels": list(action_labels(n_actions)),
                         "token_ids": [10 + i for i in range(n_actions)]},
        "prompt": {"version": posttrain.PROMPT_VERSION, "template": posttrain.PROMPT_TEMPLATE,
                   "precision": 2, "env_name": "stub", "labels": list(action_labels(n_actions)),
                   "max_tokens": 64},
        "dtype": "float32",
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
        "temperature": 1.0,
    }
    path = tmp_path / "checkpoints" / "latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": posttrain.CHECKPOINT_FORMAT,
        "trainer": "grpo_lora",
        "arch": arch,
        "policy": tensors if tensors is not None else {"adapter": torch.zeros(4)},
        "adapter_path": str(adapter_dir) if adapter_dir else "",
        "step": 10,
        "iteration": 1,
    }
    torch.save(payload, path)
    return path


def test_the_objective_loads_a_lora_checkpoint_without_importing_the_trainer(
    tmp_path, monkeypatch
):
    """`objective.load_checkpoint_policy` has to handle both trainers' output.

    The evaluator is handed a policy and must not know which trainer made it.
    Before this dispatch existed, a grpo_lora checkpoint reached the dense-MLP
    loader and raised -- so `grpo_lora` could train but never be scored, which
    makes the trainer unusable rather than merely limited.
    """
    from autobranch import lm_policy, objective

    _stub_libraries(monkeypatch, 4)
    local = _fake_volume_with_model(tmp_path, monkeypatch, "stub/base")
    adapter = tmp_path / "ws" / posttrain.ADAPTER_DIRNAME
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    import torch

    torch.save({"adapter": torch.zeros(64)}, adapter / "adapter.pt")

    path = _lora_checkpoint(tmp_path, 4, adapter_dir=adapter)
    # Point the record at the staged base model, as a real run would.
    payload = torch.load(path, weights_only=False)
    payload["arch"]["base_model_path"] = str(local)
    torch.save(payload, path)

    policy = objective.load_checkpoint_policy(path)
    assert isinstance(policy, lm_policy.LoraPolicy)
    # The duck type the evaluator relies on, identical to CheckpointPolicy's.
    assert callable(policy)
    assert hasattr(policy, "reseed") and hasattr(policy, "deterministic")
    action = policy(np.zeros(3, dtype=np.float64))
    assert isinstance(action, int) and 0 <= action < 4


def test_a_lora_artifact_rebuilds_its_adapter_from_the_checkpoint_alone(tmp_path, monkeypatch):
    """A downloaded artifact is one file; the adapter directory is long gone.

    `consolidate` ships the winning checkpoint and nothing else, so requiring
    `adapter_config.json` on disk would make the deliverable loadable during the
    run and never afterwards -- the same failure as checkpoints dying in an
    ephemeral `/workspace`.
    """
    from autobranch import objective

    _stub_libraries(monkeypatch, 4)
    local = _fake_volume_with_model(tmp_path, monkeypatch, "stub/base")
    import torch

    # No adapter directory anywhere: exactly what a downloaded artifact looks like.
    path = _lora_checkpoint(tmp_path, 4, adapter_dir=tmp_path / "gone",
                            tensors={"adapter": torch.zeros(64)})
    payload = torch.load(path, weights_only=False)
    payload["arch"]["base_model_path"] = str(local)
    torch.save(payload, path)

    policy = objective.load_checkpoint_policy(path)
    assert policy(np.zeros(3)) in range(4)


def test_an_adapter_whose_tensors_all_miss_is_refused_not_served(tmp_path, monkeypatch):
    """LoRA zero-inits B, so an unloaded adapter *is* the base model.

    That runs fine and scores like an untrained policy, and a reward search
    reading the number concludes the reward was bad. The wrong conclusion from a
    silent failure is worse than a crash, so this must raise.
    """
    _stub_libraries(monkeypatch, 4)
    import sys
    import types

    from autobranch import lm_policy

    peft = sys.modules["peft"]
    # A peft whose load matches nothing, which is what a layout change looks like.
    peft.set_peft_model_state_dict = lambda model, sd, *a, **k: types.SimpleNamespace(
        unexpected_keys=list(sd), missing_keys=[]
    )
    with pytest.raises(lm_policy.PostTrainError, match="none of the"):
        lm_policy._adapter_from_record(
            peft, object(), {"lora": {"r": 4, "alpha": 8}}, {"a": 1, "b": 2}
        )


def test_an_unknown_arch_kind_is_refused_rather_than_guessed(tmp_path):
    """A policy rebuilt to the wrong shape scores as a broken one.

    That reads as a bad reward rather than a bad load, which sends the search
    off after a phantom. Refusing names the real problem.
    """
    import torch

    from autobranch import objective

    path = tmp_path / "odd.pt"
    torch.save({"arch": {"kind": "diffusion_policy", "obs_dim": 3}, "policy": {}}, path)
    with pytest.raises(objective.ObjectiveError, match="arch.kind"):
        objective.load_checkpoint_policy(path)


def test_the_inference_half_imports_no_reward_code():
    """`lm_policy` exists so `objective` can import it. That is its whole job.

    Asserted here as well as in the import-graph walker because the walker
    tests `objective`'s closure: if someone moved reward composition into this
    module, this is the test that names the actual mistake.
    """
    import ast
    import pathlib

    source = pathlib.Path(lm_policy_module_path()).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {m for m in imported if "reward" in m}, imported
    assert "autobranch.posttrain" not in imported, (
        "importing posttrain would pull autobranch.reward back in transitively"
    )


def lm_policy_module_path() -> str:
    from autobranch import lm_policy

    return lm_policy.__file__


def test_a_vision_processor_is_left_padded_before_use():
    """The action head reads `logits[:, -1, :]`, so padding side is correctness.

    Under right padding that position is a PAD token for every row shorter than
    the batch's longest, and it moves as the batch changes. The bug this guards
    was caught by its own signature: a first-epoch importance ratio of 0.377
    where it must be 1.000, because collection batched 8 lanes and the update
    batched 4. The gradient looked healthy and meant nothing.
    """
    import types

    from autobranch import lm_policy

    tokenizer = types.SimpleNamespace(padding_side="right", pad_token=None, eos_token="<eos>")
    processor = types.SimpleNamespace(tokenizer=tokenizer, padding_side="right")

    out = lm_policy.prepare_vision_processor(processor)

    assert out is processor
    assert tokenizer.padding_side == "left"
    assert processor.padding_side == "left"
    # A model without a pad token borrows its eos, which is what makes a
    # ragged batch paddable at all.
    assert tokenizer.pad_token == "<eos>"


def test_a_processor_with_no_pad_and_no_eos_is_refused():
    """Better to fail naming the cause than to pad with token id 0."""
    import types

    from autobranch import lm_policy

    tokenizer = types.SimpleNamespace(padding_side="right", pad_token=None, eos_token=None)
    with pytest.raises(lm_policy.PostTrainError, match="neither a pad token nor an eos token"):
        lm_policy.prepare_vision_processor(types.SimpleNamespace(tokenizer=tokenizer))


def test_durable_progress_advances_only_with_the_export():
    """Resume state and resume weights must describe the same moment.

    `progress.json` was written every iteration while the export was written
    every `save_every`. A death on an odd iteration then resumed with
    iteration-4 weights while believing it was at iteration 5 -- training that
    much less than the step target implies, with nothing logged.

    Order is part of the fix: export first, then progress. A crash between them
    leaves progress *behind* the weights, so a resume redoes an iteration;
    the other order skips one, and redone work is recoverable in a way that
    skipped work is not.
    """
    import inspect
    import re

    from autobranch import posttrain_full

    src = inspect.getsource(posttrain_full.train)

    # The durable write must be inside the save_every branch, after the export.
    block = re.search(
        r"if hyp\.save_every and iteration % hyp\.save_every == 0:(.*?)last = line",
        src, re.S)
    assert block, "the periodic-save branch moved"
    body = block.group(1)
    assert "save_hf_export" in body
    assert "durable_state.write_text" in body
    assert body.index("save_hf_export") < body.index("durable_state.write_text"), (
        "progress must be written after the export, never before"
    )

    # And the workspace copy still advances every iteration -- it is the
    # fallback when no durable state exists at all.
    assert "state_path.write_text(json.dumps(progress))" in src
