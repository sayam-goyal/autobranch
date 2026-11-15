"""Retention keeps the weights a result refers to.

Every export overwrites the last, so without retention the volume holds only
the newest policy -- and the newest policy is routinely not the best one. The
SF3 run measured 0.750 at step 28,800 and 0.542 at 52,800, and by the time
anyone compared them the better weights had been overwritten four times. These
tests pin the three properties that failure needs: a copy survives the next
export, pruning is ordered by step rather than mtime, and a measured
checkpoint is never pruned at all.
"""

import json

from autobranch.posttrain_full import retain_checkpoint


def _export(tmp_path, name="hf", *, shard=b"weights"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (d / "model-00001-of-00001.safetensors").write_bytes(shard)
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}})
    )
    return d


def test_retains_a_step_stamped_copy(tmp_path):
    export = _export(tmp_path)
    dest = retain_checkpoint(export, 28800, keep=3)

    assert dest is not None and dest.is_dir()
    assert dest.name == "step-28800"
    assert (dest / "config.json").is_file()
    assert (dest / "model-00001-of-00001.safetensors").read_bytes() == b"weights"


def test_copy_survives_the_next_export(tmp_path):
    """The point of the whole feature: overwriting `hf` must not erase it."""
    export = _export(tmp_path, shard=b"old")
    retain_checkpoint(export, 28800, keep=3)

    # What save_hf_export does: replace the export directory wholesale.
    import shutil

    shutil.rmtree(export)
    _export(tmp_path, shard=b"new")

    kept = tmp_path / "checkpoints" / "step-28800"
    assert kept.is_dir()
    assert (kept / "model-00001-of-00001.safetensors").read_bytes() == b"old"


def test_prunes_oldest_by_step_not_mtime(tmp_path):
    export = _export(tmp_path)
    for step in (1000, 2000, 3000, 4000):
        retain_checkpoint(export, step, keep=2)

    root = tmp_path / "checkpoints"
    kept = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert kept == ["step-3000", "step-4000"]


def test_resume_rewriting_old_steps_does_not_delete_the_newest(tmp_path):
    """A resumed run re-exports an earlier step; mtime order would lie.

    Pruning on mtime would make the just-written old checkpoint look newest
    and delete step-4000 -- the furthest-trained weights -- first.
    """
    export = _export(tmp_path)
    for step in (3000, 4000):
        retain_checkpoint(export, step, keep=2)
    retain_checkpoint(export, 1000, keep=2)  # a resume landing behind

    root = tmp_path / "checkpoints"
    kept = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert "step-4000" in kept
    assert "step-1000" not in kept


def test_a_measured_checkpoint_is_never_pruned(tmp_path):
    export = _export(tmp_path)
    retain_checkpoint(export, 28800, keep=1)
    (tmp_path / "checkpoints" / "step-28800" / "keep.marker").write_text(
        "win_rate_vs_cpu8=0.750 n=24"
    )
    for step in (40000, 52800, 60000):
        retain_checkpoint(export, step, keep=1)

    root = tmp_path / "checkpoints"
    kept = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert "step-28800" in kept, "evidence outranks recency"
    assert "step-60000" in kept


def test_retention_disabled_writes_nothing(tmp_path):
    export = _export(tmp_path)
    assert retain_checkpoint(export, 28800, keep=0) is None
    assert not (tmp_path / "checkpoints").exists()


def test_existing_checkpoint_is_not_rewritten(tmp_path):
    export = _export(tmp_path, shard=b"first")
    first = retain_checkpoint(export, 28800, keep=3)
    (first / "model-00001-of-00001.safetensors").write_bytes(b"pinned")

    again = retain_checkpoint(export, 28800, keep=3)
    assert again == first
    assert (first / "model-00001-of-00001.safetensors").read_bytes() == b"pinned"
