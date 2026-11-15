"""Ledger tests: rate arithmetic, and aggregation under concurrent writers.

The aggregation test writes from six threads at once, each with its own
shard, and asserts that the merged total is the sum of everything written.
That is the property the predecessor project did not have.
"""

import json
import threading

import pytest

from autobranch.ledger import (
    CPU_CORE_HOURLY,
    GPU_HOURLY,
    MEMORY_GIB_HOURLY,
    Ledger,
    LedgerEntry,
    UnknownGPU,
    compute_usd,
    detect_gpu,
    gpu_count,
    hourly_rate,
    llm_usd,
    normalize_gpu,
)


# ---- rates --------------------------------------------------------------


def test_hourly_rate_cpu_only():
    # 2 cores at 0.0472 + 8 GiB at 0.008
    assert hourly_rate(None, 2.0, 8192) == pytest.approx(2 * 0.0472 + 8 * 0.008)
    assert hourly_rate(None, 2.0, 8192) == pytest.approx(0.1584)


def test_hourly_rate_adds_the_gpu():
    rate = hourly_rate("H100", 4.0, 16384)
    assert rate == pytest.approx(3.95 + 4 * CPU_CORE_HOURLY + 16 * MEMORY_GIB_HOURLY)
    assert rate == pytest.approx(4.2668)


def test_hourly_rate_multiplies_by_gpu_count():
    one = hourly_rate("L4", 1.0, 1024)
    two = hourly_rate("L4:2", 1.0, 1024)
    assert two - one == pytest.approx(GPU_HOURLY["L4"])
    assert gpu_count("L4:2") == 2
    assert gpu_count("L4") == 1


def test_published_rates_are_the_published_rates():
    assert GPU_HOURLY == {
        "L4": 0.80,
        "A10G": 1.10,
        "L40S": 1.95,
        "A100-40GB": 2.10,
        "A100-80GB": 2.50,
        "H100": 3.95,
        "H200": 4.54,
        "B200": 6.25,
    }
    assert (CPU_CORE_HOURLY, MEMORY_GIB_HOURLY) == (0.0472, 0.008)


def test_driver_product_names_fold_onto_rate_keys():
    assert normalize_gpu("NVIDIA A100-SXM4-40GB") == "A100-40GB"
    assert normalize_gpu("NVIDIA H100 80GB HBM3") == "H100"
    assert normalize_gpu("NVIDIA L4") == "L4"
    assert normalize_gpu("a100") == "A100-40GB"
    assert normalize_gpu(None) is None
    assert normalize_gpu("none") is None
    assert hourly_rate("NVIDIA A100-SXM4-40GB", 0, 0) == pytest.approx(2.10)


def test_unknown_gpu_raises_rather_than_pricing_at_zero():
    """A silent 0.00/hr is a budget cap that has quietly stopped working."""
    with pytest.raises(UnknownGPU):
        hourly_rate("RTX-4090", 2.0, 8192)


def test_compute_usd_is_rate_times_hours():
    assert compute_usd(seconds=3600, gpu="H100", cpu=4.0, memory_mib=16384) == pytest.approx(
        hourly_rate("H100", 4.0, 16384)
    )
    assert compute_usd(seconds=1800, gpu="L4", cpu=2.0, memory_mib=8192) == pytest.approx(
        hourly_rate("L4", 2.0, 8192) / 2
    )
    assert compute_usd(seconds=0, gpu="H100", cpu=4.0, memory_mib=16384) == 0.0


def test_llm_usd_prices_input_and_output_separately():
    assert llm_usd(1_000_000, 0) == pytest.approx(2.0)
    assert llm_usd(0, 1_000_000) == pytest.approx(10.0)
    assert llm_usd(500_000, 200_000) == pytest.approx(1.0 + 2.0)


def test_detect_gpu_reads_the_driver_not_the_request():
    class Proc:
        returncode = 0
        stdout = "NVIDIA A10G\n"

    assert detect_gpu(runner=lambda *a, **k: Proc()) == "A10G"

    class Multi:
        returncode = 0
        stdout = "NVIDIA H100 80GB HBM3\nNVIDIA H100 80GB HBM3\n"

    assert detect_gpu(runner=lambda *a, **k: Multi()) == "H100:2"


def test_detect_gpu_returns_none_without_a_driver():
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert detect_gpu(runner=boom) is None


# ---- recording ----------------------------------------------------------


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger("r-1", root=str(tmp_path), writer_id="w-1", branch_id="b0")


def test_record_compute_bills_the_allocated_gpu_not_the_requested_one(ledger):
    entry = ledger.record_compute(
        seconds=3600, gpu="A10G", requested_gpu="H100", cpu=2.0, memory_mib=8192
    )

    assert entry.gpu == "A10G"
    assert entry.gpu_requested == "H100"
    assert entry.usd == pytest.approx(hourly_rate("A10G", 2.0, 8192))
    assert "requested H100, allocated A10G" in entry.note
    assert ledger.total_usd() == pytest.approx(entry.usd)


def test_record_llm_counts_reasoning_tokens_as_output(ledger):
    entry = ledger.record_llm(tokens_in=10_000, tokens_out=143)

    assert entry.kind == "llm"
    assert entry.tokens == 10_143
    assert entry.usd == pytest.approx(llm_usd(10_000, 143))


def test_shard_is_append_only_jsonl(ledger):
    ledger.record_compute(seconds=10, cpu=1.0, memory_mib=1024)
    ledger.record_llm(tokens_in=1, tokens_out=1)

    lines = [json.loads(ln) for ln in open(ledger.shard_path) if ln.strip()]
    assert [raw["kind"] for raw in lines] == ["compute", "llm"]
    assert all(raw["writer_id"] == "w-1" for raw in lines)
    assert ledger.shard_path.endswith("ledger/w-1.jsonl")


def test_check_and_remaining(ledger):
    ledger.record_compute(seconds=3600, gpu="L4", cpu=1.0, memory_mib=1024)
    total = ledger.total_usd()

    assert ledger.check(total + 0.01)
    assert ledger.check(total)
    assert not ledger.check(total - 0.01)
    assert ledger.remaining(total + 1.0) == pytest.approx(1.0)
    assert ledger.remaining(0.0) == 0.0, "an exhausted budget is zero, never negative"


def test_entries_of_another_run_are_excluded(tmp_path):
    a = Ledger("r-a", root=str(tmp_path), writer_id="wa")
    b = Ledger("r-b", root=str(tmp_path), writer_id="wb")
    a.record_compute(seconds=3600, gpu="L4", cpu=1.0, memory_mib=1024)
    b.record_compute(seconds=3600, gpu="H100", cpu=1.0, memory_mib=1024)

    assert a.total_usd() == pytest.approx(hourly_rate("L4", 1.0, 1024))
    assert b.total_usd() == pytest.approx(hourly_rate("H100", 1.0, 1024))
    reader = Ledger(root=str(tmp_path), writer_id="reader")
    assert reader.total_usd() == pytest.approx(a.total_usd() + b.total_usd())


def test_a_truncated_line_is_skipped_not_fatal(ledger):
    """A writer killed mid-append must not take the aggregation down with it."""
    ledger.record_compute(seconds=3600, gpu="L4", cpu=1.0, memory_mib=1024)
    good = ledger.total_usd()
    with open(ledger.shard_path, "a") as fh:
        fh.write('{"ts": 1.0, "usd": 0.5, "kin')  # the process died here

    assert ledger.total_usd() == pytest.approx(good)
    assert ledger.skipped_lines == 1


def test_volume_is_reloaded_before_aggregation(tmp_path):
    """Without the reload, a container reports only the shards it started with."""

    class FakeVolume:
        def __init__(self):
            self.reloads = 0
            self.commits = 0

        def reload(self):
            self.reloads += 1

        def commit(self):
            self.commits += 1

    volume = FakeVolume()
    led = Ledger("r-1", root=str(tmp_path), volume=volume, writer_id="w")
    led.record_compute(seconds=1, cpu=1.0, memory_mib=1024)
    led.commit()

    led.total_usd()
    assert volume.reloads == 1
    assert volume.commits == 1
    led.total_usd(reload=False)
    assert volume.reloads == 1


# ---- the concurrency property ------------------------------------------


def test_shards_aggregate_across_concurrent_writers(tmp_path):
    """Six writers, one shard each, all appending at once: nothing is lost."""
    writers = [f"b0.{i}" for i in range(1, 7)]
    per_writer = 25
    start = threading.Barrier(len(writers))
    errors: list[BaseException] = []

    def work(branch_id: str) -> None:
        led = Ledger("r-1", root=str(tmp_path), writer_id=branch_id, branch_id=branch_id)
        try:
            start.wait(timeout=5)
            for _ in range(per_writer):
                led.record_compute(seconds=36, gpu="L4", cpu=2.0, memory_mib=8192)
                led.record_llm(tokens_in=1000, tokens_out=500)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(b,)) for b in writers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert errors == []

    reader = Ledger("r-1", root=str(tmp_path), writer_id="reader")
    entries = reader.entries()
    summary = reader.summary()

    expected_compute = compute_usd(seconds=36, gpu="L4", cpu=2.0, memory_mib=8192)
    expected_llm = llm_usd(1000, 500)
    expected_total = len(writers) * per_writer * (expected_compute + expected_llm)

    assert len(entries) == len(writers) * per_writer * 2
    assert summary["entries"] == len(entries)
    assert summary["shards"] == len(writers)  # the reader's own shard is empty
    assert summary["skipped_lines"] == 0
    assert reader.total_usd() == pytest.approx(expected_total)
    assert summary["total_usd"] == pytest.approx(expected_total)
    assert summary["tokens"] == len(writers) * per_writer * 1500

    by_branch = reader.by_branch()
    assert set(by_branch) == set(writers)
    for branch_id in writers:
        assert by_branch[branch_id] == pytest.approx(
            per_writer * (expected_compute + expected_llm)
        )
    assert sum(by_branch.values()) == pytest.approx(expected_total)
    assert reader.by_kind()["llm"] == pytest.approx(len(writers) * per_writer * expected_llm)


def test_entries_are_ordered_by_timestamp_across_shards(tmp_path):
    def led(writer, ticks):
        it = iter(ticks)
        return Ledger("r-1", root=str(tmp_path), writer_id=writer, clock=lambda: next(it))

    a = led("wa", [1.0, 3.0, 5.0])
    b = led("wb", [2.0, 4.0, 6.0])
    for _ in range(3):
        a.record_llm(tokens_in=1, tokens_out=1)
        b.record_llm(tokens_in=1, tokens_out=1)

    reader = Ledger("r-1", root=str(tmp_path), writer_id="reader")
    assert [e.ts for e in reader.entries()] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_entry_roundtrips_through_json():
    entry = LedgerEntry(
        ts=1.0, run_id="r", branch_id="b0.1", kind="compute", usd=0.5, seconds=10.0,
        gpu="L4", gpu_requested="H100", cpu=2.0, memory_mib=8192, note="n",
    )
    assert LedgerEntry.from_dict(json.loads(json.dumps(entry.to_dict()))) == entry


def test_default_writer_ids_do_not_collide():
    a = Ledger("r", root="/tmp", branch_id="b0.1")
    b = Ledger("r", root="/tmp", branch_id="b0.1")
    assert a.writer_id != b.writer_id, "two writers must never share a shard"


# ---- sandboxes are not Functions ------------------------------------

class _SandboxVolume:
    """A Volume as seen from a Sandbox: writes persist, commit/reload raise.

    Verified against real Modal: a file written inside a sandbox with no
    commit at all was readable from outside, while `volume.commit()` and
    `volume.reload()` both raise "can only be called from within a running
    function" because a Sandbox is not a Function.
    """

    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1
        raise RuntimeError("commit() can only be called from within a running function")

    def reload(self):
        self.reloads += 1
        raise RuntimeError("reload() can only be called from within a running function")


def test_spend_still_reads_inside_a_sandbox(tmp_path):
    """The bug this guards: a raising reload made every branch report $0 while
    the shards were being written correctly, so the budget guard had nothing to
    enforce against."""
    vol = _SandboxVolume()
    led = Ledger("run-sb", root=str(tmp_path), volume=vol, branch_id="b0")

    led.record_compute(seconds=60.0, gpu=None, cpu=2.0, memory_mib=8192, branch_id="b0")
    led.commit()                      # raises internally, must not propagate
    led.record_llm(tokens_in=1893, tokens_out=267, branch_id="b0")
    led.commit()

    total = led.total_usd()           # reload raises internally, must not propagate
    assert total > 0.0, "spend read as zero despite entries being written"
    assert vol.commits == 2 and vol.reloads >= 1, "the failing calls were still attempted"

    by_branch = led.by_branch()
    assert by_branch["b0"] == pytest.approx(total)
    assert led.tokens() == 1893 + 267


def test_a_failed_commit_does_not_lose_the_entry(tmp_path):
    vol = _SandboxVolume()
    led = Ledger("run-sb", root=str(tmp_path), volume=vol, branch_id="b0")
    led.record_compute(seconds=30.0, cpu=2.0, memory_mib=8192, branch_id="b0")
    led.commit()
    # A fresh reader over the same directory sees it on disk.
    reader = Ledger("run-sb", root=str(tmp_path), volume=None)
    assert len(reader.entries()) == 1
