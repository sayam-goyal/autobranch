"""Spend accounting that survives concurrency.

**An aggregate computed from every shard is the only authoritative total. The
total in any single shard file is not a total of anything.** Each writer --
each branch agent, each trainer -- appends to its own `ledger/<writer>.jsonl`
and never reads or rewrites another writer's file. `Ledger.entries()` reloads
the volume and merges all of them; that merge is the number a budget decision
may be made from.

This is not a stylistic preference. The predecessor project aggregated spend
in one shared file with read-modify-write, ran six concurrent writers, and
finished with 29 entries that all belonged to one of them. No error was
raised. Append-only per-writer shards cannot lose a neighbour's write because
no writer ever holds a neighbour's bytes.

Two further rules, both learned the same way:

* Record the **actual** allocated GPU. Modal honours a fallback list, so a
  branch that asked for an H100 may be billed as an A10G or vice versa.
  `detect_gpu()` asks the driver; the request is kept only as metadata.
* Reading tolerates a truncated last line. A writer killed mid-append leaves
  half a JSON object, and a reader that raises on it turns a rounding problem
  into a dead orchestrator.
"""

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Mirrors `infra.DATA_ROOT`. Duplicated deliberately: the ledger runs inside
# training containers, and importing `infra` would drag in `modal` (seconds of
# import time) for the sake of one string.
DEFAULT_ROOT = "/data"
LEDGER_DIR = "ledger"

# Published Modal rates, USD/hour.
GPU_HOURLY: dict[str, float] = {
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
}
CPU_CORE_HOURLY = 0.0472
MEMORY_GIB_HOURLY = 0.008

# Kimi K3.1 Pro on Modal shared endpoints, USD per million tokens.
LLM_IN_PER_MTOK = 2.0
LLM_OUT_PER_MTOK = 10.0

# What `nvidia-smi --query-gpu=name` prints, mapped to the rate table. The
# product names are not the names Modal's `gpu=` argument takes, which is
# exactly why the mapping has to be explicit.
GPU_ALIASES: dict[str, str] = {
    "A100": "A100-40GB",
    "A100-SXM4-40GB": "A100-40GB",
    "A100-PCIE-40GB": "A100-40GB",
    "A100-SXM4-80GB": "A100-80GB",
    "A100-80GB-PCIE": "A100-80GB",
    "H100-80GB-HBM3": "H100",
    "H100-PCIE": "H100",
    "H100-NVL": "H100",
    "H200-141GB-HBM3E": "H200",
    "H200-NVL": "H200",
    # Pre-release drivers report Blackwell as "Graphics Device". Mapping it to
    # the most expensive rate errs toward over-reporting spend, which is the
    # only direction a budget cap can tolerate being wrong in.
    "GRAPHICS-DEVICE": "B200",
}


class UnknownGPU(ValueError):
    """A GPU with no published rate.

    Raised rather than priced at zero: a silent 0.00/hr makes a budget cap
    stop working, and a cap that stops working is indistinguishable from a cap
    that is generous.
    """


def normalize_gpu(name: str | None) -> str | None:
    """Fold a driver product name or a Modal `gpu=` string to a rate key.

    Handles the `"H100:2"` count suffix by dropping it -- counts are handled
    by `hourly_rate`, which multiplies.
    """
    if name in (None, "", "none"):
        return None
    text = str(name).strip().upper()
    text = text.split(":", 1)[0]
    text = text.replace("NVIDIA", " ").replace("TESLA", " ").strip()
    text = "-".join(part for part in text.replace("_", "-").split() if part)
    text = text.strip("-")
    if text in GPU_HOURLY:
        return text
    return GPU_ALIASES.get(text, text)


def gpu_count(name: str | None) -> int:
    """`"H100:4"` -> 4. Modal expresses multi-GPU as a count suffix."""
    if not name or ":" not in str(name):
        return 1
    try:
        return max(1, int(str(name).split(":", 1)[1]))
    except ValueError:
        return 1


def hourly_rate(gpu: str | None, cpu: float, memory_mib: int) -> float:
    """USD/hour for one container at this shape.

    `gpu` may carry a count suffix (`"L4:2"`), which multiplies the GPU term.
    CPU and memory are always charged, GPU or not.
    """
    rate = float(cpu) * CPU_CORE_HOURLY + (float(memory_mib) / 1024.0) * MEMORY_GIB_HOURLY
    key = normalize_gpu(gpu)
    if key is not None:
        if key not in GPU_HOURLY:
            raise UnknownGPU(f"no published rate for GPU {gpu!r} (normalized {key!r})")
        rate += GPU_HOURLY[key] * gpu_count(gpu)
    return round(rate, 6)


def compute_usd(*, seconds: float, gpu: str | None, cpu: float, memory_mib: int) -> float:
    return round(hourly_rate(gpu, cpu, memory_mib) * float(seconds) / 3600.0, 6)


def llm_usd(tokens_in: int, tokens_out: int, *, in_per_mtok: float = LLM_IN_PER_MTOK,
            out_per_mtok: float = LLM_OUT_PER_MTOK) -> float:
    """Reasoning tokens are completion tokens and are billed as output.

    The model emits far more of them than it emits content (102 of 143 in a
    probe), so a caller that counts only visible output underestimates by
    multiples. Pass `tokens_out = completion_tokens`, not `len(content)`.
    """
    return round(
        (int(tokens_in) * in_per_mtok + int(tokens_out) * out_per_mtok) / 1_000_000.0, 6
    )


def detect_gpu(*, runner: Any = None) -> str | None:
    """Ask the driver what we actually got. None when there is no GPU.

    Modal's fallback lists mean the requested GPU and the allocated GPU
    differ often enough that billing from the request is simply wrong.
    """
    run = runner if runner is not None else subprocess.run
    try:
        proc = run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None  # no driver, no GPU -- a CPU branch, not an error
    if getattr(proc, "returncode", 1) != 0:
        return None
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    key = normalize_gpu(lines[0])
    return f"{key}:{len(lines)}" if len(lines) > 1 else key


@dataclass
class LedgerEntry:
    """One billable fact. Append-only, never amended."""

    ts: float
    run_id: str
    branch_id: str | None
    kind: str  # "compute" | "llm"
    usd: float
    seconds: float = 0.0
    gpu: str | None = None  # ACTUAL allocated GPU
    gpu_requested: str | None = None  # what we asked for, kept for the audit
    cpu: float = 0.0
    memory_mib: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    note: str = ""
    writer_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return int(self.tokens_in) + int(self.tokens_out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": float(self.ts),
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "kind": self.kind,
            "usd": float(self.usd),
            "seconds": float(self.seconds),
            "gpu": self.gpu,
            "gpu_requested": self.gpu_requested,
            "cpu": float(self.cpu),
            "memory_mib": int(self.memory_mib),
            "tokens_in": int(self.tokens_in),
            "tokens_out": int(self.tokens_out),
            "model": self.model,
            "note": self.note,
            "writer_id": self.writer_id,
            "extra": dict(self.extra),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "LedgerEntry":
        return LedgerEntry(
            ts=float(raw.get("ts", 0.0)),
            run_id=raw.get("run_id", "") or "",
            branch_id=raw.get("branch_id"),
            kind=raw.get("kind", "compute"),
            usd=float(raw.get("usd", 0.0)),
            seconds=float(raw.get("seconds", 0.0)),
            gpu=raw.get("gpu"),
            gpu_requested=raw.get("gpu_requested"),
            cpu=float(raw.get("cpu", 0.0)),
            memory_mib=int(raw.get("memory_mib", 0)),
            tokens_in=int(raw.get("tokens_in", 0)),
            tokens_out=int(raw.get("tokens_out", 0)),
            model=raw.get("model", "") or "",
            note=raw.get("note", "") or "",
            writer_id=raw.get("writer_id", "") or "",
            extra=dict(raw.get("extra") or {}),
        )


class Ledger:
    """Append-only spend log, one shard per writer.

    Construct one per writer (per branch agent, per trainer process) and let
    the default `writer_id` do its job -- two `Ledger`s sharing a shard path
    would interleave partial lines, which is the one way an append-only file
    can still lose data.
    """

    def __init__(self, run_id: str = "", *, root: str | None = None, volume: Any = None,
                 writer_id: str | None = None, branch_id: str | None = None,
                 clock: Any = time.time) -> None:
        self.run_id = run_id
        self.root = root if root is not None else DEFAULT_ROOT
        self.volume = volume
        self.branch_id = branch_id
        self.writer_id = writer_id or f"{branch_id or 'root'}-{uuid.uuid4().hex[:8]}"
        self._now = clock
        self.skipped_lines = 0  # truncated records seen on the last read

    # -- paths -----------------------------------------------------------

    @property
    def dir_path(self) -> str:
        return os.path.join(self.root, LEDGER_DIR)

    @property
    def shard_path(self) -> str:
        return os.path.join(self.dir_path, f"{self.writer_id}.jsonl")

    # -- writing ---------------------------------------------------------

    def record(self, entry: LedgerEntry) -> LedgerEntry:
        """Append one entry to this writer's shard and flush it.

        Flushing on every entry costs a syscall and buys the property that a
        sandbox killed by its own timeout still accounted for what it burned.
        """
        entry.writer_id = entry.writer_id or self.writer_id
        os.makedirs(self.dir_path, exist_ok=True)
        line = json.dumps(entry.to_dict(), sort_keys=True)
        with open(self.shard_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        return entry

    def record_compute(
        self,
        *,
        seconds: float,
        gpu: str | None = None,
        requested_gpu: str | None = None,
        cpu: float = 2.0,
        memory_mib: int = 8192,
        branch_id: str | None = None,
        run_id: str | None = None,
        note: str = "",
        detect: bool = False,
    ) -> LedgerEntry:
        """Record container time.

        `gpu` must be the ACTUAL allocated GPU. Pass `detect=True` to have it
        read from the driver, or pass the result of `detect_gpu()`. When the
        allocation differs from the request the note says so -- that mismatch
        is the difference between a $2.10/hr bill and a $3.95/hr one.
        """
        if detect and gpu is None:
            gpu = detect_gpu()
        if gpu is None and requested_gpu is not None:
            # Better to bill the request than to bill nothing, but say so.
            gpu = requested_gpu
            note = (note + " | gpu unverified: billed as requested").strip(" |")
        if requested_gpu and normalize_gpu(gpu) != normalize_gpu(requested_gpu):
            note = (note + f" | requested {requested_gpu}, allocated {gpu}").strip(" |")
        usd = compute_usd(seconds=seconds, gpu=gpu, cpu=cpu, memory_mib=memory_mib)
        return self.record(
            LedgerEntry(
                ts=self._now(),
                run_id=run_id or self.run_id,
                branch_id=branch_id if branch_id is not None else self.branch_id,
                kind="compute",
                usd=usd,
                seconds=float(seconds),
                gpu=gpu,
                gpu_requested=requested_gpu,
                cpu=float(cpu),
                memory_mib=int(memory_mib),
                note=note,
            )
        )

    def record_llm(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        model: str = "moonshotai/Kimi-K3.1-Pro",
        branch_id: str | None = None,
        run_id: str | None = None,
        in_per_mtok: float = LLM_IN_PER_MTOK,
        out_per_mtok: float = LLM_OUT_PER_MTOK,
        note: str = "",
    ) -> LedgerEntry:
        """Record agent token cost. `tokens_out` includes reasoning tokens."""
        usd = llm_usd(tokens_in, tokens_out, in_per_mtok=in_per_mtok, out_per_mtok=out_per_mtok)
        return self.record(
            LedgerEntry(
                ts=self._now(),
                run_id=run_id or self.run_id,
                branch_id=branch_id if branch_id is not None else self.branch_id,
                kind="llm",
                usd=usd,
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                model=model,
                note=note,
            )
        )

    def commit(self) -> None:
        """Push this shard to the Volume so other containers can read it.

        A no-op failure inside a Sandbox is expected and harmless: sandbox
        writes to a mounted Volume persist on their own (verified by probe --
        a file written with no commit at all was readable from outside), and
        `volume.commit()` raises there because a Sandbox is not a Function.
        Raising would make the caller think the spend was lost when it was not.
        """
        if self.volume is None:
            return
        try:
            self.volume.commit()
        except Exception:  # noqa: BLE001 - sandboxes auto-persist; see above
            pass

    # -- reading ---------------------------------------------------------

    def shards(self) -> list[str]:
        try:
            names = sorted(os.listdir(self.dir_path))
        except FileNotFoundError:
            return []
        return [os.path.join(self.dir_path, n) for n in names if n.endswith(".jsonl")]

    def entries(self, *, reload: bool = True, run_id: str | None = None) -> list[LedgerEntry]:
        """Every entry from every shard, oldest first.

        `volume.reload()` first: without it this container sees only the
        shards that existed when it started, and under-reports by exactly the
        branches that launched after it. A missed reload is invisible -- the
        number still looks like a number.

        But a **Sandbox is not a Function**, and Modal raises "reload() can
        only be called from within a running function" there. Letting that
        propagate made every branch's spend read as $0 -- the shards were being
        written correctly the whole time and the reader was throwing. So a
        failed reload degrades to "read what is already mounted" rather than
        to an exception: a slightly stale total beats no total, because the
        budget guard consumes this number.
        """
        if reload and self.volume is not None:
            try:
                self.volume.reload()
            except Exception:  # noqa: BLE001 - see above; sandboxes cannot reload
                pass
        wanted = run_id if run_id is not None else (self.run_id or None)
        out: list[LedgerEntry] = []
        skipped = 0
        for path in self.shards():
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except ValueError:
                            # A writer killed mid-append. One lost entry beats
                            # an aggregation that refuses to produce a number.
                            skipped += 1
                            continue
                        entry = LedgerEntry.from_dict(raw)
                        if wanted and entry.run_id and entry.run_id != wanted:
                            continue
                        out.append(entry)
            except FileNotFoundError:
                continue
        self.skipped_lines = skipped
        out.sort(key=lambda e: e.ts)
        return out

    def total_usd(self, *, reload: bool = True, run_id: str | None = None) -> float:
        """The authoritative total: every shard, summed at read."""
        return round(sum(e.usd for e in self.entries(reload=reload, run_id=run_id)), 6)

    def by_branch(self, *, reload: bool = True, run_id: str | None = None) -> dict[str, float]:
        """Spend per branch. Run-level entries land under `""`."""
        out: dict[str, float] = {}
        for entry in self.entries(reload=reload, run_id=run_id):
            key = entry.branch_id or ""
            out[key] = round(out.get(key, 0.0) + entry.usd, 6)
        return out

    def by_kind(self, *, reload: bool = True, run_id: str | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        for entry in self.entries(reload=reload, run_id=run_id):
            out[entry.kind] = round(out.get(entry.kind, 0.0) + entry.usd, 6)
        return out

    def tokens(self, *, reload: bool = True, run_id: str | None = None) -> int:
        return sum(e.tokens for e in self.entries(reload=reload, run_id=run_id))

    def check(self, cap: float, *, reload: bool = True, run_id: str | None = None) -> bool:
        """True while spend is within `cap`. False means stop launching."""
        return self.total_usd(reload=reload, run_id=run_id) <= float(cap)

    def remaining(self, cap: float, *, reload: bool = True,
                  run_id: str | None = None) -> float:
        """Budget left, floored at zero -- never a negative allowance."""
        return round(max(0.0, float(cap) - self.total_usd(reload=reload, run_id=run_id)), 6)

    def summary(self, *, reload: bool = True, run_id: str | None = None) -> dict[str, Any]:
        """One read, every aggregate -- so a caller does not pay for four scans."""
        entries = self.entries(reload=reload, run_id=run_id)
        by_branch: dict[str, float] = {}
        by_kind: dict[str, float] = {}
        for entry in entries:
            key = entry.branch_id or ""
            by_branch[key] = round(by_branch.get(key, 0.0) + entry.usd, 6)
            by_kind[entry.kind] = round(by_kind.get(entry.kind, 0.0) + entry.usd, 6)
        return {
            "total_usd": round(sum(e.usd for e in entries), 6),
            "entries": len(entries),
            "shards": len(self.shards()),
            "skipped_lines": self.skipped_lines,
            "tokens": sum(e.tokens for e in entries),
            "gpu_seconds": round(sum(e.seconds for e in entries if e.gpu), 3),
            "by_branch": by_branch,
            "by_kind": by_kind,
        }
