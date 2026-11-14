"""Git-worktree semantics on Modal Sandboxes.

The primitive pair, verified empirically before any of this was written:

    snap  = parent.snapshot_directory("/workspace")   # parent.poll() is None
    child = Sandbox.create(...)
    child.mount_image("/workspace", snap)             # child sees parent state
    # child writes and parent writes are isolated, copy-on-write

That is a worktree: the child inherits weights, optimizer state, curriculum
and history, and neither side can see the other's subsequent writes.

`snapshot_filesystem()` is the wrong primitive here -- it terminates the
sandbox it snapshots, which would mean a fork costs the parent its run.
`snapshot_directory()` does not. Directory snapshots are *not* named objects,
so the only durable reference to one is its `object_id`; it must be written to
the registry at the moment it is taken or the branch becomes unreachable.

Three guards here are paid-for lessons rather than defensive habits:

1. `fork` always mounts a snapshot. It never points a second sandbox at a
   directory a live sandbox is writing. Two writers on one workspace corrupt
   the artifact silently -- there is no error, just wrong weights -- so the
   only way to be safe is to make sharing unexpressible in the API.
2. Forking off a terminated parent requires an explicit snapshot id. A dead
   sandbox cannot produce one, and the failure would otherwise land as an
   opaque RPC error after the child had already been billed for.
3. `terminate` is idempotent, because abandonment arrives from two directions
   at once (the agent abandoning itself, and the pruner abandoning it) and
   the second caller must not see an exception.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

WORKSPACE = "/workspace"

# Modal's own ceiling for a directory snapshot RPC. Named so a caller sees
# that the limit is the platform's, not a number we picked.
SNAPSHOT_TIMEOUT = 55

BRANCH_ID_RE = re.compile(r"^b\d+(?:\.\d+)*$")


class BranchError(RuntimeError):
    """Any violation of branch lifecycle rules."""


class TerminatedParentError(BranchError):
    """A fork was requested from a parent that can no longer be snapshotted."""


class ExecError(BranchError):
    """A command run inside a sandbox failed, or did not emit the JSON asked of it."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass
class BranchHandle:
    """A live branch: one sandbox, and the snapshot its workspace came from.

    `snapshot_id` is the image currently mounted at `/workspace` (None for a
    root, whose workspace is its own filesystem). It is deliberately the
    *mounted* snapshot and not the latest snapshot taken, so `rollback`
    changes it and `snapshot` does not -- a handle always describes where the
    sandbox's workspace came from.
    """

    branch_id: str
    parent_id: str | None
    depth: int
    sandbox_id: str
    snapshot_id: str | None = None

    @property
    def lineage(self) -> list[str]:
        return lineage(self.branch_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "sandbox_id": self.sandbox_id,
            "snapshot_id": self.snapshot_id,
        }


# ---- branch ids ---------------------------------------------------------
#
# Ids are path-like so that a human reading a log line knows the depth and the
# parentage without consulting the tree: `b0.1.2` is the second child of the
# first child of root 0. Sorting them numerically segment-by-segment also puts
# siblings next to each other in any listing.


def root_branch_id(index: int = 0) -> str:
    """`b0`, `b1`, ... -- one per root of the search (usually just `b0`)."""
    return f"b{int(index)}"


def child_branch_id(parent_id: str, ordinal: int) -> str:
    """`b0` + ordinal 3 -> `b0.3`. Ordinals are 1-based and never reused."""
    if not BRANCH_ID_RE.match(parent_id):
        raise BranchError(f"{parent_id!r} is not a branch id")
    if ordinal < 1:
        raise BranchError("child ordinals are 1-based")
    return f"{parent_id}.{int(ordinal)}"


def depth_of(branch_id: str) -> int:
    """Depth is literally in the id, which is why the ids look like this."""
    if not BRANCH_ID_RE.match(branch_id):
        raise BranchError(f"{branch_id!r} is not a branch id")
    return branch_id.count(".")


def parent_of(branch_id: str) -> str | None:
    if not BRANCH_ID_RE.match(branch_id):
        raise BranchError(f"{branch_id!r} is not a branch id")
    head, _, _ = branch_id.rpartition(".")
    return head or None


def lineage(branch_id: str) -> list[str]:
    """Root-first ancestry including `branch_id` itself: `b0`, `b0.1`, `b0.1.2`."""
    if not BRANCH_ID_RE.match(branch_id):
        raise BranchError(f"{branch_id!r} is not a branch id")
    parts = branch_id.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return descendant != ancestor and descendant.startswith(ancestor + ".")


def sort_key(branch_id: str) -> tuple[int, ...]:
    """Numeric, segment-wise -- so `b0.10` sorts after `b0.2`, not before."""
    return tuple(int(p.lstrip("b") or 0) for p in branch_id.split("."))


# ---- sandbox resolution -------------------------------------------------
#
# A BranchHandle holds only a sandbox *id*, so that it stays JSON-clean and can
# live in the registry. Recovering the live object from an id costs a network
# round trip, so handles created in this process are cached here.

_LIVE: dict[str, Any] = {}
_TERMINATED: set[str] = set()
_TERMINATION_REASONS: dict[str, str] = {}


def attach(sandbox: Any) -> str:
    """Register a live sandbox so later calls can find it by id."""
    sandbox_id = sandbox.object_id
    _LIVE[sandbox_id] = sandbox
    return sandbox_id


def forget(sandbox_id: str) -> None:
    _LIVE.pop(sandbox_id, None)


def resolve_sandbox(ref: Any, *, sandboxes: Any = None) -> Any:
    """Accept a sandbox, a `BranchHandle`, or a sandbox id, return a sandbox."""
    if isinstance(ref, BranchHandle):
        ref = ref.sandbox_id
    if not isinstance(ref, str):
        return ref  # already a sandbox-like object
    cached = _LIVE.get(ref)
    if cached is not None:
        return cached
    api = sandboxes if sandboxes is not None else _sandbox_api()
    sandbox = api.from_id(ref)
    _LIVE[ref] = sandbox
    return sandbox


def _sandbox_api() -> Any:
    import modal  # imported lazily: this module is unit-testable without modal

    return modal.Sandbox


def _image_api() -> Any:
    import modal

    return modal.Image


def _resources(resources: Any) -> tuple[str | None, float, int]:
    """Accept a `spec.Resources` or a plain dict."""
    if resources is None:
        return None, 2.0, 8192
    if isinstance(resources, dict):
        gpu = resources.get("gpu")
        cpu = float(resources.get("cpu", 2.0))
        memory = int(resources.get("memory", 8192))
    else:
        gpu = resources.gpu
        cpu = float(resources.cpu)
        memory = int(resources.memory)
    return (None if gpu in (None, "", "none") else str(gpu)), cpu, memory


def _create_sandbox(
    *,
    app: Any,
    image: Any,
    resources: Any,
    run_id: str,
    branch_id: str,
    secrets: Any,
    volumes: Any,
    timeout: int,
    sandboxes: Any,
) -> Any:
    gpu, cpu, memory = _resources(resources)
    api = sandboxes if sandboxes is not None else _sandbox_api()
    sandbox = api.create(
        app=app,
        image=image,
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        timeout=timeout,
        secrets=list(secrets or ()),
        volumes=dict(volumes or {}),
        # Tags are how a crashed orchestrator finds its own sandboxes again.
        # They are also the only thing distinguishing our sandboxes from
        # anybody else's in this workspace, so they are not optional.
        tags={"autobranch_run": run_id, "autobranch_branch": branch_id},
    )
    attach(sandbox)
    return sandbox


def create_root(
    app: Any,
    image: Any,
    *,
    resources: Any,
    run_id: str,
    secrets: Any = (),
    volumes: Any = None,
    timeout: int = 3600,
    branch_id: str | None = None,
    sandboxes: Any = None,
) -> BranchHandle:
    """Start the root branch: a sandbox with an empty `/workspace`.

    The root has no snapshot mounted; `/workspace` is its own filesystem.
    Children never share it -- they mount a snapshot of it (see `fork`).
    """
    branch_id = branch_id or root_branch_id()
    sandbox = _create_sandbox(
        app=app,
        image=image,
        resources=resources,
        run_id=run_id,
        branch_id=branch_id,
        secrets=secrets,
        volumes=volumes,
        timeout=timeout,
        sandboxes=sandboxes,
    )
    # `mount_image` creates its mount point, but a root has nothing mounted,
    # so the directory has to exist before anything writes into it.
    _mkdir(sandbox, WORKSPACE)
    return BranchHandle(
        branch_id=branch_id,
        parent_id=None,
        depth=depth_of(branch_id),
        sandbox_id=sandbox.object_id,
        snapshot_id=None,
    )


def snapshot(handle_or_sandbox: Any, *, ttl: int | None = None, path: str = WORKSPACE,
             sandboxes: Any = None) -> str:
    """Snapshot `/workspace` and return the resulting image's `object_id`.

    The parent keeps running; this is the cheap half of a fork. The returned
    id is the *only* handle on the snapshot -- directory snapshots are not
    named objects -- so the caller must persist it (registry, `snapshot_id`)
    in the same breath.

    `ttl=None` means "Modal's default retention" (30 days) rather than
    "forever": `snapshot_directory` reads `ttl=None` as indefinite retention,
    which would quietly accumulate storage for every branch ever forked, so
    the argument is omitted instead of forwarded.
    """
    sandbox = resolve_sandbox(handle_or_sandbox, sandboxes=sandboxes)
    kwargs: dict[str, Any] = {}
    if ttl is not None:
        kwargs["ttl"] = ttl
    image = sandbox.snapshot_directory(path, **kwargs)
    snapshot_id = image.object_id
    if not snapshot_id:
        raise BranchError("snapshot_directory returned an image with no object_id")
    return snapshot_id


def fork(
    parent_snapshot_id: str,
    *,
    app: Any,
    image: Any,
    resources: Any,
    run_id: str,
    parent_id: str,
    depth: int | None = None,
    child_ordinal: int = 1,
    branch_id: str | None = None,
    secrets: Any = (),
    volumes: Any = None,
    timeout: int = 3600,
    sandboxes: Any = None,
    images: Any = None,
) -> BranchHandle:
    """Create a child sandbox whose `/workspace` is `parent_snapshot_id`.

    A snapshot id is mandatory and positional. That is the guard: there is no
    code path in which a child shares a directory with a live sandbox, and no
    path in which a fork is attempted against a parent that can no longer
    produce a snapshot. Use `fork_from` when the parent is still running and
    the snapshot should be taken for you.
    """
    if not parent_snapshot_id or not isinstance(parent_snapshot_id, str):
        raise TerminatedParentError(
            f"fork of {parent_id!r} needs an explicit parent snapshot id; a "
            "terminated parent cannot produce one, and sharing the parent's "
            "live workspace would corrupt both"
        )

    branch_id = branch_id or child_branch_id(parent_id, child_ordinal)
    derived = depth_of(branch_id)
    if depth is not None and int(depth) != derived:
        # Depth is encoded in the id. A disagreement means the caller's
        # bookkeeping has drifted, and the tree view would misreport parentage.
        raise BranchError(f"depth {depth} contradicts branch id {branch_id!r} (depth {derived})")

    img_api = images if images is not None else _image_api()
    sandbox = _create_sandbox(
        app=app,
        image=image,
        resources=resources,
        run_id=run_id,
        branch_id=branch_id,
        secrets=secrets,
        volumes=volumes,
        timeout=timeout,
        sandboxes=sandboxes,
    )
    sandbox.mount_image(WORKSPACE, img_api.from_id(parent_snapshot_id))
    return BranchHandle(
        branch_id=branch_id,
        parent_id=parent_id,
        depth=derived,
        sandbox_id=sandbox.object_id,
        snapshot_id=parent_snapshot_id,
    )


def fork_from(
    parent: BranchHandle,
    *,
    app: Any,
    image: Any,
    resources: Any,
    run_id: str,
    child_ordinal: int = 1,
    snapshot_id: str | None = None,
    ttl: int | None = None,
    secrets: Any = (),
    volumes: Any = None,
    timeout: int = 3600,
    sandboxes: Any = None,
    images: Any = None,
) -> BranchHandle:
    """Snapshot a *running* parent and fork from it.

    If the parent is no longer running, a snapshot cannot be taken, so an
    explicit `snapshot_id` (an earlier snapshot of that branch, or an
    ancestor's) is required. Raising here is the point: silently forking from
    a stale snapshot would produce a child whose lineage is a lie.
    """
    sandbox = resolve_sandbox(parent, sandboxes=sandboxes)
    alive = parent.sandbox_id not in _TERMINATED and sandbox.poll() is None

    if snapshot_id is None:
        if alive:
            snapshot_id = snapshot(sandbox, ttl=ttl)
        else:
            raise TerminatedParentError(
                f"branch {parent.branch_id!r} (sandbox {parent.sandbox_id}) has "
                "exited; pass snapshot_id= to fork from one of its recorded "
                "snapshots"
            )

    return fork(
        snapshot_id,
        app=app,
        image=image,
        resources=resources,
        run_id=run_id,
        parent_id=parent.branch_id,
        child_ordinal=child_ordinal,
        secrets=secrets,
        volumes=volumes,
        timeout=timeout,
        sandboxes=sandboxes,
        images=images,
    )


def rollback(branch: BranchHandle, snapshot_id: str, *, sandboxes: Any = None,
             images: Any = None) -> None:
    """Replace `/workspace` with an ancestor's snapshot, in place.

    "Go back three checkpoints and try a different reward" is this call. The
    unmount is best-effort: a root branch has nothing mounted at `/workspace`
    (it is plain sandbox filesystem), and unmounting that is an error we do
    not care about -- the subsequent mount shadows it either way.

    The handle's `snapshot_id` is updated, because it describes where the
    workspace came from and after this call that has changed.
    """
    if not snapshot_id:
        raise BranchError("rollback needs a snapshot id")
    sandbox = resolve_sandbox(branch, sandboxes=sandboxes)
    img_api = images if images is not None else _image_api()
    try:
        sandbox.unmount_image(WORKSPACE)
    except Exception:
        pass
    sandbox.mount_image(WORKSPACE, img_api.from_id(snapshot_id))
    branch.snapshot_id = snapshot_id


def terminate(branch: BranchHandle, *, reason: str, sandboxes: Any = None) -> None:
    """Stop a branch's sandbox. Safe to call any number of times.

    Abandonment arrives from two directions -- the agent calling `abandon`,
    and the pruner cutting a line that stopped paying -- and they race. The
    second caller must be a no-op, not an exception, or the pruner dies on a
    branch that already gave up.
    """
    sandbox_id = branch.sandbox_id if isinstance(branch, BranchHandle) else str(branch)
    if sandbox_id in _TERMINATED:
        return
    _TERMINATED.add(sandbox_id)
    _TERMINATION_REASONS[sandbox_id] = reason
    sandbox = _LIVE.get(sandbox_id)
    if sandbox is None:
        # A handle rebuilt from the registry after an orchestrator restart has
        # no cached object; look it up rather than leaving the sandbox running
        # and billing.
        try:
            sandbox = resolve_sandbox(sandbox_id, sandboxes=sandboxes)
        except Exception:
            sandbox = None
    forget(sandbox_id)
    if sandbox is None:
        return
    try:
        if sandbox.poll() is None:
            sandbox.terminate()
    except Exception:
        # A sandbox that hit its own timeout between the poll and the
        # terminate is already in the state we wanted.
        pass


def is_terminated(branch: BranchHandle) -> bool:
    """True once `terminate` has been called on this branch in this process."""
    sandbox_id = branch.sandbox_id if isinstance(branch, BranchHandle) else str(branch)
    return sandbox_id in _TERMINATED


def termination_reason(branch: BranchHandle) -> str | None:
    """Why this branch was stopped -- the first reason wins, as it is the cause."""
    sandbox_id = branch.sandbox_id if isinstance(branch, BranchHandle) else str(branch)
    return _TERMINATION_REASONS.get(sandbox_id)


def exec_json(sandbox: Any, argv: list[str], *, timeout: int) -> dict[str, Any]:
    """Run a command in a sandbox and parse a JSON object from its stdout.

    Trainers print progress to stdout and then their result, so the whole
    stream is rarely valid JSON. Parsing the stream, then falling back to the
    last line that parses, means a caller does not have to silence a
    trainer's logs to get its metrics back.
    """
    sandbox = resolve_sandbox(sandbox)
    proc = sandbox.exec(*argv, timeout=timeout)
    out = proc.stdout.read()
    err = ""
    try:
        err = proc.stderr.read()
    except Exception:
        pass
    code = proc.wait()
    if code != 0:
        raise ExecError(
            f"{' '.join(argv)} exited {code}", returncode=code, stderr=(err or "")[-4000:]
        )
    parsed = parse_json_tail(out)
    if parsed is None:
        raise ExecError(
            f"{' '.join(argv)} produced no JSON object on stdout",
            returncode=code,
            stderr=(err or "")[-4000:],
        )
    return parsed


def parse_json_tail(text: str) -> dict[str, Any] | None:
    """The whole stream as JSON, else the last line of it that is a JSON object."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        whole = json.loads(text)
    except ValueError:
        pass
    else:
        return whole if isinstance(whole, dict) else None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _mkdir(sandbox: Any, path: str) -> None:
    try:
        proc = sandbox.exec("mkdir", "-p", path, timeout=60)
        proc.wait()
    except Exception:
        # Not fatal: the first write into /workspace will create it, and a
        # sandbox that cannot exec at all will fail loudly on the next call.
        pass


def _reset_process_state() -> None:
    """Clear the process-local sandbox caches (tests, and orchestrator restarts)."""
    _LIVE.clear()
    _TERMINATED.clear()
    _TERMINATION_REASONS.clear()
