"""Forking a live branch from inside it.

A sandbox can read its own id from `MODAL_SANDBOX_ID` (verified), which is what
makes recursive self-forking possible: a branch snapshots its own workspace,
starts a sibling sandbox with that snapshot mounted, and hands it the child's
identity through environment variables. No central supervisor has to broker it.

Credentials. Both of these -- reaching the registry and creating a sandbox --
need workspace credentials inside the sandbox, supplied by the
`autobranch-modal-auth` secret. That is a real grant: a branch can do anything
the workspace token can. It is bounded by being the user's own workspace, by
every resource name being `autobranch-`prefixed, and by the fact that the
agent never writes shell commands -- it only picks from a fixed tool set, and
`fork` is the only tool that touches infrastructure at all.

Snapshot choice. `snapshot_directory` keeps the parent running;
`snapshot_filesystem` would terminate it, which would turn every fork into a
hand-off rather than a branch. That difference was measured before this module
was written.
"""

import json
import os
import time
from typing import Any

from autobranch.orchestrator import BranchEnv

SNAPSHOT_TTL = 7 * 24 * 3600  # a week: long enough to roll back, short enough to expire


class ForkError(RuntimeError):
    """Raised when a fork cannot be completed; always recoverable by the agent."""


def self_sandbox():
    """The `Sandbox` object for the process we are running inside.

    Raises rather than guessing: a branch that cannot identify itself must not
    silently fork from the wrong workspace.
    """
    import modal

    sandbox_id = os.environ.get("MODAL_SANDBOX_ID")
    if not sandbox_id:
        raise ForkError(
            "MODAL_SANDBOX_ID is not set, so this process is not inside a Modal "
            "sandbox and cannot snapshot itself"
        )
    return modal.Sandbox.from_id(sandbox_id)


def make_forker(*, spec, budget, registry=None, app=None, image=None, launcher=None,
                snapshotter=None):
    """Build the `forker` callable that `BranchRunner` invokes.

    Injectable throughout so the branch loop can be tested with a fake that
    records calls instead of creating sandboxes.
    """

    def forker(
        *,
        parent_env: BranchEnv,
        child_branch_id: str | None,
        reward_config: dict[str, Any] | None,
        note: str = "",
        rollback_to: str | None = None,
    ) -> dict[str, Any]:
        if rollback_to is not None:
            return _rollback_self(rollback_to)
        if not child_branch_id or reward_config is None:
            raise ForkError("a fork needs both a child branch id and a reward config")
        return _fork_child(
            parent_env=parent_env,
            child_branch_id=child_branch_id,
            reward_config=reward_config,
            note=note,
            spec=spec,
            budget=budget,
            registry=registry,
            app=app,
            image=image,
            launcher=launcher,
            snapshotter=snapshotter,
        )

    return forker


def _rollback_self(snapshot_id: str) -> dict[str, Any]:
    """Re-mount an earlier snapshot of our own workspace.

    The unmount is best effort: a root branch's `/workspace` is plain sandbox
    filesystem with nothing mounted, and Modal errors on unmounting that. The
    subsequent mount shadows the directory either way.
    """
    import modal

    sb = self_sandbox()
    workspace = os.environ.get("AUTOBRANCH_WORKSPACE", "/workspace")
    try:
        sb.unmount_image(workspace)
    except Exception:  # noqa: BLE001 - nothing mounted is the common case
        pass
    sb.mount_image(workspace, modal.Image.from_id(snapshot_id))
    return {"ok": True, "snapshot_id": snapshot_id, "rolled_back": True}


def _fork_child(
    *,
    parent_env: BranchEnv,
    child_branch_id: str,
    reward_config: dict[str, Any],
    note: str,
    spec,
    budget,
    registry,
    app,
    image,
    snapshotter,
    launcher,
) -> dict[str, Any]:
    import modal

    from autobranch import infra

    workspace = os.environ.get("AUTOBRANCH_WORKSPACE", "/workspace")

    # 1. Freeze our current state. The parent keeps running.
    #    `snapshotter` is injected only by tests; production always snapshots
    #    the live sandbox. It exists because the seam used to sit *after* this
    #    call, which made the whole fork path untestable off Modal.
    if snapshotter is not None:
        snapshot = snapshotter(workspace)
        snapshot_id = getattr(snapshot, "object_id", snapshot)
    else:
        parent = self_sandbox()
        snapshot = parent.snapshot_directory(workspace, ttl=SNAPSHOT_TTL)
        snapshot_id = snapshot.object_id

    # 2. The child's identity. It inherits the run, the spec, the constraints,
    #    the budget and -- critically -- the same absolute deadline, so a deep
    #    branch cannot extend the run by being born late.
    child_env = BranchEnv(
        run_id=parent_env.run_id,
        branch_id=child_branch_id,
        parent_id=parent_env.branch_id,
        depth=parent_env.depth + 1,
        spec=parent_env.spec,
        constraints=parent_env.constraints,
        budget=parent_env.budget,
        deadline_ts=parent_env.deadline_ts,
        reward_config=reward_config,
        snapshot_id=snapshot_id,
    )

    if launcher is not None:  # test seam
        launcher(child_env)
        return {"ok": True, "snapshot_id": snapshot_id, "child_branch_id": child_branch_id}

    # 3. Register the child before starting it, so a child that dies during
    #    boot still appears in the tree as a failed branch rather than
    #    vanishing without trace.
    if registry is not None:
        try:
            registry.put_branch(parent_env.run_id, _pending_record(child_env, note))
        except Exception:  # noqa: BLE001
            pass

    ensure_runtime_on_volume()
    the_app = app or modal.App.lookup(infra.APP_NAME, create_if_missing=True)
    the_image = image or infra.image_for(spec)

    child = modal.Sandbox.create(
        "sleep", "infinity",
        app=the_app,
        image=the_image,
        secrets=[infra.kimi_secret(), modal_auth_secret()] + list(infra.env_secrets(spec.image)),
        volumes={infra.DATA_ROOT: infra.volume, **infra.env_volumes(spec.image)},
        gpu=spec.resources.gpu,
        cpu=spec.resources.cpu,
        memory=spec.resources.memory,
        timeout=max(600, int(parent_env.deadline_ts - time.time()) + 300),
        env=child_env.to_env(),
    )
    child.mount_image(workspace, snapshot)

    # 4. Start the child's loop detached. We do not wait: the parent has its
    #    own line to continue, which is the entire point of a branch.
    child.exec(
        "bash", "-lc",
        "python -u -m autobranch.branch_main >> /workspace/branch.log 2>&1",
        env=child_env.to_env(),
        timeout=max(600, int(parent_env.deadline_ts - time.time())),
    )

    if registry is not None:
        try:
            registry.update_branch(
                parent_env.run_id, child_branch_id,
                sandbox_id=child.object_id, status="running",
            )
        except Exception:  # noqa: BLE001
            pass

    return {
        "ok": True,
        "snapshot_id": snapshot_id,
        "child_branch_id": child_branch_id,
        "sandbox_id": child.object_id,
    }


def modal_auth_secret():
    """Workspace credentials, so a branch can reach the registry and fork."""
    import modal

    return modal.Secret.from_name("autobranch-modal-auth")


RUNTIME_DIR = "lib"


def ensure_runtime_on_volume(*, data_root: str | None = None) -> str:
    """Publish the `autobranch` package to the volume for sandboxes to import.

    A sandbox's image is built from the *submitted manifest*, so it contains
    the environment's dependencies and nothing of ours -- the branch loop
    would start with `ModuleNotFoundError: No module named 'autobranch'`.
    `add_local_python_source` cannot help: it needs local files at build time,
    and these sandboxes are created at runtime from a deployed app.

    So the runtime travels on the volume that every branch already mounts, and
    `PYTHONPATH` points at it. Copying from the caller's own installed package
    means the sandboxes always run the same code as the launcher, with no
    second publishing step to forget.
    """
    import shutil

    from autobranch import infra

    root = data_root or infra.DATA_ROOT
    src = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(root, RUNTIME_DIR, "autobranch")

    # A branch forking its own child runs THIS code from the volume copy, so
    # src and dst are the same directory -- and the refresh below would delete
    # the package out from under the running process. When we are already the
    # published copy there is by definition nothing to publish.
    if os.path.abspath(src) == os.path.abspath(dst):
        return os.path.join(root, RUNTIME_DIR)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Replace wholesale rather than merge: a stale module left behind by an
    # older version is far harder to diagnose than a slightly slower copy.
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "vendor"),
    )
    try:
        infra.volume.commit()
    except Exception:  # noqa: BLE001 - off-volume (tests, local runs)
        pass
    return os.path.join(root, RUNTIME_DIR)


def _pending_record(child_env: BranchEnv, note: str):
    from autobranch.registry import BranchRecord

    now = time.time()
    return BranchRecord(
        branch_id=child_env.branch_id,
        parent_id=child_env.parent_id,
        depth=child_env.depth,
        status="pending",
        reward_config=child_env.reward_config or {},
        scores=[],
        spend_usd=0.0,
        tokens=0,
        rationale=note,
        abandoned_reason=None,
        snapshot_id=child_env.snapshot_id,
        sandbox_id=None,
        created_at=now,
        updated_at=now,
    )


def launch_root(
    *,
    spec,
    constraints: dict[str, Any],
    budget,
    run_id: str,
    registry=None,
    deadline_ts: float | None = None,
    reward_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a run's root branch. Called once, from the API or the CLI.

    After this returns, everything runs on Modal: the root sandbox owns the
    agent loop and spawns its own children.
    """
    import modal

    from autobranch import infra
    from autobranch.orchestrator import initial_reward

    deadline = deadline_ts or (time.time() + float(budget.hours) * 3600.0)
    root_id = "b0"
    env = BranchEnv(
        run_id=run_id,
        branch_id=root_id,
        parent_id=None,
        depth=0,
        spec=spec.to_dict(),
        constraints=constraints,
        budget=budget.to_dict(),
        deadline_ts=deadline,
        reward_config=reward_config or initial_reward(spec),
        snapshot_id=None,
    )

    if registry is not None:
        try:
            registry.put_branch(run_id, _pending_record(env, "root branch"))
        except Exception:  # noqa: BLE001
            pass

    ensure_runtime_on_volume()
    the_app = modal.App.lookup(infra.APP_NAME, create_if_missing=True)
    sb = modal.Sandbox.create(
        "sleep", "infinity",
        app=the_app,
        image=infra.image_for(spec),
        secrets=[infra.kimi_secret(), modal_auth_secret()] + list(infra.env_secrets(spec.image)),
        volumes={infra.DATA_ROOT: infra.volume, **infra.env_volumes(spec.image)},
        gpu=spec.resources.gpu,
        cpu=spec.resources.cpu,
        memory=spec.resources.memory,
        timeout=int(float(budget.hours) * 3600) + 600,
        env=env.to_env(),
    )
    # The bundle lives on the volume; unpack it where the entrypoint expects it.
    sb.exec(
        "bash", "-lc",
        f"mkdir -p /workspace && "
        f"(test -f {infra.DATA_ROOT}/bundles/{run_id}.zip && "
        f" python -c \"import zipfile,sys; zipfile.ZipFile('{infra.DATA_ROOT}/bundles/{run_id}.zip')"
        f".extractall('/bundle')\" || true)",
        timeout=300,
    ).wait()
    sb.exec(
        "bash", "-lc",
        "python -u -m autobranch.branch_main >> /workspace/branch.log 2>&1",
        env=env.to_env(),
        timeout=int(float(budget.hours) * 3600) + 300,
    )

    if registry is not None:
        try:
            registry.update_branch(run_id, root_id, sandbox_id=sb.object_id, status="running")
            registry.update_run(run_id, state="running")
        except Exception:  # noqa: BLE001
            pass

    return {"run_id": run_id, "branch_id": root_id, "sandbox_id": sb.object_id,
            "deadline_ts": deadline}
