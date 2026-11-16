"""Modal entrypoint: the ASGI web app, plus a CLI for the same operations.

This file is at the repository root, outside the `autobranch` package, and it
must stay there. The image mounts the package with
`add_local_python_source("autobranch")`, so an entrypoint inside the package
would be uploaded twice and the container would import two module objects with
two separate `modal.App`s. Routes on one are invisible to the other, and the
symptom is a bare HTTP 500 with no traceback.

Everything here is modal 1.6.4 API: `scaledown_window`, `max_containers`,
`modal.concurrent`. There is no `allow_concurrent_inputs`.

    modal serve app.py      # dev, hot reload
    modal deploy app.py     # production
"""

import base64
import io
import json
import os
import pathlib
import sys
import time
import zipfile

import modal

from autobranch.consolidate import ARTIFACT_DIRNAME
from autobranch.infra import VOLUMES, VOLUME_NAME, app, kimi_secret, volume, web_image
from autobranch.server import BUNDLE_DIRNAME, DEFAULT_DATA_ROOT, MANIFEST_NAME

# `numpy` is added on top of the pinned web stack so the submission gate can
# actually *step* a submitted environment in this container rather than
# deferring every bundle to its branch. The ASGI pins themselves come from
# `infra.web_image()` and are not restated here -- one place owns them.
#
# Order matters: `add_local_*` layers must come last, after every pip layer.
#
# The frontend path is absolute, resolved from this file: a relative path
# would resolve against the caller's cwd, so `modal deploy` from a parent
# directory would ship an app with no frontend and no error.
REPO_ROOT = pathlib.Path(__file__).resolve().parent

WEB_IMAGE = (
    web_image()
    .pip_install("numpy>=1.26")
    .add_local_python_source("autobranch")
    .add_local_dir(str(REPO_ROOT / "frontend"), remote_path="/root/frontend")
)

BUNDLE_ROOT = os.path.join(DEFAULT_DATA_ROOT, BUNDLE_DIRNAME)


# ---------------------------------------------------------------------------
# the web app
# ---------------------------------------------------------------------------


@app.function(
    image=WEB_IMAGE,
    volumes=VOLUMES,
    secrets=[kimi_secret()],
    scaledown_window=300,
    max_containers=2,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    """API and frontend on one origin.

    `max_containers=2` is a correctness bound, not a cost one: the registry is
    sharded per writer, but a submission that extracts a 64 MiB bundle is
    memory-heavy, and 20 concurrent inputs on two containers is already more
    than a reward search can keep busy.
    """
    from autobranch.server import build_app

    return build_app()


# ---------------------------------------------------------------------------
# the tree driver
# ---------------------------------------------------------------------------


@app.function(
    image=WEB_IMAGE,
    volumes=VOLUMES,
    secrets=[kimi_secret()],
    timeout=24 * 60 * 60,
    max_containers=8,
)
def spawn_run(run_id: str, spec_dict: dict, constraints_dict: dict, budget_dict: dict) -> dict:
    """Record the root branch, then hand the run to Modal.

    After `launch_root` returns, nothing local supervises the run: the root
    sandbox owns its agent loop and forks its own children. The run is marked
    `running` only once a sandbox genuinely exists -- a `running` run with no
    agent behind it is exactly the kind of unverified completion signal
    Never trust a completion event.
    """
    from autobranch.branch import root_branch_id
    from autobranch.orchestrator import Budget, initial_reward
    from autobranch.registry import BranchRecord, Registry
    from autobranch.spec import EnvironmentSpec

    spec = EnvironmentSpec.parse(spec_dict)
    registry = Registry(writer_id="orchestrator")

    branch_id = root_branch_id()
    registry.put_branch(
        run_id,
        BranchRecord(
            branch_id=branch_id,
            parent_id=None,
            depth=0,
            status="pending",
            reward_config=initial_reward(spec),
            rationale="auto-generated baseline; the master agent refines this first",
        ),
    )
    registry.append_event(
        run_id,
        "info",
        f"root branch {branch_id} queued for {spec.name} "
        f"(objective {spec.objective.name}, trainer {spec.trainer})",
        branch_id=branch_id,
    )

    # Hand off to Modal. `launch_root` creates the root sandbox and starts the
    # agent loop inside it; from here the run is entirely Modal's -- the root
    # branch forks its own children and nothing local supervises it.
    from autobranch.forking import launch_root

    try:
        info = launch_root(
            spec=spec,
            constraints=constraints_dict or {},
            budget=Budget.parse(budget_dict),
            run_id=run_id,
            registry=registry,
        )
    except Exception as exc:  # noqa: BLE001
        # A run that cannot start must say so rather than sit in `pending`
        # forever looking like it is thinking.
        registry.update_run(run_id, state="failed")
        registry.append_event(
            run_id, "error", f"root branch failed to launch: {type(exc).__name__}: {exc}",
            branch_id=branch_id,
        )
        raise

    registry.append_event(
        run_id, "info",
        f"root branch running in sandbox {info['sandbox_id']}; "
        f"deadline in {(info['deadline_ts'] - time.time()) / 3600:.1f}h",
        branch_id=branch_id,
    )
    return {"run_id": run_id, "root_branch_id": branch_id, "launched": True,
            "sandbox_id": info["sandbox_id"], "deadline_ts": info["deadline_ts"]}


def _launcher(spec, constraints, budget, run_id: str) -> str:
    """Launcher for calls originating inside this file.

    `server.modal_launcher` resolves `spawn_run` by name off the *deployed*
    app, which is right for the long-lived web container. Inside a function
    defined here the object is already hydrated, so spawning it directly also
    resolves under an ephemeral `modal run`, where no deployed app exists.
    """
    call = spawn_run.spawn(
        run_id=run_id,
        spec_dict=spec.to_dict(),
        constraints_dict=constraints.to_dict(),
        budget_dict=budget.to_dict(),
    )
    return getattr(call, "object_id", "")


@app.function(
    image=WEB_IMAGE,
    volumes=VOLUMES,
    secrets=[kimi_secret()],
    timeout=15 * 60,
)
def submit_bundle(payload: dict) -> dict:
    """The CLI's submit, running the browser's code path.

    It runs in a container rather than locally because the bundle has to land
    on the volume at `bundles/<run_id>.zip` for a branch sandbox to fetch the
    exact bytes that were validated.
    """
    from autobranch.registry import Registry
    from autobranch.server import LaunchError, SubmissionError, submit

    try:
        submission = submit(
            Registry(writer_id="cli"),
            payload,
            launcher=_launcher,
            bundle_root=BUNDLE_ROOT,
        )
    except SubmissionError as exc:
        return {"ok": False, "problems": exc.problems}
    except LaunchError as exc:
        return {"ok": False, "problems": [f"run could not be started: {exc}"]}
    return {
        "ok": True,
        "run_id": submission.run_id,
        "env_name": submission.spec.name,
        "objective": submission.spec.objective.name,
        "advisories": submission.advisories,
    }


# ---------------------------------------------------------------------------
# local helpers
# ---------------------------------------------------------------------------


def _zip_bytes(bundle: str) -> bytes:
    """A bundle path -- directory or `.zip` -- as zip bytes.

    Directories are zipped here rather than uploaded as a tree so the local
    and browser paths hand the server byte-identical input.
    """
    path = pathlib.Path(bundle).expanduser()
    if not path.exists():
        raise SystemExit(f"no such bundle: {path}")
    if path.is_file():
        if path.suffix != ".zip":
            raise SystemExit(f"{path} is not a .zip or a directory")
        return path.read_bytes()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(path.rglob("*")):
            if item.is_dir() or "__pycache__" in item.parts or item.name.startswith("."):
                continue
            archive.write(item, item.relative_to(path).as_posix())
    return buffer.getvalue()


def _report(problems: list, advisories: list) -> None:
    for problem in problems:
        print(f"  FAIL  {problem}")
    for note in advisories:
        print(f"  note  {note}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



# WEB_IMAGE, not web_image(): the bare image has no `autobranch` package,
# and this function imports it. That mistake crash-looped the function.
@app.function(image=WEB_IMAGE, volumes=VOLUMES, timeout=900)
def finalize_run(run_id: str) -> dict:
    """Turn a finished run into a downloadable artifact.

    Nothing used to call `consolidate()`, so a run could reach `succeeded`
    with a measured winner and still produce no weights -- the framework's
    entire deliverable. This is that missing step, and it is idempotent: a
    second call over an existing artifact just rewrites it from the same
    records.
    """
    import os
    import shutil

    from autobranch.consolidate import ARTIFACT_DIRNAME, ConsolidationError, consolidate
    from autobranch.registry import Registry
    from autobranch.spec import EnvironmentSpec

    registry = Registry(writer_id="finalizer")
    run = registry.get_run(run_id)
    if run is None:
        return {"ok": False, "reason": f"unknown run {run_id}"}

    payload = registry.api_payload(run_id)
    payload["spec"] = run.spec or {}
    payload["constraints"] = run.constraints

    # Independent evaluations, if any were run. These are the only scores in
    # the artifact the search did not produce, and they are the ones worth
    # trusting: a branch evaluates at whatever n its budget allows, and in
    # run-25ed75aa0a that was 24 -- which read 0.9583 on a checkpoint an
    # n=72 evaluation put at 0.7361, a disagreement at p=0.020. Shipping the
    # search's number alone advertises the optimistic end of its own noise.
    import json as _json

    for branch in payload.get("branches") or []:
        path = os.path.join(
            DEFAULT_DATA_ROOT, "runs", run_id, str(branch.get("branch_id")),
            "independent_eval.json",
        )
        try:
            with open(path) as fh:
                runs = _json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(runs, list) and runs:
            branch["independent_evals"] = runs

    try:
        spec = EnvironmentSpec.parse(run.spec) if run.spec else None
        objective = spec.objective if spec else _FallbackObjective(run.objective)
    except Exception:  # noqa: BLE001 - a malformed spec must not block the artifact
        objective = _FallbackObjective(run.objective)

    out_dir = os.path.join(DEFAULT_DATA_ROOT, ARTIFACT_DIRNAME, run_id)
    os.makedirs(out_dir, exist_ok=True)

    def fetch(src: str, dst: str) -> int:
        """Weights live on the Volume under runs/<run>/branches/<branch>/."""
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            return os.path.getsize(dst)
        raise ConsolidationError(
            f"the winning checkpoint is not on the volume: {src!r}. The branch "
            "that produced it may have terminated before publishing its weights."
        )

    try:
        index = consolidate(payload, objective, out_dir=out_dir, fetch_checkpoint=fetch)
    except ConsolidationError as exc:
        registry.append_event(run_id, "warn", f"artifact not built: {exc}")
        volume.commit()
        return {"ok": False, "reason": str(exc)}

    volume.commit()
    registry.append_event(
        run_id, "info",
        f"artifact ready: {len(index['files'])} files, best "
        f"{index['best']['score']:.4f} from {index['best']['branch_id']}",
    )
    return {"ok": True, **{k: index[k] for k in ("best", "run_command")},
            "files": [f["name"] for f in index["files"]]}


class _FallbackObjective:
    """Enough of an Objective to rank by, when the stored spec is unusable."""

    def __init__(self, raw: dict):
        self.name = (raw or {}).get("name", "objective")
        self.direction = (raw or {}).get("direction", "maximize")

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "maximize" else -1.0

    def better(self, a: float, b: float) -> bool:
        return a * self.sign > b * self.sign


@app.local_entrypoint()
def cli_finalize(run_id: str):
    """modal run app.py::cli_finalize --run-id run-xxxx"""
    out = finalize_run.remote(run_id)
    print(json.dumps(out, indent=2))

@app.local_entrypoint()
def cli_validate(bundle: str):
    """Manifest + environment validation. Exits non-zero if anything is wrong.

    Runs locally and on purpose: the whole point of the submission gate is
    that it answers in seconds, before any container exists.
    """
    import tempfile

    from autobranch.server import (
        SubmissionError,
        check_env,
        extract_bundle,
        find_manifest,
    )
    from autobranch.spec import EnvironmentSpec, SpecError

    data = _zip_bytes(bundle)
    print(f"bundle: {bundle} ({len(data)} bytes zipped)")

    with tempfile.TemporaryDirectory(prefix="autobranch-validate-") as tmp:
        try:
            extract_bundle(data, tmp)
        except SubmissionError as exc:
            _report(exc.problems, [])
            sys.exit(1)

        manifest_path = find_manifest(tmp)
        if manifest_path is None:
            _report([f"no {MANIFEST_NAME} at the bundle root or one level below"], [])
            sys.exit(1)

        with open(manifest_path, encoding="utf-8") as handle:
            manifest_yaml = handle.read()
        try:
            spec = EnvironmentSpec.from_yaml(manifest_yaml)
        except SpecError as exc:
            _report([str(exc)], [])
            sys.exit(1)

        print(f"manifest: {spec.name} v{spec.version} - trainer {spec.trainer}")
        print(f"objective: {spec.objective.name} ({spec.objective.direction})")
        print(f"signals: {', '.join(spec.signal_names)}")

        problems, advisories = check_env(spec, os.path.dirname(manifest_path))

    _report(problems, advisories)
    if problems:
        print(f"\n{len(problems)} problem(s); bundle rejected")
        sys.exit(1)
    print("\nok: manifest and environment both pass")


@app.local_entrypoint()
def cli_submit(
    bundle: str,
    constraints: str = "",
    usd: float = 25.0,
    hours: float = 2.0,
    max_branches: int = 12,
):
    """Submit a bundle exactly as the web form does."""
    data = _zip_bytes(bundle)
    payload = {
        "bundle_b64": base64.b64encode(data).decode("ascii"),
        "constraints": constraints,
        "budget": {"usd": usd, "hours": hours, "max_branches": max_branches},
    }
    result = submit_bundle.remote(payload)
    if not result.get("ok"):
        _report(result.get("problems") or ["submission rejected"], [])
        sys.exit(1)
    _report([], result.get("advisories") or [])
    print(f"run_id: {result['run_id']}")
    print(f"env: {result['env_name']} - objective {result['objective']}")
    print(f"status: modal run app.py::cli_status --run-id {result['run_id']}")



@app.function(
    image=WEB_IMAGE,
    volumes=VOLUMES,
    secrets=[kimi_secret()],
    timeout=60 * 30,
)
def restart_run(run_id: str) -> dict:
    """Relaunch a run's root branch in place, resuming from the Volume.

    Runs here rather than locally because `launch_root` calls
    `ensure_runtime_on_volume`, which writes the framework to `/data` -- a
    path that only exists inside a Modal container. `autobranch.supervise`
    has the same constraint for the same reason; this is the entry point that
    satisfies it from a laptop.
    """
    from autobranch.supervise import restart

    return restart(run_id)


@app.local_entrypoint()
def cli_restart(run_id: str):
    """Relaunch a dead run without losing its training.

        modal run app.py::cli_restart --run-id run-xxxx
    """
    out = restart_run.remote(run_id)
    if not out.get("ok", True):
        print(f"restart refused: {out.get('reason')}")
        return
    print(f"relaunched {run_id} in {out.get('sandbox_id')}")
    print(f"status: modal run app.py::cli_status --run-id {run_id}")

@app.local_entrypoint()
def cli_status(run_id: str, events: int = 12):
    """Poll a run's real state.

    Reads the registry Dicts directly from the client: no container, and no
    completion event to take anyone's word for.
    """
    from autobranch.registry import Registry

    registry = Registry(writer_id="cli-status")
    try:
        payload = registry.api_payload(run_id)
    except KeyError:
        raise SystemExit(f"unknown run {run_id!r}")

    best = payload.get("best") or {}
    print(f"{payload['run_id']}  {payload['state']}  {payload['env_name']}")
    print(
        f"objective {payload['objective']['name']} ({payload['objective']['direction']})"
        f"  best {best.get('score', '-')} on {best.get('branch_id', '-')}"
    )
    print(f"spend ${payload['spend_usd']:.2f}  elapsed {payload['elapsed_s']:.0f}s")
    print(f"branches: {len(payload['branches'])}")
    for branch in payload["branches"]:
        scores = [s["score"] for s in branch["scores"]]
        top = f"{max(scores):.4f}" if scores else "-"
        print(
            f"  {branch['branch_id']:<10} {branch['status']:<11} best {top:<8}"
            f" ${branch['spend_usd']:.2f}"
            + (f"  abandoned: {branch['abandoned_reason']}" if branch["abandoned_reason"] else "")
        )
    tail = payload["events"][-events:] if events > 0 else []
    for event in tail:
        print(f"  [{event['level']}] {event.get('branch_id') or 'master'}: {event['message']}")


@app.local_entrypoint()
def cli_artifact(run_id: str, out: str = "./artifacts"):
    """Download a run's artifact tree from the volume."""
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    prefix = f"{ARTIFACT_DIRNAME}/{run_id}"
    try:
        entries = volume.listdir(prefix, recursive=True)
    except Exception as exc:  # noqa: BLE001 - any failure here means "nothing to fetch"
        raise SystemExit(f"no artifact for {run_id!r} on {VOLUME_NAME}: {exc}")

    destination = pathlib.Path(out).expanduser() / run_id
    written = 0
    for entry in entries:
        if getattr(entry.type, "name", "") != "FILE":
            continue
        relative = entry.path[len(prefix):].lstrip("/")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        written += 1
        print(f"  {relative}  ({entry.size} bytes)")

    if not written:
        raise SystemExit(f"artifact directory {prefix} is empty; has the run consolidated?")

    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        objective = manifest.get("objective") or {}
        print(
            f"\n{objective.get('name')} = {objective.get('score')} "
            f"from branch {(manifest.get('winner') or {}).get('branch_id')}"
        )
    print(f"\n{written} file(s) in {destination}")
    print(f"run it: python {destination / 'serve.py'}")
