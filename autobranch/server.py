"""The web backend: one origin for the JSON API and the static frontend.

Three decisions in here are not style, they are scars.

**No `from __future__ import annotations` in this module.** FastAPI resolves an
endpoint's parameter annotations against the *module* globals. With the future
import every annotation becomes a string, and the FastAPI types below are
imported lazily inside `build_api` (so that importing this module does not
require fastapi), so the names would not be found. The route then stops
dispatching and the only symptom is a bare HTTP 500 with no traceback. Keep
annotations as real objects; the handlers are defined inside `build_api` and
work precisely because annotations are evaluated eagerly.

**The ASGI entrypoint lives outside this package**, in `app.py` at the
repository root. A Modal entrypoint inside a package that the image also
mounts is uploaded twice, and the container ends up importing two distinct
module objects with two separate `modal.App`s. See DESIGN.md section 7.

**A bundle is rejected here or nowhere.** `POST /api/runs` parses the
manifest, loads the environment and steps it before any resource is created,
and returns the whole problem list with HTTP 400. The predecessor project
burned six training runs on a measurement that was wrong from the first
minute; a validator that answers in the browser is worth hours.

Updates are **polled**, not pushed: `GET /api/runs/{id}` returns the whole run
document and the frontend asks again every few seconds. Polling real state is
DESIGN.md lesson 8 -- every silent failure in the predecessor project was
caught by comparing counters, never by a completion notification.
"""

import base64
import binascii
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from autobranch import __version__
from autobranch.consolidate import ARTIFACT_DIRNAME
from autobranch.orchestrator import Budget, Constraints, new_run_id, plan_run
from autobranch.registry import Registry
from autobranch.spec import EnvironmentSpec, SpecError

MANIFEST_NAME = "autobranch.yaml"

#: A browser upload arrives base64-encoded in a JSON body, so an oversize
#: bundle costs memory three times over (string, bytes, extracted tree). The
#: cap is enforced on the *encoded* length first, before anything is decoded.
MAX_BUNDLE_BYTES = 64 * 1024 * 1024

#: Extracted size is capped separately: a 1 MiB zip can hold a 10 GiB file.
MAX_EXTRACTED_BYTES = 4 * MAX_BUNDLE_BYTES

#: Where artifacts and bundles live when nothing is injected. The web
#: container mounts the `autobranch-data` volume at /data.
DEFAULT_DATA_ROOT = os.environ.get("AUTOBRANCH_DATA_ROOT", "/data")

BUNDLE_DIRNAME = "bundles"

FRONTEND_FILES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}

#: Artifact filenames are matched against this, not sanitised into it. A name
#: that does not match is refused rather than repaired, because "repair" is
#: where traversal bugs live.
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_GITHUB_REPO_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[\w./-]+))?/?$"
)


class SubmissionError(ValueError):
    """A submission that must fail in the browser, with every reason.

    Carries a *list*, deliberately. `validate_env` reports every problem it
    finds rather than the first; collapsing that to one line would throw away
    the part that saves someone an afternoon.
    """

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = [str(p) for p in problems] or ["bundle rejected"]
        super().__init__("\n".join(self.problems))


class LaunchError(RuntimeError):
    """The bundle was good but the run could not be started."""


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------


def bundle_urls(url: str) -> list[str]:
    """Candidate zip URLs for whatever a human pasted into the form.

    A GitHub *repo* URL is not a zip, so it is rewritten to codeload. The
    default branch is not discoverable without an API call, so `main` and
    `master` are both offered and the fetcher takes the first that answers --
    cheaper and more robust than an authenticated lookup.
    """
    url = url.strip()
    if not url:
        raise SubmissionError(["bundle_url is empty"])
    if not url.startswith(("http://", "https://")):
        raise SubmissionError([f"bundle_url {url!r} must be an http(s) URL"])

    match = _GITHUB_REPO_RE.match(url)
    if match and not url.endswith(".zip"):
        owner, repo = match.group("owner"), match.group("repo")
        ref = match.group("ref")
        refs = [ref] if ref else ["main", "master"]
        return [
            f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{r}" for r in refs
        ]
    return [url]


def fetch_url(
    url: str,
    *,
    max_bytes: int | None = None,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    """Download a zip, refusing to buffer more than `max_bytes`.

    `opener` is an injection seam so tests never touch the network. The cap
    defaults at call time, not at definition time, so a test can lower
    `MAX_BUNDLE_BYTES` without having to allocate 64 MiB to prove the cap
    works.
    """
    max_bytes = MAX_BUNDLE_BYTES if max_bytes is None else max_bytes
    open_url = opener or urllib.request.urlopen
    chunks: list[bytes] = []
    total = 0
    with open_url(url, timeout=60) as response:  # noqa: S310 - scheme checked above
        declared = response.headers.get("Content-Length") if hasattr(response, "headers") else None
        if declared and int(declared) > max_bytes:
            raise SubmissionError(
                [f"bundle at {url} is {int(declared)} bytes; the cap is {max_bytes} bytes"]
            )
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SubmissionError(
                    [f"bundle at {url} exceeds the {max_bytes}-byte cap"]
                )
            chunks.append(chunk)
    return b"".join(chunks)


def decode_bundle(
    *,
    bundle_b64: str | None = None,
    bundle_url: str | None = None,
    max_bytes: int | None = None,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    """Resolve either bundle source to zip bytes."""
    max_bytes = MAX_BUNDLE_BYTES if max_bytes is None else max_bytes
    if bool(bundle_b64) == bool(bundle_url):
        raise SubmissionError(
            ["give exactly one of bundle_b64 (an uploaded .zip) or bundle_url"]
        )

    if bundle_url:
        problems: list[str] = []
        for candidate in bundle_urls(bundle_url):
            try:
                return fetch_url(candidate, max_bytes=max_bytes, opener=opener)
            except SubmissionError:
                raise
            except Exception as exc:  # noqa: BLE001 - every fetch failure is a user error here
                problems.append(f"could not fetch {candidate}: {type(exc).__name__}: {exc}")
        raise SubmissionError(problems)

    text = str(bundle_b64)
    # 4 encoded chars per 3 decoded bytes: check before allocating the payload.
    if len(text) // 4 * 3 > max_bytes:
        raise SubmissionError(
            [
                f"bundle is about {len(text) // 4 * 3} bytes decoded; the cap is "
                f"{max_bytes} bytes. Trim the bundle: it should hold a manifest and "
                "code, not checkpoints or datasets."
            ]
        )
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SubmissionError([f"bundle_b64 is not valid base64: {exc}"]) from None
    if len(data) > max_bytes:
        raise SubmissionError([f"bundle is {len(data)} bytes; the cap is {max_bytes} bytes"])
    return data


def extract_bundle(data: bytes, dest: str, *, max_bytes: int | None = None) -> str:
    """Unpack a bundle zip into `dest` and return the directory it unpacked to.

    Entries are vetted before anything is written: an absolute path or a `..`
    segment in a zip member writes outside `dest` on extraction, and the
    uncompressed total is checked against `max_bytes` because the compressed
    size says nothing about it.
    """
    max_bytes = MAX_EXTRACTED_BYTES if max_bytes is None else max_bytes
    try:
        archive = zipfile.ZipFile(_BytesReader(data))
    except zipfile.BadZipFile as exc:
        raise SubmissionError([f"bundle is not a readable zip archive: {exc}"]) from None

    with archive:
        total = 0
        for info in archive.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or os.path.isabs(name):
                raise SubmissionError([f"zip entry {name!r} is an absolute path"])
            parts = name.replace("\\", "/").split("/")
            if ".." in parts:
                raise SubmissionError([f"zip entry {name!r} escapes the bundle directory"])
            total += info.file_size
            if total > max_bytes:
                raise SubmissionError(
                    [f"bundle expands to over {max_bytes} bytes; refusing to extract"]
                )
        archive.extractall(dest)
    return dest


class _BytesReader:
    """A minimal seekable file over bytes.

    `io.BytesIO` would do; this exists only so the zip is never copied, which
    matters at the 64 MiB cap under 20 concurrent inputs.
    """

    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return bytes(chunk)

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: len(self._data)}[whence]
        self._pos = max(0, min(len(self._data), base + offset))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True


def find_manifest(root: str) -> str | None:
    """`autobranch.yaml` at the bundle root, or one level down.

    One level down is not generosity: a GitHub zip always nests everything
    under `repo-branch/`, so rejecting that would reject every git submission.
    Deeper than one level is refused -- "which of these four manifests did you
    mean" is a question the submitter should answer, not the server.
    """
    direct = os.path.join(root, MANIFEST_NAME)
    if os.path.isfile(direct):
        return direct
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return None
    for entry in entries:
        nested = os.path.join(root, entry, MANIFEST_NAME)
        if os.path.isfile(nested):
            return nested
    return None


def _dist_to_module(requirement: str) -> str:
    """The module name a pip requirement most likely provides."""
    out = requirement.strip().lower()
    for sep in ("[", "==", ">=", "<=", "~=", "!=", ">", "<", ";", " "):
        out = out.split(sep, 1)[0]
    return out.strip().replace("-", "_")


def check_env(spec: EnvironmentSpec, bundle_dir: str) -> tuple[list[str], list[str]]:
    """Load the environment and step it. Returns `(problems, advisories)`.

    Validation is *deferred*, not failed, when the import error names a
    dependency the manifest declares in `image.pip`: the web container is a
    slim image and a bundle that needs torch cannot be stepped here, while the
    branch container will have it. Any other failure is the submitter's bug and
    is reported now. The distinction matters because the alternative is either
    rejecting every real bundle at submit time or validating none of them.
    """
    try:
        from autobranch.env_api import ADVISORY, blocking, load_env, validate_env
    except ImportError as exc:  # the slim web image may not carry numpy
        return [], [f"environment not stepped here ({exc}); the branch container will"]

    declared = {_dist_to_module(p) for p in spec.image.pip}
    before = set(sys.modules)
    env = None
    try:
        try:
            env = load_env(spec.entrypoint, search_path=bundle_dir)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing in declared:
                return [], [
                    f"environment not stepped here (needs {missing}, declared in image.pip); "
                    "the branch container will"
                ]
            return [
                f"entrypoint {spec.entrypoint!r}: no module named {missing!r}. Declare it in "
                "image.pip if it is a dependency, or fix the entrypoint."
            ], []
        except SyntaxError as exc:
            return [f"entrypoint {spec.entrypoint!r}: {exc}"], []
        except Exception as exc:  # noqa: BLE001 - any construction failure is a submission error
            return [f"entrypoint {spec.entrypoint!r} failed: {type(exc).__name__}: {exc}"], []

        try:
            found = validate_env(env, spec)
        except Exception as exc:  # noqa: BLE001 - a validator crash is still a bad bundle
            return [f"validate_env crashed on this environment: {type(exc).__name__}: {exc}"], []
        return blocking(found), [p for p in found if p.startswith(ADVISORY)]
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - a noisy close must not fail a submission
                pass
        _forget_bundle_modules(before, bundle_dir)


def _forget_bundle_modules(before: set[str], bundle_dir: str) -> None:
    """Drop the bundle's modules from `sys.modules` after validation.

    The web container is long-lived and two bundles can both define
    `grid_env`. Leaving the first one cached would validate the *previous*
    submission's code and report it as this one's -- the same two-module-
    objects confusion as DESIGN.md lesson 1, in a different disguise.
    """
    root = os.path.abspath(bundle_dir)
    for name in set(sys.modules) - before:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None) or ""
        if path and os.path.abspath(path).startswith(root):
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# submission
# ---------------------------------------------------------------------------


@dataclass
class Submission:
    """What a successful `POST /api/runs` produced."""

    run_id: str
    spec: EnvironmentSpec
    constraints: Constraints
    budget: Budget
    bundle_path: str
    advisories: list[str] = field(default_factory=list)


def validate_bundle(
    manifest_yaml: str,
    constraints_text: str,
    budget: dict[str, Any] | None,
    bundle_dir: str,
) -> tuple[EnvironmentSpec, Constraints, Budget, list[str]]:
    """The whole submission gate: manifest, then the environment itself."""
    try:
        spec, constraints, parsed_budget = plan_run(manifest_yaml, constraints_text, budget)
    except SpecError as exc:
        raise SubmissionError([str(exc)]) from None
    except (ValueError, TypeError) as exc:
        raise SubmissionError([f"budget: {exc}"]) from None

    problems, advisories = check_env(spec, bundle_dir)
    if problems:
        raise SubmissionError(problems)
    return spec, constraints, parsed_budget, advisories


def store_bundle(data: bytes, run_id: str, bundle_root: str) -> str:
    """Keep the bundle so a branch sandbox can fetch the exact bytes we vetted.

    Re-fetching a URL later would be a different experiment: the branch would
    train on whatever `main` points at then, not on what was validated now.
    """
    os.makedirs(bundle_root, exist_ok=True)
    path = os.path.join(bundle_root, f"{run_id}.zip")
    with open(path, "wb") as handle:
        handle.write(data)
    _commit_volume(path)
    return path


def _commit_volume(path: str) -> None:
    """Flush a Volume write so another container can see it.

    A Modal Volume write is only visible to other containers after a commit.
    Skipped (and never fatal) off-volume, which is how tests run.
    """
    try:
        from autobranch import infra
    except Exception:  # noqa: BLE001 - modal absent locally; nothing to commit
        return
    if not os.path.abspath(path).startswith(infra.DATA_ROOT + os.sep):
        return
    try:
        infra.volume.commit()
    except Exception:  # noqa: BLE001 - a failed commit must not lose the HTTP response
        pass


def _null_launcher(spec: EnvironmentSpec, constraints: Constraints, budget: Budget,
                   run_id: str) -> None:
    """Accept the submission and start nothing.

    The default, so a dev server and the tests can exercise the whole
    submission gate without Modal. A run launched this way stays `pending`;
    the real launcher is `modal_launcher`.
    """


def submit(
    registry: Registry,
    payload: dict[str, Any],
    *,
    launcher: Callable[[EnvironmentSpec, Constraints, Budget, str], Any] = _null_launcher,
    bundle_root: str,
    opener: Callable[..., Any] | None = None,
) -> Submission:
    """Validate, record and launch. Raises `SubmissionError` for a bad bundle.

    Shared by the HTTP route and by `app.py`'s CLI entrypoints, so `modal run
    app.py::cli_submit` and the browser cannot drift apart.
    """
    data = decode_bundle(
        bundle_b64=payload.get("bundle_b64"),
        bundle_url=payload.get("bundle_url"),
        opener=opener,
    )

    with tempfile.TemporaryDirectory(prefix="autobranch-bundle-") as tmp:
        extract_bundle(data, tmp)
        manifest_path = find_manifest(tmp)
        if manifest_path is None:
            raise SubmissionError(
                [
                    f"no {MANIFEST_NAME} found at the bundle root or one directory below it. "
                    "The manifest is the entire interface; without it there is nothing to run."
                ]
            )
        bundle_dir = os.path.dirname(manifest_path)
        with open(manifest_path, encoding="utf-8") as handle:
            manifest_yaml = handle.read()

        spec, constraints, budget, advisories = validate_bundle(
            manifest_yaml,
            str(payload.get("constraints") or ""),
            payload.get("budget"),
            bundle_dir,
        )

    run_id = new_run_id()
    bundle_path = store_bundle(data, run_id, bundle_root)

    registry.create_run(
        run_id=run_id,
        env_name=spec.name,
        objective=spec.objective,
        spec=spec.to_dict(),
        constraints=constraints.text,
        budget=budget.to_dict(),
        state="pending",
    )
    registry.append_event(
        run_id,
        "info",
        f"bundle accepted: {spec.name} v{spec.version} - {len(spec.signals)} signals - "
        f"objective {spec.objective.name} ({spec.objective.direction})",
    )
    for note in advisories:
        registry.append_event(run_id, "warn", note)
    if constraints.base_model:
        registry.append_event(
            run_id, "info", f"constraint resolved: post-train from {constraints.base_model}"
        )

    try:
        launcher(spec, constraints, budget, run_id)
    except Exception as exc:  # noqa: BLE001 - the run must not look pending forever
        registry.append_event(run_id, "error", f"launch failed: {type(exc).__name__}: {exc}")
        registry.update_run(run_id, state="failed")
        raise LaunchError(f"{type(exc).__name__}: {exc}") from exc

    registry.append_event(run_id, "info", f"run queued: {run_id}")
    return Submission(
        run_id=run_id,
        spec=spec,
        constraints=constraints,
        budget=budget,
        bundle_path=bundle_path,
        advisories=advisories,
    )


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def artifact_dir(run_id: str, artifact_root: str) -> str:
    return os.path.join(artifact_root, run_id)


def artifact_listing(run_id: str, artifact_root: str) -> dict[str, Any]:
    """`{ready, files, run_command, best}` for one run.

    `ready` is false, never a 404, while the run is unfinished: the frontend
    links to this page from the moment a run starts and "not yet" is an
    answer, not an error.
    """
    directory = artifact_dir(run_id, artifact_root)
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(manifest_path):
        return {"ready": False, "files": [], "run_command": "", "best": None}

    manifest: dict[str, Any] = {}
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        # A half-written manifest is "not ready", not a 500.
        return {"ready": False, "files": [], "run_command": "", "best": None}

    files = [
        {
            "name": name,
            "url": f"/api/runs/{urllib.parse.quote(run_id)}/artifact/"
                   f"{urllib.parse.quote(os.path.basename(name))}",
            "bytes": size,
        }
        for name, size in _walk_artifact(directory)
    ]

    objective = manifest.get("objective") or {}
    winner = manifest.get("winner") or {}
    best = None
    if winner.get("branch_id") is not None:
        best = {
            "branch_id": winner.get("branch_id"),
            "score": objective.get("score"),
            "objective": objective.get("name"),
        }

    return {
        "ready": True,
        "files": files,
        "run_command": run_command(run_id),
        "best": best,
    }


def run_command(run_id: str) -> str:
    """How to fetch the artifact and run the weights, as one copyable block."""
    return (
        f"modal volume get autobranch-data "
        f"/{ARTIFACT_DIRNAME}/{run_id} ./{run_id}\n"
        f"python {run_id}/serve.py"
    )


def _walk_artifact(directory: str) -> list[tuple[str, int]]:
    """Every file in the artifact tree, as `(relative_posix_name, bytes)`."""
    out: list[tuple[str, int]] = []
    for base, _dirs, names in os.walk(directory):
        for name in names:
            path = os.path.join(base, name)
            rel = os.path.relpath(path, directory).replace(os.sep, "/")
            try:
                out.append((rel, os.path.getsize(path)))
            except OSError:
                continue
    out.sort(key=lambda item: (item[0] != "manifest.json", item[0]))
    return out


def resolve_artifact_file(run_id: str, filename: str, artifact_root: str) -> str | None:
    """Map a requested filename onto a real file, or refuse.

    The path is never built by joining user input: the artifact tree is
    enumerated and the request is matched against that listing by basename, so
    traversal is impossible by construction rather than by sanitisation. The
    explicit check still runs first, so a traversal attempt is answered with a
    clear 400 instead of a silent 404.
    """
    if not safe_filename(filename):
        raise ValueError(
            f"{filename!r} is not an artifact filename; names may not contain '/', '..' "
            "or start with a dot"
        )
    directory = artifact_dir(run_id, artifact_root)
    for rel, _size in _walk_artifact(directory):
        if os.path.basename(rel) == filename:
            return os.path.join(directory, *rel.split("/"))
    return None


def safe_filename(name: str) -> bool:
    """Reject anything that is not a plain filename."""
    if not name or "/" in name or "\\" in name or ".." in name or name.startswith("."):
        return False
    return bool(SAFE_FILENAME_RE.match(name))


# ---------------------------------------------------------------------------
# frontend
# ---------------------------------------------------------------------------


def frontend_dir() -> str | None:
    """Where `index.html` lives, local checkout or container.

    Served from the same origin as the API on purpose: a split deployment buys
    a CORS policy, a CSP exception and an auth story for nothing.
    """
    candidates = [
        os.environ.get("AUTOBRANCH_FRONTEND"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"),
        "/root/frontend",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return None


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------


def build_api(
    registry: Registry | None = None,
    *,
    launcher: Callable[[EnvironmentSpec, Constraints, Budget, str], Any] | None = None,
    artifact_root: str | None = None,
) -> Any:
    """The FastAPI app. A factory so tests inject their own state and launcher.

    FastAPI is imported here rather than at module scope so this module stays
    importable in a container that has no ASGI stack -- a branch sandbox
    imports `autobranch.*` and must not need fastapi to do it.
    """
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    reg = registry if registry is not None else Registry()
    launch = launcher if launcher is not None else _null_launcher
    artifacts = artifact_root or os.path.join(DEFAULT_DATA_ROOT, ARTIFACT_DIRNAME)
    bundles = os.path.join(os.path.dirname(os.path.abspath(artifacts)), BUNDLE_DIRNAME)

    api = FastAPI(title="autobranch", version=__version__, docs_url=None, redoc_url=None)
    api.state.registry = reg
    api.state.artifact_root = artifacts
    api.state.bundle_root = bundles

    def _run_or_404(run_id: str):
        run = reg.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return run

    # -- api ---------------------------------------------------------------

    @api.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    @api.post("/api/runs")
    def create_run(body: dict = Body(default=None)):
        payload = body or {}
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        try:
            submission = submit(
                reg, payload, launcher=launch, bundle_root=api.state.bundle_root
            )
        except SubmissionError as exc:
            # The whole problem list, one per line: this is the message that
            # saves someone three hours of training on a broken bundle.
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except LaunchError as exc:
            raise HTTPException(status_code=502, detail=f"run could not be started: {exc}") from None
        return {"run_id": submission.run_id}

    @api.get("/api/runs")
    def list_runs():
        runs = []
        for run in reg.list_runs():
            best = run.best or {}
            runs.append(
                {
                    "run_id": run.run_id,
                    "env_name": run.env_name,
                    "state": run.state,
                    "objective_name": run.objective.get("name", ""),
                    "best_score": best.get("score"),
                    "spend_usd": run.spend_usd,
                    "created_at": run.created_at,
                }
            )
        return {"runs": runs}

    @api.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        _run_or_404(run_id)
        return reg.api_payload(run_id)

    @api.get("/api/runs/{run_id}/artifact")
    def get_artifact(run_id: str):
        run = _run_or_404(run_id)
        listing = artifact_listing(run_id, api.state.artifact_root)
        if listing["best"] is None and run.best:
            # Consolidation has not written a manifest yet, but the registry
            # already knows which branch is ahead; showing it is honest.
            listing["best"] = {
                "branch_id": run.best.get("branch_id"),
                "score": run.best.get("score"),
                "objective": run.objective.get("name", ""),
            }
        return listing

    @api.get("/api/runs/{run_id}/artifact/{filename}")
    def get_artifact_file(run_id: str, filename: str):
        _run_or_404(run_id)
        try:
            path = resolve_artifact_file(run_id, filename, api.state.artifact_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if path is None:
            raise HTTPException(
                status_code=404, detail=f"{filename!r} is not in the artifact for {run_id}"
            )
        return FileResponse(
            path, filename=filename, media_type="application/octet-stream"
        )

    @api.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str):
        run = _run_or_404(run_id)
        if run.state in ("pending", "running"):
            reg.append_event(
                run_id, "warn", "operator stop requested: halting branches and consolidating"
            )
            reg.update_run(run_id, state="stopped")
        return {"ok": True}

    # -- frontend ----------------------------------------------------------

    def _asset(name: str):
        directory = frontend_dir()
        if directory is None or name not in FRONTEND_FILES:
            raise HTTPException(status_code=404, detail=f"no such asset {name!r}")
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"no such asset {name!r}")
        return FileResponse(path, media_type=FRONTEND_FILES[name])

    @api.get("/")
    def index():
        return _asset("index.html")

    @api.get("/{asset}")
    def asset(asset: str):
        return _asset(asset)

    @api.exception_handler(SubmissionError)
    def _submission_error(request, exc):  # pragma: no cover - belt and braces
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return api


def build_app() -> Any:
    """The real app: Modal-backed registry, volume paths, deployed launcher."""
    return build_api(Registry(), launcher=modal_launcher)


def modal_launcher(spec: EnvironmentSpec, constraints: Constraints, budget: Budget,
                   run_id: str) -> str:
    """Spawn the tree driver by *name*, never by importing the entrypoint.

    `app.py` lives outside this package precisely so it is not imported twice
    (DESIGN.md lesson 1). Importing it from here to get at `spawn_run` would
    recreate that bug, so the function is looked up on the deployed app
    instead.
    """
    import modal

    from autobranch import infra

    fn = modal.Function.from_name(infra.APP_NAME, "spawn_run")
    call = fn.spawn(
        run_id=run_id,
        spec_dict=spec.to_dict(),
        constraints_dict=constraints.to_dict(),
        budget_dict=budget.to_dict(),
    )
    return getattr(call, "object_id", "")
