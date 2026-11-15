"""Web backend tests, against an in-memory registry and a fake launcher.

The ones that matter are the rejections. A submission gate that accepts a
broken bundle costs hours: in the project this framework generalises, six
training runs completed against a measurement that was wrong from the first
minute. So a malformed manifest, an environment that disagrees with its own
manifest, a bundle with no manifest and an oversized upload are each asserted
to fail *here*, with the problem text a person can act on, rather than inside
a sandbox later.

The traversal tests are the other half of the same argument: the artifact
endpoint hands out files, and the only safe filename is one that was
enumerated from the artifact directory rather than pasted by a caller.
"""

import base64
import io
import json
import os
import zipfile

import pytest

try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"the ASGI stack is not installed ({exc}); install the pinned extra with "
        "`pip install -e '.[serve]'` to run the web backend tests",
        allow_module_level=True,
    )

from autobranch import server
from autobranch.registry import MemoryBackend, Registry

# ---- bundle fixtures ----------------------------------------------------

#: A two-lane environment that agrees with MANIFEST exactly: declared signals
#: only, [n_envs, obs_dim] observations, and reproducible from a fixed seed.
#: `n_envs = 2` rather than 1 on purpose (DESIGN.md lesson 10).
GOOD_ENV = '''\
"""A tiny deterministic environment, used only by the server tests."""

import numpy as np

from autobranch.env_api import StepResult


class TinyEnv:
    def __init__(self, n_envs=2):
        self._n = n_envs
        self._t = 0

    @property
    def n_envs(self):
        return self._n

    def reset(self, *, seed=None):
        self._t = 0
        return np.zeros((self._n, 3), dtype=np.float64)

    def step(self, action):
        self._t += 1
        obs = np.full((self._n, 3), float(self._t), dtype=np.float64)
        return StepResult(
            obs=obs,
            signals={
                "hit": np.ones(self._n, dtype=np.float64),
                "cost": np.full(self._n, 0.5, dtype=np.float64),
            },
            terminated=np.zeros(self._n, dtype=bool),
            truncated=np.zeros(self._n, dtype=bool),
        )

    def close(self):
        return None


def make_env(**kwargs):
    return TinyEnv()


def evaluate(*args, **kwargs):
    return 0.0
'''

#: The same environment with one signal renamed, so the manifest and the code
#: disagree: `validate_env` must report both halves of the mismatch.
BAD_ENV = GOOD_ENV.replace('"hit"', '"bonus"')

MANIFEST = """\
name: tiny
version: 1
entrypoint: tiny_env:make_env

image:
  base: debian_slim
  pip: [numpy>=1.26]

signals:
  - {name: hit,  type: bool,  range: [0, 1], doc: "1 when the lane scored this step"}
  - {name: cost, type: float, range: [0, 1], doc: "time cost, one per step"}

objective:
  name: hit_rate
  direction: maximize
  evaluator: tiny_env:evaluate
  budget: {episodes: 4}

policy: {kind: discrete, n_actions: 2, obs_dim: 3}
trainer: ppo
"""

#: Missing the whole `objective` block: the one thing agents may never touch
#: and the manifest may never omit.
MANIFEST_NO_OBJECTIVE = "\n".join(
    MANIFEST.splitlines()[: MANIFEST.splitlines().index("objective:")]
)

BUDGET = {"usd": 25, "hours": 2, "max_branches": 4}


def make_zip(files: dict, *, prefix: str = "") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(prefix + name, text)
    return buffer.getvalue()


def payload(zip_bytes: bytes, **overrides) -> dict:
    body = {
        "bundle_b64": base64.b64encode(zip_bytes).decode("ascii"),
        "constraints": "stay under $25",
        "budget": dict(BUDGET),
    }
    body.update(overrides)
    return body


GOOD_BUNDLE = {"autobranch.yaml": MANIFEST, "tiny_env.py": GOOD_ENV}


# ---- fixtures -----------------------------------------------------------


class FakeLauncher:
    """Records what the API asked to launch, and launches nothing."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, spec, constraints, budget, run_id):
        self.calls.append(
            {
                "spec": spec,
                "constraints": constraints,
                "budget": budget,
                "run_id": run_id,
            }
        )


@pytest.fixture
def registry() -> Registry:
    return Registry(backend=MemoryBackend(), writer_id="w-test")


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
def artifacts(tmp_path) -> str:
    return str(tmp_path / "artifacts")


@pytest.fixture
def client(registry, launcher, artifacts):
    api = server.build_api(registry, launcher=launcher, artifact_root=artifacts)
    with TestClient(api) as test_client:
        yield test_client


def submit_good(client) -> str:
    response = client.post("/api/runs", json=payload(make_zip(GOOD_BUNDLE)))
    assert response.status_code == 200, response.text
    return response.json()["run_id"]


# ---- health and frontend ------------------------------------------------


def test_health_reports_a_version(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["version"]


def test_root_serves_the_frontend_html(client):
    response = client.get("/")
    assert response.status_code == 200, response.text
    assert "text/html" in response.headers["content-type"]
    assert "<title>" in response.text


def test_assets_are_same_origin_with_their_own_content_types(client):
    for name, expected in (("app.js", "javascript"), ("styles.css", "text/css")):
        response = client.get("/" + name)
        assert response.status_code == 200, name
        assert expected in response.headers["content-type"]


# ---- submission: the happy path -----------------------------------------


def test_valid_bundle_returns_a_run_id_and_launches_once(client, registry, launcher):
    run_id = submit_good(client)

    assert len(launcher.calls) == 1
    call = launcher.calls[0]
    assert call["run_id"] == run_id
    assert call["spec"].name == "tiny"
    assert call["budget"].usd == 25
    assert call["constraints"].text == "stay under $25"

    run = registry.get_run(run_id)
    assert run is not None
    assert run.env_name == "tiny"
    assert run.objective["name"] == "hit_rate"
    assert any("bundle accepted" in e["message"] for e in registry.events(run_id))


def test_bundle_nested_one_level_down_is_accepted(client):
    """A GitHub zip always nests under `repo-branch/`; rejecting that rejects git."""
    nested = make_zip(GOOD_BUNDLE, prefix="tiny-main/")
    response = client.post("/api/runs", json=payload(nested))
    assert response.status_code == 200, response.text


def test_the_validated_bytes_are_kept_for_the_branch_to_fetch(client, artifacts):
    run_id = submit_good(client)
    stored = os.path.join(os.path.dirname(artifacts), "bundles", f"{run_id}.zip")
    assert os.path.isfile(stored)
    with zipfile.ZipFile(stored) as archive:
        assert "autobranch.yaml" in archive.namelist()


# ---- submission: the rejections -----------------------------------------


def test_malformed_manifest_is_400_with_the_spec_error(client, launcher):
    bundle = make_zip({"autobranch.yaml": MANIFEST_NO_OBJECTIVE, "tiny_env.py": GOOD_ENV})
    response = client.post("/api/runs", json=payload(bundle))
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "missing required key" in detail
    assert "objective" in detail
    assert launcher.calls == []


def test_env_that_disagrees_with_its_manifest_is_400_with_every_problem(client, launcher):
    bundle = make_zip({"autobranch.yaml": MANIFEST, "tiny_env.py": BAD_ENV})
    response = client.post("/api/runs", json=payload(bundle))
    assert response.status_code == 400
    detail = response.json()["detail"]
    # Both halves of the mismatch, not just the first one found.
    assert "'hit' was not emitted" in detail
    assert "'bonus' is emitted but not declared" in detail
    assert launcher.calls == []


def test_bundle_without_a_manifest_is_400(client, launcher):
    bundle = make_zip({"tiny_env.py": GOOD_ENV, "README.md": "no manifest here"})
    response = client.post("/api/runs", json=payload(bundle))
    assert response.status_code == 400
    assert "autobranch.yaml" in response.json()["detail"]
    assert launcher.calls == []


def test_manifest_three_levels_down_is_not_hunted_for(client):
    bundle = make_zip(GOOD_BUNDLE, prefix="repo-main/envs/tiny/")
    response = client.post("/api/runs", json=payload(bundle))
    assert response.status_code == 400
    assert "autobranch.yaml" in response.json()["detail"]


def test_oversized_bundle_is_rejected_before_it_is_decoded(client, monkeypatch, launcher):
    # The cap is read at call time, so the test does not have to allocate
    # 64 MiB to prove a 64 MiB cap works.
    monkeypatch.setattr(server, "MAX_BUNDLE_BYTES", 1024)
    # Incompressible padding: a zip of repeated "x" would stay under the cap
    # and the test would pass for the wrong reason.
    bundle = make_zip({"autobranch.yaml": MANIFEST, "pad.bin": os.urandom(8192).hex()})
    response = client.post("/api/runs", json=payload(bundle))
    assert response.status_code == 400
    assert "cap" in response.json()["detail"]
    assert launcher.calls == []


def test_a_zip_bomb_is_refused_before_extraction(tmp_path):
    """Compressed size says nothing about extracted size."""
    bomb = make_zip({"autobranch.yaml": MANIFEST, "pad.txt": "x" * 2_000_000})
    assert len(bomb) < 10_000
    with pytest.raises(server.SubmissionError) as excinfo:
        server.extract_bundle(bomb, str(tmp_path), max_bytes=4096)
    assert "expands to over" in str(excinfo.value)
    assert not os.listdir(tmp_path)


def test_not_a_zip_is_rejected(client):
    response = client.post("/api/runs", json=payload(b"this is not a zip file"))
    assert response.status_code == 400
    assert "zip" in response.json()["detail"]


def test_both_or_neither_bundle_source_is_rejected(client):
    neither = {"constraints": "x", "budget": dict(BUDGET)}
    assert client.post("/api/runs", json=neither).status_code == 400

    both = payload(make_zip(GOOD_BUNDLE), bundle_url="https://example.com/x.zip")
    assert client.post("/api/runs", json=both).status_code == 400


def test_nonpositive_budget_is_rejected(client):
    body = payload(make_zip(GOOD_BUNDLE), budget={"usd": 0, "hours": 2})
    response = client.post("/api/runs", json=body)
    assert response.status_code == 400
    assert "budget" in response.json()["detail"]


def test_zip_slip_entries_never_reach_the_filesystem(tmp_path):
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../escaped.txt", "pwned")
    with pytest.raises(server.SubmissionError) as excinfo:
        server.extract_bundle(evil.getvalue(), str(tmp_path))
    assert "escapes the bundle directory" in str(excinfo.value)
    assert not (tmp_path.parent / "escaped.txt").exists()


# ---- reads --------------------------------------------------------------


def test_unknown_run_is_404_everywhere(client):
    assert client.get("/api/runs/run-nope").status_code == 404
    assert client.get("/api/runs/run-nope/artifact").status_code == 404
    assert client.get("/api/runs/run-nope/artifact/manifest.json").status_code == 404
    assert client.post("/api/runs/run-nope/stop").status_code == 404


def test_run_payload_key_set_is_exactly_the_documented_one(client):
    run_id = submit_good(client)
    body = client.get(f"/api/runs/{run_id}").json()
    assert set(body) == {
        "run_id",
        "env_name",
        "objective",
        "state",
        "created_at",
        "elapsed_s",
        "spend_usd",
        "budget",
        "best",
        "branches",
        "events",
    }
    assert set(body["objective"]) == {"name", "direction"}


def test_run_listing_shape(client):
    run_id = submit_good(client)
    body = client.get("/api/runs").json()
    assert [r["run_id"] for r in body["runs"]] == [run_id]
    assert set(body["runs"][0]) == {
        "run_id",
        "env_name",
        "state",
        "objective_name",
        "best_score",
        "spend_usd",
        "created_at",
    }
    assert body["runs"][0]["best_score"] is None


# ---- artifacts ----------------------------------------------------------


def test_artifact_is_not_ready_rather_than_missing(client):
    run_id = submit_good(client)
    body = client.get(f"/api/runs/{run_id}/artifact").json()
    assert body == {"ready": False, "files": [], "run_command": "", "best": None}


def test_artifact_lists_and_serves_files_once_consolidated(client, artifacts):
    run_id = submit_good(client)
    directory = os.path.join(artifacts, run_id)
    os.makedirs(os.path.join(directory, "weights"))
    manifest = {
        "objective": {"name": "hit_rate", "score": 0.99, "direction": "maximize"},
        "winner": {"branch_id": "b0.1", "checkpoint": "policy.pt"},
    }
    with open(os.path.join(directory, "manifest.json"), "w") as handle:
        json.dump(manifest, handle)
    with open(os.path.join(directory, "weights", "policy.pt"), "wb") as handle:
        handle.write(b"weights-bytes")

    body = client.get(f"/api/runs/{run_id}/artifact").json()
    assert body["ready"] is True
    assert body["best"] == {"branch_id": "b0.1", "score": 0.99, "objective": "hit_rate"}
    assert run_id in body["run_command"]

    names = {f["name"]: f for f in body["files"]}
    assert set(names) == {"manifest.json", "weights/policy.pt"}
    assert names["weights/policy.pt"]["bytes"] == len(b"weights-bytes")

    # A nested file is downloaded by basename: the download endpoint refuses
    # anything with a slash in it, so the listing's url is the contract.
    download = client.get(names["weights/policy.pt"]["url"])
    assert download.status_code == 200
    assert download.content == b"weights-bytes"


def test_artifact_filename_traversal_is_refused(client):
    run_id = submit_good(client)
    for attempt in ("../../etc/passwd", ".env", "..", "weights/policy.pt", "a/b"):
        with pytest.raises(ValueError):
            server.resolve_artifact_file(run_id, attempt, "/does/not/matter")

    # Over HTTP the same attempts must never return a file. Percent-encoded
    # and plain forms are both covered; either a 400 from the filename guard
    # or a 404 from routing is a refusal, a 200 would not be.
    for attempt in (
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%2F..%2Fetc%2Fpasswd",
        "../../etc/passwd",
        ".env",
    ):
        response = client.get(f"/api/runs/{run_id}/artifact/{attempt}")
        assert response.status_code in (400, 404), attempt
        assert b"root:" not in response.content


def test_unknown_artifact_file_is_404_not_400(client, artifacts):
    run_id = submit_good(client)
    os.makedirs(os.path.join(artifacts, run_id), exist_ok=True)
    response = client.get(f"/api/runs/{run_id}/artifact/nope.json")
    assert response.status_code == 404


# ---- stop ---------------------------------------------------------------


def test_stop_sets_state_and_records_an_event(client, registry):
    run_id = submit_good(client)
    response = client.post(f"/api/runs/{run_id}/stop")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    assert registry.get_run(run_id).state == "stopped"
    assert any("stop requested" in e["message"] for e in registry.events(run_id))

    # Idempotent: stopping a stopped run is still ok and adds no second event.
    before = len(registry.events(run_id))
    assert client.post(f"/api/runs/{run_id}/stop").json() == {"ok": True}
    assert len(registry.events(run_id)) == before


# ---- url bundles --------------------------------------------------------


def test_github_repo_urls_are_rewritten_to_codeload():
    assert server.bundle_urls("https://github.com/owner/repo") == [
        "https://codeload.github.com/owner/repo/zip/refs/heads/main",
        "https://codeload.github.com/owner/repo/zip/refs/heads/master",
    ]
    assert server.bundle_urls("https://github.com/owner/repo/tree/dev") == [
        "https://codeload.github.com/owner/repo/zip/refs/heads/dev"
    ]
    assert server.bundle_urls("https://example.com/x.zip") == ["https://example.com/x.zip"]


def test_a_plain_zip_url_is_fetched_and_accepted(launcher, tmp_path):
    data = make_zip(GOOD_BUNDLE)

    class FakeResponse:
        headers = {"Content-Length": str(len(data))}

        def __init__(self) -> None:
            self._body = io.BytesIO(data)

        def read(self, size=-1):
            return self._body.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fetched = []

    def opener(url, timeout=None):
        fetched.append(url)
        return FakeResponse()

    body = server.submit(
        Registry(backend=MemoryBackend(), writer_id="w-url"),
        {
            "bundle_url": "https://example.com/tiny.zip",
            "constraints": "",
            "budget": dict(BUDGET),
        },
        launcher=launcher,
        bundle_root=str(tmp_path / "bundles"),
        opener=opener,
    )
    assert fetched == ["https://example.com/tiny.zip"]
    assert body.run_id.startswith("run-")
    assert os.path.isfile(body.bundle_path)
    assert len(launcher.calls) == 1


def test_a_real_runs_nested_artifact_file_downloads_by_basename(client, registry, artifacts):
    """`consolidate` writes the checkpoint under `weights/`, so the listing
    names it `weights/latest.pt` while the download route carries one path
    segment and matches by basename -- the traversal surface is an enumerated
    tree, never a path joined from a request.

    This pins that a file in a subdirectory is actually reachable. It was not:
    the UI sent the full name, encodeURIComponent turned the slash into %2F,
    the request matched no route, and every file under `weights/` 404'd -- the
    one file a run exists to produce. Fetching a known basename directly is
    what let that survive earlier checks, because the bug was in the
    relationship between the two endpoints rather than in either one.
    """
    import os

    run_id = submit_good(client)
    directory = os.path.join(artifacts, run_id)
    os.makedirs(os.path.join(directory, "weights"), exist_ok=True)
    with open(os.path.join(directory, "weights", "latest.pt"), "wb") as fh:
        fh.write(b"not-a-real-checkpoint-but-real-bytes")
    with open(os.path.join(directory, "manifest.json"), "w") as fh:
        fh.write('{"ok": true}')

    listing = client.get(f"/api/runs/{run_id}/artifact").json()
    names = {f["name"] for f in listing["files"]}
    assert "weights/latest.pt" in names, names

    for entry in listing["files"]:
        leaf = os.path.basename(entry["name"])
        got = client.get(f"/api/runs/{run_id}/artifact/{leaf}")
        assert got.status_code == 200, f"{entry['name']} -> {got.status_code} {got.text[:120]}"
        assert len(got.content) == entry["bytes"]

    # The full path stays refused, which is what keeps traversal impossible:
    # the client strips the directory and the server never rejoins one.
    assert client.get(f"/api/runs/{run_id}/artifact/weights%2Flatest.pt").status_code == 404


def test_the_frontend_requests_artifact_files_by_basename():
    """Pins the client half of the contract above.

    The server's basename matching is deliberate, so the client has to strip
    the directory. A server-only test passes while the UI download button is
    broken, which is exactly what happened.
    """
    import pathlib

    app_js = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "app.js"
    source = app_js.read_text()
    assert 'split("/").pop()' in source, "app.js must request the basename"
    assert "encodeURIComponent(file.name)" not in source, (
        "encoding the full name percent-encodes the slash and matches no route"
    )
