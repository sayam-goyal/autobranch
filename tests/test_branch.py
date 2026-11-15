"""Branch primitives and image construction, with Modal faked out entirely.

No test here touches the network. `modal` is not even imported: importing it
costs about four seconds, and these are the tests that have to stay fast
enough to run on every save. `autobranch.branch` reaches for `modal` lazily
inside functions precisely so that injecting a fake sandbox API is enough;
`autobranch.infra` cannot (it holds module-level Modal objects by design), so
that one module is imported against a stub `modal` in `sys.modules`.
"""

import importlib
import sys
import types

import pytest

from autobranch import branch as B
from autobranch.spec import ImageSpec, Resources


# ---- fakes --------------------------------------------------------------


class RecordingImage:
    """An Image that records the layers asked of it, in order."""

    def __init__(self, calls: list, object_id: str = "im-fake") -> None:
        self.calls = calls
        self.object_id = object_id

    def _step(self, name: str, args: tuple, kwargs: dict) -> "RecordingImage":
        self.calls.append((name, args, kwargs))
        return self

    def apt_install(self, *args, **kwargs):
        return self._step("apt_install", args, kwargs)

    def pip_install(self, *args, **kwargs):
        return self._step("pip_install", args, kwargs)

    def run_commands(self, *args, **kwargs):
        return self._step("run_commands", args, kwargs)

    def env(self, vars):
        return self._step("env", (vars,), {})


class FakeImages:
    """Stands in for `modal.Image`."""

    def __init__(self) -> None:
        self.calls: list = []

    def debian_slim(self, python_version=None):
        self.calls.append(("debian_slim", (), {"python_version": python_version}))
        return RecordingImage(self.calls)

    def from_registry(self, tag, add_python=None, **kwargs):
        self.calls.append(("from_registry", (tag,), {"add_python": add_python}))
        return RecordingImage(self.calls)

    def from_id(self, image_id):
        self.calls.append(("from_id", (image_id,), {}))
        return RecordingImage(self.calls, object_id=image_id)

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]


class FakeReader:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


class FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", code: int = 0) -> None:
        self.stdout = FakeReader(stdout)
        self.stderr = FakeReader(stderr)
        self._code = code

    def wait(self) -> int:
        return self._code


class FakeSandbox:
    _seq = 0

    def __init__(self, **kwargs) -> None:
        FakeSandbox._seq += 1
        self.object_id = f"sb-{FakeSandbox._seq}"
        self.kwargs = kwargs
        self.mounts: list[tuple[str, object]] = []
        self.unmounts: list[str] = []
        self.snapshot_calls: list[tuple[str, dict]] = []
        self.execs: list[tuple[tuple, dict]] = []
        self.terminate_calls = 0
        self._returncode: int | None = None
        self.proc = FakeProc()
        self.unmount_raises = False

    def poll(self) -> int | None:
        return self._returncode

    def snapshot_directory(self, path, **kwargs):
        self.snapshot_calls.append((path, kwargs))
        # The verified behaviour: snapshotting a directory leaves the sandbox
        # running. A fake that terminated here would hide the whole point.
        return RecordingImage([], object_id=f"im-{self.object_id}-{len(self.snapshot_calls)}")

    def mount_image(self, path, image) -> None:
        self.mounts.append((path, image))

    def unmount_image(self, path) -> None:
        if self.unmount_raises:
            raise RuntimeError("nothing mounted at that path")
        self.unmounts.append(path)

    def exec(self, *argv, **kwargs):
        self.execs.append((argv, kwargs))
        return self.proc

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = 0

    def die(self, code: int = 1) -> None:
        self._returncode = code


class FakeSandboxAPI:
    def __init__(self) -> None:
        self.created: list[FakeSandbox] = []

    def create(self, **kwargs) -> FakeSandbox:
        sandbox = FakeSandbox(**kwargs)
        self.created.append(sandbox)
        return sandbox

    def from_id(self, sandbox_id: str) -> FakeSandbox:
        for sandbox in self.created:
            if sandbox.object_id == sandbox_id:
                return sandbox
        raise KeyError(sandbox_id)


@pytest.fixture(autouse=True)
def clean_branch_state():
    B._reset_process_state()
    yield
    B._reset_process_state()


@pytest.fixture
def sandboxes() -> FakeSandboxAPI:
    return FakeSandboxAPI()


@pytest.fixture
def images() -> FakeImages:
    return FakeImages()


@pytest.fixture
def infra(monkeypatch):
    """`autobranch.infra` imported against a stub `modal`."""

    class _Named:
        @staticmethod
        def from_name(name, **kwargs):
            return ("named", name, kwargs)

    stub = types.ModuleType("modal")
    stub.App = type("App", (), {"__init__": lambda self, name: setattr(self, "name", name)})
    stub.Volume = _Named
    stub.Dict = _Named
    stub.Secret = _Named
    stub.Image = FakeImages()
    monkeypatch.setitem(sys.modules, "modal", stub)
    monkeypatch.delitem(sys.modules, "autobranch.infra", raising=False)
    module = importlib.import_module("autobranch.infra")
    yield module
    sys.modules.pop("autobranch.infra", None)


def _root(sandboxes, *, run_id="r1", branch_id="b0") -> B.BranchHandle:
    return B.create_root(
        app=object(),
        image=object(),
        resources=Resources(gpu="H100", cpu=4, memory=16384),
        run_id=run_id,
        branch_id=branch_id,
        sandboxes=sandboxes,
    )


# ---- image layer order --------------------------------------------------


def test_image_layers_put_pip_before_commands(infra):
    """A command edit must never invalidate the pip layer.

    Manifest order lists commands next to pip; build order must not. The
    assertion is on the recorded builder calls, which is the only place the
    ordering is observable without building an image.
    """
    builder = FakeImages()
    spec = ImageSpec(
        base="debian_slim",
        python="3.11",
        apt=("git", "build-essential"),
        pip=("torch==2.8.0",),
        commands=("pip install -e .",),
        env={"TOKENIZERS_PARALLELISM": "false"},
    )

    infra.build_env_image(spec, images=builder)

    assert builder.names == [
        "debian_slim",
        "apt_install",
        "pip_install",
        "run_commands",
        "env",
    ]
    assert builder.names.index("pip_install") < builder.names.index("run_commands")
    assert builder.names.index("env") == len(builder.names) - 1
    assert builder.calls[0][2] == {"python_version": "3.11"}


def test_image_base_registry_reference_uses_from_registry(infra):
    builder = FakeImages()
    spec = ImageSpec(base="nvidia/cuda:12.8.1-devel-ubuntu22.04", python="3.11")

    infra.build_env_image(spec, images=builder)

    assert builder.names[0] == "from_registry"
    assert builder.calls[0][1] == ("nvidia/cuda:12.8.1-devel-ubuntu22.04",)
    assert builder.calls[0][2] == {"add_python": "3.11"}


def test_image_base_unknown_is_rejected(infra):
    with pytest.raises(ValueError, match="debian_slim"):
        infra.build_env_image(ImageSpec(base="ubuntu-ish"), images=FakeImages())


def test_framework_deps_are_injected_and_a_bundle_pin_wins(infra):
    """A sandbox image comes from the submitted manifest, which knows nothing
    about us -- so the framework's own requirements must be injected or the
    branch loop cannot import `modal` and dies before its first action."""
    plain = FakeImages()
    infra.build_env_image(ImageSpec(pip=("torch==2.8.0",)), images=plain)
    pip_args = [c for c in plain.calls if c[0] == "pip_install"][0][1]
    for dist in ("pyyaml", "modal", "numpy"):
        assert any(arg.lower().startswith(dist) for arg in pip_args), dist
    assert "torch==2.8.0" in pip_args

    # A bundle that pins one of ours wins outright: two specs for the same
    # distribution would let the resolver, not us, pick the version.
    pinned = FakeImages()
    infra.build_env_image(ImageSpec(pip=("PyYAML==5.4.1",)), images=pinned)
    pip_args = [c for c in pinned.calls if c[0] == "pip_install"][0][1]
    assert "PyYAML==5.4.1" in pip_args
    assert not any(a.lower().startswith("pyyaml=") and a != "PyYAML==5.4.1" for a in pip_args)


def test_infra_names_are_all_autobranch_prefixed(infra):
    """Guard against ever touching another project's resources."""
    names = [
        infra.APP_NAME,
        infra.VOLUME_NAME,
        infra.RUNS_DICT_NAME,
        infra.BRANCHES_DICT_NAME,
        infra.KIMI_SECRET_NAME,
    ]
    assert all(n == "autobranch" or n.startswith("autobranch-") for n in names)


# ---- branch ids ---------------------------------------------------------


def test_branch_ids_encode_depth_and_parentage():
    root = B.root_branch_id()
    assert root == "b0"
    child = B.child_branch_id(root, 1)
    grandchild = B.child_branch_id(child, 2)
    assert (child, grandchild) == ("b0.1", "b0.1.2")

    assert B.depth_of(root) == 0
    assert B.depth_of(grandchild) == 2
    assert B.parent_of(grandchild) == "b0.1"
    assert B.parent_of(root) is None
    assert B.lineage(grandchild) == ["b0", "b0.1", "b0.1.2"]
    assert B.is_ancestor("b0", grandchild)
    assert not B.is_ancestor("b0.2", grandchild)


def test_branch_ids_sort_numerically_not_lexically():
    ids = ["b0.10", "b0.2", "b0", "b0.1.1", "b0.1"]
    assert sorted(ids, key=B.sort_key) == ["b0", "b0.1", "b0.1.1", "b0.2", "b0.10"]


def test_branch_id_validation():
    with pytest.raises(B.BranchError):
        B.depth_of("branch-7")
    with pytest.raises(B.BranchError):
        B.child_branch_id("b0", 0)


# ---- create / snapshot / fork ------------------------------------------


def test_create_root_tags_sandbox_and_requests_resources(sandboxes):
    handle = _root(sandboxes)

    assert handle.branch_id == "b0"
    assert handle.parent_id is None
    assert handle.depth == 0
    assert handle.snapshot_id is None  # a root's workspace is its own filesystem

    sandbox = sandboxes.created[0]
    assert sandbox.kwargs["gpu"] == "H100"
    assert sandbox.kwargs["cpu"] == 4.0
    assert sandbox.kwargs["memory"] == 16384
    assert sandbox.kwargs["tags"] == {"autobranch_run": "r1", "autobranch_branch": "b0"}


def test_snapshot_leaves_the_parent_running(sandboxes):
    handle = _root(sandboxes)
    parent = sandboxes.created[0]

    snapshot_id = B.snapshot(handle, ttl=60)

    assert snapshot_id.startswith("im-")
    assert parent.poll() is None, "snapshot_directory must not stop the parent"
    assert parent.snapshot_calls == [(B.WORKSPACE, {"ttl": 60})]


def test_snapshot_omits_ttl_when_not_given(sandboxes):
    """`ttl=None` means Modal's 30-day default, not 'retain forever'."""
    handle = _root(sandboxes)
    B.snapshot(handle)
    assert sandboxes.created[0].snapshot_calls == [(B.WORKSPACE, {})]


def test_fork_always_mounts_a_snapshot_never_shares_the_parent(sandboxes, images):
    parent = _root(sandboxes)
    snapshot_id = B.snapshot(parent)

    child = B.fork(
        snapshot_id,
        app=object(),
        image=object(),
        resources={"gpu": None, "cpu": 2, "memory": 8192},
        run_id="r1",
        parent_id=parent.branch_id,
        child_ordinal=1,
        sandboxes=sandboxes,
        images=images,
    )

    assert child.branch_id == "b0.1"
    assert child.depth == 1
    assert child.snapshot_id == snapshot_id
    assert child.sandbox_id != parent.sandbox_id

    child_sandbox = sandboxes.from_id(child.sandbox_id)
    assert [path for path, _ in child_sandbox.mounts] == [B.WORKSPACE]
    assert child_sandbox.mounts[0][1].object_id == snapshot_id
    # The guard that matters: nothing in the fork path points the child at the
    # parent's live directory, because there is no parameter that could.
    assert images.calls == [("from_id", (snapshot_id,), {})]


def test_fork_without_a_snapshot_id_is_rejected(sandboxes, images):
    with pytest.raises(B.TerminatedParentError):
        B.fork(
            "",
            app=object(),
            image=object(),
            resources=None,
            run_id="r1",
            parent_id="b0",
            sandboxes=sandboxes,
            images=images,
        )
    assert sandboxes.created == [], "no sandbox may be billed for a rejected fork"


def test_fork_depth_must_agree_with_the_branch_id(sandboxes, images):
    with pytest.raises(B.BranchError, match="contradicts"):
        B.fork(
            "im-1",
            app=object(),
            image=object(),
            resources=None,
            run_id="r1",
            parent_id="b0",
            depth=7,
            child_ordinal=1,
            sandboxes=sandboxes,
            images=images,
        )


def test_fork_from_live_parent_snapshots_it_first(sandboxes, images):
    parent = _root(sandboxes)

    child = B.fork_from(
        parent,
        app=object(),
        image=object(),
        resources=None,
        run_id="r1",
        child_ordinal=3,
        sandboxes=sandboxes,
        images=images,
    )

    assert child.branch_id == "b0.3"
    assert sandboxes.created[0].snapshot_calls == [(B.WORKSPACE, {})]
    assert child.snapshot_id == f"im-{parent.sandbox_id}-1"
    assert child.sandbox_id != parent.sandbox_id


def test_fork_from_terminated_parent_is_rejected_without_a_snapshot(sandboxes, images):
    parent = _root(sandboxes)
    B.terminate(parent, reason="pruned")

    with pytest.raises(B.TerminatedParentError, match="snapshot_id"):
        B.fork_from(
            parent,
            app=object(),
            image=object(),
            resources=None,
            run_id="r1",
            sandboxes=sandboxes,
            images=images,
        )
    assert len(sandboxes.created) == 1, "the child sandbox must never be created"


def test_fork_from_terminated_parent_works_with_an_explicit_snapshot(sandboxes, images):
    parent = _root(sandboxes)
    snapshot_id = B.snapshot(parent)
    B.terminate(parent, reason="budget")

    child = B.fork_from(
        parent,
        app=object(),
        image=object(),
        resources=None,
        run_id="r1",
        child_ordinal=2,
        snapshot_id=snapshot_id,
        sandboxes=sandboxes,
        images=images,
    )

    assert child.branch_id == "b0.2"
    assert child.snapshot_id == snapshot_id


# ---- rollback / terminate ----------------------------------------------


def test_rollback_unmounts_then_mounts_the_ancestor_snapshot(sandboxes, images):
    handle = _root(sandboxes)
    sandbox = sandboxes.created[0]

    B.rollback(handle, "im-ancestor", images=images)

    assert sandbox.unmounts == [B.WORKSPACE]
    assert [p for p, _ in sandbox.mounts] == [B.WORKSPACE]
    assert sandbox.mounts[0][1].object_id == "im-ancestor"
    assert handle.snapshot_id == "im-ancestor"


def test_rollback_survives_a_root_with_nothing_mounted(sandboxes, images):
    handle = _root(sandboxes)
    sandboxes.created[0].unmount_raises = True

    B.rollback(handle, "im-ancestor", images=images)

    assert sandboxes.created[0].mounts[0][1].object_id == "im-ancestor"


def test_terminate_is_idempotent(sandboxes):
    handle = _root(sandboxes)
    sandbox = sandboxes.created[0]

    B.terminate(handle, reason="abandoned by agent")
    B.terminate(handle, reason="abandoned by pruner")
    B.terminate(handle, reason="run finished")

    assert sandbox.terminate_calls == 1
    assert B.is_terminated(handle)
    # The first reason is the cause; later callers are just cleanup.
    assert B.termination_reason(handle) == "abandoned by agent"


def test_terminate_tolerates_a_sandbox_that_already_exited(sandboxes):
    handle = _root(sandboxes)
    sandboxes.created[0].die(137)

    B.terminate(handle, reason="timeout")

    assert sandboxes.created[0].terminate_calls == 0


# ---- exec_json ----------------------------------------------------------


def test_exec_json_reads_the_result_past_the_trainer_logs(sandboxes):
    _root(sandboxes)
    sandbox = sandboxes.created[0]
    sandbox.proc = FakeProc(
        stdout="step 1 loss=0.4\nstep 2 loss=0.3\n{\"score\": 0.99, \"steps\": 2}\n"
    )

    out = B.exec_json(sandbox, ["python", "train.py"], timeout=600)

    assert out == {"score": 0.99, "steps": 2}
    assert sandbox.execs[-1][0] == ("python", "train.py")


def test_exec_json_raises_with_stderr_on_failure(sandboxes):
    _root(sandboxes)
    sandbox = sandboxes.created[0]
    sandbox.proc = FakeProc(stdout="", stderr="Traceback: boom", code=2)

    with pytest.raises(B.ExecError) as caught:
        B.exec_json(sandbox, ["python", "train.py"], timeout=10)

    assert caught.value.returncode == 2
    assert "boom" in caught.value.stderr


def test_exec_json_raises_when_no_json_is_emitted(sandboxes):
    _root(sandboxes)
    sandbox = sandboxes.created[0]
    sandbox.proc = FakeProc(stdout="all done, no json here\n")

    with pytest.raises(B.ExecError, match="no JSON"):
        B.exec_json(sandbox, ["python", "train.py"], timeout=10)


def test_compute_is_billed_continuously_not_only_when_a_call_returns():
    """A sandbox burns money every second, so billing has to be every second.

    `_tool_train` used to record compute when a train call returned. Two
    failures compounded from that. A 50-minute call billed nothing until it
    finished, so the ledger read $0.01 against a true $2.41 -- understated
    241x. And a sandbox **preempted mid-call never reached the billing code at
    all**: one preempted run cost ~$4.56 and was never recorded.

    The second is the serious one, because the budget cap is enforced against
    the ledger. Unbilled time is time the cap cannot see, which makes "hard
    ceilings, decremented centrally" false exactly when it matters.
    """
    import inspect

    from autobranch import branch_main

    # The biller exists and runs on a timer.
    assert hasattr(branch_main, "start_compute_biller")
    src = inspect.getsource(branch_main.start_compute_biller)
    assert "daemon=True" in src, "accounting must not keep a branch alive"
    assert "sandbox uptime" in src

    # And the per-call record is gone, or the same seconds get charged twice.
    train_src = inspect.getsource(branch_main.BranchRunner._tool_train)
    assert "record_compute" not in train_src, (
        "billing per call on top of continuous billing double-charges training time"
    )


def test_the_biller_marks_time_before_sleeping():
    """It must not double-bill an interval it already charged.

    The mark is advanced in the same statement that computes the interval, so
    a failure inside `record_compute` loses at most one interval instead of
    charging it again on the next pass.
    """
    import inspect

    from autobranch import branch_main

    src = inspect.getsource(branch_main.start_compute_biller)
    assert "seconds, last = now - last, now" in src, src
