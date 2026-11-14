"""Shared Modal objects, and the translation from `ImageSpec` to a real Image.

Every resource this framework creates is prefixed `autobranch-`. That is not
cosmetic: this workspace also runs unrelated training, and a name collision
here would stop somebody else's job. Nothing in this module may look up,
list, or act on a resource it did not name itself.

All of the module-level objects below are *lazy* -- `modal.App(...)` and
`from_name(...)` build local references and perform no network I/O until an
operation needs them -- so importing this module is safe anywhere, including
inside a sandbox that has no credentials yet.
"""

import modal

from .spec import EnvironmentSpec, ImageSpec

APP_NAME = "autobranch"
VOLUME_NAME = "autobranch-data"
RUNS_DICT_NAME = "autobranch-runs"
BRANCHES_DICT_NAME = "autobranch-branches"
KIMI_SECRET_NAME = "autobranch-kimi-auth"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATA_ROOT = "/data"
VOLUMES = {DATA_ROOT: volume}

# Pinned per DESIGN section 7.3: Modal's ASGI bridge cannot serialize
# Starlette 1.x close messages, and the failure mode is a dead websocket
# route rather than an import error, so the pins live in code, not a comment.
WEB_PINS = (
    "fastapi==0.115.12",
    "starlette<0.42",
    "uvicorn==0.34.3",
    "websockets==12.0",
)

# The manifest is YAML, so every image that touches a bundle needs a parser.
BASE_PIP = ("pyyaml==6.0.2",)

PYTHON_VERSION = "3.11"


def base_image() -> modal.Image:
    """The slim image used by the orchestrator and the web app."""
    return modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(*BASE_PIP)


def web_image() -> modal.Image:
    """`base_image()` plus the pinned ASGI stack."""
    return base_image().pip_install(*WEB_PINS)


def kimi_secret() -> modal.Secret:
    """Modal-Key / Modal-Secret for the shared Kimi endpoint."""
    return modal.Secret.from_name(KIMI_SECRET_NAME)


def runs_dict() -> modal.Dict:
    """Run records. Separate from branches so branch churn cannot contend
    with the (rarely written) run header."""
    return modal.Dict.from_name(RUNS_DICT_NAME, create_if_missing=True)


def branches_dict() -> modal.Dict:
    """Branch records and event shards -- the hot, concurrently written half."""
    return modal.Dict.from_name(BRANCHES_DICT_NAME, create_if_missing=True)


def _merge_pip(user_pip: tuple[str, ...]) -> list[str]:
    """`BASE_PIP` first, dropping any base package the bundle already pins.

    Two specs for the same distribution in one `pip_install` makes the
    resolver, not us, decide the version -- so the bundle's pin wins outright.
    """
    requested = {_dist_name(p) for p in user_pip}
    merged = [p for p in BASE_PIP if _dist_name(p) not in requested]
    merged.extend(user_pip)
    return merged


def _dist_name(requirement: str) -> str:
    out = requirement.strip().lower()
    for sep in ("[", "==", ">=", "<=", "~=", "!=", ">", "<", ";", " "):
        out = out.split(sep, 1)[0]
    return out.replace("_", "-").strip()


def build_env_image(spec: ImageSpec, *, images: object = None) -> modal.Image:
    """Turn a declarative `ImageSpec` into a real Modal Image.

    Layer order is the whole point of this function, and it is the opposite of
    the order the manifest lists things in. Modal caches image layers by their
    recipe, so a layer invalidates everything after it:

        apt -> pip -> commands -> env

    `commands` is the field a person edits ten times an hour (`pip install -e .`
    while they fix their package); `pip` is the field that costs four minutes
    to rebuild. Commands therefore go *after* pip, and `env` -- the cheapest
    and most volatile of all -- goes last. Putting them in manifest order
    would make every command tweak re-resolve torch.

    `images` is an injection seam (anything exposing `debian_slim` and
    `from_registry`); it exists so the layer order above can be unit-tested
    without building an image.
    """
    builder = images if images is not None else modal.Image

    base = (spec.base or "debian_slim").strip()
    if base in ("debian_slim", "slim", ""):
        image = builder.debian_slim(python_version=spec.python)
    elif "/" in base or ":" in base:
        # A registry reference. `add_python` installs an interpreter into
        # images (like nvidia/cuda) that ship without one.
        image = builder.from_registry(base, add_python=spec.python)
    else:
        raise ValueError(
            f"image.base {base!r} is neither 'debian_slim' nor a registry "
            "reference (a registry reference contains '/' or ':', e.g. "
            "'nvidia/cuda:12.8.1-devel-ubuntu22.04')"
        )

    if spec.apt:
        image = image.apt_install(*spec.apt)

    pip = _merge_pip(spec.pip)
    if pip:
        image = image.pip_install(*pip)

    if spec.commands:
        image = image.run_commands(*spec.commands)

    if spec.env:
        image = image.env(dict(spec.env))

    return image


def env_secrets(spec: ImageSpec) -> list[modal.Secret]:
    """Resolve the bundle's declared secret names.

    Names come from the manifest, so a missing secret must fail loudly at
    launch rather than as a `KeyError` inside the trainer an hour later;
    `from_name` is lazy, so the failure surfaces on first sandbox create.
    """
    return [modal.Secret.from_name(name) for name in spec.secrets]


def env_volumes(spec: ImageSpec) -> dict[str, modal.Volume]:
    """Resolve the bundle's declared volume mounts, plus our own data volume.

    The bundle's mounts are read-mostly inputs (ROMs, base weights). Our
    `/data` mount is where the ledger and artifacts go, and it is always
    present so a branch never has to ask for it.
    """
    mounts: dict[str, modal.Volume] = {
        path: modal.Volume.from_name(name, create_if_missing=False)
        for path, name in spec.volumes.items()
    }
    mounts.setdefault(DATA_ROOT, volume)
    return mounts


def image_for(spec: EnvironmentSpec, *, images: object = None) -> modal.Image:
    """Convenience: the image for a whole environment spec."""
    return build_env_image(spec.image, images=images)
