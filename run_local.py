"""Serve the real app locally against the live Modal registry.

    python run_local.py          # http://localhost:8080

Same code as the deployed container: the frontend and the API come from one
origin, and run/branch state is read from the same `modal.Dict` the sandboxes
write to -- so a run started on Modal appears here live. Artifact downloads
need the volume and will 404 locally; everything else is real.
"""

import os

import uvicorn

os.environ.setdefault("AUTOBRANCH_DATA_ROOT", os.path.abspath(".local-data"))
os.makedirs(os.environ["AUTOBRANCH_DATA_ROOT"], exist_ok=True)

from autobranch.server import build_app  # noqa: E402  (after env setup)

if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=8080, log_level="info")
