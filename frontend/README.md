# autobranch — frontend

Three files, no build step, no dependencies: `index.html`, `styles.css`, `app.js`.
Vanilla ES2020 and system fonts only, so it can be served as static assets from
the same Modal ASGI app that serves the API.

| screen | route | what it does |
|---|---|---|
| New run | `#/new` | bundle (`.zip` drop **or** git URL — mutually exclusive), constraints, budget caps |
| Run | `#/run/<id>` | branch tree, per-branch detail, branch table, event log, spend meter |
| Artifact | `#/artifact/<id>` | winning objective score, downloads, the "run it" snippet |

## How it's served

Everything is same-origin. The page expects the API under `/api` on whatever
host serves `index.html`:

```
GET  /api/runs                 list
POST /api/runs                 create  -> {run_id}
GET  /api/runs/{id}            poll every 3s while state is pending|running|evaluating
GET  /api/runs/{id}/artifact   downloads + run command
POST /api/runs/{id}/stop       halt and consolidate
```

Mount the directory in the ASGI app next to the API router — for example
`app.mount("/", StaticFiles(directory="frontend", html=True))` after the API
routes are registered — and the hash router handles the rest. Hash routing is
deliberate: there are no server-side paths to configure and a deep link to
`#/run/<id>` survives a refresh.

Locally:

```sh
cd frontend
python3 -m http.server 8080
# then open http://localhost:8080/#/new
```

## Pointing it at a different API base

Two `window` globals, read once at load. Set them from the console (then
reload), or from a one-line `<script>` placed **before** `app.js`:

```html
<script>
  window.AUTOBRANCH_API_BASE = "https://your-workspace--autobranch.modal.run";
  window.AUTOBRANCH_MOCK = false;
</script>
```

* `AUTOBRANCH_API_BASE` — origin (no trailing slash) prefixed to every request.
  Default `""`, i.e. same-origin. A cross-origin base needs CORS on the backend.
  The footer prints the base in use.
* `AUTOBRANCH_MOCK` — `true` serves the built-in fixture and never touches the
  network.

## Mock mode

The client latches into mock mode automatically on a network error, a 404, or a
5xx, and shows a `MOCK DATA` badge in the top bar. 4xx responses other than 404
are treated as real backend errors and surfaced to the user rather than hidden
behind the fixture.

The fixture (in `app.js`) is a nine-branch, four-level tree on
`street-fighter-iii` with two abandoned lines and the 0.000 → 0.990 arc. It is
not static: each poll advances elapsed time and spend, appends evaluations,
and — for a run you create through the form — forks new branches in one at a
time, so the tree, the log, and the meter all move. `Stop run` works, and the
artifact view becomes available once a branch has succeeded. The JSON downloads
in mock mode are real data URLs generated from the fixture.

To review the UI with no backend at all:

```sh
cd frontend
python3 -m http.server 8080
# open http://localhost:8080/#/run/run_8fc2a1
# if a backend IS running on :8080 and you still want the fixture, open the
# console and run: window.AUTOBRANCH_MOCK = true; location.reload()
```

## Theming

Light is the base palette on bare `:root`; dark is declared twice — once under
`@media (prefers-color-scheme: dark)` guarded as `:root:not([data-theme="light"])`
and once under `:root[data-theme="dark"]` — so the toggle wins in both
directions and no colour is defined only inside a conditional block. The toggle
in the top bar cycles system → light → dark and persists to `localStorage`
under `autobranch:theme`.

Accent is a single token, `--accent` (`#0b5bff` light, `#4d8dff` dark). Status
colours (`--st-running`, `--st-succeeded`, `--st-abandoned`, `--st-failed`,
`--st-evaluating`, `--st-pending`) are separate tokens so semantics never ride
on the brand colour.
