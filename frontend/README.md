# autobranch — frontend

Three files, no build step, no dependencies: `index.html`, `styles.css`, `app.js`.
Vanilla ES2020 and system fonts only, so it can be served as static assets from
the same Modal ASGI app that serves the API.

| screen | route | what it does |
|---|---|---|
| Training runs | `#/runs` | the landing tab: which runs are live, which ones scored, what they cost — one table, auto-refreshing while any run is live |
| New run | `#/new` | bundle (`.zip` drop **or** git URL), a parse preview of the manifest, constraints, budget caps |
| Run | `#/run/<id>` | identity + best objective, the sandbox tree, and one switchable region: the selected sandbox, all sandboxes, or activity |
| Artifact | `#/artifact/<id>` | winning objective score, downloads, the "run it" snippet |

Nav is two permanent tabs (**Training runs**, **New run**) plus **Run** and
**Artifact**, which appear once a run is in context and deep-link to it. An
empty hash lands on `#/runs`: the first question the app answers is "what have
I submitted", not "submit something".

## How the run screen is organised

One column, three regions, and never more than one of them asking for the same
glance:

1. **Identity and the number.** The run id, its environment and objective, then
   the best objective score set at `--fs-hero`. That score is the point of the
   product, so it is the largest thing on the page. The state, the elapsed
   clock, the sandbox count and the spend meter sit beside it as qualifiers,
   not as competitors.
2. **The sandbox tree.** The status census in its header doubles as the colour
   legend — a legend listing statuses this run does not have is noise.
3. **One switchable region.** A segmented control over three panes: **Sandbox**
   (the selected one: what it is doing, its numbers, its score history, its
   reward config, its rationale and its own timeline), **All sandboxes** (the
   full table) and **Activity** (the run feed). Picking a sandbox anywhere —
   a tree node, a table row, a branch chip in the feed — brings the Sandbox
   pane forward. Exactly one pane is visible at a time.

Earlier versions stacked a detail panel, a table and a feed permanently around
the tree. That is what made the page feel crowded, and the two-column grid left
a tall column of dead space underneath a short tree.

## How it's served

Everything is same-origin. The page expects the API under `/api` on whatever
host serves `index.html`:

```
GET  /api/runs                 list
POST /api/runs                 create  -> {run_id}, or 400 with the problem list
GET  /api/runs/{id}            poll every 3s while state is pending|running|evaluating
GET  /api/runs/{id}/artifact   downloads + run command
POST /api/runs/{id}/stop       halt and consolidate
```

Hash routing is deliberate: there are no server-side paths to configure and a
deep link to `#/run/<id>` survives a refresh.

## Sandboxes: id and activity

A node in the tree **is** a Modal sandbox. Two fields answer the only two
questions a watcher has, and each has a published source and a derived
fallback, because the published ones only exist once the backend writes them:

* **sandbox id** — `branch.sandbox_id` when `api_payload` carries it; otherwise
  the last `sb-…` id this branch's events mentioned (`launch_root` and the
  forker both name the sandbox they just created). Shown short
  (`sb-01M7…E8FN`), in full in the Sandbox pane and in the row's `title`.
* **activity** — the "doing now" line, in order of trust:
  1. `branch.activity`, written by the sandbox *before* it starts the work, so
     it stays correct across a long `train` call where the newest event still
     describes the previous step;
  2. the newest event carrying that `branch_id`, lightly rewritten — the branch
     id prefix is dropped, `reward -> …` becomes "wrote a new reward · …",
     `objective success_rate = 0.6484 at step 114688` becomes "scored
     success_rate = 0.6484 at step 114,688", a traceback is collapsed to one
     line. Anything unrecognised passes through untouched;
  3. the status, when a branch has no events yet — `pending` reads "sandbox
     starting" in a live run and "sandbox never started" in a finished one.

The same line appears on the node, in the `Doing now` block of the Sandbox
pane, and in the sandbox table's last column. The Sandbox pane also carries
that sandbox's own event timeline and the objective-score sparkline.

## The tree: nothing floats, and the canvas is the content

Two guarantees, both enforced in `layoutTree()` rather than hoped for.

**Nothing floats.** A branch id encodes its own lineage by construction —
`b0` → `b0.1` → `b0.1.2` — so a parent is always derivable by stripping the
last `.N` segment. `resolveParent()` tries, in order:

1. `parent_id`, when that branch is actually present in the payload. Drawn as
   a **solid** edge: the link was recorded.
2. the **id-derived** ancestor chain, walking `b0.1.1` → `b0.1` → `b0` until it
   reaches a branch that is present. Any intermediate id that is genuinely
   absent becomes a visible `MISSING` placeholder node, so the chain still
   reaches all the way down. Drawn **dashed**: the link was inferred.
3. the **single root**, for an id with no derivable ancestor at all. Drawn as a
   dashed `unlinked` edge — a second root is never started.

A `parent_id` cycle is cut rather than allowed to leave the forest rootless,
and the root is picked deterministically (a real branch over a placeholder,
then the shortest id). `failed` and `abandoned` branches are **ordinary members
of the tree** — same position, same edge to their parent, only de-emphasised
paint — which is the specific bug this replaced: a dead branch used to read as
a floating box because its edge was drawn in `--border` dashed over a
`--border` grid-paper background, i.e. invisible. Spent edges are now lighter
but never invisible, the grid paper is gone, and dashed means exactly one
thing: inferred.

Finally the layout asserts what it just promised — one root, and every other
node owning exactly one edge — and `console.error`s the audit if that ever
breaks, instead of silently drawing a floating box. The audit is parked on
`window.AUTOBRANCH_TREE_AUDIT` (`{nodes, edges, root, roots, placeholders,
inferred, orphans, notes, ok}`) so it can be inspected from the console, and
the tree's footer says in plain English when links had to be inferred.

**The canvas is the content.** Row depth comes from the *resolved* tree, not
from `branch.depth`, so an edge can never point upwards. The svg's `width`,
`height` and `viewBox` are computed from the node bounding box plus a 14px pad
— never a fixed height — and `.tree-scroll` takes its height from the svg. A
one-node tree is therefore a 104px strip, not a half-screen canvas, and a tall
tree scrolls inside its own container at `min(60vh, 600px)`.

## Polling

One poller exists at a time. Routing bumps a generation counter that
invalidates in-flight responses and clears the scheduled fetch, so two views
can never be fetching at once.

* **Run view** — `GET /api/runs/{id}` every 3s, and **nothing** once the run
  state is terminal. A finished run has nothing left to say.
* **Runs list** — `GET /api/runs` every 3s, plus **at most one**
  `GET /api/runs/{id}` per cycle to fill in a sandbox count, live runs first.
  Twenty runs therefore cost two requests per cycle, not twenty-one. A run
  whose tree has not been read yet shows a dash in the Sandboxes column.
* Elapsed time and row ages move every second off a local clock — that is
  arithmetic on numbers already fetched, not a request.

## The manifest preview

`POST /api/runs` validates in the browser's round trip, but the New Run screen
also parses `autobranch.yaml` locally the moment a `.zip` is picked, and shows
the environment, the declared signals with their docs, the objective and its
direction, the trainer and the resource request — the contract as the agents
will see it — before anything is submitted.

Two small readers in `app.js` do this, both free of dependencies:

* a zip reader that scans the archive tail for the end-of-central-directory
  record, walks the central directory and slices out exactly one entry,
  inflating with the platform's `DecompressionStream("deflate-raw")`. The
  manifest is looked for at the bundle root or one directory below it, matching
  `server.find_manifest` (a GitHub zip nests everything under `repo-branch/`);
* a small YAML reader: block mappings and sequences, `>` and `|` block scalars,
  one-line flow collections, quoted and plain scalars, comments. Anchors,
  aliases, tags and multiple documents throw rather than being half-supported.
  One deliberate deviation: a folded scalar comes back without its trailing
  newline.

`previewManifest()` then mirrors `autobranch/spec.py` — same regexes, same
rules, same `path: message` problem shape — with two differences, both
deliberate: it collects every problem instead of raising on the first, and it
is **advisory**. Nothing here blocks a submit. Every failure path (not a zip,
zip64, no manifest, YAML this reader does not do) degrades to a note saying the
server will parse it on submit.

`preview the gridworld reference manifest` renders the shipped example through
the same reader, so the screen can be read with no bundle in hand.

## Problems, field-adjacent

`SubmissionError` joins its problems with newlines and the route answers 400
with all of them. The client splits that back into the list and routes each
line by its path — `manifest.signals[2]`, `manifest.objective.direction`,
`budget`, `entrypoint` — to the section or field it is about: the preflight
card gets a red border and the message inline, a flagged signal gets its row
highlighted, a budget problem marks the three inputs `aria-invalid`. The full
list is always repeated above the submit button, each line tagged with where it
landed.

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

The fixture (in `app.js`) is a nine-sandbox, four-level tree on
`street-fighter-iii` with two abandoned lines and the 0.000 → 0.990 arc. It is
not static: each poll advances elapsed time and spend, appends evaluations,
and — for a run you create through the form — forks new sandboxes in one at a
time, so the tree, the activity feed, and the meter all move. Fixture branches
carry `sandbox_id` and `activity` so the live-activity path is exercised with
no backend at all. `Stop run` works, and the artifact view becomes available
once a branch has succeeded. The JSON downloads in mock mode are real data URLs
generated from the fixture.

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
colours (`--st-running`, `--st-evaluating`, `--st-pending`, `--st-succeeded`,
`--st-abandoned`, `--st-failed`) are separate tokens so semantics never ride on
the brand colour. Every status renders distinctly in the tree, the tables and
the status census; only genuinely live ones pulse, and
`prefers-reduced-motion` removes the animation entirely. `--tree-line` is its
own token because an edge has to stay legible against `--bg-subtle` in both
themes, which `--border` did not.

Wide content — both tables, the tree canvas, the signals table, code blocks —
scrolls inside its own container. The page body never scrolls sideways, at any
width.

## Type and spacing

Two short scales, declared once at the top of `styles.css` and used everywhere.
Six text sizes — `--fs-micro` 10, `--fs-xs` 11, `--fs-sm` 12, `--fs-md` 13 (the
document default), `--fs-lg` 15, plus `--fs-h1` and `--fs-hero`, both clamped —
and six spacing steps, `--sp-1` 4px through `--sp-6` 36px. The near-duplicates
the stylesheet used to carry (9/14/16px, and paddings at 0.45/0.6/0.65/0.7/0.8/
0.85/0.9rem) were collapsed onto them.

Structure is `.region`: one box per region, a hairline head, a body, an
optional hairline foot — never a card inside a card. `.region-flat` drops the
outer border for a full-width table, which wants hairlines and nothing else.
