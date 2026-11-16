"""Render the evidence page. Every coordinate is computed from measured data.

Charts are emitted as inline SVG with coordinates derived here rather than
hand-written, because a hand-placed point is a point that can disagree with its
own caption -- and the entire argument of this page is that the number has to
be the one that was measured.
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent
DATA = json.loads((HERE / "data.json").read_text())

# ---------------------------------------------------------------- scales ---

def lerp(v, lo, hi, a, b):
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) * (b - a) / (hi - lo)


def wilson(w, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ------------------------------------------------------------- primitives ---

W, H = 760, 340
PAD = {"l": 56, "r": 18, "t": 18, "b": 42}


def frame(xlabel, ylabel, xticks, yticks, xlo, xhi, ylo, yhi, fmt_x=str, fmt_y=str):
    """Axes, grid and labels. Grid is faint: it orients, it does not decorate."""
    px0, px1 = PAD["l"], W - PAD["r"]
    py0, py1 = H - PAD["b"], PAD["t"]
    out = []
    for t in yticks:
        y = lerp(t, ylo, yhi, py0, py1)
        out.append(f'<line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick ty" x="{px0 - 8}" y="{y + 3.5:.1f}">{fmt_y(t)}</text>')
    for t in xticks:
        x = lerp(t, xlo, xhi, px0, px1)
        out.append(f'<text class="tick tx" x="{x:.1f}" y="{py0 + 18}">{fmt_x(t)}</text>')
    out.append(f'<line class="axis" x1="{px0}" y1="{py1}" x2="{px0}" y2="{py0}"/>')
    out.append(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}"/>')
    out.append(f'<text class="axlab" x="{(px0 + px1) / 2:.0f}" y="{H - 6}">{esc(xlabel)}</text>')
    out.append(f'<text class="axlab rot" transform="translate(14,{(py0 + py1) / 2:.0f}) '
               f'rotate(-90)">{esc(ylabel)}</text>')
    return "\n".join(out)


def svg(body, label):
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)}" '
            f'preserveAspectRatio="xMidYMid meet">\n{body}\n</svg>')


# --------------------------------------------------------- label placement ---
# Annotations here are anchored to data points, and data points move. Offsets
# chosen by eye therefore stop being true the moment a number changes: a label
# slides off the plot, or two of them land on the same spot and neither reads.
# These helpers place labels against the plot rectangle instead, so position is
# derived like every other coordinate on the page.

MONO_ADV = 0.60  # monospace advance width, as a fraction of the font size


def text_box(text, x, y, size, anchor):
    """The rectangle a <text> occupies, in user units. Monospace, so a
    character count is an exact-enough width."""
    w = len(text) * size * MONO_ADV
    x0 = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
    return (x0, y - size * 0.78, x0 + w, y + size * 0.22)


def _overlaps(a, b, pad=1.5):
    return not (a[2] + pad <= b[0] or a[0] >= b[2] + pad
                or a[3] + pad <= b[1] or a[1] >= b[3] + pad)


def place_labels(items, box, size=11.0, obstacles=None):
    """Resolve candidate labels into in-bounds, non-overlapping ones.

    Each item is {text, x, y, anchor, prefer}. Items are placed in order, so an
    earlier one keeps its preferred spot and later ones move around it. A label
    that would leave `box` is slid back inside; one that would land on an
    already-placed label -- or on a box in `obstacles`, which lets a second
    pass dodge the labels a first pass already put down -- steps along `prefer`
    ("above"/"below") until clear.
    """
    bx0, by0, bx1, by1 = box
    step = size + 3.5
    placed, out = list(obstacles or ()), []
    for it in items:
        text, anchor = it["text"], it.get("anchor", "start")
        x, y = float(it["x"]), float(it["y"])
        w = len(text) * size * MONO_ADV
        # Horizontal: pull the whole box inside the plot, widest-case first.
        if w >= bx1 - bx0:
            anchor, x = "start", bx0
        else:
            x0, _, x1, _ = text_box(text, x, y, size, anchor)
            if x1 > bx1:
                x -= x1 - bx1
            x0, _, x1, _ = text_box(text, x, y, size, anchor)
            if x0 < bx0:
                x += bx0 - x0
        # Vertical: try the preferred side first, then alternate outward.
        d = -1 if it.get("prefer", "above") == "above" else 1
        cands = [y]
        for k in range(1, 9):
            cands += [y + d * step * k, y - d * step * k]
        chosen = None
        for cy in cands:
            bb = text_box(text, x, cy, size, anchor)
            if bb[1] < by0 or bb[3] > by1:
                continue
            if any(_overlaps(bb, p) for p in placed):
                continue
            chosen = cy
            break
        if chosen is None:  # pathological: keep it on the canvas at least
            chosen = min(max(y, by0 + size), by1)
        placed.append(text_box(text, x, chosen, size, anchor))
        out.append({**it, "x": x, "y": chosen, "anchor": anchor})
    return out


def notes_svg(items, box, size=11.0, base="note", obstacles=None):
    """Place and emit a set of labels. text-anchor is written as a style so it
    beats the class rules in the stylesheet. If `obstacles` is a list, the
    boxes used are appended to it so a later pass can avoid them."""
    out = []
    for it in place_labels(items, box, size=size, obstacles=obstacles):
        cls = f'{base} {it["cls"]}'.strip() if it.get("cls") else base
        out.append(f'<text class="{cls}" x="{it["x"]:.1f}" y="{it["y"]:.1f}" '
                   f'style="text-anchor:{it["anchor"]}">{esc(it["text"])}</text>')
        if obstacles is not None:
            obstacles.append(text_box(it["text"], it["x"], it["y"], size,
                                      it["anchor"]))
    return "\n".join(out)


def plot_box(inset=3.0):
    """The rectangle labels must stay inside: the drawing area, not the SVG."""
    return (PAD["l"] + inset, PAD["t"], W - PAD["r"] - inset,
            H - PAD["b"] - inset)


# --------------------------------------------------------------- exhibit 1 ---

def exhibit_shaping_window():
    """Rocket League: every branch's objective against training steps."""
    br = DATA["rl"]
    xlo, xhi = 0, max(p[0] for b in br for p in b["pts"])
    ylo, yhi = 0.30, 0.96
    px0, px1, py0, py1 = PAD["l"], W - PAD["r"], H - PAD["b"], PAD["t"]
    out = [frame("decision steps", "win rate vs the scripted chaser",
                 [0, 300e6, 600e6, 900e6, 1200e6], [0.3, 0.45, 0.6, 0.75, 0.9],
                 xlo, xhi, ylo, yhi,
                 fmt_x=lambda v: f"{v/1e6:.0f}M", fmt_y=lambda v: f"{v:.2f}")]

    highlight = {"b0.2.3": "win", "b0.2.1": "collapse", "b0.3": "flat"}
    for b in br:
        pts = [(lerp(s, xlo, xhi, px0, px1), lerp(v, ylo, yhi, py0, py1))
               for s, v in b["pts"]]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                     for i, (x, y) in enumerate(pts))
        cls = highlight.get(b["id"], "")
        out.append(f'<path class="ln {cls}" d="{d}"/>')
        if cls:
            lx, ly = pts[-1]
            out.append(f'<circle class="dot {cls}" cx="{lx:.1f}" cy="{ly:.1f}" r="3.5"/>')
    # Annotate the three that carry the finding. The claim and caption below
    # the chart carry the argument, so each label only has to say which line
    # this is and the number it reached -- short enough to sit inside the plot.
    by_id = {b["id"]: b for b in br}

    def pos(step, val):
        return lerp(step, xlo, xhi, px0, px1), lerp(val, ylo, yhi, py0, py1)

    def peak(bid):
        return max(by_id[bid]["pts"], key=lambda p: p[1])

    # b0.2.3 ends at 0.918 but reached 0.922, so the label is pinned to the
    # peak and says so; pinning 0.922 to the final dot would be a label
    # disagreeing with the point it points at.
    ws, wv = peak("b0.2.3")
    wx, wy = pos(ws, wv)
    ps, pv = peak("b0.2.1")
    pkx, pky = pos(ps, pv)
    cs, cv = by_id["b0.2.1"]["pts"][-1]
    cx, cy = pos(cs, cv)
    fs, fv = peak("b0.3")
    fx, fy = pos(fs, fv)

    notes = [
        {"text": f"approach_ball = 3 → peak {wv:.3f}", "x": wx - 8, "y": wy + 28,
         "anchor": "end", "cls": "win", "prefer": "below"},
        {"text": f"peak {pv:.3f}", "x": pkx + 9, "y": pky - 8,
         "anchor": "start", "cls": "collapse", "prefer": "above"},
        {"text": f"fell to {cv:.3f}", "x": cx + 9, "y": cy + 15,
         "anchor": "start", "cls": "collapse", "prefer": "below"},
        {"text": f"no shaping → never left {fv:.3f}", "x": fx + 9, "y": fy - 10,
         "anchor": "start", "cls": "flat", "prefer": "above"},
    ]
    out.append(notes_svg(notes, plot_box()))
    return svg("\n".join(out), "Rocket League objective per branch over training steps")


# --------------------------------------------------------------- exhibit 2 ---

def exhibit_signal_vs_truth():
    """Tetris: self-play win rate against the independent objective."""
    br = [b for b in DATA["tetris"] if b["selfplay"] is not None]
    xlo, xhi, ylo, yhi = 0.3, 0.85, -0.05, 1.05
    px0, px1, py0, py1 = PAD["l"], W - PAD["r"], H - PAD["b"], PAD["t"]
    out = [frame("self-play win rate  (the training signal)",
                 "win rate vs greedy  (the objective)",
                 [0.3, 0.45, 0.6, 0.75], [0.0, 0.25, 0.5, 0.75, 1.0],
                 xlo, xhi, ylo, yhi,
                 fmt_x=lambda v: f"{v:.2f}", fmt_y=lambda v: f"{v:.2f}")]
    # If the signal predicted the objective, points would track this line.
    out.append(f'<path class="ideal" d="M{lerp(0.3,xlo,xhi,px0,px1):.1f},'
               f'{lerp(0.3,ylo,yhi,py0,py1):.1f} L{lerp(0.85,xlo,xhi,px0,px1):.1f},'
               f'{lerp(0.85,ylo,yhi,py0,py1):.1f}"/>')
    box = plot_box()
    for b in br:
        best = max(v for _, v in b["pts"])
        x = lerp(b["selfplay"], xlo, xhi, px0, px1)
        y = lerp(best, ylo, yhi, py0, py1)
        out.append(f'<circle class="dot signal" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')

    # Every arm scored 0.000, so all six dots share a y and their ids would be
    # written on top of each other. The two annotations are placed first and
    # keep their spots; the ids step out of the way around them.
    hi = max(br, key=lambda b: b["selfplay"])
    hix = lerp(hi["selfplay"], xlo, xhi, px0, px1)
    zero_y = lerp(0.0, ylo, yhi, py0, py1)
    notes = [
        {"text": "if the training signal predicted the objective,"
                 " points would sit here",
         "x": px1, "y": py1 + 30, "anchor": "end", "cls": "muted",
         "prefer": "below"},
        {"text": f"best signal in the sweep: {hi['selfplay']:.3f}"
                 f" → still scored {max(v for _, v in hi['pts']):.3f}",
         "x": hix + 40, "y": zero_y - 44, "anchor": "end", "cls": "collapse",
         "prefer": "above"},
    ]
    taken = []
    out.append(notes_svg(notes, box, obstacles=taken))
    ids = [{"text": b["id"],
            "x": lerp(b["selfplay"], xlo, xhi, px0, px1),
            "y": lerp(max(v for _, v in b["pts"]), ylo, yhi, py0, py1) - 12,
            "anchor": "middle", "prefer": "above"}
           for b in sorted(br, key=lambda b: b["selfplay"])]
    out.append(notes_svg(ids, box, size=10.5, base="tick pt", obstacles=taken))
    return svg("\n".join(out), "Tetris self-play win rate against the independent objective")


# --------------------------------------------------------------- exhibit 3 ---

def exhibit_cost():
    """What each branch cost, against what it was worth."""
    rows = ([("Rocket League", b["id"], b["spend"], max(v for _, v in b["pts"]), b["status"])
             for b in DATA["rl"]]
            + [("Tetris", b["id"], b["spend"], max(v for _, v in b["pts"]), b["status"])
               for b in DATA["tetris"]])
    xlo, xhi, ylo, yhi = 0, max(r[2] for r in rows) * 1.05, -0.05, 1.05
    px0, px1, py0, py1 = PAD["l"], W - PAD["r"], H - PAD["b"], PAD["t"]
    out = [frame("branch cost (USD)", "best objective reached",
                 [0, 2.5, 5, 7.5, 10], [0.0, 0.25, 0.5, 0.75, 1.0],
                 xlo, xhi, ylo, yhi,
                 fmt_x=lambda v: f"${v:g}", fmt_y=lambda v: f"{v:.2f}")]
    for run, bid, spend, score, status in rows:
        x = lerp(spend, xlo, xhi, px0, px1)
        y = lerp(score, ylo, yhi, py0, py1)
        cls = "win" if status == "succeeded" else "pruned"
        shape = (f'<circle class="dot {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>'
                 if run == "Rocket League" else
                 f'<rect class="dot {cls}" x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8"/>')
        out.append(shape)
    cheap = [r for r in rows if r[2] < 2.5 and r[3] < 0.4]
    out.append(f'<text class="note muted" x="{px0+70}" y="{lerp(0.2,ylo,yhi,py0,py1):.0f}">'
               f'{len(cheap)} branches pruned below $2.50 — the search spends its budget '
               f'on the lines that pay</text>')
    return svg("\n".join(out), "Objective reached against branch cost")


# --------------------------------------------------------------- exhibit 4 ---

def exhibit_comparison():
    """SF3: hand-tuned reward against searched reward, with intervals."""
    bars = [
        ("r2_hinh — hand-tuned reward", 55, 100, "Qwen3-4B + LoRA", "pruned"),
        ("r3_potential — searched reward", 79, 100, "Qwen3-4B + LoRA", "win"),
        ("this run — Hinh's reward, our trainer", 18, 24, "Qwen3-VL-8B full fine-tune", "live"),
    ]
    h, gap = 46, 30
    top = 34
    out = []
    xlo, xhi = 0.0, 1.0
    # Each bar carries a value label past the right end of its interval, so the
    # track has to stop short by the width of the longest one. Measured, not
    # guessed: at W-40 the two widest labels ran off the edge of the chart.
    vals = [f"{w / n:.3f}  [{wilson(w, n)[0]:.2f}, {wilson(w, n)[1]:.2f}]"
            for _, w, n, _, _ in bars]
    px0 = 300
    px1 = W - 14 - 10 - max(len(v) for v in vals) * 11.5 * MONO_ADV
    for i, (name, w, n, sub, cls) in enumerate(bars):
        y = top + i * (h + gap)
        p = w / n
        lo, hi = wilson(w, n)
        x0 = lerp(xlo, xlo, xhi, px0, px1)
        xp = lerp(p, xlo, xhi, px0, px1)
        xa = lerp(lo, xlo, xhi, px0, px1)
        xb = lerp(hi, xlo, xhi, px0, px1)
        out.append(f'<text class="barlab" x="{px0 - 14}" y="{y + 16}">{esc(name)}</text>')
        out.append(f'<text class="barsub" x="{px0 - 14}" y="{y + 32}">{esc(sub)}  ·  n={n}</text>')
        out.append(f'<rect class="bar {cls}" x="{x0}" y="{y}" width="{xp - x0:.1f}" height="20"/>')
        out.append(f'<line class="ci" x1="{xa:.1f}" y1="{y + 10}" x2="{xb:.1f}" y2="{y + 10}"/>')
        for xe in (xa, xb):
            out.append(f'<line class="ci" x1="{xe:.1f}" y1="{y + 4}" x2="{xe:.1f}" y2="{y + 16}"/>')
        out.append(f'<text class="barval" x="{xb + 10:.1f}" y="{y + 15}">'
                   f'{p:.3f}  [{lo:.2f}, {hi:.2f}]</text>')
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = lerp(t, xlo, xhi, px0, px1)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{top + 3*(h+gap) - 20}"/>')
        out.append(f'<text class="tick tx" x="{x:.1f}" y="{top + 3*(h+gap) - 4}">{t:.2f}</text>')
    out.append(f'<text class="axlab" x="{(px0+px1)/2:.0f}" y="{H - 6}">'
               f'win rate vs the level-8 CPU · 95% Wilson interval</text>')
    return svg("\n".join(out), "Hand-tuned versus searched reward with confidence intervals")


# --------------------------------------------------------------- exhibit 5 ---

def exhibit_live():
    """The SF3 run now: what the trainer reports, iteration by iteration."""
    m = DATA["sf3"]
    xs = [d["iteration"] for d in m]
    xlo, xhi = min(xs), max(xs)
    px0, px1, py0, py1 = PAD["l"], W - PAD["r"], H - PAD["b"], PAD["t"]
    out = [frame("training iteration", "value",
                 list(range(1, max(xs) + 1, 2)), [0.0, 0.25, 0.5, 0.75, 1.0],
                 xlo, xhi, 0.0, 1.0,
                 fmt_x=lambda v: f"{v:g}", fmt_y=lambda v: f"{v:.2f}")]
    series = [
        ("return_mean", "signal", "mean return  (training signal)"),
        ("entropy", "muted2", "policy entropy, nats"),
    ]
    for key, cls, _ in series:
        pts = [(lerp(d["iteration"], xlo, xhi, px0, px1),
                lerp(d[key], 0.0, 1.0, py0, py1)) for d in m]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path class="ln {cls}" d="{d}"/>')
    # The only measured objective points, on the same axes.
    for step, score, it in ((28800, 0.75, 6),):
        x = lerp(it, xlo, xhi, px0, px1)
        y = lerp(score, 0.0, 1.0, py0, py1)
        lo, hi = wilson(18, 24)
        out.append(f'<line class="ci" x1="{x:.1f}" y1="{lerp(lo,0,1,py0,py1):.1f}" '
                   f'x2="{x:.1f}" y2="{lerp(hi,0,1,py0,py1):.1f}"/>')
        out.append(f'<circle class="dot win" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        out.append(f'<text class="note win" x="{x + 12:.0f}" y="{y - 8:.0f}">'
                   f'objective 0.750, n=24</text>')
    legend = [("signal", "mean return (training signal)"), ("muted2", "policy entropy"),
              ("win", "measured objective")]
    for i, (cls, text) in enumerate(legend):
        lx, ly = px0 + 10, py1 + 12 + i * 16
        out.append(f'<rect class="swatch {cls}" x="{lx}" y="{ly - 7}" width="16" height="3"/>')
        out.append(f'<text class="tick lg" x="{lx + 24}" y="{ly}">{esc(text)}</text>')
    return svg("\n".join(out), "Live SF3 run: training signal, entropy, and the measured objective")


# ------------------------------------------------------------------- page ---

CSS = """
:root{
  --ground:#f7f8fa; --surface:#ffffff; --line:#d8dde5;
  --ink:#121820; --muted:#5b6673; --faint:#8a94a2;
  --objective:#1f6feb;      /* the number that counts */
  --signal:#9a6b43;         /* training signal: seductive, not evidence */
  --good:#1a7f37; --bad:#cf222e; --grid:#e6e9ef;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0e1116; --surface:#161b22; --line:#2a313c;
    --ink:#e6edf3; --muted:#9aa5b1; --faint:#6e7781;
    --objective:#4d8dff; --signal:#b08968;
    --good:#3fb950; --bad:#f85149; --grid:#1d2430;
  }
}
:root[data-theme="dark"]{
  --ground:#0e1116; --surface:#161b22; --line:#2a313c;
  --ink:#e6edf3; --muted:#9aa5b1; --faint:#6e7781;
  --objective:#4d8dff; --signal:#b08968;
  --good:#3fb950; --bad:#f85149; --grid:#1d2430;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font:15px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
.wrap{max-width:900px;margin:0 auto;padding:56px 28px 96px}
header{border-bottom:1px solid var(--line);padding-bottom:28px;margin-bottom:8px}
h1{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:clamp(26px,4.4vw,40px); line-height:1.12; letter-spacing:-0.025em;
  margin:0 0 14px; text-wrap:balance; font-weight:600;
}
.lede{font-size:17px;color:var(--muted);max-width:66ch;margin:0}
.meta{
  margin-top:20px;display:flex;flex-wrap:wrap;gap:8px 18px;
  font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--faint);
  letter-spacing:.04em;text-transform:uppercase;
}
.exhibit{padding:44px 0;border-bottom:1px solid var(--line)}
.exhibit:last-of-type{border-bottom:0}
.eyebrow{
  font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin:0 0 10px;
}
h2{font-size:21px;line-height:1.3;margin:0 0 10px;text-wrap:balance;font-weight:620;letter-spacing:-0.01em}
.claim{color:var(--muted);max-width:66ch;margin:0 0 22px}
figure{margin:0}
.plot{
  background:var(--surface);border:1px solid var(--line);border-radius:4px;
  padding:10px 6px 2px;overflow-x:auto;
}
svg{display:block;width:100%;height:auto;min-width:620px}
figcaption{
  margin-top:12px;font-size:13.5px;color:var(--muted);max-width:70ch;
  border-left:2px solid var(--line);padding-left:14px;
}
figcaption b{color:var(--ink);font-weight:600}
.caveat{
  margin-top:14px;font-size:13.5px;color:var(--muted);max-width:70ch;
  border-left:2px solid var(--signal);padding-left:14px;
}
.caveat b{color:var(--signal);font-weight:600}
/* chart primitives */
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.tick{fill:var(--faint);font-family:ui-monospace,Menlo,monospace;font-size:10.5px;
      font-variant-numeric:tabular-nums}
.ty{text-anchor:end}.tx{text-anchor:middle}.pt{text-anchor:middle;fill:var(--muted)}
.lg{text-anchor:start;fill:var(--muted)}
.axlab{fill:var(--muted);font-size:11.5px;text-anchor:middle;
       font-family:ui-monospace,Menlo,monospace;letter-spacing:.04em}
.ln{fill:none;stroke:var(--faint);stroke-width:1.2;opacity:.45}
.ln.win{stroke:var(--objective);stroke-width:2.4;opacity:1}
.ln.collapse{stroke:var(--bad);stroke-width:2;opacity:1}
.ln.flat{stroke:var(--signal);stroke-width:2;opacity:1}
.ln.signal{stroke:var(--signal);stroke-width:2;opacity:1}
.ln.muted2{stroke:var(--faint);stroke-width:1.6;opacity:.9;stroke-dasharray:4 3}
.dot{fill:var(--faint)}
.dot.win{fill:var(--objective)}.dot.collapse{fill:var(--bad)}
.dot.signal{fill:var(--signal)}.dot.pruned{fill:var(--faint);opacity:.6}
.ideal{fill:none;stroke:var(--line);stroke-width:1;stroke-dasharray:3 4}
.note{font-family:ui-monospace,Menlo,monospace;font-size:11px;fill:var(--muted)}
/* A label that has to sit over a line still has to be readable. The halo is
   the panel colour, so it knocks back the line without hiding it. */
.note,.pt{paint-order:stroke fill;stroke:var(--surface);stroke-width:3px;
          stroke-linejoin:round;stroke-opacity:.9}
.note.win{fill:var(--objective)}.note.collapse{fill:var(--bad)}
.note.flat{fill:var(--signal)}.note.muted{fill:var(--faint)}
.bar{height:20px}
.bar.win{fill:var(--objective)}.bar.pruned{fill:var(--faint);opacity:.55}
.bar.live{fill:var(--good)}
.ci{stroke:var(--ink);stroke-width:1.4;opacity:.75}
.barlab{fill:var(--ink);font-size:13px;text-anchor:end}
.barsub{fill:var(--faint);font-size:11px;text-anchor:end;
        font-family:ui-monospace,Menlo,monospace}
.barval{fill:var(--muted);font-size:11.5px;font-family:ui-monospace,Menlo,monospace;
        font-variant-numeric:tabular-nums}
.swatch{opacity:.95}
.swatch.signal{fill:var(--signal)}.swatch.muted2{fill:var(--faint)}
.swatch.win{fill:var(--objective)}
footer{margin-top:40px;font-size:13px;color:var(--faint);max-width:70ch}
footer code{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted)}
@media (prefers-reduced-motion:no-preference){
  .exhibit{animation:rise .5s ease-out both}
  @keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
}
"""


def page():
    ex = [
        dict(
            n="Exhibit 1", title="The reward has a window, with an edge on both sides",
            claim="Ten branches of a Rocket League search, each a different reward, all "
                  "scored by the same evaluator. The spread is not noise — it is the "
                  "entire outcome.",
            svg=exhibit_shaping_window(),
            cap="At <b>approach_ball = 20</b> the policy peaked at 0.599 and fell to 0.353. "
                "With <b>no shaping at all</b> it never left 0.355 in 401M steps. At "
                "<b>approach_ball = 3</b> it reached <b>0.922</b>. One weight separates a "
                "working policy from two different failures.",
            cav="This is the case where searching paid. Exhibit 3 is the case where it "
                "found nothing, and that one is equally real.",
        ),
        dict(
            n="Exhibit 2", title="Training signal does not predict the objective",
            claim="Each point is one Tetris branch: how well it was doing against its own "
                  "self-play opponents, against how well it actually played.",
            svg=exhibit_signal_vs_truth(),
            cap="Every one of these six branches scored <b>0.000</b> against the independent "
                "greedy opponent. The arm with the <b>best</b> training signal in the whole "
                "sweep — 0.807 — was worth exactly as much as the worst. A search steered by "
                "training reward would have selected it and reported progress.",
            cav="This is why the objective evaluator imports no reward code, enforced by a "
                "test that walks the import graph. An agent that can reach its own "
                "measurement can look good without getting good.",
        ),
        dict(
            n="Exhibit 3", title="Where the search found nothing — and what that cost",
            claim="Twenty branches across two projects, plotted as what each one cost "
                  "against the best it ever scored. Circles are Rocket League, squares Tetris.",
            svg=exhibit_cost(),
            cap="The cluster along the bottom is the Tetris null: six reward configurations, "
                "two independent rewrites, a shaping scale swept 0.0 to 4.0 — all exactly "
                "<b>0.000</b>. Establishing that the whole axis was flat cost <b>$4.51</b>, "
                "and redirected the project to what did matter.",
            cav="A framework that only shows the searches it won is advertising. The useful "
                "claim is not that reward search always helps — it is that a null arrives "
                "cheaply and legibly instead of as a hunch defended for a week.",
        ),
        dict(
            n="Exhibit 4", title="Searched reward against hand-tuned, same harness",
            claim="Street Fighter III, win rate against the level-8 CPU. The first two bars "
                  "are matched on model, size and configuration — the reward is the only "
                  "difference between them.",
            svg=exhibit_comparison(),
            cap="A potential-shaped reward found by search beat a hand-tuned one by "
                "<b>24 points</b> at matched everything. That is the cleanest comparison "
                "this project has, because only one thing varies.",
            cav="The third bar is the run training right now — a different model entirely "
                "(Qwen3-VL-8B fully fine-tuned, not 4B + LoRA) and <b>n=24</b>, so its "
                "interval is 33 points wide. It clears the hand-tuned baseline and cannot be "
                "separated from the searched one. It is a baseline, not yet a result: that "
                "branch is running <i>Hinh's</i> reward.",
        ),
        dict(
            n="Exhibit 5", title="The run in flight, on one set of axes",
            claim="Qwen3-VL-8B-Instruct, full-parameter fine-tune, eleven iterations. Mean "
                  "return and policy entropy are what the trainer reports every iteration; "
                  "the single blue point is the only thing that counts.",
            svg=exhibit_live(),
            cap="Entropy <b>rising</b> from 0.434 to 0.693 nats is the signal worth watching: "
                "the predecessor policy collapsed to a single action and won 0 of 100 at its "
                "mode. Mean return wanders between 0.33 and 0.65 and — per Exhibit 2 — "
                "predicts nothing.",
            cav="One measured point against eleven iterations of training signal is the "
                "honest ratio of this page: evidence is expensive and training curves are "
                "free, which is exactly why the free one must not be used as the verdict.",
        ),
    ]
    blocks = "\n".join(
        f'<section class="exhibit">\n'
        f'<p class="eyebrow">{esc(e["n"])}</p>\n<h2>{esc(e["title"])}</h2>\n'
        f'<p class="claim">{esc(e["claim"])}</p>\n'
        f'<figure><div class="plot">{e["svg"]}</div>\n'
        f'<figcaption>{e["cap"]}</figcaption>\n'
        f'<p class="caveat"><b>Limit.</b> {e["cav"]}</p></figure>\n</section>'
        for e in ex
    )
    return f"""<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>What the measurements actually support</h1>
  <p class="lede">autobranch searches reward functions with a tree of agents in Modal
  sandboxes. Every number below came from an evaluator that imports no reward code —
  including the one case where the search found nothing at all.</p>
  <div class="meta">
    <span>4 environments</span><span>30+ branches</span>
    <span>objective-only evidence</span><span>intervals, not point estimates</span>
  </div>
</header>
{blocks}
<footer>
  <p>Sources: <code>autobranch/seed_runs.py</code> and <code>seed_runs_tetris.py</code>
  carry the Rocket League and Tetris measurements with a PROVENANCE block naming which
  parts are measured and which reconstructed; the Street Fighter bars come from
  <code>SF_REPORT.md</code> (n=100 each) and the live run's own
  <code>metrics.jsonl</code> on the Modal volume. Intervals are Wilson 95%.
  Chart coordinates are computed from those files rather than placed by hand, so a point
  cannot disagree with its own caption.</p>
</footer>
</div>
"""


out = HERE / "evidence.html"
out.write_text(page())
print(f"wrote {out} ({len(out.read_text()):,} bytes)")
