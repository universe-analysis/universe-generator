"""Generate a single self-contained HTML report from braid_engine curve CSVs.

Reads a directory of `attempts,n` curves (filenames containing `T<dim>` and
`s<seed>`), computes the growth law / convergence / scaling, renders plots, and
embeds everything (PNGs base64-inline) into one portable .html file.

Usage:
    python make_report.py --indir /tmp/campaign_overnight \
        --title "2+1 Braided Universe — Packing Growth" --model "2+1" \
        --out /tmp/report_2plus1.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Curve = list[tuple[float, float]]


def load(path: str) -> Curve:
    out: Curve = []
    with open(path) as f:
        next(f, None)
        for line in f:
            p = line.split(",")
            if len(p) == 2:
                out.append((float(p[0]), float(p[1])))
    return out


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sl = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return sl, (sy - sl * sx) / n


def collect(indir: str) -> dict[int, list[Curve]]:
    groups: dict[int, list[Curve]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(indir, "*.csv"))):
        m = re.search(r"[Tt](\d+)", os.path.basename(path))
        if not m:
            continue
        c = load(path)
        if len(c) >= 6:
            groups[int(m.group(1))].append(c)
    return groups


CONV_P = -1.15  # p must be clearly steeper than -1 to trust a ceiling extrapolation


def deep_metrics(curves: list[Curve]) -> dict:
    """Average deep-tail p over seeds; extrapolate N_sat only if clearly converged.

    Extrapolating N = N_sat - K*a^(p+1) is ill-conditioned when p≈-1 (the exponent
    p+1≈0 makes a^(p+1) nearly constant), so we only report a ceiling when the
    deep tail is clearly steeper than -1 *and* the result is physically sane.
    """
    ps, nfin, nsats = [], [], []
    for pts in curves:
        final_a = pts[-1][0]
        lo = final_a / 100.0
        rate = []
        for (a0, n0), (a1, n1) in zip(pts, pts[1:]):
            mid = (a0 + a1) / 2
            if mid >= lo and a1 > a0 and n1 > n0:
                rate.append((math.log10(mid), math.log10((n1 - n0) / (a1 - a0))))
        nfin.append(pts[-1][1])
        if len(rate) > 4:
            ps.append(linfit([x for x, _ in rate], [y for _, y in rate])[0])
    avg = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    pbar, nf = avg(ps), avg(nfin)
    converges = pbar == pbar and pbar < CONV_P
    nsat = float("nan")
    if converges:
        for pts in curves:
            q = pbar + 1
            lo = pts[-1][0] / 100.0
            X = [a**q for a, _ in pts if a >= lo]
            Y = [n for a, n in pts if a >= lo]
            nsats.append(linfit(X, Y)[1])
        cand = avg(nsats)
        if nf < cand < 1.6 * nf:  # sane: a few % to <60% above the current count
            nsat = cand
        else:
            converges = False  # extrapolation blew up -> treat as not yet converged
    return {"n_final": nf, "p": pbar, "n_sat": nsat, "converges": converges}


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_growth(groups: dict[int, list[Curve]]) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for ci, T in enumerate(sorted(groups)):
        for k, pts in enumerate(groups[T]):
            ax.plot([a for a, _ in pts], [n for _, n in pts], color=f"C{ci}", lw=1.5,
                    alpha=0.85, label=f"T={T}" if k == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("attempts")
    ax.set_ylabel("N (paths packed)")
    ax.set_title("Approach to jamming — straight line = log growth, plateau = convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return fig_to_b64(fig)


def plot_rate(groups: dict[int, list[Curve]]) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for ci, T in enumerate(sorted(groups)):
        pts = groups[T][0]
        xs, ys = [], []
        for (a0, n0), (a1, n1) in zip(pts, pts[1:]):
            if a1 > a0 and n1 > n0:
                xs.append((a0 + a1) / 2)
                ys.append((n1 - n0) / (a1 - a0))
        m = deep_metrics(groups[T])
        ax.plot(xs, ys, color=f"C{ci}", lw=1.4, alpha=0.85, label=f"T={T} (p={m['p']:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("attempts")
    ax.set_ylabel("acceptance rate  dN/d(attempts)")
    ax.set_title("Acceptance-rate decay  (p = -1 → logarithmic, p < -1 → converges)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return fig_to_b64(fig)


def n_at(pts: Curve, a: float) -> float:
    v = 0.0
    for att, n in pts:
        if att <= a:
            v = n
        else:
            break
    return v


def plot_scaling(rows: list[tuple[int, float]], ylabel: str) -> tuple[str, float]:
    Ts = [r[0] for r in rows]
    Ns = [r[1] for r in rows]
    D = linfit([math.log10(t) for t in Ts], [math.log10(n) for n in Ns])[0]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(Ts, Ns, "o-", ms=8)
    ax.set_xlabel("time-steps T (resolution)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs resolution:  ~ T^{D:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    return fig_to_b64(fig), D


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:0 auto;
padding:32px;color:#1a1f2b;line-height:1.55;background:#fafbfc}
h1{font-size:26px;margin-bottom:4px}h2{margin-top:34px;border-bottom:2px solid #e2e6ee;padding-bottom:6px}
.sub{color:#667;margin-top:0}img{width:100%;border:1px solid #e2e6ee;border-radius:8px;margin:10px 0}
table{border-collapse:collapse;margin:14px 0;width:100%}th,td{border:1px solid #dde2ec;padding:7px 12px;text-align:right}
th{background:#f0f3f8;text-align:center}td:first-child,th:first-child{text-align:center}
.key{background:#eef6ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:6px;margin:16px 0}
code{background:#eef1f6;padding:1px 5px;border-radius:4px}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--title", default="Braided Universe — Packing Report")
    ap.add_argument("--model", default="2+1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups = collect(args.indir)
    if not groups:
        print(f"no curves in {args.indir}")
        return
    Ts = sorted(groups)
    metrics = {T: deep_metrics(groups[T]) for T in Ts}
    ncurves = sum(len(v) for v in groups.values())

    g_img = plot_growth(groups)
    r_img = plot_rate(groups)
    dim = 2 if args.model.startswith("2") else 3
    pbar = sum(metrics[T]["p"] for T in Ts) / len(Ts)
    # T-values deeply enough converged to trust a ceiling (n_sat is finite)
    sat_rows = [(T, metrics[T]["n_sat"]) for T in Ts if metrics[T]["n_sat"] == metrics[T]["n_sat"]]
    n_settling = len(Ts) - len(sat_rows)
    converges = pbar < -1.12 and len(sat_rows) >= 2

    if converges:
        s_img, D = plot_scaling(sat_rows, "ceiling N_sat")
        ratio = D / dim
        settle = (
            f" (from the {len(sat_rows)} deeply-converged T-values; {n_settling} higher-T point"
            f"{'s' if n_settling != 1 else ''} still settling)"
            if n_settling
            else ""
        )
        verdict = (
            f"Deep-tail exponent <b>p ≈ {pbar:.2f}</b> (clearly steeper than −1): the packing "
            f"<b>converges to a finite ceiling</b> — the logarithmic-looking growth at shallow "
            f"depths is a transient."
        )
        scaling_txt = (
            f"The ceiling scales as <b>N_sat ~ T<sup>{D:.2f}</sup></b>, a fractal codimension of "
            f"<b>D/d ≈ {ratio:.2f}</b> (d={dim}){settle}."
        )
    else:
        common = min(groups[T][0][-1][0] for T in Ts)
        rows = [(T, n_at(groups[T][0], common)) for T in Ts]
        rows = [(t, n) for t, n in rows if n]
        s_img, D = (plot_scaling(rows, f"N at {common:.0e} attempts") if len(rows) >= 2 else ("", float("nan")))
        ratio = D / dim if D == D else float("nan")
        verdict = (
            f"Deep-tail exponent <b>p ≈ {pbar:.2f}</b> (≈ −1): still in the <b>logarithmic "
            f"transient</b> — <b>has NOT converged</b> at this budget. The ceiling is beyond current "
            f"reach (the {args.model} space packs far denser than fewer dimensions, pushing the "
            f"ceiling much deeper in attempts). A reliable D needs deeper runs."
        )
        scaling_txt = (
            f"<b>Provisional only:</b> at a fixed budget of {common:.0e} attempts, N(T) ~ "
            f"T<sup>{D:.2f}</sup> — this is budget-limited and will shift as it converges; it is "
            f"<b>not</b> the true converged exponent."
            if D == D
            else ""
        )

    def cell_nsat(T: int) -> str:
        m = metrics[T]
        return f"{m['n_sat']:.0f}" if m["n_sat"] == m["n_sat"] else "— (not converged)"

    def cell_head(T: int) -> str:
        m = metrics[T]
        return f"{m['n_sat'] - m['n_final']:+.0f}" if m["n_sat"] == m["n_sat"] else "—"

    trows = "".join(
        f"<tr><td>{T}</td><td>{metrics[T]['n_final']:.0f}</td>"
        f"<td>{metrics[T]['p']:.2f}</td><td>{cell_nsat(T)}</td><td>{cell_head(T)}</td></tr>"
        for T in Ts
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{args.title}</title><style>{CSS}</style></head><body>
<h1>{args.title}</h1>
<p class="sub">{args.model} model · {ncurves} curves · T ∈ {{{', '.join(map(str, Ts))}}} · GPU/CPU RSA engine</p>

<div class="key">{verdict} {scaling_txt}</div>

<h2>The model in one line</h2>
<p>Worldlines are parametric paths <code>x(z)=a·sin(b·z)+a₂·sin(z)</code> per spatial axis over a closed
loop <code>z∈(0,π)</code> (Bang→Crunch), with a slope-1 speed limit. We pack as many mutually
non-intersecting paths as possible at a fixed time-step resolution <code>T</code> (which sets both the
collision threshold <code>CELL=2/T</code> and the max frequency <code>T/2</code>), and measure how the count
<code>N</code> grows with the number of random attempts.</p>

<h2>Growth & approach to jamming</h2>
<p>On a log-x axis a logarithmic process is a straight line; a converging one bends down and plateaus.</p>
<img src="data:image/png;base64,{g_img}">

<h2>Acceptance-rate decay</h2>
<p>The acceptance rate falls as <code>attempts<sup>p</sup></code>. <code>p=−1</code> integrates to logarithmic
growth (never saturates); <code>p&lt;−1</code> means N converges to a finite ceiling.</p>
<img src="data:image/png;base64,{r_img}">

<h2>Ceilings & convergence</h2>
<table><tr><th>T</th><th>N at final budget</th><th>deep-tail p</th><th>est. ceiling N_sat</th><th>headroom</th></tr>
{trows}</table>

<h2>Scaling with resolution</h2>
{f'<img src="data:image/png;base64,{s_img}">' if s_img else '<p>(need ≥2 T values)</p>'}

</body></html>"""

    with open(args.out, "w") as f:
        f.write(html)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({size_kb:.0f} KB, self-contained)")
    print(f"  D = {D:.2f}, D/d = {ratio:.2f}, converges = {converges}")


if __name__ == "__main__":
    main()
