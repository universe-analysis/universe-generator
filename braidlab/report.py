"""Self-contained HTML report: D with error bars + the log-log fit."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from braidlab.analyze import DResult, measure_d
from braidlab.store import Store


def _plot_png(res: DResult) -> str:
    """Render the log-log N(T) fit as a base64 PNG (empty string if no mpl)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""
    fig, ax = plt.subplots(figsize=(5, 4))
    t = np.array(res.t_values, dtype=float)
    n = np.array(res.n_mean)
    ax.errorbar(t, n, yerr=res.n_sem, fmt="o", capsize=3, label="seed mean ± SEM")
    fit = np.exp(res.intercept) * t**res.d
    ax.plot(t, fit, "-", label=f"D = {res.d:.3f} ± {res.d_err:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (timesteps)")
    ax.set_ylabel("N (packed braids)")
    ax.set_title(f"{res.dim}+1  band={res.band}")
    ax.legend()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _section(res: DResult) -> str:
    rows = "".join(
        f"<tr><td>{t}</td><td>{m:.0f}</td><td>{s:.0f}</td><td>{k}</td></tr>"
        for t, m, s, k in zip(res.t_values, res.n_mean, res.n_sem, res.n_seeds)
    )
    png = _plot_png(res)
    img = f'<img src="data:image/png;base64,{png}">' if png else ""
    return f"""
    <h2>{res.dim}+1 &mdash; band {res.band}</h2>
    <p><b>D = {res.d:.3f} &plusmn; {res.d_err:.3f}</b>
       (D/d = {res.d / res.dim:.3f});
       local-slope spread = {res.local_slope_std:.3f}
       (power-law consistency check)</p>
    <table border=1 cellpadding=4>
      <tr><th>T</th><th>&lt;N&gt;</th><th>SEM</th><th>seeds</th></tr>{rows}
    </table>
    {img}
    """


def build_report(store: Store, out_path: str | Path) -> Path:
    """Write an HTML report for every (dim, band) with completed runs."""
    seen: set[tuple[int, str]] = set()
    for dim in (2, 3):
        for band in ("default", "safe", "nyq"):
            if store.results(dim, band):
                seen.add((dim, band))
    sections = []
    for dim, band in sorted(seen):
        try:
            res = measure_d(store.results(dim, band), dim, band)
            sections.append(_section(res))
        except ValueError:
            continue
    html = (
        "<html><head><meta charset='utf-8'><title>braidlab report</title>"
        "<style>body{font-family:sans-serif;max-width:760px;margin:2em auto}</style>"
        "</head><body><h1>Braided-Universe Packing</h1>"
        + "".join(sections)
        + "</body></html>"
    )
    out = Path(out_path)
    out.write_text(html)
    return out
