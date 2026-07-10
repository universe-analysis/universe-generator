"""Command-line interface for braidlab.

Usage:
    python -m braidlab run freq3d_e6 --hosts host1,host2 --db data/run.db
    python -m braidlab analyze --db data/run.db
    python -m braidlab report  --db data/run.db --out report.html
    python -m braidlab plan freq3d_e6         # dry-run: list jobs, no launch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from braidlab import campaigns, corrdim
from braidlab.analyze import measure_d
from braidlab.notify import COLORS, ENV_WEBHOOK, DiscordNotifier
from braidlab.orchestrator import Fleet, plan_assignment, run_campaign
from braidlab.report import build_report
from braidlab.store import Store


def _parse_host_max(text: str) -> dict[str, int]:
    """Parse ``host1=160,host2=160`` into a host -> max-T cap dict."""
    caps: dict[str, int] = {}
    for item in text.split(","):
        if item.strip():
            host, cap = item.split("=")
            caps[host.strip()] = int(cap)
    return caps


_SOURCE_DIR = Path(__file__).resolve().parent.parent  # braid_engine/ (has cuda/)


def _cmd_plan(args: argparse.Namespace) -> None:
    camp = campaigns.get(args.campaign)
    jobs = camp.jobs()
    hosts = args.hosts.split(",") if args.hosts else ["host0"]
    assign = plan_assignment(jobs, hosts)
    print(f"campaign {camp.name}: {len(jobs)} jobs over {len(hosts)} host(s)")
    for host, hjobs in assign.items():
        print(f"  {host}: {len(hjobs)} jobs  (T={sorted({j.t for j in hjobs})})")


def _campaign_summary(camp, hosts: list[str], db: str) -> tuple[str, dict]:
    """Build the human description + field table for the pre-flight Discord post."""
    ts = camp.t_values
    t_range = f"{min(ts)}–{max(ts)}"
    if len(ts) > 1:
        t_range += f" (step {ts[1] - ts[0]})"
    description = (
        f"{camp.dim}+1D RSA packing to jamming · torus model · band={camp.band} · "
        f"cutoff={camp.accept_rate:g}"
    )
    if getattr(camp, "euclid", False):
        description += " · sphere (L2) collision"
    fields = {
        "Hosts": ", ".join(hosts),
        "T range": t_range,
        "Seeds/T": len(camp.seeds),
        "Dumps": "yes" if camp.dump else "no",
        "DB": db,
    }
    if camp.terms_values != (2,):
        fields["Terms"] = ", ".join(str(k) for k in camp.terms_values)
    return description, fields


def _cmd_run(args: argparse.Namespace) -> None:
    camp = campaigns.get(args.campaign)
    hosts = args.hosts.split(",")
    store = Store(args.db)
    fleet = Fleet(_SOURCE_DIR)
    host_max_t = _parse_host_max(args.host_max) if args.host_max else None
    print(
        f"running {camp.name}: {len(camp.jobs())} jobs over {hosts}"
        f"{' (dumps)' if camp.dump else ''}"
    )
    description, fields = _campaign_summary(camp, hosts, args.db)
    run_campaign(
        camp.jobs(),
        store,
        fleet,
        hosts,
        poll_seconds=args.poll,
        deploy=not args.no_deploy,
        dump=camp.dump,
        host_max_t=host_max_t,
        campaign_name=camp.name,
        start_description=description,
        start_fields=fields,
    )
    print("campaign complete")


def _cmd_notify(args: argparse.Namespace) -> None:
    notifier = DiscordNotifier()
    if not notifier.enabled:
        print(f"no Discord webhook configured; set {ENV_WEBHOOK}", file=sys.stderr)
        raise SystemExit(1)
    if args.title:
        ok = notifier.send_embed(args.title, args.message, color=COLORS[args.color])
    else:
        ok = notifier.send_text(args.message)
    print("posted" if ok else "post failed")
    if not ok:
        raise SystemExit(1)


def _cmd_leaderboard(args: argparse.Namespace) -> None:
    from glob import glob

    from braidlab import leaderboard

    paths = sorted(p for pattern in args.dbs for p in glob(pattern))
    if not paths:
        print(f"no databases match {args.dbs}", file=sys.stderr)
        raise SystemExit(1)
    stats = leaderboard.gather(paths)
    board = leaderboard.format_board(stats)
    print(board)
    if args.post:
        notifier = DiscordNotifier()
        if not notifier.enabled:
            print(f"no Discord webhook configured; set {ENV_WEBHOOK}", file=sys.stderr)
            raise SystemExit(1)
        total = sum(s.jobs for s in stats)
        ok = notifier.send_embed(
            f"🏆 Fleet leaderboard — {total} jobs collected",
            board,
            color=COLORS["info"],
        )
        print("posted" if ok else "post failed")
        if not ok:
            raise SystemExit(1)


def _cmd_corrdim(args: argparse.Namespace) -> None:
    store = Store(args.db)
    # Variant campaigns share a dumps dir; restrict aggregation to the jobs this
    # store actually owns (their curve-path stems are the job names), or sibling
    # variants' dumps at the same (T, seed) leak into the average. A terms-sweep
    # store holds several term counts per (T, seed); --terms selects one so the
    # average never blends models.
    names = {
        Path(r.curve_path).stem
        for r in store.results(args.dim, args.band)
        if r.curve_path and (args.terms is None or r.terms == args.terms)
    }
    stats = corrdim.aggregate(
        store.dumps_dir, dim=args.dim, band=args.band, names=names or None
    )
    if not stats:
        print(f"no dumps found in {store.dumps_dir}")
        return
    print(f"{'T':>4} {'seeds':>5} {'sphere':>14} {'cube':>14} {'box':>14}")
    for s in stats:
        print(
            f"{s.t:>4} {s.n_seeds:>5} "
            f"{s.sphere_mean:>7.3f}+/-{s.sphere_sem:<5.3f} "
            f"{s.cube_mean:>7.3f}+/-{s.cube_sem:<5.3f} "
            f"{s.box_mean:>7.3f}+/-{s.box_sem:<5.3f}"
        )
    print(f"converged D ~ {corrdim.converged_value(stats):.3f}")
    out = corrdim.plot_convergence(stats, args.out, labels=args.labels)
    print(f"wrote {out}")


def _cmd_analyze(args: argparse.Namespace) -> None:
    store = Store(args.db)
    for dim in (2, 3):
        for band in ("default", "safe", "nyq"):
            results = store.results(dim, band)
            if len(results) < 2:
                continue
            # A terms-sweep store holds several term counts; fit each
            # separately so the N ~ T^D exponent never blends models.
            term_counts = sorted({r.terms for r in results})
            for terms in term_counts:
                subset = [r for r in results if r.terms == terms]
                if len(subset) < 2:
                    continue
                try:
                    r = measure_d(subset, dim, band)
                except ValueError:
                    continue
                label = f" terms={terms}" if len(term_counts) > 1 else ""
                print(
                    f"{dim}+1 band={band:7s}{label}  D = {r.d:.3f} ± {r.d_err:.3f}  "
                    f"(D/d={r.d / dim:.3f}, slope-spread={r.local_slope_std:.3f}, "
                    f"T={r.t_values})"
                )


def _cmd_report(args: argparse.Namespace) -> None:
    out = build_report(Store(args.db), args.out)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="braidlab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="list the jobs a campaign would run")
    sp.add_argument("campaign")
    sp.add_argument("--hosts", default="")
    sp.set_defaults(func=_cmd_plan)

    sr = sub.add_parser("run", help="run a campaign across the fleet")
    sr.add_argument("campaign")
    sr.add_argument("--hosts", required=True, help="comma-separated host list")
    sr.add_argument("--db", required=True)
    sr.add_argument("--poll", type=int, default=120)
    sr.add_argument("--no-deploy", action="store_true", help="skip engine build")
    sr.add_argument(
        "--host-max", default="", help="per-host max T, e.g. host1=160,host2=160"
    )
    sr.set_defaults(func=_cmd_run)

    sa = sub.add_parser("analyze", help="print D for all completed (dim, band)")
    sa.add_argument("--db", required=True)
    sa.set_defaults(func=_cmd_analyze)

    srep = sub.add_parser("report", help="write an HTML report")
    srep.add_argument("--db", required=True)
    srep.add_argument("--out", default="report.html")
    srep.set_defaults(func=_cmd_report)

    sc = sub.add_parser("corrdim", help="seed-averaged correlation-dimension report")
    sc.add_argument("--db", required=True)
    sc.add_argument("--dim", type=int, default=3)
    sc.add_argument("--band", default="nyq")
    sc.add_argument("--out", default="corrdim_convergence.png")
    sc.add_argument(
        "--terms",
        type=int,
        default=None,
        help="restrict to one term count (terms-sweep stores)",
    )
    sc.add_argument(
        "--labels", action="store_true", help="annotate each point with its value"
    )
    sc.set_defaults(func=_cmd_corrdim)

    sl = sub.add_parser("leaderboard", help="per-worker fleet leaderboard")
    sl.add_argument(
        "--dbs",
        nargs="+",
        required=True,
        help="store paths or globs, e.g. 'data/freq/*.db'",
    )
    sl.add_argument(
        "--post", action="store_true", help="also post to the Discord webhook"
    )
    sl.set_defaults(func=_cmd_leaderboard)

    sn = sub.add_parser("notify", help="post a message to the Discord webhook")
    sn.add_argument("--message", "-m", required=True, help="message body")
    sn.add_argument("--title", default="", help="embed title (omit for plain text)")
    sn.add_argument(
        "--color",
        default="info",
        choices=sorted(COLORS),
        help="embed color: start|progress|done|fail|info",
    )
    sn.set_defaults(func=_cmd_notify)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
