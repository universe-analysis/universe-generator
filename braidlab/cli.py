"""Command-line interface for braidlab.

Usage:
    python -m braidlab run 3plus1 --hosts host1,host2 --db data/run.db
    python -m braidlab analyze --db data/run.db
    python -m braidlab report  --db data/run.db --out report.html
    python -m braidlab plan 3plus1            # dry-run: list jobs, no launch
"""

from __future__ import annotations

import argparse
from pathlib import Path

from braidlab import campaigns, corrdim
from braidlab.analyze import measure_d
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


def _cmd_run(args: argparse.Namespace) -> None:
    camp = campaigns.get(args.campaign)
    hosts = args.hosts.split(",")
    store = Store(args.db)
    fleet = Fleet(_SOURCE_DIR)
    host_max_t = _parse_host_max(args.host_max) if args.host_max else None
    print(f"running {camp.name}: {len(camp.jobs())} jobs over {hosts}"
          f"{' (dumps)' if camp.dump else ''}")
    run_campaign(
        camp.jobs(),
        store,
        fleet,
        hosts,
        poll_seconds=args.poll,
        deploy=not args.no_deploy,
        dump=camp.dump,
        host_max_t=host_max_t,
    )
    print("campaign complete")


def _cmd_corrdim(args: argparse.Namespace) -> None:
    store = Store(args.db)
    stats = corrdim.aggregate(store.dumps_dir, dim=args.dim, band=args.band)
    if not stats:
        print(f"no dumps found in {store.dumps_dir}")
        return
    print(f"{'T':>4} {'seeds':>5} {'sphere':>14} {'cube':>14} {'box':>14}")
    for s in stats:
        print(f"{s.t:>4} {s.n_seeds:>5} "
              f"{s.sphere_mean:>7.3f}+/-{s.sphere_sem:<5.3f} "
              f"{s.cube_mean:>7.3f}+/-{s.cube_sem:<5.3f} "
              f"{s.box_mean:>7.3f}+/-{s.box_sem:<5.3f}")
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
            try:
                r = measure_d(results, dim, band)
            except ValueError:
                continue
            print(
                f"{dim}+1 band={band:7s}  D = {r.d:.3f} ± {r.d_err:.3f}  "
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
    sr.add_argument("--host-max", default="",
                    help="per-host max T, e.g. host1=160,host2=160")
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
    sc.add_argument("--labels", action="store_true",
                    help="annotate each point with its value")
    sc.set_defaults(func=_cmd_corrdim)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
