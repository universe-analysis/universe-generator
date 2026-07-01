"""Command-line driver for the constrained braiding solver.

Examples
--------
    # 2D with frequencies 3 and 10 (permutation model):
    python solve_braid.py --dims 2 --freqs 3 10

    # 3D, free assignment, two frequencies reused across axes:
    python solve_braid.py --dims 3 --freqs 3 10 --free

    # Sweep the canonical 1D/2D/3D questions:
    python solve_braid.py --report
"""

from __future__ import annotations

import argparse

from braid import format_path, solve


def _run_one(dims: int, freqs: list[int], permutation: bool) -> None:
    mode = "permutation" if permutation else "free"
    res = solve(dims, freqs, permutation=permutation)
    print(f"dims={dims} freqs={freqs} mode={mode}: max = {res.count}")
    for p in res.paths:
        print(f"    {format_path(p)}")


def _report() -> None:
    
    print("== 1D (one frequency) ==")
    _run_one(1, [100], permutation=True)
    print("\n== 2D (two frequencies, permutation) ==")
    _run_one(2, [3, 10], permutation=True)
    print("\n== 2D (two frequencies, free assignment) ==")
    _run_one(2, [3, 10], permutation=False)
    print("\n== 3D (three frequencies, permutation) ==")
    _run_one(3, [3, 7, 10], permutation=True)
    print("\n== 3D (two frequencies, free -- matches chris1.png) ==")
    _run_one(3, [3, 10], permutation=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, help="number of spatial dimensions")
    parser.add_argument(
        "--freqs", type=int, nargs="+", help="allowed integer frequencies"
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="free assignment (each axis any freq) instead of permutation",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="run the canonical 1D/2D/3D sweep and exit",
    )
    args = parser.parse_args()

    if args.report:
        _report()
        return
    if args.dims is None or args.freqs is None:
        parser.error("provide --dims and --freqs, or use --report")
    _run_one(args.dims, args.freqs, permutation=not args.free)


if __name__ == "__main__":
    main()
