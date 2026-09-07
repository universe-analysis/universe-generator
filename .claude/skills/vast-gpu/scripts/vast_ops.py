#!/usr/bin/env python3
"""vast.ai instance lifecycle helper: pick offers, provision, watch, destroy.

Wraps the ``vastai`` CLI (auto-falls back to ``uvx vastai`` if not installed)
and resolves the API key from, in order: the ``VAST_API_KEY`` environment
variable, a ``.env`` file in the current directory, or the CLI's own default
key file (``~/.config/vastai/vast_api_key``).

Subcommands:
    check      Verify CLI, API key, and registered SSH keys.
    pick       Search offers and select the N cheapest matching instances.
    provision  Create instances from offer IDs, wait until SSH-ready.
    status     List instances with burn rate and estimated cost so far.
    destroy    Destroy instances (by id or --label) and verify they are gone.

Everything billing-related is loud on purpose: provisioning prints the total
$/hr, and destroy reports estimated spend and any instances still alive on
the account afterward.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

POLL_SECONDS = 15


def _read_env_file_key(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("VAST_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def resolve_api_key() -> str | None:
    """Find the API key; None means 'let the CLI use its default key file'."""
    import os

    if os.environ.get("VAST_API_KEY"):
        return os.environ["VAST_API_KEY"]
    for name in (".env", ".env.local"):
        key = _read_env_file_key(Path(name))
        if key:
            return key
    return None


def _cli_base() -> list[str]:
    if shutil.which("vastai"):
        return ["vastai"]
    if shutil.which("uvx"):
        return ["uvx", "vastai"]
    sys.exit("error: neither 'vastai' nor 'uvx' found; pip install vastai")


def vast(*args: str, raw: bool = True, check: bool = True) -> Any:
    """Run a vastai CLI command; with raw=True, parse and return its JSON."""
    cmd = _cli_base() + list(args)
    key = resolve_api_key()
    if key:
        cmd += ["--api-key", key]
    if raw:
        cmd += ["--raw"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        if check:
            sys.exit(f"error: vastai {' '.join(args[:2])} failed: {msg}")
        return None
    if not raw:
        return proc.stdout
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        if check:
            detail = proc.stdout.strip()[:500] or (
                "(empty output — usually a missing/invalid API key; put "
                "VAST_API_KEY=... in ./.env or run 'vastai set api-key')"
            )
            sys.exit(f"error: vastai {' '.join(args[:2])} returned non-JSON: {detail}")
        return None


def offer_field(offer: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if offer.get(name) is not None:
            return offer[name]
    return default


# ---------------------------------------------------------------- check


def cmd_check(args: argparse.Namespace) -> None:
    user = vast("show", "user")
    email = user.get("email", "?")
    balance = user.get("credit", user.get("balance", "?"))
    print(f"API key OK: {email}, credit balance ${balance}")

    keys = vast("show", "ssh-keys")
    if keys:
        print(f"{len(keys)} SSH key(s) registered with the account.")
        return
    pub = _default_pubkey()
    if pub is None:
        sys.exit(
            "error: no SSH keys registered and no local public key found in "
            "~/.ssh; create one, then re-run"
        )
    if args.register_ssh_key:
        _register_pubkey(pub)
    else:
        print(
            f"WARNING: no SSH key registered; run with --register-ssh-key to "
            f"register {pub} (needed before provisioning)."
        )


def _register_pubkey(pub: Path) -> None:
    # 'create ssh-key' prints progress text around its JSON even under --raw,
    # so run it unparsed and confirm via 'show ssh-keys' instead.
    vast("create", "ssh-key", pub.read_text().strip(), raw=False, check=False)
    if not vast("show", "ssh-keys"):
        sys.exit(f"error: registering {pub} with the account failed")
    print(f"Registered {pub} with the account.")


def _default_pubkey() -> Path | None:
    ssh_dir = Path.home() / ".ssh"
    for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
        if (ssh_dir / name).is_file():
            return ssh_dir / name
    return None


# ---------------------------------------------------------------- pick


def cmd_pick(args: argparse.Namespace) -> None:
    query = (
        f"gpu_name={args.gpu} num_gpus={args.num_gpus} "
        f"rentable=true verified=true reliability>{args.min_reliability}"
    )
    if args.query:
        query += f" {args.query}"
    offers = vast("search", "offers", query, "--limit", "200")
    if not offers:
        sys.exit(f"error: no offers match: {query}")

    if args.max_dph is not None:
        offers = [o for o in offers if o.get("dph_total", 1e9) <= args.max_dph]
        if not offers:
            sys.exit(f"error: offers exist but none under ${args.max_dph}/hr")

    # One offer per physical machine (cheapest), then cheapest overall first.
    by_machine: dict[int, dict[str, Any]] = {}
    for o in offers:
        mid = o.get("machine_id", o["id"])
        if mid not in by_machine or o["dph_total"] < by_machine[mid]["dph_total"]:
            by_machine[mid] = o
    picked = sorted(by_machine.values(), key=lambda o: o["dph_total"])[: args.count]

    if args.json:
        print(json.dumps(picked, indent=2))
        return
    print(
        f"{'OFFER_ID':>10}  {'GPU':<18} {'$/hr':>6}  {'VRAM':>5} "
        f"{'CUDA':>5} {'REL':>5} {'DOWN Mb/s':>9}  GEO"
    )
    for o in picked:
        gpu = f"{o.get('gpu_name', '?')} x{o.get('num_gpus', '?')}"
        print(
            f"{o['id']:>10}  {gpu:<18} {o['dph_total']:>6.3f}  "
            f"{offer_field(o, 'gpu_ram', default=0) / 1024:>4.0f}G "
            f"{offer_field(o, 'cuda_max_good', 'cuda_vers', default='?'):>5} "
            f"{offer_field(o, 'reliability2', 'reliability', default=0):>5.3f} "
            f"{offer_field(o, 'inet_down', default=0):>9.0f}  "
            f"{offer_field(o, 'geolocation', default='?')}"
        )
    total = sum(o["dph_total"] for o in picked)
    print(f"\nTotal if all {len(picked)} rented: ${total:.3f}/hr")
    if len(picked) < args.count:
        print(f"NOTE: only {len(picked)} distinct machines matched, not {args.count}")


# ---------------------------------------------------------------- provision


def cmd_provision(args: argparse.Namespace) -> None:
    _ensure_ssh_key()
    created: list[int] = []
    for offer_id in args.offer_ids:
        cmd = [
            "create", "instance", str(offer_id),
            "--image", args.image,
            "--disk", str(args.disk),
            "--label", args.label,
            "--ssh", "--direct",
            "--cancel-unavail",
        ]  # fmt: skip
        if args.onstart_cmd:
            cmd += ["--onstart-cmd", args.onstart_cmd]
        if args.env:
            cmd += ["--env", args.env]
        result = vast(*cmd, check=False)
        if result and result.get("success"):
            created.append(result["new_contract"])
            print(f"offer {offer_id} -> instance {result['new_contract']}")
        else:
            print(f"offer {offer_id} FAILED to rent (likely taken); skipping")
    if not created:
        sys.exit("error: no instances created")
    print(f"\n{len(created)} instance(s) created — BILLING HAS STARTED.")
    if args.no_wait:
        print("Not waiting (--no-wait). Poll with: vast_ops.py status")
        return
    _wait_ready(created, args.timeout)


def _ensure_ssh_key() -> None:
    if vast("show", "ssh-keys"):
        return
    pub = _default_pubkey()
    if pub is None:
        sys.exit("error: no SSH key on the account and none in ~/.ssh to register")
    _register_pubkey(pub)


def _instances(label: str | None = None) -> list[dict[str, Any]]:
    rows = vast("show", "instances") or []
    if label:
        rows = [r for r in rows if r.get("label") == label]
    return rows


def _ssh_endpoint(inst_id: int) -> tuple[str, int] | None:
    url = vast("ssh-url", str(inst_id), raw=False, check=False)
    if not url or "://" not in url:
        return None
    hostport = url.strip().split("://", 1)[1].split("@")[-1]
    host, _, port = hostport.partition(":")
    return host, int(port)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


def _wait_ready(ids: list[int], timeout: int) -> None:
    deadline = time.time() + timeout
    ready: dict[int, tuple[str, int]] = {}
    while time.time() < deadline and len(ready) < len(ids):
        rows = {r["id"]: r for r in _instances()}
        for inst_id in ids:
            if inst_id in ready:
                continue
            row = rows.get(inst_id)
            if not row or row.get("actual_status") != "running":
                continue
            endpoint = _ssh_endpoint(inst_id)
            if endpoint and _port_open(*endpoint):
                ready[inst_id] = endpoint
        if len(ready) < len(ids):
            waiting = [i for i in ids if i not in ready]
            print(f"waiting on {waiting} ({int(deadline - time.time())}s left)...")
            time.sleep(POLL_SECONDS)

    rows = {r["id"]: r for r in _instances()}
    print()
    for inst_id in ids:
        row = rows.get(inst_id, {})
        if inst_id in ready:
            host, port = ready[inst_id]
            print(
                f"instance {inst_id} READY  "
                f"ssh -o StrictHostKeyChecking=accept-new -p {port} root@{host}  "
                f"(${row.get('dph_total', 0):.3f}/hr)"
            )
        else:
            print(
                f"instance {inst_id} NOT READY after {timeout}s "
                f"(status={row.get('actual_status')}); investigate or destroy it"
            )
    total = sum(rows[i].get("dph_total", 0) for i in ids if i in rows)
    print(f"\nTotal burn rate: ${total:.3f}/hr — destroy when the job is done.")
    if len(ready) < len(ids):
        sys.exit(1)


# ---------------------------------------------------------------- status


def cmd_status(args: argparse.Namespace) -> None:
    rows = _instances(args.label)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        scope = f" with label '{args.label}'" if args.label else ""
        print(f"No instances{scope}. Nothing is billing.")
        return
    now = time.time()
    total = 0.0
    for r in rows:
        dph = r.get("dph_total", 0)
        hours = max(0.0, now - r.get("start_date", now)) / 3600
        total += dph
        print(
            f"instance {r['id']}  label={r.get('label') or '-'}  "
            f"{r.get('gpu_name', '?')} x{r.get('num_gpus', '?')}  "
            f"status={r.get('actual_status')}  ${dph:.3f}/hr  "
            f"up {hours:.2f}h  est ${dph * hours:.2f} so far"
        )
    print(f"\nTotal burn rate: ${total:.3f}/hr across {len(rows)} instance(s)")


# ---------------------------------------------------------------- destroy


def cmd_destroy(args: argparse.Namespace) -> None:
    if args.label:
        targets = _instances(args.label)
    else:
        all_rows = {r["id"]: r for r in _instances()}
        targets = [all_rows.get(i, {"id": i}) for i in args.ids]
    if not targets:
        print("Nothing to destroy.")
        return

    now = time.time()
    for r in targets:
        dph = r.get("dph_total", 0)
        hours = max(0.0, now - r.get("start_date", now)) / 3600
        # --yes: newer CLIs interactively confirm destroys; without it the
        # prompt reads captured stdin as EOF and the destroy silently no-ops.
        vast("destroy", "instance", str(r["id"]), "--yes", check=False)
        print(
            f"destroy requested: {r['id']} (ran {hours:.2f}h, est ${dph * hours:.2f})"
        )

    target_ids = {r["id"] for r in targets}
    deadline = time.time() + 120
    while time.time() < deadline:
        alive = {r["id"] for r in _instances()} & target_ids
        if not alive:
            break
        print(f"waiting for {sorted(alive)} to disappear...")
        time.sleep(10)

    remaining = _instances()
    still_targeted = [r for r in remaining if r["id"] in target_ids]
    if still_targeted:
        sys.exit(
            f"error: {[r['id'] for r in still_targeted]} STILL EXIST and are "
            f"billing — retry destroy or check the vast.ai console"
        )
    print(f"Verified destroyed: {sorted(target_ids)}")
    if remaining:
        print(
            f"NOTE: {len(remaining)} other instance(s) still on the account "
            f"(still billing): {[(r['id'], r.get('label')) for r in remaining]}"
        )
    else:
        print("Account has zero instances. Nothing is billing.")


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="verify CLI, API key, SSH keys")
    p.add_argument("--register-ssh-key", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("pick", help="search offers, select N cheapest machines")
    p.add_argument("--gpu", required=True, help="e.g. RTX_3090 (underscores)")
    p.add_argument("--num-gpus", default="1", help="GPUs per instance, e.g. 1 or >=4")
    p.add_argument("--count", type=int, default=1, help="number of instances")
    p.add_argument("--max-dph", type=float, help="max $/hr per instance")
    p.add_argument("--min-reliability", default="0.98")
    p.add_argument("--query", help="extra raw query terms, e.g. 'cuda_vers>=12.1'")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("provision", help="create instances and wait for SSH")
    p.add_argument("offer_ids", nargs="+", type=int)
    p.add_argument("--image", required=True)
    p.add_argument("--disk", type=int, default=20)
    p.add_argument("--label", required=True, help="tag for status/destroy --label")
    p.add_argument("--onstart-cmd")
    p.add_argument("--env", help="docker env/port args, e.g. '-e FOO=1'")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(func=cmd_provision)

    p = sub.add_parser("status", help="list instances, burn rate, cost so far")
    p.add_argument("--label")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("destroy", help="destroy by id or --label, verify gone")
    p.add_argument("ids", nargs="*", type=int)
    p.add_argument("--label")
    p.set_defaults(func=cmd_destroy)

    args = parser.parse_args()
    if args.command == "destroy" and not args.ids and not args.label:
        parser.error("destroy needs instance ids or --label")
    args.func(args)


if __name__ == "__main__":
    main()
