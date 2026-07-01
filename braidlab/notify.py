"""Discord notifications for campaign lifecycle events.

Posts to a Discord *webhook* (a post-only channel URL — no bot, no gateway).
The webhook URL is a secret and is read from the ``BRAIDLAB_DISCORD_WEBHOOK``
environment variable; it is never hard-coded or committed. If the variable is
unset the notifier is simply *disabled* and every call is a silent no-op, so
campaigns run identically with or without notifications configured.

Posting is fail-safe: a network error logs a warning to stderr and returns
``False`` rather than raising, because a down Discord must never kill a job.

Only the standard library is used (``urllib``), so nothing new is required on
the fleet hosts.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

#: Environment variable holding the secret webhook URL.
ENV_WEBHOOK = "BRAIDLAB_DISCORD_WEBHOOK"

#: Embed side-bar colors, keyed by event kind.
COLORS = {
    "start": 0x5865F2,     # blurple  -- a campaign is dispatching
    "progress": 0x3498DB,  # blue     -- a periodic progress ping
    "done": 0x2ECC71,      # green    -- finished cleanly
    "fail": 0xE74C3C,      # red      -- a failure or host drop
    "info": 0x95A5A6,      # grey     -- ad-hoc / manual message
}


def _http_post(url: str, data: bytes, timeout: int = 10) -> int:
    """POST ``data`` (JSON bytes) to ``url``; return the HTTP status code.

    Isolated as a module function so tests can monkeypatch it without touching
    the network.
    """
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            # Discord/Cloudflare rejects the default Python-urllib UA with 403.
            "User-Agent": "braidlab-notify/0.1 (+https://github.com/universe-analysis)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def _humanize_duration(seconds: float) -> str:
    """Render a duration as a compact ``1h 12m 03s`` string."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class DiscordNotifier:
    """Posts campaign lifecycle events to a Discord webhook (or no-ops)."""

    def __init__(self, webhook: str | None = None, username: str = "braidlab") -> None:
        self.webhook = webhook or os.environ.get(ENV_WEBHOOK)
        self.username = username

    @property
    def enabled(self) -> bool:
        """True when a webhook URL is configured."""
        return bool(self.webhook)

    # -- low-level senders ----------------------------------------------------

    def send_embed(
        self,
        title: str,
        description: str = "",
        fields: dict[str, object] | None = None,
        color: int = COLORS["info"],
    ) -> bool:
        """Post a single rich embed. Returns True on success, False otherwise."""
        embed: dict[str, object] = {"title": title, "color": color}
        if description:
            embed["description"] = description
        if fields:
            embed["fields"] = [
                {"name": name, "value": str(value), "inline": True}
                for name, value in fields.items()
            ]
        return self._post({"username": self.username, "embeds": [embed]})

    def send_text(self, content: str) -> bool:
        """Post a plain-text message."""
        return self._post({"username": self.username, "content": content})

    def _post(self, payload: dict) -> bool:
        if not self.enabled:
            return False
        try:
            _http_post(self.webhook, json.dumps(payload).encode("utf-8"))
            return True
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"[notify] Discord post failed: {exc}", file=sys.stderr)
            return False

    # -- lifecycle helpers ----------------------------------------------------

    def campaign_start(
        self, name: str, description: str, fields: dict[str, object]
    ) -> bool:
        """Pre-flight: what is about to run, on which hosts."""
        return self.send_embed(
            f"▶ Starting campaign: {name}", description, fields, COLORS["start"]
        )

    def campaign_progress(
        self, name: str, done: int, total: int, elapsed: float
    ) -> bool:
        """Periodic progress ping."""
        pct = int(round(100 * done / total)) if total else 0
        fields = {
            "Progress": f"{done}/{total} jobs ({pct}%)",
            "Elapsed": _humanize_duration(elapsed),
        }
        return self.send_embed(
            f"… {name} running", "", fields, COLORS["progress"]
        )

    def campaign_done(
        self, name: str, total: int, duration: float, output: str
    ) -> bool:
        """Completion summary."""
        fields = {
            "Jobs": str(total),
            "Duration": _humanize_duration(duration),
            "Output": output,
        }
        return self.send_embed(
            f"✅ Campaign complete: {name}", "", fields, COLORS["done"]
        )

    def campaign_failed(self, name: str, message: str) -> bool:
        """A failure or host drop worth surfacing."""
        return self.send_embed(
            f"⚠ {name}: problem", message, None, COLORS["fail"]
        )
