"""Tests for the Discord notifier (no real network)."""

import json

import pytest

from braidlab import notify
from braidlab.notify import COLORS, DiscordNotifier, _humanize_duration


@pytest.fixture
def captured(monkeypatch) -> list[dict]:
    """Capture the JSON payload of every _http_post instead of sending it."""
    posts: list[dict] = []

    def fake_post(url: str, data: bytes, timeout: int = 10) -> int:
        posts.append({"url": url, "payload": json.loads(data.decode("utf-8"))})
        return 204

    monkeypatch.setattr(notify, "_http_post", fake_post)
    return posts


def test_disabled_without_webhook_is_noop(monkeypatch, captured) -> None:
    monkeypatch.delenv("BRAIDLAB_DISCORD_WEBHOOK", raising=False)
    n = DiscordNotifier()
    assert not n.enabled
    assert n.send_text("hello") is False
    assert n.campaign_start("c", "desc", {"Hosts": "a"}) is False
    assert captured == []  # nothing hit the wire


def test_embed_payload_shape(captured) -> None:
    n = DiscordNotifier(webhook="https://example.test/webhook")
    assert n.enabled
    ok = n.send_embed("Title", "Body", {"Hosts": "a, b", "Jobs": 5}, COLORS["start"])
    assert ok is True
    assert len(captured) == 1
    embed = captured[0]["payload"]["embeds"][0]
    assert embed["title"] == "Title"
    assert embed["description"] == "Body"
    assert embed["color"] == COLORS["start"]
    # dict fields become inline name/value pairs (values stringified)
    assert {"name": "Jobs", "value": "5", "inline": True} in embed["fields"]


def test_start_progress_done_use_distinct_colors(captured) -> None:
    n = DiscordNotifier(webhook="https://example.test/webhook")
    n.campaign_start("camp", "desc", {"Hosts": "a"})
    n.campaign_progress("camp", done=5, total=20, elapsed=90)
    n.campaign_done("camp", total=20, duration=3723, output="data/x")
    colors = [c["payload"]["embeds"][0]["color"] for c in captured]
    assert colors == [COLORS["start"], COLORS["progress"], COLORS["done"]]
    # progress embed reports the percentage
    progress_fields = captured[1]["payload"]["embeds"][0]["fields"]
    assert any("25%" in f["value"] for f in progress_fields)


def test_failed_uses_fail_color(captured) -> None:
    n = DiscordNotifier(webhook="https://example.test/webhook")
    n.campaign_failed("camp", "host down")
    embed = captured[0]["payload"]["embeds"][0]
    assert embed["color"] == COLORS["fail"]
    assert embed["description"] == "host down"


def test_post_failure_is_swallowed(monkeypatch) -> None:
    def boom(url: str, data: bytes, timeout: int = 10) -> int:
        raise OSError("network down")

    monkeypatch.setattr(notify, "_http_post", boom)
    n = DiscordNotifier(webhook="https://example.test/webhook")
    assert n.send_text("x") is False  # error swallowed, returns False


def test_webhook_read_from_env(monkeypatch, captured) -> None:
    monkeypatch.setenv("BRAIDLAB_DISCORD_WEBHOOK", "https://env.test/hook")
    n = DiscordNotifier()
    assert n.enabled
    n.send_text("hi")
    assert captured[0]["url"] == "https://env.test/hook"


@pytest.mark.parametrize(
    "seconds,expected",
    [(5, "5s"), (90, "1m 30s"), (3723, "1h 02m 03s")],
)
def test_humanize_duration(seconds: int, expected: str) -> None:
    assert _humanize_duration(seconds) == expected
