"""Incident log (F11): logs/incidents.jsonl, one JSON line per alert / caution / override / overlay_shown.

Each line: time (local ISO), ts, kind, incident, score, band, reason_ids, plus
  alert:          decision_ms       (trigger -> engine decision)
  overlay_shown:  time_to_alert_ms  (trigger -> warning on screen; the user-facing latency)
  override:       override_until, types
  overlay:*       UI events from the overlay (e.g. overlay:injected_click_blocked): incident, button id
Built only from the fusion events' numbers and reason ids - never screen text, transcripts, window
titles, images or audio. Reason ids are re-checked against a strict pattern before writing, so a
malformed id (e.g. text that slipped into a tool name) is dropped instead of logged.

    log = IncidentLog()                 # path from config fusion.incident_log
    log.write(alert_event)              # AlertEvent | CautionEvent | OverrideEvent (fusion.risk)

Self-test:  python -m fusion.incidents   # prints the records it would write (does not write)
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fusion.risk import ALERT

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("alert", "caution", "override", "overlay_shown")
DEFAULT_PATH = "logs/incidents.jsonl"
REASON_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}(:[a-z0-9_]{1,40})?$")
UI_KIND = re.compile(r"^overlay:[a-z_]{1,40}$")
UI_ID = re.compile(r"^[a-z_]{1,20}$")


@dataclass(frozen=True)
class UiEvent:
    """Derived overlay event, e.g. UiEvent(ts, "overlay:injected_click_blocked", incident=3, button="safe").
    Fixed ids only: no coordinates, window text or input data."""
    ts: float
    kind: str
    incident: int = 0
    button: str = ""


def is_incident_kind(kind: Any) -> bool:
    return isinstance(kind, str) and (kind in KINDS or bool(UI_KIND.match(kind)))


def safe_reason_ids(ids: Any) -> list[str]:
    return [i for i in (ids or ()) if isinstance(i, str) and REASON_ID.match(i)]


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) + f".{int(ts % 1 * 1000):03d}"


def record_of(event: Any) -> dict[str, Any]:
    """Event -> log record (ids and numbers only)."""
    kind = getattr(event, "kind", None)
    if not is_incident_kind(kind):
        raise TypeError(f"not an incident event: {type(event).__name__}")
    ts = float(event.ts)
    if kind.startswith("overlay:"):
        rec = {"time": _iso(ts), "ts": round(ts, 3), "kind": kind, "incident": int(getattr(event, "incident", 0) or 0)}
        button = getattr(event, "button", "")
        if button and UI_ID.match(button):
            rec["button"] = button
        return rec
    rec: dict[str, Any] = {
        "time": _iso(ts),
        "ts": round(ts, 3),
        "kind": kind,
        "incident": int(getattr(event, "incident", 0) or 0),
        "score": int(event.score),
        "band": ALERT if kind in ("alert", "overlay_shown") else ("caution" if kind == "caution" else str(event.band)),
        "reason_ids": safe_reason_ids(event.reason_ids),
    }
    if kind == "alert":
        rec["decision_ms"] = float(event.decision_ms)
    if kind == "overlay_shown":
        rec["time_to_alert_ms"] = float(event.time_to_alert_ms)
    if kind == "alert" and getattr(event, "new_types", ()):
        rec["new_types"] = safe_reason_ids(event.new_types)
    if kind == "override":
        rec["override_until"] = round(float(event.until), 3)
        rec["types"] = safe_reason_ids(event.types)
    return rec


class IncidentLog:
    def __init__(self, path: str | Path | None = None, config: Mapping[str, Any] | None = None):
        if path is None:
            if config is None:
                from kavach_config import get_config
                config = get_config()
            path = (config.get("fusion", {}) or {}).get("incident_log") or DEFAULT_PATH
        p = Path(path)
        self.path = p if p.is_absolute() else ROOT / p
        self._lock = threading.Lock()
        self.written = 0

    def write(self, event: Any) -> dict[str, Any]:
        rec = record_of(event)
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self.written += 1
        return rec

    def __call__(self, event: Any) -> None:
        """Subscriber form: ignores non-incident events."""
        if is_incident_kind(getattr(event, "kind", None)):
            self.write(event)


if __name__ == "__main__":
    from fusion.risk import AlertEvent, CautionEvent, OverlayShownEvent, OverrideEvent

    now = time.time()
    for ev in (CautionEvent(now - 20, 55, ("screen:bank", "remote_tool:anydesk")),
               AlertEvent(1, now, now - 0.84, 100, ("call:authority", "combo", "remote_tool:anydesk")),
               OverlayShownEvent(1, now + 0.2, now - 0.84, 100, ("call:authority", "combo", "remote_tool:anydesk")),
               OverrideEvent(now + 5, now + 605, 100, "alert", ("combo",), ("call_tactic", "remote_tool"), 1)):
        print(json.dumps(record_of(ev)))
