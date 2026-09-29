"""Incident log (F11): logs/incidents.jsonl, one JSON line per alert / caution / override.

Each line: time (local ISO), ts, kind, incident, score, band, reason_ids, time_to_alert_ms.
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
from pathlib import Path
from typing import Any, Mapping

from fusion.risk import ALERT

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = "logs/incidents.jsonl"
REASON_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}(:[a-z0-9_]{1,40})?$")


def safe_reason_ids(ids: Any) -> list[str]:
    return [i for i in (ids or ()) if isinstance(i, str) and REASON_ID.match(i)]


def record_of(event: Any) -> dict[str, Any]:
    """Event -> log record (ids and numbers only)."""
    kind = getattr(event, "kind", None)
    if kind not in ("alert", "caution", "override"):
        raise TypeError(f"not an incident event: {type(event).__name__}")
    ts = float(event.ts)
    rec: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) + f".{int(ts % 1 * 1000):03d}",
        "ts": round(ts, 3),
        "kind": kind,
        "incident": int(getattr(event, "incident", 0) or 0),
        "score": int(event.score),
        "band": ALERT if kind == "alert" else ("caution" if kind == "caution" else str(event.band)),
        "reason_ids": safe_reason_ids(event.reason_ids),
        "time_to_alert_ms": float(event.time_to_alert_ms) if kind == "alert" else None,
    }
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
        if getattr(event, "kind", None) in ("alert", "caution", "override"):
            self.write(event)


if __name__ == "__main__":
    from fusion.risk import AlertEvent, CautionEvent, OverrideEvent

    now = time.time()
    for ev in (CautionEvent(now - 20, 55, ("screen:bank", "remote_tool:anydesk")),
               AlertEvent(1, now, now - 0.84, 100, ("call:authority", "combo", "remote_tool:anydesk")),
               OverrideEvent(now + 5, now + 605, 100, "alert", ("combo",), ("call_tactic", "remote_tool"), 1)):
        print(json.dumps(record_of(ev)))
