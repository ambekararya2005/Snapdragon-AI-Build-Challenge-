"""kavach.pipeline fusion thread with capture stages off: events posted directly, no models or devices."""

import json
import threading
import time
from types import SimpleNamespace

from fusion.incidents import IncidentLog
from fusion.risk import RemoteToolSignal
from kavach.pipeline import Pipeline, status_line
from kavach_config import load_config

CFG = load_config(env={})


def _label(tactics=(), **labels):
    return SimpleNamespace(labels=labels, screen_tactics=frozenset(tactics))


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_fusion_thread_publishes_states_alerts_and_override(tmp_path):
    log = IncidentLog(tmp_path / "incidents.jsonl")
    pipe = Pipeline(CFG, screen=False, audio=False, processes=False, incident_log=log)
    states, events = [], []
    lock = threading.Lock()
    pipe.subscribe(on_state=lambda s: states.append(s), on_event=lambda e: (lock.acquire(), events.append(e), lock.release()))
    pipe.start()
    try:
        assert _wait(lambda: len(states) >= 2), "ticks publish states without any events"
        t = time.time()
        pipe.post("processes", RemoteToolSignal("AnyDesk", True, True), t)
        pipe.post("screen", _label(bank=0.9, otp_card=0.7, tactics={"threat"}), t - 0.3)
        assert _wait(lambda: any(e.kind == "alert" for e in events))
        alert = next(e for e in events if e.kind == "alert")
        assert alert.score == 100 and alert.time_to_alert_ms >= 300
        assert pipe.state.band == "alert"
        pipe.override()
        assert _wait(lambda: any(e.kind == "override" for e in events))
        assert _wait(lambda: pipe.state.override_active)
        h = pipe.health()
        assert h["fusion"]["alive"] and h["fusion"]["events"] > 0 and h["fusion"]["errors"] == 0
        assert h["screen"]["enabled"] is False and h["processes"]["events"] == 1 and h["screen"]["events"] == 1
    finally:
        pipe.stop()
    assert not pipe.health()["fusion"]["alive"]
    kinds = [json.loads(l)["kind"] for l in log.path.read_text(encoding="utf-8").splitlines()]
    assert kinds == ["alert", "override"]
    assert [e.kind for e in events].count("alert") == 1


def test_bad_event_is_counted_not_fatal():
    pipe = Pipeline(CFG, screen=False, audio=False, processes=False, incident_log=False)
    pipe.start()
    try:
        pipe.post("asr", "raw transcript text")                 # not a signal type -> fusion error, keeps going
        assert _wait(lambda: pipe.health()["fusion"]["errors"] == 1)
        pipe.post("processes", RemoteToolSignal("AnyDesk", True))
        assert _wait(lambda: pipe.state is not None and pipe.state.score == 30)
        assert "raw transcript" not in (pipe.health()["fusion"]["last_error"] or "")
    finally:
        pipe.stop()


def test_status_line_has_ids_only():
    pipe_state = SimpleNamespace(score=45, band="quiet", override_active=False,
                                 contributions=[{"signal": "screen:bank", "points": 25}, {"signal": "screen:otp_card", "points": 20}])
    line = status_line(pipe_state, "[chrome.exe] <title hidden>")
    assert "score  45  quiet" in line and "screen:bank(25)" in line and line.endswith("<title hidden>")
