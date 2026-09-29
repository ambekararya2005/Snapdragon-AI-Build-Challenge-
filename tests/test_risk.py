"""Fusion truth table (F6/F11): scripted timelines on a fake clock -> RiskEngine.

Screen and call signals come from the real detectors on the text fixtures (screen classifier on
tests/fixtures/screen/*.txt, intent detector on demo/scripts/*.txt fed line by line), so a lexicon
change that moves a scenario shows up here. No models, threads or Windows APIs.
The table is printed at the end of the pytest run (tests/conftest.py).
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from detect.intent import detect, read_script
from detect.screen_classifier import classify
from fusion.incidents import REASON_ID, IncidentLog, record_of
from fusion.risk import ALERT, CAUTION, QUIET, AlertEvent, CautionEvent, RemoteToolSignal, RiskEngine, fade
from kavach_config import load_config
from tests.conftest import TRUTH_TABLE

ROOT = Path(__file__).resolve().parent.parent
SCREEN = ROOT / "tests" / "fixtures" / "screen"
SCRIPTS = ROOT / "demo" / "scripts"
CFG = load_config(env={})
T0 = 1000.0              # fake clock origin (small numbers keep log timestamps short)
TICK = 0.5               # the pipeline re-scores every 0.5 s


def screen(name: str):
    return classify((SCREEN / f"{name}.txt").read_text(encoding="utf-8"), None, CFG.screen_classifier)


def call_lines(name: str) -> list[str]:
    return [l for l in read_script(SCRIPTS / f"{name}.txt").splitlines() if l.strip()]


def remote(live: bool = True, tool: str = "AnyDesk"):
    return RemoteToolSignal(tool, live, live)


class Sim:
    """Fake clock + engine; score() every TICK like the pipeline's fusion thread."""

    def __init__(self, log: IncidentLog | None = None):
        self.engine = RiskEngine(CFG)
        self.t = 0.0
        self.state = self.engine.score(T0)
        self.max = 0
        self.events: list = []
        self.log = log
        self.segments: list[tuple[float, float, str]] = []

    def _score(self):
        self.state = self.engine.score(T0 + self.t)
        self.max = max(self.max, self.state.score)
        self.events += self.state.events
        if self.log:
            for ev in self.state.events:
                self.log.write(ev)

    def advance(self, to: float) -> "Sim":
        while self.t + TICK <= to + 1e-9:
            self.t += TICK
            self._score()
        return self

    def at(self, t: float, event, lag: float = 0.0) -> "Sim":
        """Signal observed at t - lag, delivered to fusion at t (lag = OCR / ASR time)."""
        self.advance(t)
        self.engine.update(event, T0 + t, ts=T0 + t - lag)
        self._score()
        return self

    def say(self, t: float, line: str) -> "Sim":
        """One more utterance in the rolling transcript -> intent over the whole window (like the ASR stage)."""
        self.segments.append((T0 + t - 3.0, T0 + t, line))
        return self.at(t, detect(list(self.segments), CFG.intent))

    def call(self, name: str, start: float, end: float) -> "Sim":
        lines = call_lines(name)
        step = (end - start) / max(1, len(lines) - 1)
        for i, line in enumerate(lines):
            self.say(start + i * step, line)
        return self

    def override(self, t: float):
        self.advance(t)
        ev = self.engine.user_override(T0 + t)
        if self.log:
            self.log.write(ev)
        return ev

    @property
    def alerts(self) -> list[AlertEvent]:
        return [e for e in self.events if isinstance(e, AlertEvent)]


def record(scenario: str, expect: str, sim: Sim, ok: bool) -> None:
    alerts = sim.alerts
    TRUTH_TABLE.append({
        "scenario": scenario, "expect": expect, "final": sim.state.score, "max": sim.max, "band": sim.state.band,
        "alerts": len(alerts), "t2a_ms": f"{alerts[0].time_to_alert_ms:.0f}" if alerts else "-", "ok": ok,
        "reasons": ", ".join(sim.state.reason_ids) or "-",
    })


# ---------------------------------------------------------------- truth table

def test_hero_scam_call():
    """Remote live + bank transfer + OTP screen + 5 call tactics within 60 s -> 100, alert, combo."""
    sim = Sim()
    sim.at(0, remote())
    sim.at(5, screen("bank_transfer"), lag=0.8)          # bank + otp_card, no screen tactics
    sim.call("hero_call", 8, 60)
    st = sim.state
    ok = (st.score == 100 and st.band == ALERT and "combo" in st.reason_ids and len(sim.alerts) == 1
          and {"call:authority", "call:threat", "call:secrecy", "call:urgency", "call:money_move"} <= set(st.reason_ids))
    record("hero", "100 alert combo", sim, ok)
    assert st.score == 100 and st.band == ALERT
    assert "combo" in st.reason_ids and "remote_tool:anydesk" in st.reason_ids
    assert {"screen:bank", "screen:otp_card"} <= set(st.reason_ids)
    assert {f"call:{t}" for t in ("authority", "threat", "secrecy", "urgency", "money_move")} <= set(st.reason_ids)
    assert len(sim.alerts) == 1, "exactly one AlertEvent per incident"
    # remote 30 + bank 25 + otp 20 = 75 crosses 70 at the screen event; t_trigger = when it was captured
    assert sim.alerts[0].t_trigger == pytest.approx(T0 + 5 - 0.8)
    assert sim.alerts[0].time_to_alert_ms == pytest.approx(800)
    assert ok


def test_genuine_it_help():
    """Remote live + an IT-support call with no scam tactics -> 30, quiet."""
    sim = Sim().at(0, remote())
    sim.call("normal_it_call", 2, 60)
    ok = sim.state.score == 30 and sim.state.band == QUIET and sim.max == 30 and not sim.events
    record("genuine IT help", "30 quiet", sim, ok)
    assert sim.state.score == 30 and sim.state.band == QUIET
    assert sim.max == 30 and not sim.events
    assert ok


def test_news_video():
    """A news clip about scams (browser article + narration): no remote, no money screen -> <= 40, quiet."""
    sim = Sim().at(0, screen("news_digital_arrest"))
    sim.call("news_clip", 2, 60)
    call_tactics = [r for r in sim.state.reason_ids if r.startswith("call:")]
    ok = sim.max <= 40 and sim.state.band == QUIET and len(call_tactics) <= 1 and not sim.alerts
    record("news video", "<=40 quiet", sim, ok)
    assert len(call_tactics) <= 1
    assert sim.max <= 40 and sim.state.band == QUIET and not sim.alerts
    assert ok


def test_normal_banking():
    """Genuine net banking with an OTP field, no remote tool, no call -> 45, quiet."""
    sim = Sim().at(0, screen("genuine_netbanking"))
    sim.advance(60)
    ok = sim.state.score == 45 and sim.state.band == QUIET and not sim.events
    record("normal banking", "45 quiet", sim, ok)
    assert sim.state.score == 45 and sim.state.band == QUIET and not sim.events
    assert set(sim.state.reason_ids) == {"screen:bank", "screen:otp_card"}
    assert ok


def test_tech_support_scam():
    """Fake virus alert page (label + >= 2 screen tactics) + remote tool live -> >= 80, alert."""
    sim = Sim().at(0, remote(tool="Quick Assist")).at(10, screen("fake_virus_alert"), lag=1.2)
    sim.advance(30)
    tactics = [r for r in sim.state.reason_ids if r.startswith("screen_tactic:")]
    ok = sim.state.score >= 80 and sim.state.band == ALERT and len(tactics) >= 2 and len(sim.alerts) == 1
    record("tech-support scam", ">=80 alert", sim, ok)
    assert "screen:fake_alert" in sim.state.reason_ids and "remote_tool:quick_assist" in sim.state.reason_ids
    assert len(tactics) >= 2
    assert sim.state.score >= 80 and sim.state.band == ALERT and len(sim.alerts) == 1
    assert sim.alerts[0].time_to_alert_ms == pytest.approx(1200)
    assert ok


def test_fake_kyc_page_alone():
    """A fake KYC page (bank + OTP + threat/urgency/authority/secrecy) with nothing else -> >= 70, alert."""
    sim = Sim().at(0, screen("fake_kyc"))
    sim.advance(30)
    ok = sim.state.score >= 70 and sim.state.band == ALERT and len(sim.alerts) == 1
    record("fake KYC alone", ">=70 alert", sim, ok)
    assert sim.state.score >= 70 and sim.state.band == ALERT and len(sim.alerts) == 1
    assert "combo" not in sim.state.reason_ids
    assert ok


def test_decay_and_alert_hysteresis():
    """Signals stop: full weight for decay_s (120 s), fade over 20 s, 0 after 140 s.
    Alert clears only once the score has been < 60 for 10 s."""
    sim = Sim().at(0, screen("fake_kyc"))     # 75 = bank 25 + otp 20 + 4 tactics capped 30
    sim.advance(120)
    assert sim.state.score == 75 and sim.state.band == ALERT
    sim.advance(126.5)
    assert sim.state.score >= 60 and sim.state.band == ALERT        # fading, still >= 60
    sim.advance(127)
    first_below = sim.state.score
    assert first_below < 60 and sim.state.band == ALERT             # < 60 starts the 10 s hold
    sim.advance(136.5)
    assert sim.state.band == ALERT and sim.state.score < 60         # held 9.5 s: still alert
    sim.advance(137)
    assert sim.state.band != ALERT                                  # held 10 s: cleared
    sim.advance(140)
    ok = sim.state.score == 0 and sim.state.band == QUIET and not sim.state.reason_ids and len(sim.alerts) == 1
    record("decay", "0 quiet @140s", sim, ok)
    assert sim.state.score == 0 and sim.state.band == QUIET and not sim.state.reason_ids
    assert len(sim.alerts) == 1
    assert ok


def test_hysteresis_timer_resets_when_score_recovers():
    sim = Sim().at(0, remote()).at(0, screen("genuine_netbanking"))      # 75: alert
    assert sim.state.band == ALERT
    sim.at(10, remote(False))                                            # 45: < 60
    sim.advance(18)
    sim.at(18, remote())                                                 # back to 75 before 10 s
    sim.at(19, remote(False))
    sim.advance(28)                                                      # < 60 for 9 s since 19
    assert sim.state.band == ALERT
    sim.advance(29.5)
    assert sim.state.band == CAUTION or sim.state.band == QUIET
    assert len(sim.alerts) == 1, "same incident: no second alert"


def test_override_suppresses_same_signals_but_not_new_type():
    """After "I'm safe, continue" the same signals do not re-alert (even as a new incident);
    a signal type that was not active at override time does."""
    sim = Sim()
    sim.at(0, remote()).at(5, screen("bank_transfer"))
    sim.call("hero_call", 8, 60)
    assert len(sim.alerts) == 1
    ov = sim.override(62)
    assert set(ov.types) == {"remote_tool", "money_screen", "otp_card", "call_tactic"}
    sim.at(90, screen("bank_transfer"))
    sim.advance(200)
    assert len(sim.alerts) == 1 and sim.state.override_active
    # Tool closed, everything decays, incident clears ...
    sim.at(200, remote(False))
    sim.advance(420)
    assert sim.state.band == QUIET
    # ... same signal types come back within 10 minutes: new incident, but suppressed
    sim.at(430, remote()).at(435, screen("bank_transfer"))
    sim.advance(450)
    assert sim.state.score >= 70 and sim.state.band == ALERT and len(sim.alerts) == 1
    # A new signal type (fake alert label + screen tactics): alert again
    sim.at(460, screen("fake_virus_alert"))
    new = sim.alerts[-1]
    ok = len(sim.alerts) == 2 and {"fake_alert_label", "screen_tactic"} <= set(new.new_types)
    record("override", "no re-alert; new", sim, ok)
    assert len(sim.alerts) == 2
    assert {"fake_alert_label", "screen_tactic"} <= set(new.new_types)
    assert not sim.state.override_active
    assert ok


def test_override_expires_after_override_minutes():
    sim = Sim().at(0, screen("fake_kyc"))
    sim.override(5)
    sim.advance(200)                                    # decays, clears
    assert sim.state.band == QUIET
    sim.at(300, screen("fake_kyc"))                     # within 10 min: suppressed
    assert len(sim.alerts) == 1 and sim.state.band == ALERT
    sim.advance(700)                                    # clears; override ended at 605
    assert not sim.state.override_active and sim.state.band == QUIET
    sim.at(710, screen("fake_kyc"))
    assert len(sim.alerts) == 2


def test_caution_toast_at_most_once_per_minute():
    sim = Sim().say(0, "This is the police, there is an arrest warrant against you, transfer the money now")
    sim.at(1, remote())
    assert sim.state.band == CAUTION
    sim.at(10, remote(False)).at(30, remote())          # quiet -> caution again within 60 s
    sim.at(40, remote(False)).at(70, remote())          # 69 s after the first toast
    toasts = [e for e in sim.events if isinstance(e, CautionEvent)]
    assert [round(e.ts - T0) for e in toasts] == [1, 70]


def test_fade_and_caps():
    assert fade(10, 120, 20) == 1.0 and fade(130, 120, 20) == 0.5 and fade(141, 120, 20) == 0.0
    assert fade(1e6, None, 20) == 1.0
    sim = Sim()
    sim.say(0, "I am calling from the CBI, you are under digital arrest, do not tell anyone, "
               "transfer the money to a safe account immediately or you will be arrested")
    calls = [c for c in sim.state.contributions if c["type"] == "call_tactic"]
    assert sum(c["points"] for c in calls) <= 40 + 1e-6
    assert sim.state.score <= 100


def test_engine_accepts_capture_remote_state_and_rejects_text():
    from capture.processes import RemoteToolState

    eng = RiskEngine(CFG)
    eng.update(RemoteToolState("TeamViewer", [1], {1: "user"}, True, None, 0.0, 0.0), T0)
    assert eng.score(T0).reason_ids == ["remote_tool:teamviewer"]
    with pytest.raises(TypeError):
        eng.update("Your OTP is 482913", T0)


# ---------------------------------------------------------------- incident log (F11)

def _shingles(text: str, n: int = 3) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def test_incident_log_has_ids_and_numbers_only(tmp_path):
    log = IncidentLog(tmp_path / "incidents.jsonl")
    sim = Sim(log)
    sim.at(0, remote()).at(5, screen("bank_transfer"), lag=0.5)
    sim.call("hero_call", 8, 60)
    sim.override(62)
    sim.at(100, screen("fake_virus_alert"))
    sim2 = Sim(log)
    sim2.at(0, screen("fake_kyc"))
    sim3 = Sim(log).say(0, "This is the police, there is an arrest warrant against you, transfer the money now")
    sim3.at(1, remote())

    lines = log.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines]
    kinds = [r["kind"] for r in records]
    assert kinds.count("alert") == 3 and "override" in kinds and "caution" in kinds
    base = {"time", "ts", "kind", "incident", "score", "band", "reason_ids", "time_to_alert_ms"}
    for r in records:
        assert base <= set(r) <= base | {"new_types", "override_until", "types"}
        assert all(REASON_ID.match(i) for i in r["reason_ids"])
        assert (r["time_to_alert_ms"] is not None) == (r["kind"] == "alert")
    assert records[kinds.index("alert")]["time_to_alert_ms"] == pytest.approx(500)

    # No fixture text leaks: no 3-word phrase, no 6+ digit run (OTP, account, phone) from any input.
    logged = log.path.read_text(encoding="utf-8").lower()
    logged_shingles = _shingles(logged)
    fixtures = [(SCREEN / f"{n}.txt").read_text(encoding="utf-8")
                for n in ("bank_transfer", "fake_virus_alert", "fake_kyc")]
    fixtures.append(read_script(SCRIPTS / "hero_call.txt"))
    for text in fixtures:
        assert not (_shingles(text) & logged_shingles)
        for digits in re.findall(r"\d{6,}", text):
            assert digits not in logged
        for line in text.splitlines():
            if len(line.strip()) >= 12:
                assert line.strip().lower() not in logged


def test_record_drops_malformed_reason_ids():
    ev = AlertEvent(1, T0, T0, 90, ("screen:bank", "Your OTP is 482913", "call:threat"))
    assert record_of(ev)["reason_ids"] == ["screen:bank", "call:threat"]
    with pytest.raises(TypeError):
        record_of(SimpleNamespace(kind="transcript", text="hello"))
