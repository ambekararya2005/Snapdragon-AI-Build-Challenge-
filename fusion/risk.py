"""Risk fusion (F6): weighted, time-decayed, capped signals plus a combo bonus, mapped to bands.

Pure: no threads, no I/O, no clock of its own. The caller passes `now` (and the event's own
timestamp), so the engine runs the same under a fake clock in tests and time.time() in the pipeline.

    engine = RiskEngine(config)
    engine.update(screen_label, now, ts=frame.ts)     # ScreenLabel (detect.screen_classifier)
    engine.update(intent_result, now, ts=chunk.ts_end) # IntentResult (detect.intent)
    engine.update(RemoteToolSignal("AnyDesk", live=True), now)
    state = engine.score(now)                          # RiskState; state.events has Alert/Caution events
    engine.user_override(now)                          # "I'm safe, continue" -> OverrideEvent

Signals (config fusion.signals: weight, decay_s, cap):
    remote_tool       remote-access tool live (user process); counts while live, no decay
    money_screen      screen label bank or upi_payment >= label_threshold
    otp_card          screen label otp_card
    fake_alert_label  screen label fake_alert
    screen_tactic     per tactic in ScreenLabel.screen_tactics
    call_tactic       per tactic in IntentResult.tactics
    combo_bonus       while remote live + money screen active + any tactic (screen or call) active
A hit keeps full weight for decay_s after it was last seen, then fades linearly to 0 over fade_s.
Each signal is capped, then the total is capped at 100.

Tactic gate: without at least one active tactic (screen_tactic / call_tactic) or an active
fake_alert label, the score is clamped to fusion.no_tactic_max (69, just under alert) and the reason
id "gate:no_tactic" is reported while the clamp is active. So remote + bank + OTP alone (a genuine IT
helper while a bank page is open) is a caution, and the alert fires at the first scam signal.

Bands: quiet < caution (50) <= caution < alert (70). Alert has hysteresis: it clears only after the
score stays below alert_clear_below (60) for alert_clear_hold_s (10 s). One AlertEvent per incident.

Output is ids and numbers only (reason ids like remote_tool:anydesk, screen:bank, call:authority,
combo, gate:no_tactic). No screen text, transcript or window title ever enters the engine's state.
reason_id_catalog(config) lists every id the engine can emit (for user-facing string tables).

Self-test:  python -m fusion.risk        # replays a short scripted timeline, prints the score per step
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SIGNAL_TYPES = ("remote_tool", "money_screen", "otp_card", "fake_alert_label", "screen_tactic", "call_tactic")
COMBO = "combo_bonus"
GATE_TYPES = frozenset({"screen_tactic", "call_tactic", "fake_alert_label"})   # any of these opens the gate
GATE_REASON = "gate:no_tactic"
TACTIC_NAMES = ("authority", "threat", "secrecy", "urgency", "money_move")   # detect.tactic_lexicon.TACTICS
MONEY_LABELS = ("bank", "upi_payment")
QUIET, CAUTION, ALERT = "quiet", "caution", "alert"

DEFAULT_SIGNALS: dict[str, dict[str, Any]] = {
    "remote_tool": {"weight": 30, "decay_s": None, "cap": 30},
    "money_screen": {"weight": 25, "decay_s": 120, "cap": 25},
    "otp_card": {"weight": 20, "decay_s": 120, "cap": 20},
    "fake_alert_label": {"weight": 20, "decay_s": 120, "cap": 20},
    "screen_tactic": {"weight": 15, "decay_s": 120, "cap": 30},
    "call_tactic": {"weight": 10, "decay_s": 180, "cap": 40},
    "combo_bonus": {"weight": 20, "decay_s": None, "cap": 20},
}
DEFAULTS: dict[str, Any] = {
    "fade_s": 20.0, "label_threshold": 0.5, "bands": {"caution": 50, "alert": 70},
    "alert_clear_below": 60, "alert_clear_hold_s": 10.0, "caution_cooldown_s": 60.0, "override_minutes": 10.0,
    "no_tactic_max": 69,
}
_ID_PART = re.compile(r"[^a-z0-9]+")


def slug(name: str) -> str:
    """Tool / label / tactic name -> reason-id part ("Quick Assist" -> "quick_assist")."""
    return _ID_PART.sub("_", str(name).lower()).strip("_") or "unknown"


def reason_id_catalog(config: Mapping[str, Any] | None = None, extra_tools: Iterable[str] = ()) -> list[str]:
    """Every reason id RiskEngine can emit: fixed ids + remote_tool:<slug> for processes.known_tools.
    User-facing string tables (act/strings.yaml) are checked against this list."""
    if config is None:
        from kavach_config import get_config
        config = get_config()
    tools = [t.get("name", "") for t in (config.get("processes", {}) or {}).get("known_tools", []) or []]
    ids = [f"remote_tool:{slug(t)}" for t in [*tools, *extra_tools] if t]
    ids += [f"screen:{name}" for name in (*MONEY_LABELS, "otp_card", "fake_alert")]
    ids += [f"screen_tactic:{t}" for t in TACTIC_NAMES] + [f"call:{t}" for t in TACTIC_NAMES]
    ids += ["combo", GATE_REASON]
    return list(dict.fromkeys(ids))


def fade(age_s: float, decay_s: float | None, fade_s: float) -> float:
    """Weight factor 0..1: 1 for decay_s after the hit, then linear to 0 over fade_s."""
    if decay_s is None or age_s <= decay_s:
        return 1.0
    if fade_s <= 0:
        return 0.0
    return max(0.0, 1.0 - (age_s - decay_s) / fade_s)


# ---------------------------------------------------------------- events in / out

@dataclass(frozen=True)
class RemoteToolSignal:
    """One remote-access tool's state (from capture.processes.RemoteToolState: live = has_user_process)."""
    tool: str
    live: bool
    active_session: bool | None = None


@dataclass(frozen=True)
class AlertEvent:
    incident: int
    ts: float                          # when the engine decided (score() time)
    t_trigger: float                   # timestamp of the event that pushed the score across the alert band
    score: int
    reason_ids: tuple[str, ...]
    new_types: tuple[str, ...] = ()    # set when the alert broke through a user override
    kind: str = "alert"

    @property
    def decision_ms(self) -> float:
        """Trigger -> engine decision. The user-facing time_to_alert_ms is on OverlayShownEvent."""
        return round(max(0.0, self.ts - self.t_trigger) * 1000.0, 1)


@dataclass(frozen=True)
class OverlayShownEvent:
    """The overlay reports when the alert window was actually on screen (RiskEngine.overlay_shown)."""
    incident: int
    ts: float                          # overlay shown
    t_trigger: float
    score: int
    reason_ids: tuple[str, ...]
    kind: str = "overlay_shown"

    @property
    def time_to_alert_ms(self) -> float:
        """Trigger (first scam signal observed) -> warning on screen."""
        return round(max(0.0, self.ts - self.t_trigger) * 1000.0, 1)


@dataclass(frozen=True)
class CautionEvent:
    ts: float
    score: int
    reason_ids: tuple[str, ...]
    kind: str = "caution"


@dataclass(frozen=True)
class OverrideEvent:
    ts: float
    until: float
    score: int
    band: str
    reason_ids: tuple[str, ...]
    types: tuple[str, ...]             # signal types active at override time
    incident: int = 0
    kind: str = "override"


@dataclass
class RiskState:
    ts: float
    score: int
    band: str                          # quiet | caution | alert (alert includes the hysteresis hold)
    contributions: list[dict[str, Any]]   # [{signal, type, points, age_s}], freshest first within a type
    reason_ids: list[str]
    in_alert: bool = False
    incident: int = 0                  # current / last incident number
    override_active: bool = False
    events: list[Any] = field(default_factory=list)   # AlertEvent / CautionEvent raised by this score()


# ---------------------------------------------------------------- engine

class RiskEngine:
    def __init__(self, config: Mapping[str, Any] | None = None):
        """config: the whole Kavach config or just its `fusion` section (None -> get_config())."""
        if config is None:
            from kavach_config import get_config
            config = get_config()
        f = config.get("fusion", config) if isinstance(config, Mapping) else {}
        raw_signals = f.get("signals") or {}
        self.signals = {name: {**DEFAULT_SIGNALS[name], **(raw_signals.get(name) or {})} for name in DEFAULT_SIGNALS}
        get = lambda k: f.get(k) if f.get(k) is not None else DEFAULTS[k]  # noqa: E731
        bands = {**DEFAULTS["bands"], **(f.get("bands") or {})}
        self.fade_s = float(get("fade_s"))
        self.label_threshold = float(get("label_threshold"))
        self.caution_at = int(bands["caution"])
        self.alert_at = int(bands["alert"])
        self.clear_below = int(get("alert_clear_below"))
        self.clear_hold_s = float(get("alert_clear_hold_s"))
        self.caution_cooldown_s = float(get("caution_cooldown_s"))
        self.override_s = float(get("override_minutes")) * 60.0
        self.no_tactic_max = float(get("no_tactic_max"))
        self.reset()

    def reset(self) -> None:
        self._items: dict[str, tuple[str, float]] = {}       # reason id -> (signal type, last seen)
        self._remote: dict[str, float] = {}                  # live tool reason id -> live since
        self._pending_trigger: float | None = None
        self._in_alert = False
        self._below_since: float | None = None
        self._incident = 0
        self._band = QUIET
        self._last_caution: float | None = None
        self._override_until: float | None = None
        self._override_types: frozenset[str] = frozenset()
        self._alerts: dict[int, AlertEvent] = {}             # incident -> its AlertEvent (for overlay_shown)

    # -- input

    def update(self, event: Any, now: float, ts: float | None = None) -> None:
        """Apply one signal event. ts = when the signal was observed (frame / chunk time), default now."""
        ts = now if ts is None else float(ts)
        before = self._compute(now)[0]
        if isinstance(event, RemoteToolSignal) or hasattr(event, "has_user_process"):
            self._update_remote(event, ts)
        elif hasattr(event, "labels") and hasattr(event, "screen_tactics"):
            self._update_screen(event, ts)
        elif hasattr(event, "tactics"):
            self._update_call(event, ts)
        else:
            raise TypeError(f"unsupported signal event: {type(event).__name__}")
        after = self._compute(now)[0]
        if not self._in_alert and self._pending_trigger is None and round(before) < self.alert_at <= round(after):
            self._pending_trigger = ts

    def _update_remote(self, ev: Any, ts: float) -> None:
        rid = f"remote_tool:{slug(ev.tool)}"
        live = bool(ev.live if isinstance(ev, RemoteToolSignal) else ev.has_user_process)
        if live:
            self._remote.setdefault(rid, ts)
        else:
            self._remote.pop(rid, None)

    def _seen(self, rid: str, kind: str, ts: float) -> None:
        prev = self._items.get(rid)
        self._items[rid] = (kind, max(ts, prev[1]) if prev else ts)

    def _update_screen(self, label: Any, ts: float) -> None:
        labels = getattr(label, "labels", {}) or {}
        thr = self.label_threshold
        for name in MONEY_LABELS:
            if labels.get(name, 0) >= thr:
                self._seen(f"screen:{name}", "money_screen", ts)
        if labels.get("otp_card", 0) >= thr:
            self._seen("screen:otp_card", "otp_card", ts)
        if labels.get("fake_alert", 0) >= thr:
            self._seen("screen:fake_alert", "fake_alert_label", ts)
        for tactic in getattr(label, "screen_tactics", ()) or ():
            self._seen(f"screen_tactic:{slug(tactic)}", "screen_tactic", ts)

    def _update_call(self, result: Any, ts: float) -> None:
        for tactic, hit in (getattr(result, "tactics", {}) or {}).items():
            # The rolling transcript repeats old tactics in every result: count from when the tactic
            # was last said (TacticHit.last_ts), not from when this result arrived.
            last = getattr(hit, "last_ts", None)
            seen = min(float(last), ts) if isinstance(last, (int, float)) else ts
            self._seen(f"call:{slug(tactic)}", "call_tactic", seen)

    # -- scoring

    def _compute(self, now: float) -> tuple[float, list[dict[str, Any]], frozenset[str]]:
        """(total 0..100, contributions, active signal types). Pure read of the current items."""
        per_type: dict[str, list[tuple[float, str, float]]] = {}      # type -> [(age, rid, raw points)]
        for rid, since in self._remote.items():
            per_type.setdefault("remote_tool", []).append((max(0.0, now - since), rid, float(self.signals["remote_tool"]["weight"])))
        for rid, (kind, seen) in self._items.items():
            s = self.signals[kind]
            age = max(0.0, now - seen)
            pts = float(s["weight"]) * fade(age, s["decay_s"], self.fade_s)
            if pts > 0:
                per_type.setdefault(kind, []).append((age, rid, pts))

        contributions: list[dict[str, Any]] = []
        totals: dict[str, float] = {}
        for kind in SIGNAL_TYPES:
            hits = sorted(per_type.get(kind, ()))
            raw = sum(pts for _, _, pts in hits)
            if raw <= 0:
                continue
            cap = float(self.signals[kind]["cap"])
            scale = min(1.0, cap / raw)         # over the cap: every hit scaled down, all stay listed
            totals[kind] = min(raw, cap)
            for age, rid, pts in hits:
                contributions.append({"signal": rid, "type": kind, "points": round(pts * scale, 1),
                                      "age_s": round(age, 1)})
        active = frozenset(totals)
        if "remote_tool" in active and "money_screen" in active and active & {"screen_tactic", "call_tactic"}:
            c = self.signals[COMBO]
            pts = float(min(c["weight"], c["cap"]))
            totals[COMBO] = pts
            contributions.append({"signal": "combo", "type": COMBO, "points": round(pts, 1), "age_s": 0.0})
        total = min(100.0, sum(totals.values()))
        if not (active & GATE_TYPES) and total > self.no_tactic_max:
            # Tactic gate: no scam signal yet -> stay below the alert band.
            contributions.append({"signal": GATE_REASON, "type": "gate",
                                  "points": round(self.no_tactic_max - total, 1), "age_s": 0.0})
            total = self.no_tactic_max
        return total, contributions, active

    def _prune(self, now: float) -> None:
        for rid, (kind, seen) in list(self._items.items()):
            d = self.signals[kind]["decay_s"]
            if d is not None and now - seen > d + self.fade_s:
                del self._items[rid]

    def active_types(self, now: float) -> frozenset[str]:
        return self._compute(now)[2]

    def score(self, now: float) -> RiskState:
        """Current risk; advances the band / incident state machine. Call after updates and on a timer."""
        self._prune(now)
        total, contributions, types = self._compute(now)
        score = int(round(total))
        reason_ids = [c["signal"] for c in sorted(contributions, key=lambda c: (-c["points"], c["signal"]))]
        events: list[Any] = []

        if self._override_until is not None and now >= self._override_until:
            self._override_until, self._override_types = None, frozenset()
        override = self._override_until is not None
        new_types = tuple(sorted(types - self._override_types)) if override else ()

        if not self._in_alert:
            if score >= self.alert_at:
                self._in_alert, self._below_since = True, None
                self._incident += 1
                t_trigger = self._pending_trigger if self._pending_trigger is not None else now
                self._pending_trigger = None
                if not override or new_types:
                    events.append(AlertEvent(self._incident, now, t_trigger, score, tuple(reason_ids), new_types))
                    self._override_until, self._override_types = None, frozenset()
        else:
            if override and new_types:
                # A signal type that was not there when the user said "I'm safe" -> alert again.
                self._incident += 1
                events.append(AlertEvent(self._incident, now, now, score, tuple(reason_ids), new_types))
                self._override_until, self._override_types = None, frozenset()
            if score < self.clear_below:
                if self._below_since is None:
                    self._below_since = now
                if now - self._below_since >= self.clear_hold_s:
                    self._in_alert, self._below_since = False, None
            else:
                self._below_since = None
        if score < self.alert_at and not self._in_alert:
            self._pending_trigger = None

        override = self._override_until is not None
        band = ALERT if self._in_alert else CAUTION if score >= self.caution_at else QUIET
        if (band == CAUTION and self._band == QUIET and not override
                and (self._last_caution is None or now - self._last_caution >= self.caution_cooldown_s)):
            self._last_caution = now
            events.append(CautionEvent(now, score, tuple(reason_ids)))
        for ev in events:
            if isinstance(ev, AlertEvent):
                self._alerts[ev.incident] = ev
                for old in [i for i in self._alerts if i < ev.incident - 20]:
                    del self._alerts[old]
        self._band = band
        return RiskState(now, score, band, contributions, reason_ids, self._in_alert, self._incident, override, events)

    def overlay_shown(self, incident: int, shown_ts: float) -> OverlayShownEvent | None:
        """Overlay on screen for an incident -> event with time_to_alert_ms = shown - t_trigger.
        Once per incident; None for unknown / already reported incidents."""
        alert = self._alerts.pop(incident, None)
        if alert is None:
            return None
        return OverlayShownEvent(incident, float(shown_ts), alert.t_trigger, alert.score, alert.reason_ids)

    def user_override(self, now: float) -> OverrideEvent:
        """"I'm safe, continue": no new alerts for override_minutes unless a new signal type appears."""
        total, contributions, types = self._compute(now)
        self._override_until = now + self.override_s
        self._override_types = types
        reason_ids = tuple(c["signal"] for c in sorted(contributions, key=lambda c: (-c["points"], c["signal"])))
        return OverrideEvent(now, self._override_until, int(round(total)), self._band, reason_ids,
                             tuple(sorted(types)), self._incident)


# ---------------------------------------------------------------- self-test

def format_state(state: RiskState) -> str:
    """One line, ids and numbers only (safe to print/log)."""
    parts = " ".join(f"{c['signal']}({c['points']:g})" for c in
                     sorted(state.contributions, key=lambda c: (-c["points"], c["signal"])))
    flags = " [override]" if state.override_active else ""
    return f"score {state.score:3d}  {state.band:<7}{flags}  {parts or '-'}"


def _main() -> int:
    from types import SimpleNamespace

    def screen(**labels: float) -> Any:
        tactics = frozenset(labels.pop("tactics", ()))  # type: ignore[arg-type]
        return SimpleNamespace(labels=labels, screen_tactics=tactics)

    def call(*tactics: str) -> Any:
        return SimpleNamespace(tactics={t: SimpleNamespace(last_ts=None) for t in tactics})

    timeline = [
        (0, RemoteToolSignal("AnyDesk", True)),
        (10, call("authority")),
        (20, call("authority", "threat", "urgency")),
        (30, screen(bank=0.9, otp_card=0.8)),
        (45, call("authority", "threat", "urgency", "secrecy", "money_move")),
        (60, RemoteToolSignal("AnyDesk", False)),
    ]
    engine = RiskEngine()
    t, i = 0.0, 0
    while t <= 400:
        while i < len(timeline) and timeline[i][0] <= t:
            engine.update(timeline[i][1], t)
            i += 1
        st = engine.score(t)
        for ev in st.events:
            extra = f" time_to_alert {ev.decision_ms:.0f} ms" if isinstance(ev, AlertEvent) else ""
            print(f"t={t:5.1f}  >> {ev.kind.upper()} score {ev.score}{extra}")
        if t % 10 == 0:
            print(f"t={t:5.1f}  {format_state(st)}")
        t += 0.5
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
