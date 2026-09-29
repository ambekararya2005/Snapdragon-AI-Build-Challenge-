"""act.overlay: strings cover every engine reason id (en + hi); keys never trigger actions; mouse does."""

import re

import pytest

from act.overlay import (GUARDED_BUTTONS, LLMHF_INJECTED, LLMHF_LOWER_IL_INJECTED, NO_KEYS, MouseGuard, Overlay,
                         click_decision, is_injected, load_strings, pick_reasons, reason_text, toast_reason,
                         tool_names, ui)
from act.remote_control import SuspendResult
from fusion.risk import reason_id_catalog
from kavach_config import load_config

CFG = load_config(env={})
S = load_strings()
TOOLS = tool_names(CFG)
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def test_every_engine_reason_id_has_en_and_hi():
    for rid in reason_id_catalog(CFG):
        en, hi = reason_text(rid, "en", S, TOOLS), reason_text(rid, "hi", S, TOOLS)
        assert en and hi, rid
        assert DEVANAGARI.search(hi), f"{rid}: hi text has no Devanagari"
        assert "{" not in en + hi, f"{rid}: unfilled placeholder"
    assert reason_text("remote_tool:quick_assist", "en", S, TOOLS).startswith("Quick Assist ")
    assert reason_text("call:authority", "en", S) == "The caller claims to be from police/CBI/RBI"
    assert reason_text("call:authority", "hi", S) == "कॉल करने वाला खुद को पुलिस/CBI/RBI बता रहा है"


def test_ui_strings_bilingual_and_required_text():
    for key, entry in S["ui"].items():
        assert entry.get("en") and entry.get("hi") and DEVANAGARI.search(entry["hi"]), key
    assert ui("headline", "hi", S) == "रुकिए! यह धोखाधड़ी हो सकती है"
    assert ui("headline", "en", S) == "Stop! This may be a scam"
    assert "1930" in ui("advice", "en", S) and "cybercrime.gov.in" in ui("advice", "en", S)
    assert ui("btn_help", "hi", S) == "रुकें और मदद लें" and ui("btn_safe", "hi", S) == "मैं सुरक्षित हूँ, जारी रखें"
    assert ui("could_not_pause", "en", S, tool="AnyDesk").startswith("Could not pause AnyDesk")


def test_pick_reasons_priority_limit_and_gate_hidden():
    ids = ["combo", "screen:bank", "gate:no_tactic", "call:authority", "remote_tool:anydesk", "screen:otp_card",
           "call:money_move", "not:a_known_id"]
    picked = pick_reasons(ids, S)
    assert picked == ["remote_tool:anydesk", "call:authority", "call:money_move", "screen:otp_card"]
    assert "gate:no_tactic" not in pick_reasons(["gate:no_tactic", "screen:bank", "remote_tool:anydesk"], S)
    assert toast_reason(["remote_tool:anydesk", "screen:bank", "gate:no_tactic"], S) == "gate:no_tactic"
    assert toast_reason(["screen:bank", "call:urgency"], S) == "call:urgency"
    assert toast_reason([], S) is None


# ---------------------------------------------------------------- Tk (skipped without a display)

class FakeRemote:
    def __init__(self, denied=False):
        self.calls = []
        self.denied = denied

    def suspend_live(self):
        self.calls.append("suspend")
        if self.denied:
            return [SuspendResult("AnyDesk", denied=[12])]
        return [SuspendResult("AnyDesk", suspended=[12])]

    def resume_all(self):
        self.calls.append("resume_all")
        return [12]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def override(self):
        self.calls.append("override")

    def alert_shown(self, incident, ts):
        self.calls.append(("shown", incident))


@pytest.fixture
def ov():
    tk = pytest.importorskip("tkinter")
    try:
        o = Overlay(CFG, remote=FakeRemote(), pipeline=FakePipeline(), speak=False, fullscreen=False, input_guard=False)
        o.make_root()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    closed = []
    o.on_closed = closed.append
    o.closed = closed
    yield o
    o.close_toast()
    if o.alert_win is not None:
        o.alert_win.destroy()
    o.root.destroy()


def click(widget):
    widget.update()
    widget.event_generate("<ButtonPress-1>", x=5, y=5)
    widget.event_generate("<ButtonRelease-1>", x=5, y=5)
    widget.update()


def press_keys(ov, *widgets):
    for w in (ov.alert_win, *widgets):
        w.focus_force()
        for seq in NO_KEYS:
            w.event_generate(seq)
        w.update()


def test_alert_shows_reasons_and_reports_shown(ov):
    ov.show_alert(["remote_tool:anydesk", "call:authority", "screen:bank", "gate:no_tactic"], incident=7)
    texts = [l.cget("text") for l in ov.reason_labels]
    assert len(texts) == 3 and "AnyDesk" in texts[0] and "पुलिस" in texts[1]
    assert not any("banking page is open" in t and "remote access" in t for t in texts)   # gate hidden
    assert ov.pipeline.calls == [("shown", 7)] and ov.shown_log[0][0] == 7


def test_enter_space_escape_do_not_trigger_buttons(ov):
    ov.show_alert(["remote_tool:anydesk", "call:threat"], incident=1)
    press_keys(ov, ov.btn_help, ov.btn_safe)
    assert ov.remote.calls == [] and ov.pipeline.calls == [("shown", 1)]
    assert ov.alert_win is not None and ov.closed == []
    assert str(ov.btn_help.cget("takefocus")) == "0" and str(ov.btn_safe.cget("takefocus")) == "0"


def test_click_safe_resumes_overrides_and_closes(ov):
    ov.show_alert(["remote_tool:anydesk", "call:threat"], incident=1)
    click(ov.btn_safe)
    assert ov.remote.calls == ["resume_all"] and "override" in ov.pipeline.calls
    assert ov.alert_win is None and ov.closed == ["safe"]


def test_click_help_pauses_then_resume(ov):
    ov.show_alert(["remote_tool:anydesk", "call:threat"], incident=1)
    click(ov.btn_help)
    assert ov.remote.calls == ["suspend"] and ov.alert_win is not None
    assert "AnyDesk" in ov.help_lines[0].cget("text") and ov.btn_resume is not None
    press_keys(ov, ov.btn_resume, ov.btn_close)                  # keys still do nothing on the help screen
    assert ov.remote.calls == ["suspend"] and ov.alert_win is not None
    click(ov.btn_resume)
    assert ov.remote.calls == ["suspend", "resume_all"] and ov.closed == ["resumed"]
    assert "override" not in ov.pipeline.calls


def test_help_when_pause_denied_says_close_it_yourself(ov):
    ov.remote = FakeRemote(denied=True)
    ov.show_alert(["remote_tool:anydesk"], incident=1)
    click(ov.btn_help)
    text = ov.help_lines[0].cget("text")
    assert "Could not pause AnyDesk" in text and "खुद बंद करें" in text
    assert ov.btn_resume is None


def test_press_and_release_outside_the_button_does_nothing(ov):
    ov.show_alert(["remote_tool:anydesk"], incident=1)
    ov.btn_safe.update()
    ov.btn_safe.event_generate("<ButtonPress-1>", x=5, y=5)
    ov.btn_safe.event_generate("<ButtonRelease-1>", x=-50, y=-50)     # dragged off
    ov.btn_safe.update()
    assert ov.pipeline.calls == [("shown", 1)] and ov.alert_win is not None


def test_toast_bilingual_and_auto_dismiss(ov):
    w = ov.show_toast(["remote_tool:anydesk", "screen:bank", "gate:no_tactic"], seconds=0.2)
    texts = [c.cget("text") for c in w.winfo_children() if c.winfo_class() == "Label"]
    assert "Someone has remote access while a banking page is open" in texts
    assert any(DEVANAGARI.search(t) for t in texts)
    ov.root.after(400, ov.root.quit)
    ov.root.mainloop()
    assert ov.toast_win is None


def test_poll_routes_pipeline_events(ov):
    from fusion.risk import AlertEvent, CautionEvent

    ov.q.put(CautionEvent(0.0, 55, ("screen:bank", "gate:no_tactic")))
    ov._poll()
    assert ov.toast_win is not None
    ov.q.put(AlertEvent(3, 0.0, 0.0, 90, ("remote_tool:anydesk", "call:threat")))
    ov._poll()
    assert ov.alert_win is not None and ov.toast_win is None and ("shown", 3) in ov.pipeline.calls


# ---------------------------------------------------------------- injected-input guard

def test_click_decision_table():
    table = {("safe", True): False, ("resume", True): False, ("help", True): True, ("close", True): True,
             ("safe", False): True, ("resume", False): True, ("help", False): True,
             ("safe", None): True, ("resume", None): True}          # None = hook unavailable -> as before
    for (button, injected), allowed in table.items():
        assert click_decision(button, injected) is allowed, (button, injected)
    assert GUARDED_BUTTONS == {"safe", "resume"}


def test_injected_flag_bits():
    assert is_injected(LLMHF_INJECTED) and is_injected(LLMHF_LOWER_IL_INJECTED)
    assert is_injected(LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED) and not is_injected(0)
    assert not is_injected(0x10)                                     # other flag bits don't count


def test_ui_event_incident_record_has_no_coordinates():
    from fusion.incidents import UiEvent, record_of

    rec = record_of(UiEvent(1000.5, "overlay:injected_click_blocked", 3, "safe"))
    assert set(rec) == {"time", "ts", "kind", "incident", "button"}
    assert rec["kind"] == "overlay:injected_click_blocked" and rec["button"] == "safe" and rec["incident"] == 3
    assert "button" not in record_of(UiEvent(1.0, "overlay:injected_click_blocked", 1, "x=10,y=20"))
    with pytest.raises(TypeError):
        record_of(UiEvent(1.0, "overlay:Bad Kind"))


class FakeGuard:
    available = True

    def __init__(self, injected):
        self.last_down_injected = injected

    def start(self):
        return True

    def stop(self):
        pass


def test_injected_click_on_safe_is_blocked_with_note_and_event(ov):
    ov.pipeline.report_event = lambda ev: ov.pipeline.calls.append(("report", ev.kind, ev.button))
    ov.show_alert(["remote_tool:anydesk", "call:threat"], incident=4)
    ov.guard = FakeGuard(True)
    click(ov.btn_safe)
    assert ov.alert_win is not None and ov.remote.calls == [] and "override" not in ov.pipeline.calls
    assert ("report", "overlay:injected_click_blocked", "safe") in ov.pipeline.calls
    assert ov.blocked == ["safe"]
    note = ov.note.cget("text")
    assert "This click came from the remote connection" in note and "रिमोट कनेक्शन" in note
    ov.guard = FakeGuard(False)                                    # the user's own mouse
    click(ov.btn_safe)
    assert ov.closed == ["safe"] and "override" in ov.pipeline.calls


def test_injected_click_on_help_is_allowed_resume_is_blocked(ov):
    ov.show_alert(["remote_tool:anydesk"], incident=1)
    ov.guard = FakeGuard(True)
    click(ov.btn_help)                                             # stopping is always safe
    assert ov.remote.calls == ["suspend"] and ov.btn_resume is not None
    click(ov.btn_resume)
    assert ov.remote.calls == ["suspend"] and ov.blocked == ["resume"] and ov.alert_win is not None
    click(ov.btn_close)                                            # closing (tool stays paused) is fine
    assert ov.closed == ["help_close"]


def test_guard_unavailable_falls_back_to_allow(ov):
    g = FakeGuard(True)
    g.available = False                                            # hook failed to install
    ov.show_alert(["remote_tool:anydesk"], incident=1)
    ov.guard = g
    click(ov.btn_safe)
    assert ov.closed == ["safe"] and ov.blocked == []


def test_mouse_guard_start_stop_real_hook():
    import sys

    if sys.platform != "win32":
        pytest.skip("Windows only")
    g = MouseGuard()
    ok = g.start()
    try:
        assert ok is g.available            # installs (or reports why not) without raising
    finally:
        g.stop()
    assert not g.available and g._thread is None
