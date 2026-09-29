"""bench.scenarios without hardware: spec + assets, signal records, replay, metrics, reports, CLI dry run."""

import re
import time
from types import SimpleNamespace

import pytest

from bench import scenarios as S
from fusion.risk import RemoteToolSignal
from kavach_config import load_config

CFG = load_config(env={}).to_dict()


def _label(tactics=(), **labels):
    return SimpleNamespace(labels=labels, screen_tactics=frozenset(tactics))


def _call(**last_ts):
    return SimpleNamespace(tactics={t: SimpleNamespace(last_ts=v) for t, v in last_ts.items()})


def _record(signals, stimuli=(), duration=40.0, type_="scam", expected="alert"):
    return {"id": "x", "type": type_, "expected": expected, "duration_s": duration,
            "signals": list(signals), "stimuli": list(stimuli)}


# ---------------------------------------------------------------- spec + assets

def test_spec_has_ten_scam_and_ten_normal_with_existing_assets():
    spec = S.load_spec()
    assert len(spec) == 20 and len({s.id for s in spec}) == 20
    assert sum(s.type == "scam" for s in spec) == 10 and sum(s.type == "normal" for s in spec) == 10
    for s in spec:
        assert s.page_path is None or s.page_path.is_file()
        assert s.call_path is None or S.script_lines(s.call_path)
        assert s.type == "normal" or s.expected in ("alert", "caution")
        assert s.type == "scam" or s.expected == "quiet"


def test_demo_pages_are_labelled_demo_and_have_unique_titles():
    pages = sorted(S.PAGES_DIR.glob("*.html"))
    titles = [S.page_title(p) for p in pages]
    assert len(set(titles)) == len(titles), "the runner finds page windows by title"
    for p, title in zip(pages, titles):
        text = p.read_text(encoding="utf-8")
        assert "DEMO" in title and 'class="demo-banner"' in text, p.name
        assert not re.search(r"https?://", text), f"{p.name}: demo pages load nothing from the network"


def test_call_scripts_are_marked_fictional():
    for p in S.SCRIPTS_DIR.glob("*.txt"):
        head = p.read_text(encoding="utf-8").splitlines()[0]
        assert head.startswith("#") and "FICTIONAL" in head, p.name


def test_spec_validation_errors(tmp_path):
    bad = tmp_path / "spec.yaml"
    bad.write_text("scenarios:\n  - {id: a, type: scam, page: nope.html, remote: none, call: null, duration_s: 10, expected: alert}\n")
    with pytest.raises(S.SpecError, match="page not found"):
        S.load_spec(bad)
    bad.write_text("scenarios:\n  - {id: a, type: scam, page: null, remote: teleport, call: null, duration_s: 10, expected: alert}\n")
    with pytest.raises(S.SpecError, match="remote"):
        S.load_spec(bad)
    with pytest.raises(S.SpecError, match="unknown"):
        S.select(S.load_spec(), "digital_arrest,nope")
    assert [s.id for s in S.select(S.load_spec(), "work_call, email")] == ["work_call", "email"]


# ---------------------------------------------------------------- records + replay

def test_encode_decode_roundtrip_keeps_only_derived_fields():
    t0 = 1000.0
    screen = S.encode_signal("screen", _label({"threat"}, bank=0.91234, otp_card=0.7, normal=0.1), 1002.0, 1002.4, t0)
    assert screen == {"t": 2.4, "ts": 2.0, "stage": "screen", "kind": "screen",
                      "labels": {"bank": 0.912, "otp_card": 0.7}, "tactics": ["threat"]}
    call = S.encode_signal("asr", _call(authority=1003.5, threat=None), 1005.0, 1005.2, t0)
    assert call["tactics"] == {"authority": 3.5, "threat": None}
    remote = S.encode_signal("processes", RemoteToolSignal("FakeRemote", True, None), 1001.0, 1001.0, t0)
    assert remote["tool"] == "FakeRemote" and remote["live"] is True
    assert S.encode_signal("x", object(), 1, 1, 0) is None
    assert S.decode_signal(screen).screen_tactics == frozenset({"threat"})
    assert S.decode_signal(call).tactics["authority"].last_ts == 3.5
    assert S.decode_signal(remote) == RemoteToolSignal("FakeRemote", True)


def test_replay_scores_like_the_engine_and_measures_time_to_alert():
    sigs = [
        {"t": 2.0, "ts": 2.0, "stage": "processes", "kind": "remote", "tool": "FakeRemote", "live": True},
        {"t": 4.3, "ts": 3.5, "stage": "screen", "kind": "screen", "labels": {"bank": 0.9, "otp_card": 0.8}, "tactics": []},
        {"t": 12.6, "ts": 12.0, "stage": "asr", "kind": "call", "tactics": {"authority": 11.0}},
    ]
    stimuli = [{"t": 1.0, "kind": "tool"}, {"t": 3.0, "kind": "page"}, {"t": 9.0, "kind": "line", "i": 1},
               {"t": 11.5, "kind": "line", "i": 2}, {"t": 14.0, "kind": "line", "i": 3}]
    o = S.replay(_record(sigs, stimuli), CFG)
    assert o.band == "alert" and o.alert and o.caution            # 75 -> gate 69 caution, then the call tactic
    assert o.max_score == 100 and "combo" in o.reasons
    assert o.trigger_s == 12.0 and o.alert_s == 12.6 and o.decision_ms == 600.0
    assert o.stimulus == "line 2" and o.e2e_ms == 1100.0            # line 2 ended 11.5 -> alert 12.6
    quiet = S.replay(_record(sigs[1:2], stimuli[1:2], type_="normal", expected="quiet"), CFG)
    assert quiet.band == "quiet" and quiet.max_score == 45 and quiet.e2e_ms is None


def test_speech_trigger_is_dated_by_the_script_onset():
    lines = ["Hello, good morning.", "This is the CBI calling about your parcel."]
    onsets = S.speech_onsets(lines, starts=[0.0, 2.0], ends=[1.5, 6.0], intent_cfg=CFG["intent"])
    assert list(onsets) == ["authority"]
    assert onsets["authority"] == round(2.0 + len("This is the CBI") / len(lines[1]) * 4.0, 3)
    sigs = [{"t": 1.0, "ts": 1.0, "stage": "processes", "kind": "remote", "tool": "FakeRemote", "live": True},
            {"t": 1.5, "ts": 1.2, "stage": "screen", "kind": "screen", "labels": {"bank": 0.9, "otp_card": 0.8}, "tactics": []},
            {"t": 7.4, "ts": 7.0, "stage": "asr", "kind": "call", "tactics": {"authority": 6.0}}]
    stimuli = [{"t": 0.5, "kind": "tool"}, {"t": 1.0, "kind": "page"}, {"t": 2.0, "kind": "line_start", "i": 2},
               {"t": 6.0, "kind": "line", "i": 2}]
    o = S.replay({**_record(sigs, stimuli, duration=10.0), "onsets": onsets}, CFG)
    assert o.stimulus == "speech:authority" and o.e2e_ms == round((7.4 - onsets["authority"]) * 1000, 1)


def test_replay_follows_the_config():
    sigs = [{"t": 1.0, "ts": 1.0, "stage": "asr", "kind": "call",
             "tactics": {t: 1.0 for t in ("authority", "threat", "secrecy", "urgency", "money_move")}}]
    assert S.replay(_record(sigs), CFG).max_score == 40             # call-only: capped below caution
    tuned = {**CFG, "fusion": {**CFG["fusion"], "signals": {**CFG["fusion"]["signals"],
                                                            "call_tactic": {"weight": 12, "decay_s": 180, "cap": 60}}}}
    assert S.replay(_record(sigs), tuned).band == "caution"


def test_recording_pipeline_records_signals_the_live_outcome_matches_replay():
    t0 = time.time()
    signals, tracker = [], S.Tracker(t0)
    pipe = S.RecordingPipeline(CFG, screen=False, audio=False, processes=False, incident_log=False,
                               record=lambda st, p, ts, tp: signals.append(S.encode_signal(st, p, ts, tp, t0)))
    pipe.subscribe(on_state=tracker.state, on_event=tracker.event)
    pipe.start()
    try:
        pipe.post("processes", RemoteToolSignal("FakeRemote", True, None), t0 + 0.1)
        pipe.post("screen", _label({"threat"}, bank=0.9, otp_card=0.8), t0 + 0.2)
        end = time.monotonic() + 3
        while not tracker.o.alert and time.monotonic() < end:
            time.sleep(0.02)
    finally:
        pipe.stop()
    live = tracker.finish([{"t": 0.0, "kind": "page"}])
    assert live.alert and live.stimulus == "page"
    assert [s["kind"] for s in signals] == ["remote", "screen"]
    replayed = S.replay(_record(signals, [{"t": 0.0, "kind": "page"}], duration=2.0), CFG)
    assert (replayed.band, replayed.max_score) == (live.band, live.max_score)


# ---------------------------------------------------------------- metrics + reports

def _row(id_, type_, expected, outcome, e2e=None, dec=None):
    return {"id": id_, "type": type_, "expected": expected, "outcome": outcome,
            "ok": S.is_ok(type_, expected, outcome), "alert": outcome == "alert", "e2e_ms": e2e, "decision_ms": dec}


def test_is_ok_and_summary_targets():
    assert S.is_ok("scam", "caution", "alert") and not S.is_ok("scam", "alert", "caution")
    assert S.is_ok("normal", "quiet", "quiet") and not S.is_ok("normal", "quiet", "caution")
    rows = [_row(f"s{i}", "scam", "alert", "alert", e2e=1000.0 * (i + 1), dec=500.0) for i in range(9)]
    rows.append(_row("s9", "scam", "alert", "quiet"))
    rows += [_row(f"n{i}", "normal", "quiet", "quiet") for i in range(9)] + [_row("n9", "normal", "quiet", "caution")]
    s = S.summarize(rows)
    assert s["scam_caught_pct"] == 90.0 and s["pass_caught"]
    assert (s["normal_alerts"], s["normal_cautions"]) == (0, 1) and s["pass_false_alarms"]
    assert s["tta_n"] == 9 and s["tta_p50_s"] == 5.0 and s["tta_p95_s"] == 8.6 and not s["pass_tta"]
    assert not s["pass_all"]
    assert S.percentile([], 0.5) is None and S.percentile([3.0], 0.95) == 3.0


def test_reports_and_compare_are_written(tmp_path):
    sigs = [{"t": 1.0, "ts": 1.0, "stage": "asr", "kind": "call",
             "tactics": {t: 1.0 for t in ("authority", "threat", "secrecy", "urgency", "money_move")}}]
    rec = {**_record(sigs, [{"t": 0.5, "kind": "line", "i": 1}], expected="caution"), "id": "kyc_call_no_page",
           "run_at": "2026-09-29 10:00", "providers": {"asr": "aihub_whisper:DmlExecutionProvider"}, "notes": []}
    rec["live"] = S.asdict(S.replay(rec, CFG))
    rows, summary = S.write_reports({rec["id"]: rec}, [rec["id"]], tmp_path, CFG)
    assert rows[0]["outcome"] == "quiet" and not rows[0]["ok"]
    md = (tmp_path / "scenarios.md").read_text(encoding="utf-8")
    assert "scam caught" in md and "kyc_call_no_page" in md and "MISS" in md
    assert (tmp_path / "scenarios.csv").read_text(encoding="utf-8").splitlines()[0].startswith("id,type,expected,outcome")
    tuned = {**CFG, "fusion": {**CFG["fusion"], "signals": {**CFG["fusion"]["signals"],
                                                            "call_tactic": {"weight": 12, "decay_s": 180, "cap": 60}}}}
    sb, sa, lines = S.compare({rec["id"]: rec}, [rec["id"]], CFG, tuned, tmp_path / "tuning.md")
    assert sb["scam_caught"] == 0 and sa["scam_caught"] == 1
    assert any("fusion.signals.call_tactic.cap" in l and "40" in l and "60" in l for l in lines)


def test_tuning_overlays_merge_and_validate():
    for path in (S.ROOT / "bench" / "tuning").glob("*.yaml"):
        overlay = S.yaml.safe_load(path.read_text(encoding="utf-8"))
        merged = S.deep_merge(CFG, overlay)
        changes = S.config_changes(CFG, merged)
        assert changes and all(k.startswith("fusion.") for k, _, _ in changes), path.name
    merged = S.deep_merge(CFG, {"fusion": {"signals": {"call_tactic": {"weight": 13}}}})
    assert merged["fusion"]["signals"]["call_tactic"] == {**CFG["fusion"]["signals"]["call_tactic"], "weight": 13}
    assert CFG["fusion"]["signals"]["call_tactic"]["weight"] == 10          # base untouched


def test_records_roundtrip_in_spec_order(tmp_path):
    path = tmp_path / "runs.jsonl"
    S.save_records({"b": {"id": "b"}, "a": {"id": "a"}, "z": {"id": "z"}}, ["a", "b"], path)
    assert list(S.load_records(path)) == ["a", "b", "z"]
    assert S.load_records(tmp_path / "missing.jsonl") == {}


def test_cli_dry_run_runs_nothing(capsys):
    assert S.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "20 scenarios (10 scam, 10 normal)" in out and "Nothing was run" in out
    assert S.main(["--dry-run", "--only", "nope"]) == 2
