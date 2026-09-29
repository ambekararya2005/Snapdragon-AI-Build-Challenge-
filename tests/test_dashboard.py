"""dashboard.app view-model functions (pure) + main.py argument handling."""

from types import SimpleNamespace

import pytest

from dashboard import app as D


def test_band_colours_and_display_band():
    assert D.band_colour("quiet") == "#22C55E" and D.band_colour("caution") == "#F59E0B"
    assert D.band_colour("alert") == "#EF4444" and D.band_colour("paused") == "#64748B"
    assert D.band_colour(None) == D.band_colour("quiet") and D.band_colour("weird") == D.band_colour("quiet")
    st = SimpleNamespace(score=72, band="alert")
    assert D.display_band(st, None) == (72, "alert")
    assert D.display_band(st, 9999.0) == (0, "paused")
    assert D.display_band(None, None) == (0, "quiet")


def test_formatting():
    assert D.fmt_age(None, 100) == "never" and D.fmt_age(99.6, 100) == "just now"
    assert D.fmt_age(97, 100) == "3 s ago" and D.fmt_age(-500, 100) == "10 min ago"
    assert D.fmt_age(105, 100) == "just now"                        # clock skew: never negative
    assert D.fmt_ms(12.345) == "12.3" and D.fmt_ms(None) == "-"
    assert D.provider_short("DmlExecutionProvider") == "GPU (DirectML)"
    assert D.provider_short("QNNExecutionProvider") == "NPU (QNN)" and D.provider_short(None) == "-"
    assert D.gauge_extent(0) == 0 and D.gauge_extent(100) == -270 and D.gauge_extent(50) == -135
    assert D.gauge_extent(150) == -270 and D.gauge_extent(-5) == 0
    assert D.pause_text(None, 0) == "Monitoring" and D.pause_text(10, 20) == "Monitoring"
    assert D.pause_text(3600 + 1000, 1000).startswith("Paused until ") and "60 min left" in D.pause_text(3600 + 1000, 1000)


def _tool(name, roles, live, session):
    return SimpleNamespace(tool=name, roles=roles, has_user_process=live, active_session=session)


def test_remote_lines():
    assert D.remote_lines([]) == ["No remote-access tool running"]
    lines = D.remote_lines([_tool("TeamViewer", {1: "service"}, False, False),
                            _tool("AnyDesk", {2: "user", 3: "service"}, True, True),
                            _tool("RustDesk", {4: "tray"}, False, False),
                            _tool("Quick Assist", {5: "user"}, True, None)])
    assert lines == ["AnyDesk: LIVE (active session)", "Quick Assist: LIVE (session unknown)",
                     "RustDesk: idle (tray)", "TeamViewer: idle (service)"]


def test_screen_and_call_lines_show_ids_and_numbers_only():
    label = SimpleNamespace(labels={"bank": 1.0, "otp_card": 0.8, "upi_payment": 0.2, "fake_alert": 0.5, "normal": 0.0},
                            top_label="bank", screen_tactics=frozenset({"urgency", "threat"}), news_context=False,
                            evidence=["ctx:browser", "bank:netbanking:ifsc", "pattern:ifsc", "a", "b", "c", "d", "e"])
    lines = D.screen_lines((label, 97.0), 100.0)
    assert lines[0] == "Top label: bank"
    assert lines[1] == "Labels ≥ 0.5: bank 1.00, otp_card 0.80, fake_alert 0.50"
    assert lines[2] == "Tactics: threat, urgency" and lines[3] == "Last OCR: 3 s ago"
    assert lines[4] == "Evidence (7 ids): a, b, …"
    assert D.screen_lines((SimpleNamespace(**{**vars(label), "evidence": ["pattern:ifsc"]}), 97.0), 100.0)[4] ==         "Evidence (1 id): pattern:ifsc"
    assert "ctx:browser" not in lines[4]
    assert D.screen_lines(None, 0) == ["Waiting for the first OCR frame"]

    res = SimpleNamespace(tactics={"threat": SimpleNamespace(score=0.8), "authority": SimpleNamespace(score=1.0)})
    call = D.call_lines((res, 95.0), (98.0, True), {"speech": 3, "silent": 7}, 100.0)
    assert call == ["Tactics: authority 1.00, threat 0.80", "Last chunk: 2 s ago (speech)", "Chunks: 3 speech / 7 silent"]
    assert D.call_lines(None, (99.9, False), {}, 100.0)[:2] == ["Tactics: no speech yet", "Last chunk: just now (silent)"]
    assert D.call_lines(None, None, {}, 0)[1] == "Last chunk: never"


def test_aihub_mapping_and_model_rows():
    assert D.aihub_key("ocr.det") == "det_static" and D.aihub_key("ocr.rec_640") == "rec_static_640"
    assert D.aihub_key("asr_whisper_base_encoder") == "whisper_base_encoder" and D.aihub_key("other") is None
    idx = D.aihub_index([
        {"kind": "profile", "status": "SUCCESS", "model": "det_static", "inference_ms": 14.139,
         "ops": {"NPU": 201, "GPU": 0, "CPU": 0}, "device": "Snapdragon X Elite CRD"},
        {"kind": "inference", "status": "SUCCESS", "model": "det_static"},
        {"kind": "profile", "status": "FAILED", "model": "rec_static_320"},
    ])
    assert idx == {"det_static": {"ms": 14.139, "npu_ops": 201, "total_ops": 201, "device": "Snapdragon X Elite CRD"}}
    stats = [{"name": "asr_whisper_base_decoder", "actual_provider": "DmlExecutionProvider", "p50_ms": 9.1, "p95_ms": 12, "n": 40},
             {"name": "ocr.rec_1280", "actual_provider": "DmlExecutionProvider", "p50_ms": None, "p95_ms": None, "n": 0},
             {"name": "ocr.det", "actual_provider": "CPUExecutionProvider", "p50_ms": 50.0, "p95_ms": 70.25, "n": 9},
             {"name": "ocr.rec_320", "actual_provider": "DmlExecutionProvider", "p50_ms": 4, "p95_ms": 6, "n": 9},
             {"name": "ocr.native", "actual_provider": "DmlExecutionProvider", "p50_ms": 200, "p95_ms": 220, "n": 9},
             {"name": "dummy", "actual_provider": "CPUExecutionProvider"}]
    rows = D.model_rows(stats, idx)
    assert [r[0] for r in rows] == ["ocr.det", "ocr.rec_320", "ocr.rec_1280", "ocr total (det+rec)",
                                    "whisper_base dec"]
    assert rows[0] == ("ocr.det", "CPU", "50.0", "70.2", "9", "14.1", "201/201")
    assert rows[2][2:] == ("-", "-", "0", "-", "-")
    table = D.format_model_table(rows)
    assert table.splitlines()[0].startswith("model") and len(table.splitlines()) == 6
    assert D.format_model_table([]) == "no models loaded yet"


def test_real_aihub_file_covers_the_runtime_models():
    idx = D.load_aihub()
    for name in ("ocr.det", "ocr.rec_320", "ocr.rec_640", "ocr.rec_1280", "asr_whisper_base_encoder",
                 "asr_whisper_base_decoder"):
        hub = idx.get(D.aihub_key(name))
        assert hub and hub["ms"] > 0 and hub["npu_ops"] == hub["total_ops"] > 0, name


def test_net_status_and_live_process_has_no_connections():
    assert D.net_status([]) == (0, "Network connections opened by Kavach: 0", True)
    n, text, ok = D.net_status([object(), object()])
    assert n == 2 and not ok and text.endswith(": 2")
    assert D.net_status(None)[2] is False
    conns = D.kavach_connections()
    assert conns is not None and D.net_status(conns)[2], "the test process must not hold inet sockets"

    class Denied:
        def net_connections(self, kind):
            import psutil
            raise psutil.AccessDenied(1)
    assert D.kavach_connections(Denied()) is None


def test_timeline_points():
    hist = [(0.0, 100.0), (60.0, 50.0), (119.0, 0.0), (120.0, 25.0), (130.0, 10.0)]
    pts = D.timeline_points(hist, now=120.0, x0=0, y0=0, w=120, h=100)
    # (0,100) is exactly 120 s old -> x=0, y=0; (60,50) -> x=60, y=50; (119,0) -> x=119, y=100; future dropped
    assert pts == pytest.approx([0, 0, 60, 50, 119, 100, 120, 75])
    assert D.timeline_points([(0.0, 50.0)], now=200.0, x0=0, y0=0, w=10, h=10) == []


def test_main_parse_args_and_env(monkeypatch):
    import main

    a = main.parse_args(["--no-dashboard", "--provider", "cpu", "--seconds", "5"])
    assert a.no_dashboard and a.provider == "cpu" and a.seconds == 5 and not a.plain
    with pytest.raises(SystemExit):
        main.parse_args(["--provider", "tpu"])


def test_mask_text_and_capture_flag():
    assert D.mask_text(None) == "" and D.mask_text((0, 100)) == "" and D.mask_text((5, 0)) == ""
    assert D.mask_text((38, 100)) == " · 38% masked (Kavach)"
    label = SimpleNamespace(labels={"bank": 1.0}, top_label="bank", screen_tactics=frozenset(), evidence=[])
    assert D.screen_lines((label, 97.0), 100.0, mask=(25, 100))[3] == "Last OCR: 3 s ago · 25% masked (Kavach)"
    from act.overlay import hide_from_capture_enabled
    from kavach_config import load_config

    assert hide_from_capture_enabled(load_config(env={})) is False          # recorders (OBS / Game Bar) see Kavach
    assert hide_from_capture_enabled({"screen": {"hide_kavach_from_capture": True}}) is True
    assert hide_from_capture_enabled(None) is False
