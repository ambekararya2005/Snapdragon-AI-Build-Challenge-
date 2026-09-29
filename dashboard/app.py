"""Kavach live dashboard: a normal Tk window ("Kavach — live") on the overlay's Tk root.

    dash = Dashboard(root, pipeline, overlay)     # built on the main thread; refreshes itself at 2 Hz

Panels: risk gauge (0-100, band colour) + 120 s score timeline; System / Screen / Call signal cards
(ids, labels, scores and ages only); models (provider actually used, p50/p95 ms from
models.runtime.all_stats()) next to the Snapdragon NPU numbers from bench/results/aihub_results.json;
privacy (network connections opened by this process - live, must stay 0; raw data on disk; CPU %);
controls (Pause 1 hour / Resume, Resume remote app when something is suspended).

The dashboard never blocks the pipeline: the only pipeline callback is a deque append (score
history); everything else is read from pipeline.snapshot() on the Tk thread. When the dashboard
covers part of the watched window, the screen sampler masks it (white) before OCR, so Kavach never
reads its own text; the Screen card shows how much was masked. Screen recorders still see it.
"Pause" and "Resume remote app" are mouse-only and ignore injected clicks (a remote session could
otherwise pause Kavach).

The view-model functions below are pure (tested in tests/test_dashboard.py).

Self-test (fake data, no pipeline):  python -m dashboard.app --demo [--seconds 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AIHUB_RESULTS = ROOT / "bench" / "results" / "aihub_results.json"
TITLE = "Kavach — live"
WINDOW_S = 120.0
REFRESH_MS = 500
PAUSE_S = 3600.0
DASH_GUARDED = frozenset({"pause", "resume_remote"})

BG, CARD, FG, MUTED, GRID = "#0F172A", "#1E293B", "#F8FAFC", "#94A3B8", "#334155"
BAND_COLOURS = {"quiet": "#22C55E", "caution": "#F59E0B", "alert": "#EF4444", "paused": "#64748B"}
PROVIDERS = {"QNNExecutionProvider": "NPU (QNN)", "DmlExecutionProvider": "GPU (DirectML)",
             "CUDAExecutionProvider": "GPU (CUDA)", "CPUExecutionProvider": "CPU"}
MONO, UI = ("Consolas", 10), "Segoe UI"


# ---------------------------------------------------------------- view model (pure)

def band_colour(band: str | None) -> str:
    return BAND_COLOURS.get(band or "quiet", BAND_COLOURS["quiet"])


def fmt_age(ts: float | None, now: float) -> str:
    if ts is None:
        return "never"
    age = max(0.0, now - ts)
    if age < 1:
        return "just now"
    if age < 120:
        return f"{age:.0f} s ago"
    return f"{age / 60:.0f} min ago"


def fmt_ms(v: Any) -> str:
    return f"{float(v):.1f}" if isinstance(v, (int, float)) else "-"


def provider_short(name: str | None) -> str:
    return PROVIDERS.get(name or "", name or "-")


def gauge_extent(score: float, sweep: float = 270.0) -> float:
    """Arc extent in degrees (negative = clockwise in Tk) for a 0-100 score."""
    return -sweep * max(0.0, min(100.0, float(score))) / 100.0


def display_band(state: Any, paused_until: float | None) -> tuple[int, str]:
    if paused_until is not None:
        return 0, "paused"
    if state is None:
        return 0, "quiet"
    return int(state.score), str(state.band)


def pause_text(paused_until: float | None, now: float) -> str:
    if paused_until is None or paused_until <= now:
        return "Monitoring"
    return f"Paused until {time.strftime('%H:%M', time.localtime(paused_until))} ({(paused_until - now) / 60:.0f} min left)"


def remote_lines(states: Iterable[Any]) -> list[str]:
    """System card: one line per known tool (live / idle tray / idle service, active_session)."""
    lines = []
    for s in sorted(states, key=lambda s: s.tool):
        roles = set((getattr(s, "roles", {}) or {}).values())
        if s.has_user_process:
            session = {True: "active session", False: "no session", None: "session unknown"}[s.active_session]
            lines.append(f"{s.tool}: LIVE ({session})")
        elif "tray" in roles:
            lines.append(f"{s.tool}: idle (tray)")
        else:
            lines.append(f"{s.tool}: idle (service)")
    return lines or ["No remote-access tool running"]


def mask_text(mask: tuple[int, int] | None) -> str:
    """(masked px, frame px) of the last OCR'd frame -> "" or " · 38% masked (Kavach)"."""
    if not mask or not mask[0] or not mask[1]:
        return ""
    return f" · {100 * mask[0] / mask[1]:.0f}% masked (Kavach)"


def screen_lines(last: tuple[Any, float] | None, now: float, threshold: float = 0.5, max_evidence: int = 2,
                 mask: tuple[int, int] | None = None) -> list[str]:
    """Screen card: top label, labels >= threshold, screen tactics, OCR age (+ share of the frame masked
    because a Kavach window covered it), evidence ids."""
    if last is None:
        return ["Waiting for the first OCR frame"]
    label, ts = last
    present = sorted(((k, v) for k, v in label.labels.items() if k != "normal" and v >= threshold), key=lambda kv: -kv[1])
    ev = [e for e in label.evidence if not e.startswith("ctx:")]
    shown = sorted(ev, key=lambda e: (len(e), e))[:max_evidence]     # shortest ids fit on one line
    return [
        f"Top label: {label.top_label}" + ("  [news]" if getattr(label, "news_context", False) else ""),
        "Labels ≥ 0.5: " + (", ".join(f"{k} {v:.2f}" for k, v in present) or "none"),
        "Tactics: " + (", ".join(sorted(label.screen_tactics)) or "none"),
        f"Last OCR: {fmt_age(ts, now)}{mask_text(mask)}",
        f"Evidence ({len(ev)} id{'' if len(ev) == 1 else 's'}): " + (", ".join(shown) + (", …" if len(ev) > len(shown) else "") if ev else "-"),
    ]


def call_lines(last: tuple[Any, float] | None, chunk: tuple[float, bool] | None, counts: Mapping[str, int],
               now: float) -> list[str]:
    """Call card: tactics with scores, last chunk age, speech / silent."""
    if last is None:
        tactics = "no speech yet"
    else:
        res = last[0]
        tactics = ", ".join(f"{t} {h.score:.2f}" for t, h in sorted(res.tactics.items(), key=lambda kv: -kv[1].score)) or "none"
    if chunk is None:
        chunk_line = "Last chunk: never"
    else:
        chunk_line = f"Last chunk: {fmt_age(chunk[0], now)} ({'speech' if chunk[1] else 'silent'})"
    return [f"Tactics: {tactics}", chunk_line,
            f"Chunks: {counts.get('speech', 0)} speech / {counts.get('silent', 0)} silent"]


def aihub_key(session_name: str) -> str | None:
    """models.runtime session name -> AI Hub model name (ocr.det -> det_static, ocr.rec_640 ->
    rec_static_640, asr_whisper_base_encoder -> whisper_base_encoder)."""
    if session_name == "ocr.det":
        return "det_static"
    if session_name.startswith("ocr.rec_"):
        return "rec_static_" + session_name.rsplit("_", 1)[1]
    if session_name.startswith("asr_"):
        return session_name[len("asr_"):]
    return None


def aihub_index(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest successful AI Hub profile per model: {model: {ms, npu_ops, total_ops, device}}."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("kind") != "profile" or r.get("status") != "SUCCESS" or r.get("inference_ms") is None:
            continue
        ops = r.get("ops") or {}
        out[r["model"]] = {"ms": float(r["inference_ms"]), "npu_ops": int(ops.get("NPU", 0)),
                           "total_ops": int(sum(ops.values())) if ops else 0, "device": r.get("device", "")}
    return out


def load_aihub(path: Path = AIHUB_RESULTS) -> dict[str, dict[str, Any]]:
    try:
        return aihub_index(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


DISPLAY_NAMES = {"ocr.native": "ocr total (det+rec)"}     # whole OCR call, not a single model


def display_name(name: str) -> str:
    """Short table name: asr_whisper_base_encoder -> whisper_base enc."""
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    if name.startswith("asr_"):
        return name[4:].replace("_encoder", " enc").replace("_decoder", " dec")
    return name


def _model_order(name: str) -> tuple[int, int, str]:
    if name in DISPLAY_NAMES:
        return (1, 10**6, name)
    if name == "ocr.det":
        return (0, 0, name)
    if name.startswith("ocr.rec_"):
        tail = name.rsplit("_", 1)[1]
        return (1, int(tail) if tail.isdigit() else 0, name)
    return (2, 0 if name.endswith("encoder") else 1, name)


def model_rows(stats: Iterable[Mapping[str, Any]], aihub: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, ...]]:
    """(model, provider used, p50 ms, p95 ms, runs, NPU ms, NPU ops) for OCR / ASR sessions."""
    rows = []
    for st in stats:
        name = str(st.get("name", ""))
        if not (name.startswith("ocr.") or name.startswith("asr_")):
            continue
        hub = aihub.get(aihub_key(name) or "")
        rows.append((name, provider_short(st.get("actual_provider")), fmt_ms(st.get("p50_ms")), fmt_ms(st.get("p95_ms")),
                     str(st.get("n", 0)), fmt_ms(hub["ms"]) if hub else "-",
                     f"{hub['npu_ops']}/{hub['total_ops']}" if hub else "-"))
    rows.sort(key=lambda r: _model_order(r[0]))
    return [(display_name(r[0]), *r[1:]) for r in rows]


def format_model_table(rows: Sequence[tuple[str, ...]]) -> str:
    head = ("model", "provider used", "p50", "p95", "runs", "NPU ms", "NPU ops")
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) if rows else len(head[i]) for i in range(len(head))]
    fmt = lambda r: "  ".join(c.ljust(widths[i]) if i < 2 else c.rjust(widths[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([fmt(head)] + [fmt(r) for r in rows]) if rows else "no models loaded yet"


def net_status(conns: Sequence[Any] | None) -> tuple[int | None, str, bool]:
    """(count, text, ok). conns = this process's inet connections (None = could not read)."""
    if conns is None:
        return None, "Network connections opened by Kavach: unreadable", False
    n = len(conns)
    return n, f"Network connections opened by Kavach: {n}", n == 0


def kavach_connections(proc: Any = None) -> list[Any] | None:
    """psutil: this process's TCP/UDP sockets (listening or connected), or None if unreadable."""
    import psutil

    proc = proc or psutil.Process()
    try:
        fn = getattr(proc, "net_connections", None) or proc.connections
        return list(fn(kind="inet"))
    except (psutil.Error, OSError):
        return None


def timeline_points(history: Iterable[tuple[float, float]], now: float, x0: float, y0: float, w: float, h: float,
                    window_s: float = WINDOW_S) -> list[float]:
    """Flat [x1, y1, x2, y2, ...] canvas coords for (ts, score) points in the last window_s seconds."""
    pts: list[float] = []
    for ts, score in history:
        age = now - ts
        if age < 0 or age > window_s:
            continue
        pts += [x0 + w * (1.0 - age / window_s), y0 + h * (1.0 - max(0.0, min(100.0, score)) / 100.0)]
    return pts


# ---------------------------------------------------------------- Tk window

class Dashboard:
    def __init__(self, root: Any, pipeline: Any, overlay: Any = None, config: Mapping[str, Any] | None = None,
                 topmost: bool = False, input_guard: bool = True):
        import tkinter as tk

        import psutil

        from act.overlay import MouseGuard, hide_from_capture_enabled, mouse_button, set_capture_excluded

        self.root, self.pipeline, self.overlay = root, pipeline, overlay
        self.cfg = config or getattr(pipeline, "cfg", {}) or {}
        self.bands = (self.cfg.get("fusion", {}) or {}).get("bands", {"caution": 50, "alert": 70})
        self.history: deque[tuple[float, float]] = deque(maxlen=2000)
        self.aihub = load_aihub()
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.ncpu = psutil.cpu_count() or 1
        self.guard = MouseGuard() if input_guard else None
        self.blocked: list[str] = []
        self.net_max = 0
        self.refresh_ms: deque[float] = deque(maxlen=50)
        self.closed = False
        self.on_close: Any = None
        pipeline.subscribe(on_state=self._on_state)          # fusion thread: a deque append, nothing else

        w = self.win = tk.Toplevel(root, bg=BG)
        w.title(TITLE)
        w.geometry("1280x720+40+40")
        w.minsize(1100, 640)
        if topmost:
            w.attributes("-topmost", True)
        w.protocol("WM_DELETE_WINDOW", self._close_clicked)
        w.columnconfigure(0, weight=1)

        head = tk.Frame(w, bg=BG)
        head.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 4))
        tk.Label(head, text="Kavach — live", bg=BG, fg=FG, font=(UI, 18, "bold")).pack(side="left")
        self.status = tk.Label(head, text="", bg=BG, fg=MUTED, font=(UI, 11))
        self.status.pack(side="right")

        top = tk.Frame(w, bg=BG)
        top.grid(row=1, column=0, sticky="ew", padx=16)
        self.gauge = tk.Canvas(top, width=280, height=196, bg=CARD, highlightthickness=0)
        self.gauge.pack(side="left")
        self.timeline = tk.Canvas(top, height=196, bg=CARD, highlightthickness=0)
        self.timeline.pack(side="left", fill="x", expand=True, padx=(12, 0))

        cards = tk.Frame(w, bg=BG)
        cards.grid(row=2, column=0, sticky="ew", padx=16, pady=10)
        self.card_text = {}
        for i, name in enumerate(("System", "Screen", "Call")):
            cards.columnconfigure(i, weight=1, uniform="cards")
            f = tk.Frame(cards, bg=CARD, padx=12, pady=8)
            f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0 if i == 2 else 6))
            tk.Label(f, text=name, bg=CARD, fg=FG, font=(UI, 13, "bold")).pack(anchor="w")
            lbl = tk.Label(f, text="", bg=CARD, fg=FG, font=(UI, 10), justify="left", anchor="nw", wraplength=380, height=5)
            lbl.pack(anchor="w", fill="both")
            self.card_text[name] = lbl

        bottom = tk.Frame(w, bg=BG)
        bottom.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 12))
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)
        mf = tk.Frame(bottom, bg=CARD, padx=12, pady=8)
        mf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        dev = next((v["device"] for v in self.aihub.values() if v.get("device")), "not profiled")
        tk.Label(mf, text="Models: this PC vs Snapdragon NPU", bg=CARD, fg=FG, font=(UI, 13, "bold")).pack(anchor="w")
        tk.Label(mf, text=f"p50/p95 ms measured live here  ·  NPU ms from Qualcomm AI Hub ({dev}, QNN)", bg=CARD,
                 fg=MUTED, font=(UI, 9)).pack(anchor="w")
        self.models = tk.Label(mf, text="", bg=CARD, fg=FG, font=MONO, justify="left", anchor="nw")
        self.models.pack(anchor="w")

        pf = tk.Frame(bottom, bg=CARD, padx=12, pady=8)
        pf.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(pf, text="Privacy", bg=CARD, fg=FG, font=(UI, 13, "bold")).pack(anchor="w")
        self.net = tk.Label(pf, text="", bg=CARD, fg=FG, font=(UI, 11, "bold"), anchor="w")
        self.net.pack(anchor="w")
        raw = "none" if not (self.cfg.get("privacy", {}) or {}).get("write_raw_to_disk", False) else "ENABLED"
        tk.Label(pf, text=f"Raw data written to disk: {raw}\n(screens, text and audio stay in RAM)", bg=CARD, fg=FG,
                 font=(UI, 11), anchor="w", justify="left").pack(anchor="w")
        self.cpu = tk.Label(pf, text="", bg=CARD, fg=FG, font=(UI, 11), anchor="w")
        self.cpu.pack(anchor="w")
        ctl = tk.Frame(pf, bg=CARD)
        ctl.pack(anchor="w", pady=(10, 0))
        self.btn_pause = mouse_button(ctl, "Pause 1 hour", ("#334155", FG), lambda: self._click("pause", self._pause), 12)
        self.btn_resume_mon = mouse_button(ctl, "Resume", ("#334155", FG), lambda: self._click("resume", self._resume), 12)
        self.btn_resume_remote = mouse_button(ctl, "Resume remote app", ("#7A0000", FG),
                                              lambda: self._click("resume_remote", self._resume_remote), 12)
        for b in (self.btn_pause, self.btn_resume_mon, self.btn_resume_remote):
            b.config(padx=14, pady=6, bd=2)
        for b in (self.btn_pause, self.btn_resume_remote):     # hook only while the pointer is over them
            b.bind("<Enter>", lambda e: self.guard and self.guard.start(), add="+")
            b.bind("<Leave>", lambda e: self.guard and self.root.after(300, self.guard.stop), add="+")
        self.btn_pause.pack(side="left")
        self.note = tk.Label(pf, text="", bg=CARD, fg="#FBBF24", font=(UI, 10), anchor="w", wraplength=420, justify="left")
        self.note.pack(anchor="w", pady=(6, 0))
        self._ctl_state = (False, False)                     # (paused, remote suspended) as shown

        w.update_idletasks()
        if hide_from_capture_enabled(self.cfg):              # off by default: the sampler masks it
            set_capture_excluded(w)
        self.root.after(100, self.refresh)

    # -- data in

    def _on_state(self, state: Any) -> None:
        self.history.append((state.ts, float(state.score)))

    # -- controls

    def _click(self, button: str, action: Any) -> None:
        from act.overlay import click_decision

        g = self.guard
        injected = g.last_down_injected if g is not None and g.available else None
        if click_decision(button, injected, DASH_GUARDED):
            action()
            return
        from fusion.incidents import UiEvent

        self.blocked.append(button)
        self.note.config(text="That click came from the remote connection and was ignored. Use your own mouse.")
        if hasattr(self.pipeline, "report_event"):
            self.pipeline.report_event(UiEvent(time.time(), "overlay:injected_click_blocked", 0, button))

    def _pause(self) -> None:
        self.pipeline.pause(PAUSE_S)
        self.note.config(text="Paused for 1 hour. Kavach is not watching until you press Resume.")

    def _resume(self) -> None:
        self.pipeline.resume_monitoring()
        self.note.config(text="")

    def _remote(self) -> Any:
        return getattr(self.overlay, "remote", None)

    def _resume_remote(self) -> None:
        r = self._remote()
        if r is not None:
            r.resume_all()
        self.note.config(text="Remote app resumed.")

    def _close_clicked(self) -> None:
        if self.on_close:
            self.on_close()
        else:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.guard is not None:
            self.guard.stop()
        try:
            self.win.destroy()
        except Exception:  # noqa: BLE001
            pass

    # -- drawing

    def _draw_gauge(self, score: int, band: str) -> None:
        c = self.gauge
        c.delete("all")
        col = band_colour(band)
        box = (52, 28, 228, 204)
        c.create_arc(*box, start=225, extent=-270, style="arc", outline=GRID, width=20)
        if score > 0:
            c.create_arc(*box, start=225, extent=gauge_extent(score), style="arc", outline=col, width=20)
        c.create_text(140, 108, text=str(score), fill=FG, font=(UI, 44, "bold"))
        c.create_text(140, 156, text=band.upper(), fill=col, font=(UI, 15, "bold"))
        c.create_text(140, 12, text="Risk", fill=MUTED, font=(UI, 11))

    def _draw_timeline(self, now: float) -> None:
        c = self.timeline
        c.delete("all")
        W, H = max(200, c.winfo_width()), max(100, c.winfo_height())
        x0, y0, w, h = 44, 24, W - 60, H - 48
        c.create_text(x0, 12, text=f"Score, last {WINDOW_S:.0f} s", fill=MUTED, font=(UI, 11), anchor="w")
        for level, name in ((self.bands.get("alert", 70), "alert"), (self.bands.get("caution", 50), "caution")):
            y = y0 + h * (1 - level / 100)
            c.create_line(x0, y, x0 + w, y, fill=band_colour(name), dash=(4, 4))
            c.create_text(x0 - 6, y, text=str(level), fill=band_colour(name), font=(UI, 9), anchor="e")
        c.create_rectangle(x0, y0, x0 + w, y0 + h, outline=GRID)
        for sec in (0, 30, 60, 90, 120):
            x = x0 + w * (1 - sec / WINDOW_S)
            c.create_text(x, y0 + h + 12, text="now" if sec == 0 else f"-{sec}s", fill=MUTED, font=(UI, 9))
        pts = timeline_points(list(self.history), now, x0, y0, w, h)
        if len(pts) >= 4:
            c.create_line(*pts, fill=FG, width=2)
        elif len(pts) == 2:
            c.create_oval(pts[0] - 2, pts[1] - 2, pts[0] + 2, pts[1] + 2, fill=FG, outline=FG)

    def refresh(self) -> None:
        if self.closed:
            return
        t0 = time.perf_counter()
        try:
            now = time.time()
            snap = self.pipeline.snapshot()
            paused_until = snap.get("paused_until")
            score, band = display_band(snap.get("state"), paused_until)
            while self.history and now - self.history[0][0] > WINDOW_S + 5:
                self.history.popleft()
            self._draw_gauge(score, band)
            self._draw_timeline(now)
            self.card_text["System"].config(text="\n".join(remote_lines(snap.get("remote", []))))
            self.card_text["Screen"].config(text="\n".join(screen_lines(snap.get("screen"), now, mask=snap.get("mask"))))
            self.card_text["Call"].config(text="\n".join(call_lines(snap.get("call"), snap.get("chunk"),
                                                                    snap.get("chunk_counts", {}), now)))
            from models import runtime

            self.models.config(text=format_model_table(model_rows(runtime.all_stats(), self.aihub)))
            n, text, ok = net_status(kavach_connections(self.proc))
            self.net_max = max(self.net_max, n or 0)
            self.net.config(text=text, fg=BAND_COLOURS["quiet"] if ok else BAND_COLOURS["alert"])
            cpu = self.proc.cpu_percent(None) / self.ncpu
            self.cpu.config(text=f"Kavach process CPU: {cpu:.1f}% of all cores")
            self.status.config(text=f"{pause_text(paused_until, now)}   ·   {time.strftime('%H:%M:%S')}")
            self._update_controls(paused_until is not None)
        finally:
            self.refresh_ms.append((time.perf_counter() - t0) * 1000.0)
            self.root.after(REFRESH_MS, self.refresh)

    def _update_controls(self, paused: bool) -> None:
        r = self._remote()
        suspended = bool(r is not None and getattr(r, "suspended", {}))
        if (paused, suspended) == self._ctl_state:
            return
        for btn in (self.btn_pause, self.btn_resume_mon, self.btn_resume_remote):
            btn.pack_forget()
        (self.btn_resume_mon if paused else self.btn_pause).pack(side="left")
        if suspended:
            self.btn_resume_remote.pack(side="left", padx=(10, 0))
        self._ctl_state = (paused, suspended)


# ---------------------------------------------------------------- demo (fake data)

def _demo(seconds: float) -> int:
    import math
    import threading
    from types import SimpleNamespace

    from act.overlay import Overlay

    class FakePipeline:
        cfg: dict = {}

        def __init__(self) -> None:
            self.subs: list[Any] = []
            self.paused_until = None
            self.stop = threading.Event()

        def subscribe(self, on_state=None, on_event=None):
            if on_state:
                self.subs.append(on_state)

        def pause(self, s):
            self.paused_until = time.time() + s

        def resume_monitoring(self):
            self.paused_until = None

        def snapshot(self):
            t = time.time()
            score = int(45 + 40 * max(0.0, math.sin(t / 8)))
            band = "alert" if score >= 70 else "caution" if score >= 50 else "quiet"
            st = SimpleNamespace(ts=t, score=score, band=band)
            for cb in self.subs:
                cb(st)
            label = SimpleNamespace(labels={"bank": 1.0, "otp_card": 0.8, "upi_payment": 0.2, "normal": 0.0},
                                    top_label="bank", screen_tactics=frozenset(), news_context=False,
                                    evidence=["bank:netbanking:ifsc", "pattern:ifsc", "otp_card:otp_entry:otp_has_been_sent"])
            res = SimpleNamespace(tactics={"authority": SimpleNamespace(score=1.0)})
            tool = SimpleNamespace(tool="AnyDesk", roles={1: "user"}, has_user_process=True, active_session=True)
            return {"state": st, "remote": [tool], "screen": (label, t - 3), "call": (res, t - 2),
                    "chunk": (t - 2, True), "chunk_counts": {"speech": 4, "silent": 9},
                    "paused_until": self.paused_until if self.paused_until and self.paused_until > t else None}

    ov = Overlay(speak=False)
    root = ov.make_root()
    dash = Dashboard(root, FakePipeline())
    dash.on_close = root.quit
    if seconds:
        root.after(int(seconds * 1000), root.quit)
    root.mainloop()
    dash.close()
    root.destroy()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m dashboard.app", description="Kavach dashboard self-test")
    ap.add_argument("--demo", action="store_true", required=True, help="fake data, no pipeline")
    ap.add_argument("--seconds", type=float, default=0)
    raise SystemExit(_demo(ap.parse_args().seconds))
