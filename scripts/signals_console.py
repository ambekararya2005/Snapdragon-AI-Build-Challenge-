"""Live signals console (end of Day 1): every detector running together, no fusion yet.

Threads (each producer feeds one thread-safe queue that the main thread renders from):
  - ProcessMonitor (own thread)            -> remote-access tool state
  - ScreenSampler (own thread) -> OCR worker -> screen_classifier
  - AudioCapture (own threads) -> ASR worker -> intent.detect over the rolling transcript

A status block refreshes every second: remote tool live? (tool, active_session), screen top label +
tactics + OCR ms, last call tactics + ASR ms, providers, and this process's CPU %. The preview score
line adds config.fusion weights without decay; the real fusion comes on Day 2. Ctrl+C (or --seconds)
prints a summary.

Privacy: only labels, scores, ids, counts and latencies are printed. Window titles and transcripts
appear only when privacy.debug_show_text is true. Nothing is written to disk.

Usage:  python scripts\\signals_console.py [--seconds 30] [--plain] [--ocr-backend native]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PREVIEW_NOTE = "preview - fusion comes on Day 2"


# ---------------------------------------------------------------- preview score (no decay)

def preview_score(remote_live: bool, screen_label: Any, call_tactics: set[str] | frozenset[str],
                  signals: Mapping[str, Mapping[str, Any]], threshold: float = 0.5) -> tuple[int, dict[str, int]]:
    """Sum of config.fusion.signals weights for what is currently present, each capped, total <= 100:
    remote_tool live, money screen (bank or upi_payment), otp_card, fake_alert label, screen tactics
    (per tactic), call tactics (per tactic), combo bonus when remote + money screen + any tactic."""
    def w(name: str, default: int, key: str = "weight") -> int:
        return int((signals.get(name) or {}).get(key, default))

    parts: dict[str, int] = {}
    labels = getattr(screen_label, "labels", {}) or {}
    screen_tactics = set(getattr(screen_label, "screen_tactics", ()) or ())
    money = labels.get("bank", 0) >= threshold or labels.get("upi_payment", 0) >= threshold
    if remote_live:
        parts["remote_tool"] = min(w("remote_tool", 30), w("remote_tool", 30, "cap"))
    if money:
        parts["money_screen"] = min(w("money_screen", 25), w("money_screen", 25, "cap"))
    if labels.get("otp_card", 0) >= threshold:
        parts["otp_card"] = min(w("otp_card", 20), w("otp_card", 20, "cap"))
    if labels.get("fake_alert", 0) >= threshold:
        parts["fake_alert_label"] = min(w("fake_alert_label", 20), w("fake_alert_label", 20, "cap"))
    if screen_tactics:
        parts["screen_tactics"] = min(w("screen_tactic", 15) * len(screen_tactics), w("screen_tactic", 30, "cap"))
    if call_tactics:
        parts["call_tactics"] = min(w("call_tactic", 10) * len(call_tactics), w("call_tactic", 40, "cap"))
    if remote_live and money and (screen_tactics or call_tactics):
        parts["combo_bonus"] = min(w("combo_bonus", 20), w("combo_bonus", 20, "cap"))
    return min(100, sum(parts.values())), parts


# ---------------------------------------------------------------- shared state

@dataclass
class State:
    started: float = field(default_factory=time.time)
    remote: list = field(default_factory=list)                 # RemoteToolState
    remote_live: bool = False
    remote_events: int = 0
    screen_label: Any = None
    screen_ts: float | None = None
    screen_window: tuple[str, str] = ("", "")                  # process, title (title shown only if allowed)
    ocr_ms: list[float] = field(default_factory=list)
    ocr_lines: int = 0
    ocr_provider: str = "-"
    ocr_dropped: int = 0
    call_intent: Any = None
    call_ts: float | None = None
    asr_ms: list[float] = field(default_factory=list)
    asr_audio_s: float = 0.0
    asr_provider: str = "-"
    asr_last: Any = None
    chunks: int = 0
    silent_chunks: int = 0
    asr_dropped: int = 0
    cpu: list[float] = field(default_factory=list)
    preview_max: int = 0
    errors: list[str] = field(default_factory=list)


def _latest_put(q: queue.Queue, item: Any) -> bool:
    """Keep only the newest item (drop the stale one). Returns False if something was dropped."""
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            return not dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                pass


def _p(ms: list[float], q: float) -> str:
    return f"{np.percentile(ms, q):.0f}" if ms else "-"


def _ago(ts: float | None) -> str:
    return f"{time.time() - ts:4.0f}s ago" if ts else "never"


# ---------------------------------------------------------------- rendering

def render(st: State, signals: Mapping, show: bool, cpu_now: float) -> list[str]:
    from kavach_privacy import redact_text, redact_title

    now = time.time()
    up = int(now - st.started)
    lines = [f"Kavach signals  {time.strftime('%H:%M:%S')}  up {up // 60:02d}:{up % 60:02d}"]
    live = [s for s in st.remote if s.has_user_process]
    idle = [s for s in st.remote if not s.has_user_process]
    if live:
        rt = ", ".join(f"{s.tool} (active_session={s.active_session})" for s in live)
        lines.append(f"Remote tool   : LIVE  {rt}")
    else:
        lines.append("Remote tool   : no" + (f"  (installed/idle: {', '.join(s.tool for s in idle)})" if idle else ""))

    lab = st.screen_label
    if lab is None:
        lines.append("Screen        : waiting for first OCR frame")
    else:
        scores = " ".join(f"{k}={v:.2f}" for k, v in sorted(lab.labels.items(), key=lambda kv: -kv[1])
                          if k != "normal" and v >= 0.1) or "-"
        proc, title = st.screen_window
        lines.append(f"Screen        : [{proc}] {redact_title(title, show)[:40]}  top={lab.top_label}  {scores}  "
                     f"tactics={','.join(sorted(lab.screen_tactics)) or '-'}")
        lines.append(f"                OCR {st.ocr_ms[-1]:.0f} ms ({st.ocr_lines} lines), p50 {_p(st.ocr_ms, 50)} ms, "
                     f"{len(st.ocr_ms)} frames, last {_ago(st.screen_ts)}")
    res = st.call_intent
    if res is None:
        lines.append(f"Call          : no speech yet  ({st.chunks} chunks, {st.silent_chunks} silent)")
    else:
        tac = ", ".join(f"{t} {h.score:.2f}" for t, h in res.tactics.items()) or "none"
        t = st.asr_last
        lines.append(f"Call          : tactics {res.distinct_tactics}: {tac}")
        lines.append(f"                ASR {t.total_ms:.0f} ms (p50 {_p(st.asr_ms, 50)}), last chunk {_ago(st.call_ts)}, "
                     f"{st.chunks} chunks ({st.silent_chunks} silent), "
                     + (f"text: {redact_text(t.text, show)[:60]}" if show else f"{t.words} words"))
    lines.append(f"Providers     : OCR {st.ocr_provider} | ASR {st.asr_provider}")
    lines.append(f"Process CPU   : {cpu_now:5.1f}% of all cores ({cpu_now * psutil.cpu_count():.0f}% of one core)")
    total, parts = preview_score(st.remote_live, lab, set(res.tactics) if res else set(), signals)
    st.preview_max = max(st.preview_max, total)
    detail = " + ".join(f"{k} {v}" for k, v in parts.items()) or "nothing present"
    lines.append(f"Preview score : {total:3d}/100  ({detail})  [{PREVIEW_NOTE}]")
    if st.errors:
        lines.append(f"Errors        : {st.errors[-1]}")
    return lines


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="All Kavach detectors together (no fusion yet)")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (default: until Ctrl+C)")
    p.add_argument("--plain", action="store_true", help="print blocks instead of a live refreshing panel")
    p.add_argument("--plain-every", type=float, default=1.0, help="with --plain: seconds between blocks")
    p.add_argument("--ocr-backend", choices=["rapidocr", "native"], help="override config ocr.backend")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from capture.audio import AudioCapture, AudioChunk
    from capture.processes import ProcessMonitor
    from capture.screen import ScreenFrame, ScreenSampler
    from detect.intent import detect as detect_intent
    from detect.screen_classifier import classify
    from kavach_config import get_config
    from kavach_privacy import show_text
    from models.asr import ASR
    from models.ocr import create_backend

    cfg = get_config().to_dict()
    if args.ocr_backend:
        cfg["ocr"]["backend"] = args.ocr_backend
    show = show_text()
    signals = cfg.get("fusion", {}).get("signals", {})
    print("loading models ...", flush=True)
    ocr = create_backend(cfg)
    asr = ASR(cfg)
    st = State(ocr_provider=f"{ocr.name}:{ocr.provider}", asr_provider=f"{asr.backend_name}:{asr.provider}")

    updates: queue.Queue[tuple[str, Any]] = queue.Queue()      # every thread -> main thread
    frames: queue.Queue[ScreenFrame] = queue.Queue(maxsize=1)   # newest screen frame only
    chunks: queue.Queue[AudioChunk] = queue.Queue(maxsize=3)    # ASR backlog, oldest dropped
    stop = threading.Event()

    def on_process_event(ev: Any) -> None:
        updates.put(("remote_event", ev.type))

    def on_frame(frame: ScreenFrame) -> None:
        # The sampler clears frame.image when this returns (frames never linger); keep our own
        # reference for the OCR worker. At most one frame waits in RAM (queue size 1).
        if not _latest_put(frames, dataclasses.replace(frame)):
            updates.put(("ocr_dropped", 1))

    def ocr_worker() -> None:
        while not stop.is_set():
            try:
                frame = frames.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                r = ocr(frame.image)
                label = classify(r, frame.window)
                updates.put(("screen", (label, r.total_ms, len(r.lines), frame.ts,
                                        (frame.window.process_name or "?", frame.window.title or ""))))
            except Exception as e:  # noqa: BLE001
                updates.put(("error", f"OCR: {type(e).__name__}: {e}"[:120]))

    def on_chunk(c: AudioChunk) -> None:
        if not c.is_speech:
            updates.put(("silent", 1))
            return
        if not _latest_put(chunks, c):
            updates.put(("asr_dropped", 1))

    def asr_worker() -> None:
        while not stop.is_set():
            try:
                c = chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                t = asr.transcribe(c)
                res = detect_intent(asr.rolling)
                updates.put(("call", (t, res, c.ts_end - c.ts_start)))
            except Exception as e:  # noqa: BLE001
                updates.put(("error", f"ASR: {type(e).__name__}: {e}"[:120]))

    monitor = ProcessMonitor(cfg, callback=on_process_event)
    sampler = ScreenSampler(cfg, on_frame=on_frame)
    audio = AudioCapture(cfg, on_chunk)
    workers = [threading.Thread(target=ocr_worker, name="ocr", daemon=True),
               threading.Thread(target=asr_worker, name="asr", daemon=True)]
    for t in workers:
        t.start()
    monitor.start()
    sampler.start()
    audio.start()
    proc = psutil.Process()
    proc.cpu_percent(None)

    def drain() -> None:
        while True:
            try:
                kind, val = updates.get_nowait()
            except queue.Empty:
                break
            if kind == "screen":
                st.screen_label, ms, st.ocr_lines, st.screen_ts, st.screen_window = val
                st.ocr_ms.append(ms)
            elif kind == "call":
                t, res, dur = val
                st.chunks += 1
                st.asr_last, st.call_intent, st.call_ts = t, res, time.time()
                st.asr_ms.append(t.total_ms)
                st.asr_audio_s += dur
            elif kind == "silent":
                st.chunks += 1
                st.silent_chunks += 1
            elif kind == "remote_event":
                st.remote_events += 1
            elif kind == "ocr_dropped":
                st.ocr_dropped += 1
            elif kind == "asr_dropped":
                st.asr_dropped += 1
            elif kind == "error":
                st.errors.append(val)
        st.remote = monitor.current()
        st.remote_live = monitor.is_remote_access_live()

    def block() -> list[str]:
        drain()
        cpu = proc.cpu_percent(None) / psutil.cpu_count()
        st.cpu.append(cpu)
        return render(st, signals, show, cpu)

    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        if args.plain:
            next_print = 0.0
            while deadline is None or time.monotonic() < deadline:
                time.sleep(1.0)
                lines = block()
                if time.monotonic() >= next_print:
                    print("\n".join(lines) + "\n", flush=True)
                    next_print = time.monotonic() + args.plain_every - 0.01
        else:
            from rich.live import Live
            from rich.panel import Panel
            from rich.text import Text
            with Live(refresh_per_second=4, transient=False) as live:
                while deadline is None or time.monotonic() < deadline:
                    live.update(Panel(Text("\n".join(block())), title="Kavach - Day 1 signals", expand=False))
                    time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        audio.stop()
        sampler.stop()
        monitor.stop()
        for t in workers:
            t.join(timeout=10)
        drain()

    s = sampler.stats()
    print("\n--- summary ---")
    print(f"runtime       : {time.time() - st.started:.0f} s, process CPU avg {np.mean(st.cpu) if st.cpu else 0:.1f}% "
          f"/ max {max(st.cpu) if st.cpu else 0:.1f}% of all cores")
    print(f"remote tools  : live at end={st.remote_live}, {st.remote_events} events")
    print(f"screen        : {len(st.ocr_ms)} frames OCR'd ({s['skipped']} unchanged skipped, {st.ocr_dropped} dropped "
          f"while busy), OCR p50 {_p(st.ocr_ms, 50)} / p95 {_p(st.ocr_ms, 95)} ms, last top label "
          f"{st.screen_label.top_label if st.screen_label else '-'}")
    rtf = sum(st.asr_ms) / 1000 / st.asr_audio_s if st.asr_audio_s else 0
    tac = ", ".join(f"{t} {h.score:.2f}" for t, h in st.call_intent.tactics.items()) if st.call_intent else "-"
    print(f"call          : {st.chunks} chunks ({st.silent_chunks} silent, {st.asr_dropped} dropped while busy), "
          f"ASR p50 {_p(st.asr_ms, 50)} / p95 {_p(st.asr_ms, 95)} ms, RTF {rtf:.3f}; final tactics: {tac or 'none'}")
    print(f"providers     : OCR {st.ocr_provider} | ASR {st.asr_provider}")
    print(f"preview score : max {st.preview_max}/100 [{PREVIEW_NOTE}]")
    if st.errors:
        print(f"errors        : {len(st.errors)} (last: {st.errors[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
