"""Live system audio -> ASR loop: AudioCapture chunks go to models.asr on a separate thread.

Per chunk prints: time, source, audio level, and for speech chunks the provider, total ASR latency and
the text (only if privacy.debug_show_text is true; otherwise the word count). Silent chunks are skipped
(never sent to ASR). Ctrl+C (or --seconds) prints p50/p95 latency and the real-time factor
(processing time / audio length). Nothing is written to disk.

Usage:  python scripts\\run_audio_asr.py [--seconds 30] [--provider cpu|dml|qnn] [--backend faster_whisper]
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.audio import AudioCapture, AudioChunk  # noqa: E402
from kavach_config import get_config  # noqa: E402
from kavach_privacy import show_text  # noqa: E402
from models.asr import ASR, Transcript  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="System audio -> ASR live loop")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (default: until Ctrl+C)")
    p.add_argument("--provider", help="override runtime.provider for the ASR sessions")
    p.add_argument("--backend", choices=["aihub_whisper", "faster_whisper"], help="override config asr.backend")
    p.add_argument("--mic", action="store_true", help="also capture the microphone (source=mic)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    cfg = get_config().to_dict()
    if args.backend:
        cfg.setdefault("asr", {})["backend"] = args.backend
    if args.mic:
        cfg["audio"]["capture_mic"] = True
    show = show_text()
    asr = ASR(cfg, provider=args.provider)
    print(f"ASR backend {asr.backend_name}, provider {asr.provider}; chunk {cfg['audio']['chunk_s']} s, "
          f"overlap {cfg['audio'].get('overlap_s', 1.0)} s; text {'shown' if show else 'hidden (word counts)'}",
          flush=True)

    jobs: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=8)
    results: list[Transcript] = []
    counts = {"silent": 0, "backlog_dropped": 0}

    def on_chunk(c: AudioChunk) -> None:                  # capture thread: never block it
        stamp = time.strftime("%H:%M:%S", time.localtime(c.ts_start))
        if not c.is_speech:
            counts["silent"] += 1
            print(f"{stamp} {c.source:<8} rms={c.rms:.4f} voiced={c.voiced_fraction:4.0%}  silent (not sent to ASR)",
                  flush=True)
            return
        try:
            jobs.put_nowait(c)
        except queue.Full:
            counts["backlog_dropped"] += 1

    def worker() -> None:
        while True:
            c = jobs.get()
            if c is None:
                return
            try:
                t = asr.transcribe(c)
            except Exception as e:  # noqa: BLE001
                print(f"ASR error: {type(e).__name__}: {e}", flush=True)
                continue
            if t is None:
                continue
            results.append(t)
            stamp = time.strftime("%H:%M:%S", time.localtime(t.ts_start))
            if t.dropped:
                body = f"[dropped: {t.dropped}]"
            else:
                body = repr(t.text) if show else f"<{t.words} words>"
            print(f"{stamp} {t.source:<8} rms={c.rms:.4f} {t.provider:<22} total {t.total_ms:6.0f} ms "
                  f"(enc {t.encoder_ms:4.0f}, dec {t.decoder_ms:5.0f}, {t.n_tokens:>2} tok)  "
                  f"RTF {t.total_ms / 1000 / (t.ts_end - t.ts_start):.3f}  {body}", flush=True)

    th = threading.Thread(target=worker, name="asr", daemon=True)
    th.start()
    cap = AudioCapture(cfg, on_chunk)
    cap.start()
    t0 = time.monotonic()
    try:
        while not args.seconds or time.monotonic() - t0 < args.seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        jobs.put(None)
        th.join(timeout=30)

    lat = np.array([t.total_ms for t in results])
    audio_s = sum(t.ts_end - t.ts_start for t in results)
    print("\n--- summary ---")
    print(f"chunks: {cap.counts['chunks']} total, {len(results)} transcribed, {counts['silent']} silent, "
          f"{sum(1 for t in results if t.dropped)} dropped by hallucination guard, "
          f"{counts['backlog_dropped']} dropped (ASR backlog)")
    if len(lat):
        print(f"ASR latency: p50 {np.percentile(lat, 50):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms, "
              f"max {lat.max():.0f} ms  ({asr.provider})")
        print(f"real-time factor: {lat.sum() / 1000 / audio_s:.3f}  ({lat.sum() / 1000:.1f} s processing "
              f"for {audio_s:.0f} s of speech chunks)")
        words = len(asr.rolling.text().split())
        print(f"rolling transcript: {words} words" + (f"\n  {asr.rolling.text()}" if show else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
