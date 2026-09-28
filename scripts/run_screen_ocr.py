"""Live screen -> OCR loop: ScreenSampler feeds changed frames to the OCR backend.

Per OCR'd frame prints: time, process, window title, line count, OCR latency. The title and the
first 200 characters of text are shown only if privacy.debug_show_text is true. Ctrl+C (or
--seconds) prints OCR p50/p95 latency and the frame skip rate. Nothing is written to disk.

Usage:  python scripts\\run_screen_ocr.py [--seconds 20]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.screen import ScreenFrame, ScreenSampler  # noqa: E402
from kavach_config import get_config  # noqa: E402
from kavach_privacy import redact_title, show_text  # noqa: E402
from models.ocr import create_backend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Screen -> OCR live loop")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (default: until Ctrl+C)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    cfg = get_config()
    show = show_text()
    t0 = time.perf_counter()
    ocr = create_backend(cfg)
    print(f"OCR: {ocr.name} on {ocr.provider} (init {(time.perf_counter() - t0) * 1000:.0f} ms); "
          f"debug_show_text={show}; sampling every {cfg.screen.interval_s}s")

    def on_frame(frame: ScreenFrame) -> None:
        result = ocr(frame.image)
        w = frame.window
        ts = time.strftime("%H:%M:%S", time.localtime(frame.ts))
        print(f"{ts}  [{w.process_name}] {redact_title(w.title, show)[:60]!s:<60}  "
              f"{frame.image.shape[1]}x{frame.image.shape[0]}  lines={len(result.lines):<3} "
              f"ocr={result.total_ms:6.0f} ms (det {result.det_ms:.0f} / rec {result.rec_ms:.0f})  "
              f"capture={frame.capture_ms:.0f} ms", flush=True)
        if show and result.full_text:
            print("          text: " + result.full_text[:200].replace("\n", " | "), flush=True)

    sampler = ScreenSampler(cfg, on_frame=on_frame)
    sampler.start()
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
        o, s = ocr.stats(), sampler.stats()
        print()
        print(f"OCR:    n={o['n']}  p50={o['p50_ms']} ms  p95={o['p95_ms']} ms  provider={o['actual_provider']}")
        print(f"frames: {s['frames']} sampled, {s['skipped']} skipped (skip rate {s['skip_rate']:.0%}), "
              f"{s['no_window']} with no capturable window, last capture {s['last_capture_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
