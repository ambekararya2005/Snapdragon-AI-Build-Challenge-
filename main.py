"""Kavach entry point: pipeline (capture -> models -> detect -> fusion) + alert overlay + live dashboard.

    python main.py                       # everything; close the dashboard or Ctrl+C to quit
    python main.py --no-dashboard        # overlay only
    python main.py --plain               # headless: one status line per second, no windows
    python main.py --provider cpu        # override config runtime.provider (sets KAVACH_PROVIDER)
    python main.py --config other.yaml   # alternate config (sets KAVACH_CONFIG)
Demo helpers: --seconds N (quit after N s), --screenshot PATH (dashboard PNG 2 s before quitting),
--dashboard-topmost (keep the dashboard above the page being watched, without taking focus).

Tk runs on the main thread (Overlay.run owns the mainloop); pipeline stages run in their own threads.
Shutdown (window close, Ctrl+C, --seconds): stop the pipeline threads, resume anything suspended,
and flush queued incident-log events. Startup prints providers and model load times only.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python main.py", description="Kavach: offline, on-device scam shield")
    p.add_argument("--no-dashboard", action="store_true", help="overlay only, no dashboard window")
    p.add_argument("--plain", action="store_true", help="headless: status lines, no windows")
    p.add_argument("--provider", choices=["qnn", "cuda", "dml", "cpu"], help="override config runtime.provider")
    p.add_argument("--config", help="alternate config file")
    p.add_argument("--no-speak", action="store_true", help="no audio read-out of alerts")
    p.add_argument("--seconds", type=float, default=0, help="quit after N seconds (demo / tests)")
    p.add_argument("--screenshot", help="save a dashboard PNG 2 s before quitting (needs --seconds)")
    p.add_argument("--dashboard-topmost", action="store_true", help="keep the dashboard above other windows")
    return p.parse_args(argv)


def startup_report(pipe, t_start: float) -> list[str]:
    """Providers actually used + model load times (no titles, text or paths)."""
    from models import runtime

    h = pipe.health()
    lines = [f"started in {time.perf_counter() - t_start:.1f} s"]
    for stage, label in (("ocr", "OCR"), ("asr", "ASR")):
        ms = pipe.load_ms.get(stage)
        info = h[stage]["info"] or ("disabled" if not h[stage]["enabled"] else "not loaded")
        lines.append(f"  {label}: {info}" + (f", loaded in {ms / 1000:.1f} s" if ms is not None else ""))
    by_provider: dict[str, list[str]] = {}
    for st in runtime.all_stats():
        by_provider.setdefault(st.get("actual_provider", "?"), []).append(st["name"])
    for prov, names in sorted(by_provider.items()):
        lines.append(f"  {prov}: {len(names)} sessions ({', '.join(sorted(names))})")
    for name, st in h.items():
        if st["last_error"]:
            lines.append(f"  stage {name}: {st['last_error']}")
    if pipe.incident_log is not None:
        lines.append(f"  incident log: {pipe.incident_log.path.relative_to(ROOT) if pipe.incident_log.path.is_relative_to(ROOT) else pipe.incident_log.path}")
    return lines


def preflight() -> str | None:
    """Config errors, or a strict config (runtime.fallback_to_cpu: false) whose provider is missing here.
    Returns a message for a loud exit; None when Kavach may start."""
    from kavach_config import ConfigError, get_config

    try:
        cfg = get_config()
    except ConfigError as e:
        return f"config error: {e}"
    from models import runtime

    return runtime.provider_problem(cfg.runtime)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.config:
        os.environ["KAVACH_CONFIG"] = str(Path(args.config).resolve())
    if args.provider:
        os.environ["KAVACH_PROVIDER"] = args.provider
    if args.config or args.provider:
        from kavach_config import get_config

        get_config.cache_clear()                           # the overrides above must win over a cached config
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    problem = preflight()
    if problem:
        print(f"Kavach cannot start: {problem}", file=sys.stderr, flush=True)
        return 2

    if args.plain:
        from kavach.pipeline import _main as run_plain

        return run_plain(["--plain"] + (["--seconds", str(args.seconds)] if args.seconds else []))

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from act.overlay import Overlay, save_window_png
    from kavach.pipeline import Pipeline
    from kavach_config import get_config

    cfg = get_config()
    pipe = Pipeline(cfg)
    overlay = Overlay(cfg, speak=not args.no_speak)
    root = overlay.make_root()

    print(f"Kavach starting: provider={cfg.runtime.provider} (loading models) ...", flush=True)
    t0 = time.perf_counter()
    pipe.start()
    for line in startup_report(pipe, t0):
        print(line, flush=True)
    failed = pipe.strict_start_failures()
    if failed:
        pipe.stop()
        root.destroy()
        print("Kavach cannot start (runtime.fallback_to_cpu is false, so no stage may fall back or be skipped):\n  "
              + "\n  ".join(failed), file=sys.stderr, flush=True)
        return 2

    dash = None
    if not args.no_dashboard:
        from dashboard.app import BG as DASH_BG, CARD as DASH_CARD, Dashboard

        dash = Dashboard(root, pipe, overlay, cfg, topmost=args.dashboard_topmost)
        dash.on_close = root.quit                          # closing the dashboard quits Kavach

    def request_quit(*_: object) -> None:
        root.after(0, root.quit)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):         # remote_control chains its resume_all() onto these
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, request_quit)
    if args.seconds:
        root.after(int(args.seconds * 1000), root.quit)
        if args.screenshot and dash is not None:
            def grab() -> None:
                path = Path(args.screenshot)
                ok = save_window_png(dash.win, path, (DASH_BG, DASH_CARD), min_bg_fraction=0.5)
                print(f"screenshot: {path if ok else 'REFUSED (dashboard not fully visible)'}", flush=True)
            root.after(max(1000, int((args.seconds - 2) * 1000)), grab)

    print("running - close the dashboard or press Ctrl+C to quit", flush=True)
    t_run = time.time()
    try:
        overlay.run(pipe, start_pipeline=False)            # Tk mainloop on this (main) thread
    finally:
        print("shutting down ...", flush=True)
        if dash is not None:
            dash.close()
        pipe.stop()                                        # threads joined, queued incident events flushed
        if overlay.remote is not None:
            resumed = overlay.remote.resume_all()
            if resumed:
                print(f"  resumed {len(resumed)} suspended process(es)")
        state = pipe.state
        written = pipe.incident_log.written if pipe.incident_log is not None else 0
        print(f"  ran {time.time() - t_run:.0f} s; last score {state.score if state else '-'} "
              f"({state.band if state else '-'}); incident log lines written: {written}")
        if pipe.sampler is not None:
            st = pipe.sampler.stats()
            last = pipe.last_screen[0] if pipe.last_screen else None
            present = ", ".join(last.present) if last and last.present else "none"
            mask = pipe.last_mask
            pct = f"{100 * mask[0] / mask[1]:.0f}%" if mask and mask[1] else "0%"
            print(f"  screen: {st['frames']} frames, {st['masked_frames']} with Kavach windows masked; last OCR'd "
                  f"frame {pct} masked -> top label {last.top_label if last else '-'} (labels >= 0.5: {present})")
        if dash is not None:
            n = len(dash.refresh_ms)
            avg = sum(dash.refresh_ms) / n if n else 0
            print(f"  dashboard: refresh avg {avg:.1f} ms, max network connections seen {dash.net_max}")
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
