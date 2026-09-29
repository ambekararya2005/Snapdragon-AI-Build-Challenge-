"""Speak a demo script through the speakers with Windows SAPI TTS, so WASAPI loopback captures it.

One line at a time with a short pause between lines (like a caller's turns). Lines starting with '#'
are comments and are skipped. Nothing is written to disk; audio goes straight to the default output.

Usage:
    python scripts\\play_scenario.py                            # demo/scripts/hero_call.txt
    python scripts\\play_scenario.py --script demo/scripts/news_clip.txt --pause 0.4 --rate 1
    python scripts\\play_scenario.py --list-voices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = ROOT / "demo" / "scripts" / "hero_call.txt"


def script_lines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Speak a demo script via SAPI TTS (for loopback capture)")
    p.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    p.add_argument("--pause", type=float, default=0.6, help="seconds of silence between lines")
    p.add_argument("--rate", type=int, default=0, help="SAPI rate, -10 (slow) .. 10 (fast)")
    p.add_argument("--voice", help="substring of the voice name to use (see --list-voices)")
    p.add_argument("--delay", type=float, default=0.0, help="seconds to wait before the first line")
    p.add_argument("--show", action="store_true", help="print each line as it is spoken (it is our demo script)")
    p.add_argument("--list-voices", action="store_true")
    args = p.parse_args(argv)
    if sys.platform != "win32":
        print("SAPI TTS needs Windows")
        return 1
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    try:
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voices = [voice.GetVoices().Item(i) for i in range(voice.GetVoices().Count)]
        if args.list_voices:
            for v in voices:
                print(v.GetDescription())
            return 0
        if args.voice:
            match = [v for v in voices if args.voice.lower() in v.GetDescription().lower()]
            if not match:
                print(f"no voice matching {args.voice!r}; use --list-voices")
                return 1
            voice.Voice = match[0]
        voice.Rate = max(-10, min(10, args.rate))
        lines = script_lines(args.script)
        print(f"speaking {args.script.name}: {len(lines)} lines, voice {voice.Voice.GetDescription()}, "
              f"rate {voice.Rate}, pause {args.pause}s", flush=True)
        time.sleep(args.delay)
        t0 = time.monotonic()
        for i, line in enumerate(lines, 1):
            print(f"  [{i:>2}/{len(lines)}] {line if args.show else f'{len(line.split())} words'}", flush=True)
            voice.Speak(line)                       # synchronous: returns when the line has been played
            time.sleep(args.pause)
        print(f"done in {time.monotonic() - t0:.0f} s", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        pythoncom.CoUninitialize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
