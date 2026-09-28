"""Measure OCR text-line widths to choose static rec widths (ocr.rec_buckets).

Captures the active window (browser tab strip cropped like ScreenSampler does), runs OCR, and
prints the distribution of line-crop widths after resizing to height 48 (what the rec model sees).
Prints only numbers, never the text. Nothing is written to disk.

Usage:
    python scripts\\measure_rec_widths.py                       # 3 s countdown, then the active window
    python scripts\\measure_rec_widths.py --activate "NOT A REAL BANK" --process chrome.exe
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.screen import crop_rect_top, get_active_window, grab, top_crop_px  # noqa: E402
from kavach_config import get_config  # noqa: E402
from models.ocr import create_backend  # noqa: E402

REC_H = 48


def line_widths(boxes: list, rec_h: int = REC_H) -> np.ndarray:
    """Width each crop gets when resized to rec_h, keeping aspect (same crop geometry as rapidocr)."""
    out = []
    for box in boxes:
        p = np.asarray(box, dtype=np.float32)
        w = max(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[2] - p[3]))
        h = max(np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[1] - p[2]))
        if h > 0:
            out.append(rec_h * w / h)
    return np.asarray(out, dtype=np.float64)


def pick_width(widths: np.ndarray, pct: float = 95, multiple: int = 32, lo: int = 320) -> int:
    """p95 width rounded up to a multiple of 32, at least `lo`. No upper cap: report what the data says."""
    if widths.size == 0:
        return lo
    w = int(math.ceil(np.percentile(widths, pct) / multiple) * multiple)
    return max(lo, w)


def _activate(title_sub: str, process: str | None = None) -> bool:
    import psutil
    import win32api
    import win32con
    import win32gui
    import win32process

    def owner(h: int) -> str:
        try:
            return psutil.Process(win32process.GetWindowThreadProcessId(h)[1]).name().lower()
        except (psutil.Error, OSError):
            return ""

    hits = []
    win32gui.EnumWindows(lambda h, _: hits.append(h) if win32gui.IsWindowVisible(h)
                         and title_sub in win32gui.GetWindowText(h)
                         and (not process or owner(h) == process.lower()) else None, None)
    if not hits:
        return False
    win32api.keybd_event(0x12, 0, 0, 0)  # Alt: lets SetForegroundWindow succeed
    win32gui.ShowWindow(hits[0], win32con.SW_SHOWMAXIMIZED)
    win32gui.SetForegroundWindow(hits[0])
    win32api.keybd_event(0x12, 0, 2, 0)
    time.sleep(1.5)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure rec line widths on the active window")
    p.add_argument("--activate", help="bring the first window whose title contains this to the front")
    p.add_argument("--process", help="with --activate: only windows of this exe, e.g. chrome.exe")
    args = p.parse_args(argv)
    cfg = get_config()

    if args.activate:
        ok = _activate(args.activate, args.process)
        print(f"activated: {ok}")
        if not ok:
            return 1
    else:
        for i in (3, 2, 1):
            print(f"capturing in {i}...", flush=True)
            time.sleep(1)
    window = get_active_window()
    if window is None:
        print("no capturable window")
        return 1
    crop = top_crop_px(window, cfg.screen.get("browser_top_crop_px"))
    target = dataclasses.replace(window, rect=crop_rect_top(window.rect, crop)) if crop else window
    image = grab(target, int(cfg.screen.max_side))
    ocr = create_backend(cfg)
    result = ocr(image)
    widths = line_widths([l.box for l in result.lines])
    del image

    print(f"window: {window.process_name}  dpi={window.dpi}  top crop={crop}px  lines={len(result.lines)}")
    if widths.size == 0:
        print("no text lines found")
        return 1
    pct = {q: np.percentile(widths, q) for q in (50, 75, 90, 95, 99)}
    print(f"rec width @h={REC_H}:  min={widths.min():.0f}  p50={pct[50]:.0f}  p75={pct[75]:.0f}  "
          f"p90={pct[90]:.0f}  p95={pct[95]:.0f}  p99={pct[99]:.0f}  max={widths.max():.0f}")
    edges = [0, 160, 320, 480, 640, 800, 960, 1280, 10_000]
    hist, _ = np.histogram(widths, bins=edges)
    for lo, hi, n in zip(edges, edges[1:], hist):
        label = f"{lo}-{hi}" if hi < 10_000 else f">{lo}"
        print(f"  {label:>9}: {n:3d} {'#' * int(n)}")
    for w in (320, 480, 640, 960):
        print(f"  lines wider than {w}: {int((widths > w).sum())}/{widths.size} ({(widths > w).mean():.0%})")
    print(f"p95 width rounded up to a multiple of 32: {pick_width(widths)}")
    buckets = [int(w) for w in cfg.ocr.get("rec_buckets", [])]
    if buckets:
        left = np.ones(widths.size, dtype=bool)
        for b in buckets:
            fits = left & (widths <= b)
            print(f"  bucket {b}: {int(fits.sum())} lines")
            left &= ~fits
        print(f"  squashed to {buckets[-1]}: {int(left.sum())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
