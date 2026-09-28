"""Compare the rapidocr and native OCR backends on the same in-memory frames.

Frames: the synthetic "Enter OTP 482913" image, and a capture of demo/test_pages/mock_bank_transfer.html
(brought to the front in Chrome, tab strip cropped exactly like ScreenSampler does). Nothing is saved.

For each provider (cpu, dml) prints per backend: lines found, det/rec/total ms (median of --runs after
a warm-up), native rec bucket counts, and the character similarity of the two full texts after
fold_confusables (rapidocr = reference). Text itself is shown only if privacy.debug_show_text.

Usage:  python scripts\\compare_ocr.py [--providers cpu dml] [--runs 3] [--browser chrome.exe]
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kavach_config  # noqa: E402
from capture.screen import crop_rect_top, get_active_window, grab, top_crop_px  # noqa: E402
from kavach_privacy import redact_text, show_text  # noqa: E402
from models import runtime  # noqa: E402
from models.ocr import create_backend, fold_confusables, synthetic_image  # noqa: E402
from scripts.measure_rec_widths import _activate  # noqa: E402


def similarity(a: str, b: str) -> float:
    fa, fb = fold_confusables(a.replace("\n", " ")), fold_confusables(b.replace("\n", " "))
    if not fa and not fb:
        return 1.0
    return difflib.SequenceMatcher(None, fa, fb, autojunk=False).ratio()


def line_diff(ref, other) -> tuple[int, int]:
    """(# folded lines only in ref, # only in other), multiset comparison."""
    from collections import Counter
    a = Counter(fold_confusables(l.text) for l in ref.lines)
    b = Counter(fold_confusables(l.text) for l in other.lines)
    return sum((a - b).values()), sum((b - a).values())


def capture_bank(browser: str, attempts: int = 3) -> np.ndarray | None:
    """Bring the bank page to the front and verify it is really the foreground window before grabbing."""
    window = None
    for _ in range(attempts):
        if not _activate("NOT A REAL BANK", browser):
            print(f"could not find a {browser} window titled '...NOT A REAL BANK...'; open "
                  "demo/test_pages/mock_bank_transfer.html first")
            return None
        w = get_active_window()
        if w and (w.process_name or "").lower() == browser.lower() and "NOT A REAL BANK" in w.title:
            window = w
            break
    if window is None:
        print(f"bank page did not stay in the foreground (another window took focus); not capturing")
        return None
    cfg = kavach_config.get_config()
    crop = top_crop_px(window, cfg.screen.get("browser_top_crop_px"))
    target = dataclasses.replace(window, rect=crop_rect_top(window.rect, crop)) if crop else window
    image = grab(target, int(cfg.screen.max_side))
    print(f"bank frame: {window.process_name} dpi={window.dpi} crop={crop}px -> {image.shape[1]}x{image.shape[0]}")
    return image


def run_backend(backend, image: np.ndarray, runs: int):
    backend(image)  # warm-up (DML compiles per shape)
    results = [backend(image) for _ in range(runs)]
    med = {k: float(np.median([getattr(r, k) for r in results])) for k in ("det_ms", "rec_ms", "total_ms")}
    return results[-1], med


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="rapidocr vs native OCR")
    p.add_argument("--providers", nargs="+", default=["cpu", "dml"])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--browser", default="chrome.exe")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    show = show_text()

    frames = {"synthetic OTP": synthetic_image()}
    bank = capture_bank(args.browser)
    if bank is not None:
        frames["bank.html"] = bank

    summary = []
    for prov in args.providers:
        cfg = kavach_config.load_config(env={"KAVACH_PROVIDER": prov})
        backends = {}
        for name in ("rapidocr", "native"):
            cfg.ocr["backend"] = name
            backends[name] = create_backend(cfg)
        print(f"\n=== provider {prov}:  rapidocr on {backends['rapidocr'].provider}  |  "
              f"native on {backends['native'].provider}")
        for fname, image in frames.items():
            res = {n: run_backend(b, image, args.runs) for n, b in backends.items()}
            (ra, ma), (na, mn) = res["rapidocr"], res["native"]
            sim = similarity(ra.full_text, na.full_text)
            only_r, only_n = line_diff(ra, na)
            print(f"  [{fname}]  similarity {sim:.3f}   lines rapidocr={len(ra.lines)} native={len(na.lines)}  "
                  f"(differing lines: {only_r} only in rapidocr, {only_n} only in native)")
            for n, (r, m) in (("rapidocr", res["rapidocr"]), ("native", res["native"])):
                extra = f"  buckets {r.extra['buckets']}" if "buckets" in r.extra else ""
                print(f"    {n:<8} det {m['det_ms']:6.0f} ms  rec {m['rec_ms']:6.0f} ms  total {m['total_ms']:6.0f} ms{extra}")
            if show and sim < 1.0:
                print("    rapidocr: " + redact_text(ra.full_text, show)[:300].replace("\n", " | "))
                print("    native:   " + redact_text(na.full_text, show)[:300].replace("\n", " | "))
            summary.append((prov, fname, sim))
        runtime.clear_registry()

    print("\nsimilarity (fold_confusables, rapidocr = reference):")
    for prov, fname, sim in summary:
        print(f"  {prov:<4} {fname:<14} {sim:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
