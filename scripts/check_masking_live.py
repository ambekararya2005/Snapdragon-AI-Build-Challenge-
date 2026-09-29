"""Live check for masking Kavach's own windows in screen capture.

Opens the mock bank page, puts a real Kavach dashboard (fake data, full text) topmost over the left
half of it, then samples the page through the real sampler -> OCR -> screen classifier twice: with
masking (normal) and without (to show what Kavach would read otherwise). Finally a recorder-style
screen grab of the dashboard area shows that screen recorders (OBS / Game Bar) still see it.

Prints labels, ids and percentages only; nothing is written to disk. Don't touch the mouse or
keyboard for ~5 s while it runs (it needs the bank page to stay the active window).

Usage:  python scripts\\check_masking_live.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGE = ROOT / "demo" / "test_pages" / "mock_bank_transfer.html"
PAGE_TITLE = "NOT A REAL BANK"


class FakePipeline:
    """Enough of kavach.pipeline.Pipeline for the dashboard to draw realistic text."""

    def __init__(self, cfg):
        self.cfg = cfg

    def subscribe(self, on_state=None, on_event=None):
        pass

    def snapshot(self):
        t = time.time()
        label = SimpleNamespace(labels={"bank": 1.0, "otp_card": 0.8, "normal": 0.0}, top_label="bank",
                                screen_tactics=frozenset(), news_context=False,
                                evidence=["bank:netbanking:ifsc", "pattern:ifsc", "bank:netbanking:transfer_funds"])
        return {"state": SimpleNamespace(ts=t, score=45, band="quiet"), "remote": [], "screen": (label, t - 2),
                "call": None, "chunk": None, "chunk_counts": {}, "paused_until": None, "mask": None}


def main() -> int:
    import mss
    import win32gui

    from act.overlay import Overlay, colour_fraction
    from capture.screen import ScreenSampler, get_active_window, kavach_window_rects
    from dashboard.app import BG, CARD, Dashboard
    from detect.screen_classifier import classify
    from kavach_config import get_config
    from models.ocr import create_backend

    cfg = get_config().to_dict()
    scfg = cfg["screen"]
    ocr = create_backend(cfg)
    ov = Overlay(cfg, speak=False)
    root = ov.make_root()

    os.startfile(PAGE)                                                   # default browser, comes to front
    time.sleep(2.0)
    w = get_active_window(scfg["exclude_title_prefixes"], scfg["exclude_window_classes"])
    if not w or PAGE_TITLE not in (w.title or ""):
        print("the bank page is not the active window - nothing measured; run again without touching the PC")
        return 2
    l, t, r, b = w.rect
    dash = Dashboard(root, FakePipeline(cfg), ov, cfg, topmost=True, input_guard=False)
    dash.win.minsize(300, 200)
    dash.win.geometry(f"{(r - l) // 2}x{b - t - 60}+{l}+{t + 60}")    # left half of the page
    for _ in range(10):
        root.update()
        time.sleep(0.1)
    win32gui.SetForegroundWindow(w.hwnd)                                 # the page stays the watched window
    for _ in range(5):
        root.update()
        time.sleep(0.1)
    dx, dy, dw, dh = dash.win.winfo_rootx(), dash.win.winfo_rooty(), dash.win.winfo_width(), dash.win.winfo_height()
    print(f"dashboard {dw}x{dh} on top of the {r - l}x{b - t} bank page")

    def sample(mask: bool):
        got = []
        s = ScreenSampler(cfg, on_frame=lambda f: got.append((ocr(f.image), f.masked_px, f.image.shape[0] * f.image.shape[1])),
                          mask_rects_fn=kavach_window_rects if mask else None)
        s.sample_once()
        if not got:
            return None
        res, masked, px = got[0]
        lab = classify(res, get_active_window(scfg["exclude_title_prefixes"], scfg["exclude_window_classes"]),
                       cfg["screen_classifier"])
        present = {k: round(v, 2) for k, v in lab.labels.items() if k != "normal" and v >= 0.5}
        return lab, present, masked / px, len(res.lines)

    ok = True
    m = sample(mask=True)
    if m is None:
        print("no frame (the page lost focus) - run again without touching the PC")
        return 2
    lab, present, frac, n = m
    print(f"masked  : {frac:.0%} of the frame masked, {n} OCR lines -> top={lab.top_label} labels>=0.5={present}")
    ok &= lab.top_label in ("bank", "otp_card") and "bank" in present and frac > 0.2
    u = sample(mask=False)
    if u is not None:
        lab2, present2, _, n2 = u
        extra = sorted(set(lab2.evidence) - set(lab.evidence))
        print(f"unmasked: {n2} OCR lines (incl. the dashboard's own text) -> top={lab2.top_label} labels>=0.5={present2}; "
              f"{len(extra)} evidence ids come only from the dashboard text: {extra[:6]}")
    with mss.MSS() as sct:
        rgb = sct.grab({"left": dx, "top": dy, "width": dw, "height": dh}).rgb
    seen = colour_fraction(rgb, [BG, CARD])
    print(f"recorder: screen grab of the dashboard area shows {seen:.0%} dashboard colours "
          f"(0% when display affinity hid it)")
    ok &= seen > 0.5
    dash.close()
    root.destroy()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
