import sys

import numpy as np
import pytest

from capture import screen as S

VIRTUAL = {"left": 0, "top": 0, "width": 1920, "height": 1080}


def frame(value=0, h=720, w=1280):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------- change detector

def test_first_frame_is_changed_and_identical_is_skipped():
    d = S.ChangeDetector(threshold=4.0)
    assert d.update(frame(100), "A", 1)[0] is True
    changed, diff = d.update(frame(100), "A", 1)
    assert changed is False and diff == 0.0
    assert d.frames == 2 and d.skipped == 1 and d.skip_rate == 0.5


def test_threshold():
    d = S.ChangeDetector(threshold=4.0)
    d.update(frame(100), "A", 1)
    assert d.update(frame(103), "A", 1) == (False, 3.0)       # below threshold
    assert d.update(frame(110), "A", 1) == (True, 10.0)       # above threshold


def test_compares_against_last_sent_frame():
    d = S.ChangeDetector(threshold=4.0)
    d.update(frame(100), "A", 1)
    assert not d.update(frame(103), "A", 1)[0]
    changed, diff = d.update(frame(106), "A", 1)              # +3 per step, +6 since last sent
    assert changed and diff == 6.0


def test_local_change_region():
    d = S.ChangeDetector(threshold=4.0)
    img = frame(0)
    d.update(img, "A", 1)
    img2 = img.copy()
    img2[:360, :640] = 255                                    # a quarter of the window changes
    changed, diff = d.update(img2, "A", 1)
    assert changed and diff == pytest.approx(255 / 4, rel=0.02)


def test_title_or_window_change_forces():
    d = S.ChangeDetector(threshold=4.0, force_on_title_change=True)
    d.update(frame(100), "Inbox", 1)
    assert d.update(frame(100), "Bank login", 1)[0] is True
    assert d.update(frame(100), "Bank login", 2)[0] is True   # other window, same title
    assert d.update(frame(100), "Bank login", 2)[0] is False

    d = S.ChangeDetector(threshold=4.0, force_on_title_change=False)
    d.update(frame(100), "Inbox", 1)
    assert d.update(frame(100), "Bank login", 1)[0] is False


def test_thumbnail_shape_and_grayscale_input():
    assert S.thumbnail(frame(5)).shape == (36, 64)
    assert S.thumbnail(np.zeros((100, 200), np.uint8)).shape == (36, 64)


# ---------------------------------------------------------------- rect clamping / downscale

@pytest.mark.parametrize("rect,bounds,expected", [
    ((100, 100, 500, 400), VIRTUAL, (100, 100, 500, 400)),                    # inside
    ((-8, -8, 1928, 1088), VIRTUAL, (0, 0, 1920, 1080)),                      # maximized overhang
    ((1800, 1000, 2200, 1300), VIRTUAL, (1800, 1000, 1920, 1080)),            # partly off-screen
    ((2000, 0, 2400, 300), VIRTUAL, None),                                    # fully off-screen
    ((100, 100, 100, 400), VIRTUAL, None),                                    # zero width
    ((-1500, 200, -100, 900), {"left": -1920, "top": 0, "width": 3840, "height": 1080},
     (-1500, 200, -100, 900)),                                                # monitor left of primary
    ((-2000, -50, -1800, 100), {"left": -1920, "top": 0, "width": 3840, "height": 1080},
     (-1920, 0, -1800, 100)),
])
def test_clamp_rect(rect, bounds, expected):
    assert S.clamp_rect(rect, bounds) == expected


def test_downscale():
    assert S.downscale(frame(0, 2160, 3840), 1920).shape == (1080, 1920, 3)
    assert S.downscale(frame(0, 3000, 1000), 1920).shape == (1920, 640, 3)   # portrait
    small = frame(0, 500, 800)
    assert S.downscale(small, 1920) is small
    assert S.downscale(small, 0) is small


# ---------------------------------------------------------------- sampler (fake window + grab)

def make_sampler(images, titles=None):
    win = [S.WindowInfo(1, "A", 10, "app.exe", (0, 0, 1280, 720))]
    it = iter(images)

    def window_fn():
        return win[0]

    def grab_fn(window, max_side):
        return next(it)

    cfg = {"screen": {"interval_s": 0.01, "change_threshold": 4.0, "force_on_title_change": True,
                      "exclude_title_prefixes": ["Kavach"], "max_side": 1920}}
    got = []
    sampler = S.ScreenSampler(cfg, on_frame=lambda f: got.append((f.changed, f.image.shape)),
                              window_fn=window_fn, grab_fn=grab_fn)
    return sampler, got, win


def test_sampler_only_sends_changed_frames():
    sampler, got, _ = make_sampler([frame(100), frame(100), frame(101), frame(200)])
    metas = [sampler.sample_once() for _ in range(4)]
    assert [m.changed for m in metas] == [True, False, False, True]
    assert all(m.image is None for m in metas)
    assert got == [(True, (720, 1280, 3)), (True, (720, 1280, 3))]
    st = sampler.stats()
    assert st["frames"] == 4 and st["skipped"] == 2 and st["skip_rate"] == 0.5
    assert st["last_capture_ms"] is not None


def test_sampler_no_window_and_bad_callback():
    sampler, got, win = make_sampler([frame(1)])
    win[0] = None
    assert sampler.sample_once() is None
    assert sampler.stats()["no_window"] == 1

    sampler, _, _ = make_sampler([frame(1)])
    sampler.on_frame = lambda f: 1 / 0
    assert sampler.sample_once().changed is True               # callback error does not break sampling


def test_sampler_thread_start_stop():
    import threading
    seen = threading.Event()
    sampler, _, _ = make_sampler(frame((i * 10) % 256) for i in range(1000))   # lazy: frames built on demand
    sampler.on_frame = lambda f: seen.set()
    sampler.start()
    assert seen.wait(2)
    sampler.stop()


# ---------------------------------------------------------------- Windows API

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows API")


@windows_only
def test_get_active_window_real():
    w = S.get_active_window([])
    assert w is None or (w.size[0] > 0 and w.size[1] > 0)


@windows_only
def test_get_active_window_filters(monkeypatch):
    g = S.win32gui
    monkeypatch.setattr(g, "GetForegroundWindow", lambda: 1234)
    monkeypatch.setattr(g, "GetClassName", lambda h: "Notepad")
    monkeypatch.setattr(g, "IsIconic", lambda h: False)
    monkeypatch.setattr(g, "GetWindowText", lambda h: "Kavach warning")
    monkeypatch.setattr(g, "GetWindowRect", lambda h: (0, 0, 800, 600))
    monkeypatch.setattr(S, "_dwm_frame_rect", lambda h: None)
    monkeypatch.setattr(S.win32process, "GetWindowThreadProcessId", lambda h: (1, 999999999))
    assert S.get_active_window(["Kavach"]) is None               # excluded title

    monkeypatch.setattr(g, "GetWindowText", lambda h: "Notepad")
    w = S.get_active_window(["Kavach"])
    assert w.title == "Notepad" and w.rect == (0, 0, 800, 600) and w.process_name is None  # pid gone

    monkeypatch.setattr(g, "IsIconic", lambda h: True)
    assert S.get_active_window(["Kavach"]) is None               # minimized

    monkeypatch.setattr(g, "IsIconic", lambda h: False)
    monkeypatch.setattr(g, "GetWindowRect", lambda h: (10, 10, 10, 600))
    assert S.get_active_window(["Kavach"]) is None               # zero size

    monkeypatch.setattr(g, "GetForegroundWindow", lambda: 0)
    assert S.get_active_window(["Kavach"]) is None               # no window


@windows_only
@pytest.mark.parametrize("cls", ["Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW"])
def test_get_active_window_skips_shell_classes(monkeypatch, cls):
    g = S.win32gui
    monkeypatch.setattr(g, "GetForegroundWindow", lambda: 1234)
    monkeypatch.setattr(g, "IsIconic", lambda h: False)
    monkeypatch.setattr(g, "GetClassName", lambda h: cls)
    monkeypatch.setattr(g, "GetWindowText", lambda h: "")
    monkeypatch.setattr(g, "GetWindowRect", lambda h: (0, 1020, 1920, 1080))
    monkeypatch.setattr(S, "_dwm_frame_rect", lambda h: None)
    monkeypatch.setattr(S.win32process, "GetWindowThreadProcessId", lambda h: (1, 999999999))
    assert S.get_active_window([], list(S.DEFAULT_EXCLUDE_CLASSES)) is None
    assert S.get_active_window([], []) is not None               # same window, no class filter


@windows_only
def test_get_active_window_skips_own_process(monkeypatch):
    import os
    g = S.win32gui
    monkeypatch.setattr(g, "GetForegroundWindow", lambda: 1234)
    monkeypatch.setattr(g, "IsIconic", lambda h: False)
    monkeypatch.setattr(g, "GetClassName", lambda h: "TkTopLevel")
    monkeypatch.setattr(g, "GetWindowText", lambda h: "Some overlay")
    monkeypatch.setattr(g, "GetWindowRect", lambda h: (0, 0, 800, 600))
    monkeypatch.setattr(S, "_dwm_frame_rect", lambda h: None)
    monkeypatch.setattr(S.win32process, "GetWindowThreadProcessId", lambda h: (1, os.getpid()))
    assert S.get_active_window([], []) is None


# ---------------------------------------------------------------- browser tab-strip crop

CROPS = {"chrome.exe": 41, "msedge.exe": 40}


def win(process, dpi=96, h=1020):
    return S.WindowInfo(1, "t", 10, process, (0, 0, 1920, h), dpi)


@pytest.mark.parametrize("process,dpi,expected", [
    ("chrome.exe", 96, 41),
    ("chrome.exe", 120, 51),          # measured value at 125%
    ("Chrome.EXE", 144, 62),          # case-insensitive, 150%
    ("msedge.exe", 120, 50),          # measured value at 125%
    ("notepad.exe", 120, 0),
    (None, 120, 0),
])
def test_top_crop_px(process, dpi, expected):
    assert S.top_crop_px(win(process, dpi), CROPS) == expected


def test_top_crop_px_never_more_than_half():
    assert S.top_crop_px(win("chrome.exe", 96, h=60), CROPS) == 30
    assert S.top_crop_px(win("chrome.exe"), None) == 0


def test_crop_rect_top():
    assert S.crop_rect_top((0, 0, 1920, 1020), 51) == (0, 51, 1920, 1020)
    assert S.crop_rect_top((0, 0, 100, 10), 50) == (0, 10, 100, 10)       # never past the bottom


def test_sampler_crops_browser_but_keeps_window_info():
    grabbed = []
    window = S.WindowInfo(1, "DEMO – NOT A REAL BANK - Google Chrome", 10, "chrome.exe", (0, 0, 1920, 1020), 120)
    cfg = {"screen": {"interval_s": 1, "change_threshold": 4.0, "max_side": 1920,
                      "browser_top_crop_px": CROPS, "exclude_window_classes": []}}
    frames = []

    def grab_fn(w, max_side):
        grabbed.append(w.rect)
        return frame(0, w.size[1], w.size[0])

    sampler = S.ScreenSampler(cfg, on_frame=lambda f: frames.append((f.image.shape, f.crop_top, f.window.rect)),
                              window_fn=lambda: window, grab_fn=grab_fn)
    meta = sampler.sample_once()
    assert grabbed == [(0, 51, 1920, 1020)]
    assert frames == [((969, 1920, 3), 51, (0, 0, 1920, 1020))]           # title/rect of the full window
    assert meta.crop_top == 51


@windows_only
def test_grab_real_screen():
    w = S.WindowInfo(0, "", None, None, (-50, -50, 400, 300))   # partly off-screen
    img = S.grab(w, max_side=200)
    assert img is not None and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert max(img.shape[:2]) <= 200
    S.close_thread_mss()
