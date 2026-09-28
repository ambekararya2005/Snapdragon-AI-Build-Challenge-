"""Active-window screen capture with change detection (mss + pywin32). Frames stay in memory.

    sampler = ScreenSampler(get_config(), on_frame=ocr_frame)   # on_frame(ScreenFrame) for changed frames only
    sampler.start()
    sampler.stats()                                             # frames, skipped, skip_rate, last_capture_ms

Screenshots are never written to disk; ScreenSampler drops its image reference after the callback.
Window titles are screen text: logs never contain them, and CLI output shows "<title hidden>"
unless privacy.debug_show_text is true (see kavach_privacy).

Self-test / CLI:
    python -m capture.screen --once            # 3 s countdown, then capture the active window
    python -m capture.screen --watch [--preview]
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np

IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    from ctypes import wintypes

    import psutil
    import win32gui
    import win32process

log = logging.getLogger("capture.screen")

THUMB_W, THUMB_H = 64, 36
DEFAULT_EXCLUDE_CLASSES = ("Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW")  # taskbars, desktop
DWMWA_EXTENDED_FRAME_BOUNDS = 9
Rect = tuple[int, int, int, int]  # left, top, right, bottom (physical pixels)


def _set_dpi_awareness() -> str:
    """Per-monitor DPI aware, so window rects match physical pixels on 125%/150% scaled screens."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return "per-monitor"
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "system"
    except (AttributeError, OSError):
        return "unaware"


DPI_AWARENESS = _set_dpi_awareness() if IS_WINDOWS else "n/a"


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    pid: int | None
    process_name: str | None
    rect: Rect
    dpi: int = 96                      # window DPI (96 = 100% scaling)
    class_name: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]


@dataclass
class ScreenFrame:
    ts: float
    window: WindowInfo                 # full window; the image may have its top cropped (crop_top)
    image: np.ndarray | None           # BGR uint8; None in on_sample metadata callbacks
    changed: bool
    capture_ms: float
    diff: float = 0.0                  # mean abs diff of the 64x36 thumbnail vs the last sent frame
    crop_top: int = 0                  # physical px removed from the top (browser tab strip)


# ---------------------------------------------------------------- pure helpers (unit-tested)

def top_crop_px(window: WindowInfo, crop_table: Mapping[str, int] | None) -> int:
    """Browser tab-strip height in physical px: config value (at 96 DPI) scaled by the window DPI.

    Never removes more than half the window, so a tiny/odd window is still captured.
    """
    if not crop_table or not window.process_name:
        return 0
    base = {k.lower(): v for k, v in crop_table.items()}.get(window.process_name.lower())
    if not base:
        return 0
    px = round(float(base) * (window.dpi or 96) / 96)
    return max(0, min(px, window.size[1] // 2))


def crop_rect_top(rect: Rect, px: int) -> Rect:
    l, t, r, b = rect
    return l, min(t + px, b), r, b


def clamp_rect(rect: Rect, bounds: Mapping[str, int]) -> Rect | None:
    """Intersect rect with the virtual screen {left, top, width, height}; None if nothing is left."""
    bl, bt = bounds["left"], bounds["top"]
    br, bb = bl + bounds["width"], bt + bounds["height"]
    l, t, r, b = max(rect[0], bl), max(rect[1], bt), min(rect[2], br), min(rect[3], bb)
    if r <= l or b <= t:
        return None
    return l, t, r, b


def downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if not max_side or longest <= max_side:
        return image
    s = max_side / longest
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def thumbnail(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32)


class ChangeDetector:
    """Decides whether a frame is worth sending to OCR.

    Compares against the last *sent* frame (not the last sampled one), so slow changes such as
    typing or gradual scrolling accumulate until they cross the threshold instead of never firing.
    """

    def __init__(self, threshold: float = 4.0, force_on_title_change: bool = True):
        self.threshold = float(threshold)
        self.force_on_title_change = force_on_title_change
        self._thumb: np.ndarray | None = None
        self._key: tuple[Any, str] | None = None
        self.frames = 0
        self.skipped = 0

    def update(self, image: np.ndarray, title: str = "", hwnd: Any = None) -> tuple[bool, float]:
        thumb = thumbnail(image)
        self.frames += 1
        key = (hwnd, title)
        if self._thumb is None:
            changed, diff = True, float("inf")
        else:
            diff = float(np.mean(np.abs(thumb - self._thumb)))
            changed = diff > self.threshold or (self.force_on_title_change and key != self._key)
        if changed:
            self._thumb, self._key = thumb, key
        else:
            self.skipped += 1
        return changed, diff

    @property
    def skip_rate(self) -> float:
        return self.skipped / self.frames if self.frames else 0.0

    def reset(self) -> None:
        self._thumb, self._key = None, None


# ---------------------------------------------------------------- Windows API

def _dwm_frame_rect(hwnd: int) -> Rect | None:
    """Visible window bounds without the invisible resize borders (Win10/11)."""
    try:
        r = wintypes.RECT()
        hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS), ctypes.byref(r), ctypes.sizeof(r)
        )
        if hr == 0:
            return r.left, r.top, r.right, r.bottom
    except (AttributeError, OSError):
        pass
    return None


def _window_dpi(hwnd: int) -> int:
    try:
        return int(ctypes.windll.user32.GetDpiForWindow(wintypes.HWND(hwnd))) or 96
    except (AttributeError, OSError):
        return 96


def _screen_cfg(key: str, default: Any) -> Any:
    from kavach_config import get_config

    return get_config().screen.get(key, default)


def get_active_window(
    exclude_title_prefixes: Iterable[str] | None = None,
    exclude_classes: Iterable[str] | None = None,
) -> WindowInfo | None:
    """Foreground window, or None if there is none, it is minimized, zero-size or excluded.

    Excluded: titles starting with screen.exclude_title_prefixes (Kavach windows), window classes in
    screen.exclude_window_classes (taskbar, desktop), and any window owned by this process.
    """
    if not IS_WINDOWS:
        return None
    if exclude_title_prefixes is None:
        exclude_title_prefixes = _screen_cfg("exclude_title_prefixes", [])
    if exclude_classes is None:
        exclude_classes = _screen_cfg("exclude_window_classes", DEFAULT_EXCLUDE_CLASSES)
    prefixes = tuple(exclude_title_prefixes)
    classes = set(exclude_classes)
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd or win32gui.IsIconic(hwnd):
            return None
        class_name = win32gui.GetClassName(hwnd) or ""
        if class_name in classes:
            return None
        title = win32gui.GetWindowText(hwnd) or ""
        if prefixes and title.startswith(prefixes):
            return None
        pid: int | None = None
        try:
            pid = win32process.GetWindowThreadProcessId(hwnd)[1]
        except (OSError, ValueError):
            pass
        if pid == os.getpid():
            return None
        rect = _dwm_frame_rect(hwnd) or tuple(win32gui.GetWindowRect(hwnd))
        if rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
            return None
        name: str | None = None
        try:
            name = psutil.Process(pid).name() if pid else None
        except (psutil.Error, OSError, ValueError):
            pass
        return WindowInfo(int(hwnd), title, pid, name, tuple(int(v) for v in rect), _window_dpi(hwnd), class_name)
    except Exception as e:  # pywintypes.error when the window vanishes mid-call
        log.debug("active window unreadable: %s", type(e).__name__)
        return None


# ---------------------------------------------------------------- capture

_tls = threading.local()


def _mss():
    """One mss instance per thread (mss is not thread-safe)."""
    sct = getattr(_tls, "sct", None)
    if sct is None:
        import mss

        sct = (getattr(mss, "MSS", None) or mss.mss)()
        _tls.sct = sct
    return sct


def close_thread_mss() -> None:
    sct = getattr(_tls, "sct", None)
    if sct is not None:
        sct.close()
        _tls.sct = None


def grab(window: WindowInfo, max_side: int | None = None) -> np.ndarray | None:
    """BGR image of the window rect, clamped to the virtual screen, downscaled to max_side. In memory only."""
    sct = _mss()
    rect = clamp_rect(window.rect, sct.monitors[0])
    if rect is None:
        return None
    l, t, r, b = rect
    shot = sct.grab({"left": l, "top": t, "width": r - l, "height": b - t})
    image = np.asarray(shot)[:, :, :3]  # BGRA -> BGR view
    del shot
    return np.ascontiguousarray(downscale(image, max_side or 0))


# ---------------------------------------------------------------- sampler

class ScreenSampler:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        on_frame: Callable[[ScreenFrame], None] | None = None,
        on_sample: Callable[[ScreenFrame], None] | None = None,
        window_fn: Callable[[], WindowInfo | None] | None = None,
        grab_fn: Callable[[WindowInfo, int], np.ndarray | None] = grab,
    ):
        """on_frame gets changed frames with the image; on_sample gets every sample as metadata (image=None)."""
        if config is None:
            from kavach_config import get_config
            config = get_config()
        scfg = config["screen"]
        self.interval_s = float(scfg.get("interval_s", 2.5))
        self.max_side = int(scfg.get("max_side", 1920))
        prefixes = list(scfg.get("exclude_title_prefixes", []))
        classes = list(scfg.get("exclude_window_classes", DEFAULT_EXCLUDE_CLASSES))
        self.crop_table = dict(scfg.get("browser_top_crop_px") or {})
        self.detector = ChangeDetector(scfg.get("change_threshold", 4.0), bool(scfg.get("force_on_title_change", True)))
        self.on_frame = on_frame
        self.on_sample = on_sample
        self._window_fn = window_fn or (lambda: get_active_window(prefixes, classes))
        self._grab_fn = grab_fn
        self.no_window = 0
        self.last_capture_ms: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> ScreenFrame | None:
        """Capture one frame, run change detection, fire callbacks. Returns metadata (image=None)."""
        window = self._window_fn()
        if window is None:
            self.no_window += 1
            return None
        crop = top_crop_px(window, self.crop_table)
        target = dataclasses.replace(window, rect=crop_rect_top(window.rect, crop)) if crop else window
        t0 = time.perf_counter()
        image = self._grab_fn(target, self.max_side)
        capture_ms = (time.perf_counter() - t0) * 1000.0
        if image is None:
            self.no_window += 1
            return None
        self.last_capture_ms = capture_ms
        changed, diff = self.detector.update(image, window.title, window.hwnd)
        ts = time.time()
        if changed and self.on_frame:
            frame = ScreenFrame(ts, window, image, True, capture_ms, diff, crop)
            try:
                self.on_frame(frame)
            except Exception:
                log.exception("on_frame callback failed")
            finally:
                frame.image = None
                del frame
        del image
        meta = ScreenFrame(ts, window, None, changed, capture_ms, diff, crop)
        if self.on_sample:
            try:
                self.on_sample(meta)
            except Exception:
                log.exception("on_sample callback failed")
        return meta

    def stats(self) -> dict[str, Any]:
        d = self.detector
        return {
            "frames": d.frames,
            "skipped": d.skipped,
            "skip_rate": round(d.skip_rate, 3),
            "no_window": self.no_window,
            "last_capture_ms": round(self.last_capture_ms, 1) if self.last_capture_ms is not None else None,
        }

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    self.sample_once()
                except Exception:
                    log.exception("screen sample failed")
                self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - t0)))
        finally:
            close_thread_mss()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="kavach-screen", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            self._thread = None


# ---------------------------------------------------------------- CLI

def _short(text: str | None, n: int = 48) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _show_preview(image: np.ndarray) -> None:
    # Display only; never cv2.imwrite.
    cv2.imshow("Kavach preview (not saved)", image)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m capture.screen", description="Active-window capture self-test")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="3 s countdown, capture the active window (default)")
    mode.add_argument("--watch", action="store_true", help="sample every screen.interval_s until Ctrl+C")
    p.add_argument("--preview", action="store_true", help="show frames in a window (never saved)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # titles may contain characters the console can't encode

    if not IS_WINDOWS:
        print("capture.screen needs Windows")
        return 1

    from kavach_config import get_config
    from kavach_privacy import redact_title
    cfg = get_config()
    print(f"DPI awareness: {DPI_AWARENESS}")

    if args.watch:
        import queue
        previews: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

        def on_frame(frame: ScreenFrame) -> None:
            if args.preview and previews.empty():
                previews.put_nowait(frame.image)

        def on_sample(f: ScreenFrame) -> None:
            st = sampler.stats()
            ts = time.strftime("%H:%M:%S", time.localtime(f.ts))
            print(f"{ts}  {'changed' if f.changed else 'skipped':<7}  {f.capture_ms:6.1f} ms  diff={min(f.diff, 999):5.1f}  "
                  f"skip_rate={st['skip_rate']:.0%}  [{f.window.process_name}] {_short(redact_title(f.window.title))}",
                  flush=True)

        sampler = ScreenSampler(cfg, on_frame=on_frame, on_sample=on_sample)
        print(f"sampling every {sampler.interval_s}s, threshold {sampler.detector.threshold} (Ctrl+C to stop)", flush=True)
        sampler.start()
        try:
            while True:
                if args.preview:
                    try:
                        _show_preview(previews.get(timeout=0.1))
                    except queue.Empty:
                        pass
                    cv2.waitKey(1)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            sampler.stop()
            if args.preview:
                cv2.destroyAllWindows()
            print(f"stats: {sampler.stats()}")
        return 0

    for i in (3, 2, 1):
        print(f"capturing in {i}... (switch to the window to capture)", flush=True)
        time.sleep(1)
    window = get_active_window()
    if window is None:
        print("no capturable active window (none, minimized, zero-size, shell or excluded)")
        return 0
    crop = top_crop_px(window, cfg.screen.get("browser_top_crop_px"))
    target = dataclasses.replace(window, rect=crop_rect_top(window.rect, crop)) if crop else window
    print(f"window:  hwnd={window.hwnd} pid={window.pid} process={window.process_name} class={window.class_name}")
    print(f"title:   {_short(redact_title(window.title), 100)}")
    print(f"rect:    {window.rect}  size={window.size[0]}x{window.size[1]}  dpi={window.dpi} "
          f"({window.dpi / 96:.0%})  top crop={crop}px")
    t0 = time.perf_counter()
    image = grab(target, int(cfg.screen.get("max_side", 1920)))
    capture_ms = (time.perf_counter() - t0) * 1000
    if image is None:
        print("window is outside the virtual screen")
        return 0
    print(f"image:   shape={image.shape} dtype={image.dtype}  (max_side={cfg.screen.max_side})")
    print(f"capture: {capture_ms:.1f} ms (first grab includes mss init)")
    t0 = time.perf_counter()
    grab(target, int(cfg.screen.get("max_side", 1920)))
    print(f"capture: {(time.perf_counter() - t0) * 1000:.1f} ms (warm)")
    if args.preview:
        _show_preview(image)
        print("press any key in the preview window to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    del image
    close_thread_mss()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
