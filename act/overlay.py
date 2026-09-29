"""Full-screen scam alert + caution toast (Tkinter, standard library only), bilingual en/hi.

    Overlay(config).run(pipeline)      # owns the Tk mainloop on the MAIN thread; pipeline events
                                        # arrive through a queue polled with after()

Alert window: topmost, covers the primary screen, dark red, white text. Hindi + English headline,
2-4 plain-language reasons from reason ids (act/strings.yaml; never screen text or the call), advice
(hang up, call the bank, 1930 / cybercrime.gov.in) and two big mouse-only buttons:
  "Stop and get help"   -> suspends the live remote tool (act.remote_control) and shows the helpline,
                           with "Resume remote app" and "Close"
  "I'm safe, continue"  -> resumes anything suspended, sends user_override to the engine, closes
The buttons are labels, not Tk buttons: Enter / Space / Escape do nothing, only a mouse click works
(someone on a remote session could press keys). The window takes keyboard focus while shown, so keys
typed by a remote user land here, not in the banking page.
Injected-input guard: while the alert is shown a WH_MOUSE_LL hook (own thread) records whether the
latest button-down was injected (LLMHF_INJECTED - how remote-access tools deliver the remote mouse).
Injected clicks on "I'm safe" / "Resume remote app" are ignored with a bilingual note and an
overlay:injected_click_blocked incident-log line; "Stop and get help" accepts any click. If the hook
can't be installed, a warning is logged and every click is accepted as before.

Caution toast: small topmost window bottom-right, one bilingual reason, x to close, gone after 12 s.
Read-out: English headline + first reason via Windows SAPI (System.Speech, PowerShell in a
background thread); Hindi too if a hi-IN SAPI voice is installed. Never blocks the UI thread.
When the alert is on screen, pipeline.alert_shown(incident, t) lets the engine log time_to_alert_ms.

Demos (no pipeline):
    python -m act.overlay --demo alert   [--screenshot demo/screenshots/overlay.png] [--seconds 8]
    python -m act.overlay --demo caution
    python -m act.overlay --demo alert --inject-test   # SendInput click on "I'm safe" after 3 s: must be blocked
    python -m act.overlay --voices        # list SAPI voices
Live:  python -m act.overlay --live       # pipeline + overlay
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("act.overlay")

STRINGS_PATH = ROOT / "act" / "strings.yaml"
FONT = "Nirmala UI"                     # Windows font with Devanagari shaping (and Latin)
BG, FG = "#7A0000", "#FFFFFF"           # dark red / white: ~12:1 contrast
BTN_HELP = ("#FFFFFF", "#7A0000")       # (bg, fg)
BTN_SAFE = ("#3D0000", "#FFFFFF")
TOAST_BG, TOAST_FG = "#FFD54A", "#1A1A1A"   # amber caution
TOAST_S = 12.0
NO_KEYS = ("<Return>", "<KP_Enter>", "<space>", "<Escape>", "<Tab>")
HEADLINE_MIN_PT = 32
DEMO_REASONS = ("remote_tool:anydesk", "call:authority", "call:money_move", "screen:otp_card", "screen:bank", "combo")


# ---------------------------------------------------------------- strings

@lru_cache(maxsize=4)
def load_strings(path: str | Path = STRINGS_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def tool_names(config: Mapping[str, Any] | None) -> dict[str, str]:
    """slug -> display name for processes.known_tools ("quick_assist" -> "Quick Assist")."""
    from fusion.risk import slug

    tools = ((config or {}).get("processes", {}) or {}).get("known_tools", []) or []
    return {slug(t["name"]): t["name"] for t in tools if t.get("name")}


def _entry(rid: str, strings: Mapping[str, Any]) -> Mapping[str, Any] | None:
    reasons = strings.get("reasons", {})
    if rid in reasons:
        return reasons[rid]
    prefix = rid.split(":", 1)[0]
    return reasons.get(f"{prefix}:*")


def reason_text(rid: str, lang: str, strings: Mapping[str, Any], tools: Mapping[str, str] | None = None) -> str | None:
    e = _entry(rid, strings)
    if not e or not e.get(lang):
        return None
    tool_slug = rid.split(":", 1)[1] if ":" in rid else ""
    tool = (tools or {}).get(tool_slug) or tool_slug.replace("_", " ").title()
    return str(e[lang]).replace("{tool}", tool)


def _rank(rid: str, strings: Mapping[str, Any]) -> int:
    prio = list(strings.get("priority", []))
    for key in (rid, f"{rid.split(':', 1)[0]}:*"):
        if key in prio:
            return prio.index(key)
    return len(prio)


def pick_reasons(reason_ids: Iterable[str], strings: Mapping[str, Any], n: int = 4, overlay: bool = True) -> list[str]:
    """Up to n known reason ids in priority order (overlay=True skips entries marked overlay: false)."""
    ids = [r for r in dict.fromkeys(reason_ids) if _entry(r, strings)]
    if overlay:
        ids = [r for r in ids if _entry(r, strings).get("overlay", True)]
    return sorted(ids, key=lambda r: _rank(r, strings))[:n]


def toast_reason(reason_ids: Iterable[str], strings: Mapping[str, Any]) -> str | None:
    ids = list(reason_ids)
    if "gate:no_tactic" in ids:
        return "gate:no_tactic"
    picked = pick_reasons(ids, strings, n=1, overlay=False)
    return picked[0] if picked else None


def ui(key: str, lang: str, strings: Mapping[str, Any], **fmt: str) -> str:
    text = str(strings["ui"][key][lang])
    for k, v in fmt.items():
        text = text.replace("{" + k + "}", v)
    return text


# ---------------------------------------------------------------- speech (SAPI via PowerShell)

_PS_SPEAK = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | Where-Object { $_.Enabled }
$en = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'en*' } | Select-Object -First 1
if ($en) { $s.SelectVoice($en.VoiceInfo.Name) }
$s.Speak($env:KAVACH_SAY_EN)
$hi = $voices | Where-Object { $_.VoiceInfo.Culture.Name -eq 'hi-IN' } | Select-Object -First 1
if ($hi -and $env:KAVACH_SAY_HI) { $s.SelectVoice($hi.VoiceInfo.Name); $s.Speak($env:KAVACH_SAY_HI) }
"""
_PS_VOICES = r"""
Add-Type -AssemblyName System.Speech
(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() |
  ForEach-Object { "{0}`t{1}`t{2}" -f $_.VoiceInfo.Name, $_.VoiceInfo.Culture.Name, $_.Enabled }
"""


def _powershell(script: str, env: Mapping[str, str] | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    # -EncodedCommand (UTF-16LE base64): multi-line scripts run as one block. Spoken text goes in env
    # vars, so no quoting / injection issues.
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                          text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout,
                          env={**os.environ, **(env or {})}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def list_voices() -> list[tuple[str, str, bool]]:
    out = _powershell(_PS_VOICES, timeout=30).stdout
    rows = [l.split("\t") for l in out.splitlines() if l.count("\t") == 2]
    return [(n, c, e.strip().lower() == "true") for n, c, e in rows]


class Speaker:
    """Speaks in a daemon thread; a new request while speaking is dropped (no queue of stale alerts)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.platform == "win32"
        self._busy = threading.Lock()

    def say(self, en: str, hi: str = "") -> bool:
        if not self.enabled or not self._busy.acquire(blocking=False):
            return False

        def work() -> None:
            try:
                _powershell(_PS_SPEAK, {"KAVACH_SAY_EN": en, "KAVACH_SAY_HI": hi}, timeout=60)
            except Exception as e:  # noqa: BLE001
                log.warning("speech failed: %s", type(e).__name__)
            finally:
                self._busy.release()

        threading.Thread(target=work, name="kavach-speech", daemon=True).start()
        return True


# ---------------------------------------------------------------- Tk helpers

def set_dpi_awareness() -> None:
    """Before Tk(): physical pixels, so the overlay really covers the screen and text is sharp."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001 - already set (e.g. by capture.screen) or older Windows
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def work_area() -> tuple[int, int, int, int] | None:
    """Primary monitor work area (excludes the taskbar), physical px."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    r = wintypes.RECT()
    if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0):   # SPI_GETWORKAREA
        return r.left, r.top, r.right, r.bottom
    return None


def mouse_button(parent: Any, text: str, colors: tuple[str, str], action: Callable[[], None], size: int = 24) -> Any:
    """A big clickable label: only a mouse click (press + release inside) runs `action`. No keyboard
    activation at all (not focusable, no Enter/Space binding)."""
    import tkinter as tk

    bg, fg = colors
    b = tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, size, "bold"), padx=36, pady=18, cursor="hand2",
                 takefocus=0, relief="raised", bd=4, highlightthickness=3, highlightbackground=FG)
    b._pressed = False  # type: ignore[attr-defined]

    def press(_e: Any) -> str:
        b._pressed = True  # type: ignore[attr-defined]
        b.config(relief="sunken")
        return "break"

    def release(e: Any) -> str:
        inside = 0 <= e.x < b.winfo_width() and 0 <= e.y < b.winfo_height()
        was = b._pressed  # type: ignore[attr-defined]
        b._pressed = False  # type: ignore[attr-defined]
        b.config(relief="raised")
        if was and inside:
            action()
        return "break"

    b.bind("<ButtonPress-1>", press)
    b.bind("<ButtonRelease-1>", release)
    return b


# ---------------------------------------------------------------- injected-input guard

# A remote-access tool delivers the remote person's mouse as injected input (SendInput); a
# WH_MOUSE_LL hook sees LLMHF_INJECTED on those events. "I'm safe" and "Resume remote app" must be
# clicked with the local mouse; "Stop and get help" accepts any click (stopping is always safe).
GUARDED_BUTTONS = frozenset({"safe", "resume"})
WH_MOUSE_LL, WM_QUIT = 14, 0x0012
BUTTON_DOWN_MSGS = frozenset({0x0201, 0x0204, 0x0207, 0x020B})     # L / R / M / X button down
LLMHF_INJECTED, LLMHF_LOWER_IL_INJECTED = 0x01, 0x02


def click_decision(button: str, injected: bool | None) -> bool:
    """True = run the button's action. injected None = unknown (hook unavailable) -> allow."""
    return not (injected and button in GUARDED_BUTTONS)


def is_injected(flags: int) -> bool:
    return bool(flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED))


class MouseGuard:
    """WH_MOUSE_LL hook on its own thread (with a message loop) while the alert is shown. Records only
    whether the latest button-down was injected - no coordinates, nothing else. start() never raises:
    if the hook can't be installed it logs a warning and the overlay behaves as before."""

    def __init__(self) -> None:
        self.available = False
        self.last_down_injected: bool | None = None
        self.downs = 0
        self._thread: threading.Thread | None = None
        self._tid: int | None = None
        self._ready = threading.Event()

    def start(self, timeout: float = 2.0) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return self.available
        if sys.platform != "win32":
            return False
        self._ready.clear()
        self.last_down_injected = None
        self._thread = threading.Thread(target=self._run, name="kavach-mousehook", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            log.warning("mouse hook: no answer from hook thread; injected-click guard off")
        return self.available

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        if self._tid is not None:
            import ctypes
            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        self._thread.join(timeout)
        self._thread = None
        self._tid = None
        self.available = False

    def _run(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class MSLLHOOKSTRUCT(ctypes.Structure):
                _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                            ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

            LRESULT = ctypes.c_ssize_t
            HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
            user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
            user32.SetWindowsHookExW.restype = ctypes.c_void_p
            user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
            user32.CallNextHookEx.restype = LRESULT
            user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE

            def proc(n_code: int, wparam: int, lparam: int) -> int:
                try:
                    if n_code == 0 and wparam in BUTTON_DOWN_MSGS:
                        info = ctypes.cast(lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                        self.last_down_injected = is_injected(info.flags)
                        self.downs += 1
                except Exception:  # noqa: BLE001 - never break the system's mouse input
                    pass
                return user32.CallNextHookEx(None, n_code, wparam, lparam)

            self._proc_ref = HOOKPROC(proc)              # keep alive while hooked
            self._tid = kernel32.GetCurrentThreadId()
            hook = user32.SetWindowsHookExW(WH_MOUSE_LL, self._proc_ref, kernel32.GetModuleHandleW(None), 0)
            if not hook:
                log.warning("mouse hook: SetWindowsHookExW failed (error %d); injected-click guard off",
                            ctypes.get_last_error())
                self._ready.set()
                return
            self.available = True
            self._ready.set()
            msg = wintypes.MSG()
            try:
                while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
            finally:
                user32.UnhookWindowsHookEx(hook)
                self.available = False
        except Exception as e:  # noqa: BLE001
            log.warning("mouse hook: %s; injected-click guard off", type(e).__name__)
            self.available = False
            self._ready.set()


def send_injected_click(x: int, y: int) -> bool:
    """Demo / test only: move the cursor to (x, y) and SendInput one left click (flagged injected,
    like a remote-access tool's input)."""
    import ctypes
    from ctypes import wintypes

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetCursorPos(int(x), int(y))
    down, up = INPUT(0, MOUSEINPUT(0, 0, 0, 0x0002, 0, 0)), INPUT(0, MOUSEINPUT(0, 0, 0, 0x0004, 0, 0))
    arr = (INPUT * 2)(down, up)
    return user32.SendInput(2, arr, ctypes.sizeof(INPUT)) == 2


# ---------------------------------------------------------------- overlay

class Overlay:
    def __init__(self, config: Mapping[str, Any] | None = None, strings: Mapping[str, Any] | None = None,
                 remote: Any = None, pipeline: Any = None, speak: bool = True, fullscreen: bool = True,
                 input_guard: bool = True):
        if config is None:
            from kavach_config import get_config
            config = get_config()
        self.cfg = config
        self.strings = strings or load_strings()
        self.tools = tool_names(config)
        self.remote = remote                 # act.remote_control.RemoteControl (or a stand-in)
        self.pipeline = pipeline             # needs .override() and .alert_shown(incident, ts)
        self.speaker = Speaker(speak)
        self.fullscreen = fullscreen
        self.root: Any = None
        self.alert_win: Any = None
        self.toast_win: Any = None
        self.incident = 0
        self.shown_log: list[tuple[int, float]] = []       # (incident, overlay shown ts)
        self.q: queue.Queue[Any] = queue.Queue()
        self.last_state: Any = None
        self._scale = 1.0                   # font scale; shrunk until the content fits the screen
        self.on_closed: Callable[[str], None] | None = None   # "safe" | "help_close" | "resumed"
        self.guard: Any = MouseGuard() if input_guard else None   # WH_MOUSE_LL while the alert is shown
        self.blocked: list[str] = []                         # buttons whose injected clicks were ignored
        self.note: Any = None

    # -- Tk root / main loop

    def make_root(self) -> Any:
        import tkinter as tk

        set_dpi_awareness()
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Kavach")
        return self.root

    def run(self, pipeline: Any = None, start_pipeline: bool = True) -> None:
        """Own the Tk mainloop (call from the main thread). Pipeline callbacks only enqueue."""
        root = self.root or self.make_root()
        if pipeline is not None:
            self.pipeline = pipeline
            pipeline.subscribe(on_state=self._on_state, on_event=self.q.put)
            if start_pipeline:
                pipeline.start()
            if self.remote is None and getattr(pipeline, "monitor", None) is not None:
                from act.remote_control import RemoteControl
                self.remote = RemoteControl(pipeline.monitor)
        root.after(100, self._poll)
        try:
            root.mainloop()
        finally:
            if self.guard is not None:
                self.guard.stop()
            if self.remote is not None:
                self.remote.resume_all()
            if pipeline is not None and start_pipeline:
                pipeline.stop()

    def _on_state(self, state: Any) -> None:
        self.last_state = state                              # latest only; read on the Tk thread

    def _poll(self) -> None:
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            kind = getattr(ev, "kind", None)
            if kind == "alert":
                self.show_alert(ev.reason_ids, ev.incident)
            elif kind == "caution" and self.alert_win is None:
                self.show_toast(ev.reason_ids)
        if self.root is not None:
            self.root.after(100, self._poll)

    # -- alert

    def show_alert(self, reason_ids: Iterable[str], incident: int = 0) -> Any:
        import tkinter as tk

        s, root = self.strings, self.root or self.make_root()
        self.incident = incident
        self.close_toast()
        if self.alert_win is None:
            w = tk.Toplevel(root, bg=BG)
            w.title("Kavach alert")
            w.overrideredirect(True)
            w.attributes("-topmost", True)
            if self.fullscreen:
                w.geometry(f"{w.winfo_screenwidth()}x{w.winfo_screenheight()}+0+0")
            else:
                w.geometry("1100x800+40+40")
            w.protocol("WM_DELETE_WINDOW", lambda: None)     # Alt+F4 does nothing
            for seq in NO_KEYS:                              # keys never act; mouse only
                w.bind(seq, lambda e: "break")
            self.alert_win = w
            self._keep_on_top()
            if self.guard is not None and not self.guard.start():
                log.warning("injected-click guard unavailable; buttons accept any click")
        self._fit(lambda: self._build_alert(reason_ids))
        w = self.alert_win
        w.deiconify()
        w.lift()
        w.focus_force()
        w.update()
        shown = time.time()
        self.shown_log.append((incident, shown))
        if self.pipeline is not None and hasattr(self.pipeline, "alert_shown"):
            self.pipeline.alert_shown(incident, shown)
        picked = pick_reasons(reason_ids, s, n=1)
        first_en = reason_text(picked[0], "en", s, self.tools) if picked else ""
        first_hi = reason_text(picked[0], "hi", s, self.tools) if picked else ""
        self.speaker.say(f"{ui('headline', 'en', s)}. {first_en}.", f"{ui('headline', 'hi', s)}। {first_hi}।")
        return w

    def _clear(self) -> Any:
        import tkinter as tk

        for child in self.alert_win.winfo_children():
            child.destroy()
        box = tk.Frame(self.alert_win, bg=BG)
        box.place(relx=0.5, rely=0.5, anchor="center")
        self._box = box
        return box

    def _fit(self, build: Callable[[], None]) -> None:
        """Build at full size; shrink fonts until the content fits in 94% of the window height."""
        self._scale = 1.0
        for _ in range(8):
            build()
            self.alert_win.update_idletasks()
            avail = (self.alert_win.winfo_screenheight() if self.fullscreen else 800) * 0.94
            if self._box.winfo_reqheight() <= avail:
                return
            self._scale *= 0.9

    def _pt(self, size: int, floor: int = 12) -> int:
        return max(floor, round(size * self._scale))

    def _text(self, parent: Any, text: str, size: int, bold: bool = False, pady: Any = 4, floor: int = 12) -> Any:
        import tkinter as tk

        wrap = int(self.alert_win.winfo_screenwidth() * 0.85) if self.fullscreen else 1000
        pady = tuple(round(v * self._scale) for v in pady) if isinstance(pady, tuple) else round(pady * self._scale)
        lbl = tk.Label(parent, text=text, bg=BG, fg=FG, font=(FONT, self._pt(size, floor), "bold" if bold else "normal"),
                       wraplength=wrap, justify="center")
        lbl.pack(pady=pady)
        return lbl

    def _build_alert(self, reason_ids: Iterable[str]) -> None:
        import tkinter as tk

        s = self.strings
        box = self._clear()
        self._text(box, f"⚠  {ui('headline', 'hi', s)}", 44, True, 0, floor=HEADLINE_MIN_PT)
        self._text(box, ui("headline", "en", s), 40, True, (0, 24), floor=HEADLINE_MIN_PT)
        self.reason_labels = []
        for rid in pick_reasons(reason_ids, s, n=4):
            en, hi = reason_text(rid, "en", s, self.tools), reason_text(rid, "hi", s, self.tools)
            self.reason_labels.append(self._text(box, f"•  {en}\n    {hi}", 20, False, 6))
        self._text(box, ui("advice", "en", s), 20, True, (24, 0))
        self._text(box, ui("advice", "hi", s), 20, True, (0, 24))
        row = tk.Frame(box, bg=BG)
        row.pack(pady=(8, 8))
        self.btn_help = mouse_button(row, f"{ui('btn_help', 'en', s)}\n{ui('btn_help', 'hi', s)}", BTN_HELP,
                                     lambda: self._click("help", self.on_help), self._pt(24, 16))
        self.btn_safe = mouse_button(row, f"{ui('btn_safe', 'en', s)}\n{ui('btn_safe', 'hi', s)}", BTN_SAFE,
                                     lambda: self._click("safe", self.on_safe), self._pt(24, 16))
        self.btn_help.pack(side="left", padx=24)
        self.btn_safe.pack(side="left", padx=24)
        self._text(box, f"{ui('mouse_hint', 'en', s)}  /  {ui('mouse_hint', 'hi', s)}", 14, False, (8, 0))

    def _keep_on_top(self) -> None:
        if self.alert_win is not None:
            self.alert_win.attributes("-topmost", True)
            self.alert_win.lift()
            self.alert_win.after(1000, self._keep_on_top)

    # -- button actions

    def on_help(self) -> None:
        """Pause the live remote tool(s) and show the helpline."""
        s = self.strings
        results = self.remote.suspend_live() if self.remote is not None else []
        lines: list[tuple[str, str]] = []
        for r in results:
            key = "could_not_pause" if (r.denied or not r.suspended) else "paused"
            lines.append((ui(key, "en", s, tool=r.tool), ui(key, "hi", s, tool=r.tool)))
        if not results:
            lines.append((ui("no_remote", "en", s), ui("no_remote", "hi", s)))
        self._fit(lambda: self._build_help(lines, resumable=any(r.suspended for r in results)))

    def _build_help(self, lines: list[tuple[str, str]], resumable: bool) -> None:
        import tkinter as tk

        s = self.strings
        box = self._clear()
        self._text(box, ui("help_title", "hi", s), 40, True, 0, floor=HEADLINE_MIN_PT)
        self._text(box, ui("help_title", "en", s), 36, True, (0, 24), floor=HEADLINE_MIN_PT)
        self.help_lines = [self._text(box, f"{en}\n{hi}", 22, False, 8) for en, hi in lines]
        self._text(box, ui("helpline", "en", s), 26, True, (24, 0))
        self._text(box, ui("helpline", "hi", s), 26, True, (0, 12))
        self._text(box, ui("advice", "en", s), 18, False, (12, 0))
        self._text(box, ui("advice", "hi", s), 18, False, (0, 24))
        row = tk.Frame(box, bg=BG)
        row.pack(pady=8)
        self.btn_close = mouse_button(row, f"{ui('btn_close', 'en', s)}\n{ui('btn_close', 'hi', s)}", BTN_HELP,
                                      lambda: self.close_alert("help_close"), self._pt(24, 16))
        self.btn_close.pack(side="left", padx=24)
        self.btn_resume = None
        if resumable:
            self.btn_resume = mouse_button(row, f"{ui('btn_resume', 'en', s)}\n{ui('btn_resume', 'hi', s)}", BTN_SAFE,
                                           lambda: self._click("resume", self.on_resume), size=self._pt(18, 14))
            self.btn_resume.pack(side="left", padx=24)

    def on_resume(self) -> None:
        if self.remote is not None:
            self.remote.resume_all()
        self.close_alert("resumed")

    def on_safe(self) -> None:
        """"I'm safe, continue": resume anything suspended, tell the engine, close."""
        if self.remote is not None:
            self.remote.resume_all()
        if self.pipeline is not None:
            self.pipeline.override()
        self.close_alert("safe")

    def _click(self, button: str, action: Callable[[], None]) -> None:
        """Mouse click on a big button: injected clicks on guarded buttons are ignored."""
        guard = self.guard
        injected = guard.last_down_injected if guard is not None and guard.available else None
        allowed = click_decision(button, injected)
        log.info("click %s: injected=%s -> %s", button, injected, "allow" if allowed else "block")
        if allowed:
            action()
        else:
            self._block(button)

    def _block(self, button: str) -> None:
        import tkinter as tk

        from fusion.incidents import UiEvent

        self.blocked.append(button)
        log.warning("injected click on %r ignored", button)
        if self.pipeline is not None and hasattr(self.pipeline, "report_event"):
            self.pipeline.report_event(UiEvent(time.time(), "overlay:injected_click_blocked", self.incident, button))
        s = self.strings
        if self.note is not None and self.note.winfo_exists():
            self.note.destroy()
        text = ui("injected_click", "en", s) + "\n" + ui("injected_click", "hi", s)
        self.note = tk.Label(self.alert_win, text=text, bg=TOAST_BG, fg=TOAST_FG,
                             font=(FONT, self._pt(18, 14), "bold"), padx=20, pady=10)
        self.note.place(relx=0.5, rely=0.995, anchor="s")        # over the mouse hint, not the headline
        note = self.note
        self.alert_win.after(8000, lambda: note.winfo_exists() and note.destroy())

    def close_alert(self, how: str = "") -> None:
        if self.guard is not None:
            self.guard.stop()
        self.note = None
        if self.alert_win is not None:
            self.alert_win.destroy()
            self.alert_win = None
        if self.on_closed:
            self.on_closed(how)

    # -- caution toast

    def show_toast(self, reason_ids: Iterable[str], seconds: float = TOAST_S) -> Any:
        import tkinter as tk

        s, root = self.strings, self.root or self.make_root()
        rid = toast_reason(reason_ids, s)
        if rid is None:
            return None
        self.close_toast()
        w = tk.Toplevel(root, bg=TOAST_BG)
        w.title("Kavach caution")
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        head = tk.Frame(w, bg=TOAST_BG)
        head.pack(fill="x", padx=14, pady=(10, 0))
        tk.Label(head, text=f"⚠  {ui('toast_title', 'en', s)}  /  {ui('toast_title', 'hi', s)}", bg=TOAST_BG,
                 fg=TOAST_FG, font=(FONT, 12, "bold")).pack(side="left")
        x = tk.Label(head, text="✕", bg=TOAST_BG, fg=TOAST_FG, font=(FONT, 14, "bold"), cursor="hand2", takefocus=0)
        x.pack(side="right")
        x.bind("<ButtonRelease-1>", lambda e: self.close_toast())
        for lang in ("en", "hi"):
            tk.Label(w, text=reason_text(rid, lang, s, self.tools), bg=TOAST_BG, fg=TOAST_FG, font=(FONT, 14),
                     wraplength=760, justify="left").pack(anchor="w", padx=14)
        tk.Frame(w, bg=TOAST_BG, height=10).pack()
        w.update_idletasks()
        ww, wh = max(560, w.winfo_reqwidth()), w.winfo_reqheight()
        left, top, right, bottom = work_area() or (0, 0, w.winfo_screenwidth(), w.winfo_screenheight() - 48)
        w.geometry(f"{ww}x{wh}+{right - ww - 16}+{bottom - wh - 16}")
        w.after(int(seconds * 1000), self.close_toast)
        self.toast_win = w
        w.update()
        return w

    def close_toast(self) -> None:
        if self.toast_win is not None:
            try:
                self.toast_win.destroy()
            except Exception:  # noqa: BLE001 - already gone
                pass
            self.toast_win = None


# ---------------------------------------------------------------- demo screenshot (demo strings only)

def save_window_png(win: Any, path: Path, bg: str, min_bg_fraction: float = 0.4) -> bool:
    """Grab exactly the window's rectangle and save it - only if it really shows our window (enough
    pixels in its background colour), so no real screen content can end up on disk."""
    import mss
    import mss.tools

    win.update()
    x, y, w, h = win.winfo_rootx(), win.winfo_rooty(), win.winfo_width(), win.winfo_height()
    with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
    want = tuple(int(bg[i:i + 2], 16) for i in (1, 3, 5))
    px = shot.rgb
    n = len(px) // 3
    step = max(1, n // 20000)
    hits = sum(1 for i in range(0, n, step) if all(abs(px[3 * i + c] - want[c]) <= 6 for c in range(3)))
    frac = hits / len(range(0, n, step))
    if frac < min_bg_fraction:
        log.error("screenshot refused: only %.0f%% of pixels are the overlay background", frac * 100)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    mss.tools.to_png(shot.rgb, shot.size, output=str(path))
    return True


class DemoRemote:
    """Stand-in for RemoteControl in --demo: pretends AnyDesk was paused (nothing is touched)."""

    def __init__(self) -> None:
        self.paused = False

    def suspend_live(self) -> list[Any]:
        from act.remote_control import SuspendResult

        self.paused = True
        return [SuspendResult("AnyDesk", suspended=[0])]

    def resume_all(self) -> list[int]:
        self.paused = False
        return []


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m act.overlay", description="Kavach overlay")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", choices=["alert", "caution"], help="show a window with sample reasons (no pipeline)")
    g.add_argument("--live", action="store_true", help="run the pipeline with the overlay")
    g.add_argument("--voices", action="store_true", help="list SAPI voices")
    p.add_argument("--screenshot", help="demo only: save a PNG of the demo window (and the help screen)")
    p.add_argument("--seconds", type=float, default=0, help="demo only: close after N seconds")
    p.add_argument("--no-speak", action="store_true")
    p.add_argument("--inject-test", action="store_true",
                   help="demo alert: after 3 s send one injected (SendInput) click at \"I'm safe\" - it must be blocked")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.voices:
        for name, culture, enabled in list_voices():
            print(f"{culture:<6} {'enabled ' if enabled else 'disabled'} {name}")
        return 0
    from kavach_config import get_config

    cfg = get_config()
    if args.live:
        from kavach.pipeline import Pipeline

        Overlay(cfg, speak=not args.no_speak).run(Pipeline(cfg))
        return 0

    class DemoPipeline:
        def override(self) -> None:
            print("demo: user_override sent")

        def alert_shown(self, incident: int, ts: float) -> None:
            print(f"demo: overlay shown for incident {incident} at {ts:.3f}")

        def report_event(self, ev: Any) -> None:
            import json

            from fusion.incidents import record_of
            print(f"demo: incident log line: {json.dumps(record_of(ev))}")

    ov = Overlay(cfg, remote=DemoRemote(), pipeline=DemoPipeline(), speak=not args.no_speak)
    root = ov.make_root()
    ov.on_closed = lambda how: (print(f"demo: closed ({how})"), root.after(300, root.destroy))
    shot = Path(args.screenshot) if args.screenshot else None
    if args.demo == "alert":
        root.after(50, lambda: ov.show_alert(DEMO_REASONS, incident=1))
        if args.inject_test:
            def inject() -> None:
                b = ov.btn_safe
                x, y = b.winfo_rootx() + b.winfo_width() // 2, b.winfo_rooty() + b.winfo_height() // 2
                print(f"inject-test: hook installed={ov.guard.available}; sending an injected click at \"I'm safe\"")
                threading.Thread(target=send_injected_click, args=(x, y), daemon=True).start()
                root.after(1500, check)

            def check() -> None:
                open_ = ov.alert_win is not None
                print(f"inject-test: hook saw injected={ov.guard.last_down_injected}, blocked={ov.blocked}, "
                      f"overlay still open={open_} -> {'PASS' if ov.blocked == ['safe'] and open_ else 'FAIL'}")
                if shot and open_:
                    inj = shot.with_name(shot.stem + "_injected" + shot.suffix)
                    print(f"screenshot: {inj if save_window_png(ov.alert_win, inj, BG) else 'REFUSED'}")
            root.after(3000, inject)
        elif shot:
            def grab_main() -> None:
                ok = save_window_png(ov.alert_win, shot, BG)
                print(f"screenshot: {shot if ok else 'REFUSED'}")
                ov.on_help()
                root.after(700, grab_help)

            def grab_help() -> None:
                help_path = shot.with_name(shot.stem + "_help" + shot.suffix)
                ok = save_window_png(ov.alert_win, help_path, BG)
                print(f"screenshot: {help_path if ok else 'REFUSED'}")
            root.after(1500, grab_main)
    else:
        root.after(50, lambda: ov.show_toast(["remote_tool:anydesk", "screen:bank", "screen:otp_card", "gate:no_tactic"],
                                             seconds=args.seconds or TOAST_S))
        if shot:
            root.after(1200, lambda: print(f"screenshot: {shot if save_window_png(ov.toast_win, shot, TOAST_BG) else 'REFUSED'}"))
        root.after(int((args.seconds or TOAST_S) * 1000) + 500, root.destroy)
    if args.seconds and args.demo == "alert":
        root.after(int(args.seconds * 1000), root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
