"""Pause / resume a live remote-access tool (the overlay's "Stop and get help" button).

Rules:
  - Only USER-role processes of a tool that ProcessMonitor knows (processes.known_tools; extra exe
    names only when the monitor was built with them for testing). Service / SYSTEM / tray / other-user
    processes are never touched. Nothing is ever killed.
  - Right before suspending, each pid is re-checked: its exe name must still map to the same tool
    (the pid could have been reused since the last poll).
  - Every suspended pid is recorded, so resume_all() can always undo it. resume_all() also runs on
    exit (atexit + SIGINT/SIGTERM/SIGBREAK handlers), so nothing is left frozen.
  - AccessDenied is reported, not raised: the overlay then says "could not pause AnyDesk - please
    close it yourself".

    rc = RemoteControl(monitor)
    res = rc.suspend("AnyDesk")   # SuspendResult(tool, suspended=[pids], denied=[pids], gone=[pids])
    rc.resume_all()

Self-test (open Notepad first):  python -m act.remote_control --test notepad.exe
"""

from __future__ import annotations

import argparse
import atexit
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from capture.processes import ROLE_USER

log = logging.getLogger("act.remote_control")


@dataclass
class SuspendResult:
    tool: str
    suspended: list[int] = field(default_factory=list)
    denied: list[int] = field(default_factory=list)       # AccessDenied: tell the user to close it
    gone: list[int] = field(default_factory=list)          # exited / pid reused since the last poll
    skipped: list[int] = field(default_factory=list)       # not user-role (service, tray, other user)
    error: str | None = None                               # e.g. unknown tool

    @property
    def ok(self) -> bool:
        return bool(self.suspended) and not self.denied and self.error is None

    @property
    def could_not_pause(self) -> bool:
        return bool(self.denied) or (self.error is None and not self.suspended)


class RemoteControl:
    def __init__(self, monitor: Any, psutil_mod: Any = psutil, install_exit_hooks: bool = True):
        self.monitor = monitor
        self.psutil = psutil_mod
        self._suspended: dict[int, str] = {}                # pid -> tool
        self._lock = threading.Lock()
        if install_exit_hooks:
            _register(self)

    # -- targets

    def known_tool(self, tool: str) -> bool:
        return tool in set(self.monitor.index.values())

    def user_pids(self, tool: str) -> tuple[list[int], list[int]]:
        """(user-role pids, other pids) for a tool from the monitor's latest poll."""
        pids = set(self.monitor.get_pids_for_tool(tool))
        state = next((s for s in self.monitor.current() if s.tool == tool), None)
        roles = state.roles if state else {}
        user = sorted(p for p in pids if roles.get(p) == ROLE_USER)
        return user, sorted(pids - set(user))

    def _still_same_tool(self, proc: Any, tool: str) -> bool:
        name = (proc.name() or "").lower()
        return self.monitor.index.get(name) == tool

    def live_tools(self) -> list[str]:
        return sorted(s.tool for s in self.monitor.current() if s.has_user_process)

    # -- actions

    def suspend(self, tool: str) -> SuspendResult:
        res = SuspendResult(tool)
        if not self.known_tool(tool):
            res.error = "unknown tool"
            log.warning("refusing to suspend unknown tool")
            return res
        user, other = self.user_pids(tool)
        res.skipped = other
        for pid in user:
            try:
                proc = self.psutil.Process(pid)
                if not self._still_same_tool(proc, tool):
                    res.gone.append(pid)
                    continue
                proc.suspend()
            except self.psutil.NoSuchProcess:
                res.gone.append(pid)
            except self.psutil.AccessDenied:
                res.denied.append(pid)
            else:
                with self._lock:
                    self._suspended[pid] = tool
                res.suspended.append(pid)
        log.info("suspend %s: %d suspended, %d denied, %d gone, %d skipped", tool, len(res.suspended),
                 len(res.denied), len(res.gone), len(res.skipped))
        return res

    def suspend_live(self) -> list[SuspendResult]:
        """Suspend every tool that is live right now (has a user-role process)."""
        return [self.suspend(t) for t in self.live_tools()]

    def resume(self, tool: str) -> list[int]:
        with self._lock:
            pids = [p for p, t in self._suspended.items() if t == tool]
        return self._resume(pids)

    def resume_all(self) -> list[int]:
        with self._lock:
            pids = list(self._suspended)
        return self._resume(pids)

    def _resume(self, pids: list[int]) -> list[int]:
        done = []
        for pid in pids:
            try:
                self.psutil.Process(pid).resume()
                done.append(pid)
            except self.psutil.NoSuchProcess:
                done.append(pid)                             # gone: nothing left frozen
            except self.psutil.AccessDenied:
                log.warning("resume denied for a suspended pid; will retry on exit")
                continue
            with self._lock:
                self._suspended.pop(pid, None)
        return done

    @property
    def suspended(self) -> dict[int, str]:
        with self._lock:
            return dict(self._suspended)


# ---------------------------------------------------------------- exit hooks

_instances: list[RemoteControl] = []
_hooks_installed = False


def resume_everything() -> None:
    for rc in list(_instances):
        try:
            rc.resume_all()
        except Exception:
            log.exception("resume_all on exit failed")


def _register(rc: RemoteControl) -> None:
    global _hooks_installed
    _instances.append(rc)
    if _hooks_installed:
        return
    _hooks_installed = True
    atexit.register(resume_everything)
    if threading.current_thread() is not threading.main_thread():
        return                                               # signal handlers only from the main thread
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        prev = signal.getsignal(sig)

        def handler(signum, frame, _prev=prev):
            resume_everything()
            if callable(_prev):
                _prev(signum, frame)
            elif _prev == signal.SIG_DFL:
                raise SystemExit(128 + signum)
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------- self-test

def _status(pid: int) -> str:
    try:
        return psutil.Process(pid).status()
    except psutil.Error as e:
        return type(e).__name__


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m act.remote_control", description="suspend/resume self-test")
    p.add_argument("--test", metavar="EXE", required=True, help="exe name to treat as a remote tool, e.g. notepad.exe")
    p.add_argument("--seconds", type=float, default=5.0, help="how long to keep it suspended")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from capture.processes import ProcessMonitor
    from kavach_config import get_config

    mon = ProcessMonitor(get_config(), extra_exe_names=[args.test])
    mon.poll()
    tool = mon.index[args.test.lower()]
    user, other = RemoteControl(mon, install_exit_hooks=False).user_pids(tool)
    if not user:
        print(f"no user-role {args.test} running - open it first")
        return 1
    rc = RemoteControl(mon)
    print(f"{tool}: user pids {user}, not touched {other}")
    print(f"before: {[(p, _status(p)) for p in user]}")
    res = rc.suspend(tool)
    print(f"suspend: suspended={res.suspended} denied={res.denied} gone={res.gone}")
    print(f"frozen: {[(p, _status(p)) for p in res.suspended]}  (try typing in it now)")
    time.sleep(args.seconds)
    print(f"resume_all: {rc.resume_all()}")
    print(f"after: {[(p, _status(p)) for p in user]}")
    return 0 if res.suspended else 1


if __name__ == "__main__":
    raise SystemExit(_main())
