"""Poll running processes for known remote-access tools (psutil).

Each matched process gets a role: "service" (SYSTEM / LOCAL SERVICE / NETWORK SERVICE, unreadable
user, or exe name containing "Service"), "tray" (current user, but cmdline has one of the tool's
config `idle_args`, e.g. AnyDesk --tray / --control), "user" (current user, no idle args) or "other"
(another user). Installed AnyDesk/TeamViewer keep service and tray processes running all day, so only
a user-role process counts as remote access being live. An ESTABLISHED non-loopback TCP connection on
a user-role process marks an active session (extra evidence). Only local socket state is read;
nothing is sent. Cmdlines are read for matched tool processes only and are never logged; CLI output
shows only the exe name and flag names unless privacy.debug_show_text is true (see kavach_privacy).

Events: tool_started / tool_stopped (per tool), session_active / session_ended (per tool),
live_started / live_ended (when is_remote_access_live() flips; tool = the tool that caused it).

    mon = ProcessMonitor(get_config(), callback=print)
    mon.start()
    mon.is_remote_access_live()   # fusion signal
    mon.current()                 # [RemoteToolState, ...]

Self-test / CLI:
    python -m capture.processes --once
    python -m capture.processes --watch --extra notepad.exe
"""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import PureWindowsPath
from typing import Any, Callable, Iterable, Mapping

import psutil

from kavach_privacy import redact_cmdline

log =logging.getLogger("capture.processes")

PROC_ATTRS = ["pid", "name", "exe", "create_time", "username", "cmdline"]
SERVICE_ACCOUNTS = {"SYSTEM", "LOCAL SERVICE", "NETWORK SERVICE", "LOCALSERVICE", "NETWORKSERVICE"}
ROLE_SERVICE, ROLE_TRAY, ROLE_USER, ROLE_OTHER = "service", "tray", "user", "other"
EVENT_TYPES = ("tool_started", "tool_stopped", "session_active", "session_ended", "live_started", "live_ended")


@dataclass
class RemoteToolState:
    tool: str
    pids: list[int]
    roles: dict[int, str]              # pid -> service | tray | user | other
    has_user_process: bool
    active_session: bool | None        # None = unknown (connections unreadable)
    first_seen: float
    last_seen: float
    cmdlines: dict[int, list[str] | None] = field(default_factory=dict, repr=False)  # None = AccessDenied

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessEvent:
    type: str                          # one of EVENT_TYPES
    tool: str
    ts: float
    state: RemoteToolState | None = field(default=None, repr=False)


# ---------------------------------------------------------------- pure helpers (unit-tested)

def build_exe_index(known_tools: Iterable[Mapping[str, Any]], extra_exe_names: Iterable[str] = ()) -> dict[str, str]:
    """Lower-cased exe name -> tool name. Extra exe names map to themselves."""
    index: dict[str, str] = {}
    for tool in known_tools:
        for exe in tool.get("exe_names", []):
            index[exe.lower()] = tool["name"]
    for exe in extra_exe_names:
        index.setdefault(exe.lower(), exe)
    return index


def build_idle_args(known_tools: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Tool name -> lower-cased cmdline args that mark an idle (installed, not live) process."""
    return {t["name"]: tuple(a.lower() for a in (t.get("idle_args") or [])) for t in known_tools}


def has_idle_arg(cmdline: Iterable[str] | None, idle_args: Iterable[str]) -> bool:
    """True if any argument (argv[1:]) equals an idle arg, or is `<idle arg>=value`. Case-insensitive."""
    idle = tuple(idle_args)
    if not cmdline or not idle:
        return False
    for arg in list(cmdline)[1:]:
        a = arg.strip().lower()
        if any(a == i or a.startswith(i + "=") for i in idle):
            return True
    return False


def _exe_basename(info: Mapping[str, Any]) -> str | None:
    exe = info.get("exe")
    if exe:
        return PureWindowsPath(exe).name
    return info.get("name")


def match_process(info: Mapping[str, Any], index: Mapping[str, str]) -> str | None:
    """Tool name for a process-info dict (from process_iter attrs), or None."""
    for candidate in (info.get("name"), _exe_basename(info)):
        if candidate and candidate.lower() in index:
            return index[candidate.lower()]
    return None


def _short_user(username: str | None) -> str | None:
    if not username:
        return None
    return username.replace("/", "\\").rsplit("\\", 1)[-1].strip().upper()


def current_user_names() -> set[str]:
    names: set[str] = set()
    for getter in (os.getlogin, getpass.getuser, lambda: psutil.Process().username()):
        try:
            short = _short_user(getter())
        except Exception:
            continue
        if short:
            names.add(short)
    return names


def classify_role(info: Mapping[str, Any], current_users: set[str], idle_args: Iterable[str] = ()) -> str:
    """service | tray | user | other. Unreadable username counts as service (conservative against false alarms).

    A current-user process whose cmdline has an idle arg is "tray". An unreadable cmdline (None) stays
    "user", so a hidden cmdline can't mask a live session.
    """
    user = _short_user(info.get("username"))
    exe = _exe_basename(info) or ""
    if user is None or user in SERVICE_ACCOUNTS or "service" in exe.lower():
        return ROLE_SERVICE
    if user in current_users:
        return ROLE_TRAY if has_idle_arg(info.get("cmdline"), idle_args) else ROLE_USER
    return ROLE_OTHER


def is_loopback(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    return addr.is_loopback or bool(mapped and mapped.is_loopback)


def has_external_established(conns: Iterable[Any]) -> bool:
    for c in conns:
        raddr = getattr(c, "raddr", None)
        if getattr(c, "status", None) == psutil.CONN_ESTABLISHED and raddr and not is_loopback(raddr[0] if isinstance(raddr, tuple) else raddr.ip):
            return True
    return False


def check_pid_connections(pid: int) -> bool | None:
    """True/False if the process has an ESTABLISHED non-loopback TCP connection; None if unreadable."""
    try:
        proc = psutil.Process(pid)
        get = getattr(proc, "net_connections", None) or proc.connections
        return has_external_established(get(kind="tcp"))
    except (psutil.AccessDenied, psutil.ZombieProcess):
        return None
    except psutil.NoSuchProcess:
        return False
    except Exception as e:
        log.debug("connections for pid %d unreadable: %s", pid, type(e).__name__)
        return None


def combine_session(results: Iterable[bool | None]) -> bool | None:
    results = list(results)
    if any(r is True for r in results):
        return True
    if any(r is None for r in results):
        return None
    return False


# ---------------------------------------------------------------- monitor

class ProcessMonitor:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        callback: Callable[[ProcessEvent], None] | None = None,
        extra_exe_names: Iterable[str] = (),
        process_iter: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
        conn_checker: Callable[[int], bool | None] = check_pid_connections,
    ):
        if config is None:
            from kavach_config import get_config
            config = get_config()
        pcfg = config["processes"]
        self.poll_interval_s = float(pcfg.get("poll_interval_s", 2.0))
        self.index = build_exe_index(pcfg.get("known_tools", []), [*pcfg.get("extra_exe_names", []), *extra_exe_names])
        self.idle_args = build_idle_args(pcfg.get("known_tools", []))
        self.callback = callback
        self.events: queue.Queue[ProcessEvent] = queue.Queue(maxsize=1000)
        self._process_iter = process_iter or self._psutil_iter
        self._conn_checker = conn_checker
        self._current_users = current_user_names()
        self._states: dict[str, RemoteToolState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _psutil_iter(self) -> Iterable[Mapping[str, Any]]:
        # Cheap pass on pid+name (~1 ms for 300 procs); exe/username/create_time/cmdline only for
        # matches (reading those for every process costs ~25 ms per poll, and cmdlines of unrelated
        # processes are none of our business).
        # ad_value=None: attrs that raise AccessDenied come back as None instead of aborting.
        for p in psutil.process_iter(["pid", "name"], ad_value=None):
            name = (p.info.get("name") or "").lower()
            if name not in self.index:
                continue
            try:
                yield p.as_dict(PROC_ATTRS, ad_value=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def _scan(self) -> dict[str, dict[int, tuple[str, list[str] | None]]]:
        """tool -> {pid: (role, cmdline)} for matched processes."""
        found: dict[str, dict[int, tuple[str, list[str] | None]]] = {}
        try:
            for info in self._process_iter():
                try:
                    tool = match_process(info, self.index)
                    if tool:
                        role = classify_role(info, self._current_users, self.idle_args.get(tool, ()))
                        cmdline = info.get("cmdline")
                        found.setdefault(tool, {})[int(info["pid"])] = (role, list(cmdline) if cmdline else None)
                except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
                    continue
        except (psutil.Error, OSError) as e:
            log.warning("process scan failed: %s", type(e).__name__)
        return found

    def poll(self) -> list[ProcessEvent]:
        """One scan: update state, emit and return events."""
        now = time.time()
        found = self._scan()
        events: list[ProcessEvent] = []
        with self._lock:
            live_before = {t: s for t, s in self._states.items() if s.has_user_process}
            for tool, procs in found.items():
                roles = {pid: procs[pid][0] for pid in sorted(procs)}
                user_pids = [pid for pid, role in roles.items() if role == ROLE_USER]
                session = combine_session(self._conn_checker(pid) for pid in user_pids) if user_pids else False
                prev = self._states.get(tool)
                state = RemoteToolState(
                    tool=tool, pids=sorted(roles), roles=roles,
                    has_user_process=bool(user_pids), active_session=session,
                    first_seen=prev.first_seen if prev else now, last_seen=now,
                    cmdlines={pid: procs[pid][1] for pid in sorted(procs)},
                )
                self._states[tool] = state
                if prev is None:
                    events.append(ProcessEvent("tool_started", tool, now, state))
                was_active = prev is not None and prev.active_session is True
                if session is True and not was_active:
                    events.append(ProcessEvent("session_active", tool, now, state))
                elif was_active and session is False:
                    events.append(ProcessEvent("session_ended", tool, now, state))
            for tool in [t for t in self._states if t not in found]:
                prev = self._states.pop(tool)
                if prev.active_session is True:
                    events.append(ProcessEvent("session_ended", tool, now, prev))
                events.append(ProcessEvent("tool_stopped", tool, now, prev))
            live_after = {t: s for t, s in self._states.items() if s.has_user_process}
            if live_after and not live_before:
                tool = min(live_after)
                events.append(ProcessEvent("live_started", tool, now, live_after[tool]))
            elif live_before and not live_after:
                tool = min(live_before)
                events.append(ProcessEvent("live_ended", tool, now, self._states.get(tool, live_before[tool])))
        for ev in events:
            self._emit(ev)
        return events

    def _emit(self, ev: ProcessEvent) -> None:
        log.info("%s: %s", ev.type, ev.tool)
        try:
            self.events.put_nowait(ev)
        except queue.Full:
            try:
                self.events.get_nowait()
                self.events.put_nowait(ev)
            except (queue.Empty, queue.Full):
                pass
        if self.callback:
            try:
                self.callback(ev)
            except Exception:
                log.exception("process event callback failed")

    def drain_events(self) -> list[ProcessEvent]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def current(self) -> list[RemoteToolState]:
        with self._lock:
            return list(self._states.values())

    def is_remote_access_live(self) -> bool:
        with self._lock:
            return any(s.has_user_process for s in self._states.values())

    def get_pids_for_tool(self, tool: str) -> list[int]:
        with self._lock:
            state = self._states.get(tool)
            return list(state.pids) if state else []

    # -- thread

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                log.exception("process poll failed")
            self._stop.wait(self.poll_interval_s)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="kavach-processes", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            self._thread = None


def _fmt_state(s: RemoteToolState, indent: str = "    ", show_text: bool | None = None) -> str:
    """Cmdlines are redacted to exe + flag names unless privacy.debug_show_text (or show_text=True)."""
    lines = [f"{s.tool}: live={s.has_user_process} active_session={s.active_session}"]
    for pid, role in s.roles.items():
        lines.append(f"{indent}{pid:>6} {role:<7} {redact_cmdline(s.cmdlines.get(pid), show_text)}")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m capture.processes", description="Detect remote-access tools")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="print one snapshot and exit (default)")
    mode.add_argument("--watch", action="store_true", help="print events live until Ctrl+C")
    p.add_argument("--extra", action="append", default=[], metavar="EXE", help="extra exe name to match, e.g. notepad.exe")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.watch:
        def show(ev: ProcessEvent) -> None:
            ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
            body = _fmt_state(ev.state, indent=" " * 30) if ev.state else ev.tool
            print(f"{ts}  {ev.type:<15} {body}", flush=True)

        mon = ProcessMonitor(callback=show, extra_exe_names=args.extra)
        print(f"watching {len(mon.index)} exe names every {mon.poll_interval_s}s (Ctrl+C to stop)", flush=True)
        mon.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            mon.stop()
        return 0

    mon = ProcessMonitor(extra_exe_names=args.extra)
    t0 = time.perf_counter()
    mon.poll()
    ms = (time.perf_counter() - t0) * 1000
    print(f"scanned for {len(mon.index)} exe names in {ms:.0f} ms; current user(s): {sorted(mon._current_users)}")
    states = mon.current()
    if not states:
        print("no remote-access tools running")
    for s in states:
        print(_fmt_state(s))
    print(f"is_remote_access_live: {mon.is_remote_access_live()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
