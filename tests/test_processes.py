from collections import namedtuple

import psutil
import pytest

from capture import processes as P

ANYDESK_IDLE = ["--service", "--control", "--tray", "--backend"]
KNOWN = [
    {"name": "AnyDesk", "exe_names": ["AnyDesk.exe"], "idle_args": ANYDESK_IDLE},
    {"name": "TeamViewer", "exe_names": ["TeamViewer.exe", "TeamViewer_Service.exe"], "idle_args": []},
]
ME = {"ARYA"}
AD_EXE = r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe"
Addr = namedtuple("Addr", "ip port")
Conn = namedtuple("Conn", "status raddr")


def proc(pid, name, username="PC\\arya", exe=None, cmdline=None):
    return {"pid": pid, "name": name, "exe": exe, "create_time": 0.0, "username": username, "cmdline": cmdline}


@pytest.fixture
def index():
    return P.build_exe_index(KNOWN, ["notepad.exe"])


def test_match_is_case_insensitive(index):
    assert P.match_process(proc(1, "anydesk.EXE"), index) == "AnyDesk"
    assert P.match_process(proc(2, "TEAMVIEWER_SERVICE.exe"), index) == "TeamViewer"
    assert P.match_process(proc(3, "Notepad.exe"), index) == "notepad.exe"
    assert P.match_process(proc(4, "chrome.exe"), index) is None
    assert P.match_process(proc(5, None), index) is None


def test_match_falls_back_to_exe_path(index):
    assert P.match_process(proc(1, "renamed", exe=r"C:\Program Files\AnyDesk\AnyDesk.exe"), index) == "AnyDesk"


@pytest.mark.parametrize("username,exe_name,expected", [
    ("NT AUTHORITY\\SYSTEM", "AnyDesk.exe", "service"),
    ("NT AUTHORITY\\LOCAL SERVICE", "AnyDesk.exe", "service"),
    ("NT AUTHORITY\\NETWORK SERVICE", "AnyDesk.exe", "service"),
    (None, "AnyDesk.exe", "service"),                       # AccessDenied -> None
    ("PC\\arya", "TeamViewer_Service.exe", "service"),       # exe name contains Service
    ("PC\\Arya", "AnyDesk.exe", "user"),
    ("arya", "AnyDesk.exe", "user"),
    ("PC\\someoneelse", "AnyDesk.exe", "other"),
])
def test_classify_role(username, exe_name, expected):
    assert P.classify_role(proc(1, exe_name, username=username), ME) == expected


def test_loopback_and_connections():
    assert P.is_loopback("127.0.0.1") and P.is_loopback("::1") and P.is_loopback("::ffff:127.0.0.1")
    assert not P.is_loopback("52.1.2.3")
    est = psutil.CONN_ESTABLISHED
    assert P.has_external_established([Conn(est, Addr("52.1.2.3", 443))])
    assert not P.has_external_established([Conn(est, Addr("127.0.0.1", 7070))])
    assert not P.has_external_established([Conn(psutil.CONN_LISTEN, ())])
    assert not P.has_external_established([Conn(psutil.CONN_SYN_SENT, Addr("52.1.2.3", 443))])
    assert P.combine_session([False, None]) is None
    assert P.combine_session([None, True]) is True
    assert P.combine_session([False]) is False


class FakeWorld:
    def __init__(self):
        self.procs = []
        self.conns = {}

    def iter(self):
        return list(self.procs)

    def check(self, pid):
        return self.conns.get(pid, False)


@pytest.fixture
def world_monitor():
    world = FakeWorld()
    cfg = {"processes": {"poll_interval_s": 0.01, "known_tools": KNOWN, "extra_exe_names": []}}
    mon = P.ProcessMonitor(cfg, process_iter=world.iter, conn_checker=world.check)
    mon._current_users = ME
    return world, mon


def types(events):
    return [(e.type, e.tool) for e in events]


def test_service_only_is_not_live(world_monitor):
    world, mon = world_monitor
    world.procs = [proc(10, "AnyDesk.exe", username="NT AUTHORITY\\SYSTEM")]
    assert types(mon.poll()) == [("tool_started", "AnyDesk")]
    assert not mon.is_remote_access_live()
    (state,) = mon.current()
    assert state.roles == {10: "service"} and state.active_session is False


def test_event_lifecycle(world_monitor):
    world, mon = world_monitor
    world.procs = [proc(10, "AnyDesk.exe", username="NT AUTHORITY\\SYSTEM"), proc(11, "AnyDesk.exe")]
    mon.poll()
    assert mon.is_remote_access_live()
    assert mon.get_pids_for_tool("AnyDesk") == [10, 11]
    assert mon.get_pids_for_tool("TeamViewer") == []

    world.conns[11] = True
    assert types(mon.poll()) == [("session_active", "AnyDesk")]
    assert mon.poll() == []                                   # no repeat while unchanged

    world.conns[11] = False
    assert types(mon.poll()) == [("session_ended", "AnyDesk")]

    world.conns[11] = True
    mon.poll()
    world.procs = []
    assert types(mon.poll()) == [("session_ended", "AnyDesk"), ("tool_stopped", "AnyDesk"), ("live_ended", "AnyDesk")]
    assert mon.current() == [] and not mon.is_remote_access_live()

    drained = types(mon.drain_events())
    assert drained[:2] == [("tool_started", "AnyDesk"), ("live_started", "AnyDesk")]
    assert drained[-1] == ("live_ended", "AnyDesk")


def test_unknown_connection_state(world_monitor):
    world, mon = world_monitor
    world.procs = [proc(11, "AnyDesk.exe")]
    world.conns[11] = None
    mon.poll()
    assert mon.current()[0].active_session is None


def test_scan_survives_bad_process_and_callback(world_monitor):
    world, mon = world_monitor

    def bad_iter():
        yield {"name": "AnyDesk.exe"}                         # missing pid
        yield proc(12, "AnyDesk.exe")

    mon._process_iter = bad_iter
    mon.callback = lambda ev: 1 / 0
    assert types(mon.poll()) == [("tool_started", "AnyDesk"), ("live_started", "AnyDesk")]
    assert mon.get_pids_for_tool("AnyDesk") == [12]


def test_thread_start_stop(world_monitor):
    world, mon = world_monitor
    world.procs = [proc(11, "AnyDesk.exe")]
    mon.start()
    ev = mon.events.get(timeout=2)
    mon.stop()
    assert ev.type == "tool_started"


# ---------------------------------------------------------------- tray role (idle_args)

def test_build_idle_args():
    assert P.build_idle_args(KNOWN) == {"AnyDesk": tuple(ANYDESK_IDLE), "TeamViewer": ()}
    assert P.build_idle_args([{"name": "X", "exe_names": ["x.exe"]}]) == {"X": ()}


@pytest.mark.parametrize("cmdline,expected", [
    ([AD_EXE, "--control"], True),
    ([AD_EXE, "--TRAY"], True),
    ([AD_EXE, "--backend=1"], True),
    ([AD_EXE, "--service", "--foo"], True),
    ([AD_EXE], False),                                     # plain launch = session window
    ([AD_EXE, "--controller"], False),                     # prefix is not a match
    ([r"C:\tools\--tray\AnyDesk.exe"], False),             # argv[0] is ignored
    (None, False),                                         # AccessDenied
    ([], False),
])
def test_has_idle_arg(cmdline, expected):
    assert P.has_idle_arg(cmdline, ANYDESK_IDLE) is expected


@pytest.mark.parametrize("username,cmdline,idle,expected", [
    ("PC\\arya", [AD_EXE, "--control"], ANYDESK_IDLE, "tray"),
    ("PC\\arya", [AD_EXE, "--tray"], ANYDESK_IDLE, "tray"),
    ("PC\\arya", [AD_EXE], ANYDESK_IDLE, "user"),
    ("PC\\arya", None, ANYDESK_IDLE, "user"),              # unreadable cmdline can't hide a session
    ("PC\\arya", [AD_EXE, "--control"], [], "user"),       # tool without idle_args
    ("NT AUTHORITY\\SYSTEM", [AD_EXE, "--service"], ANYDESK_IDLE, "service"),
    ("PC\\someoneelse", [AD_EXE, "--tray"], ANYDESK_IDLE, "other"),
])
def test_classify_role_tray(username, cmdline, idle, expected):
    info = proc(1, "AnyDesk.exe", username=username, cmdline=cmdline)
    assert P.classify_role(info, ME, idle) == expected


def test_installed_anydesk_is_not_live(world_monitor):
    world, mon = world_monitor
    world.procs = [
        proc(10, "AnyDesk.exe", username="NT AUTHORITY\\SYSTEM", cmdline=[AD_EXE, "--service"]),
        proc(11, "AnyDesk.exe", cmdline=[AD_EXE, "--control"]),
        proc(12, "AnyDesk.exe", cmdline=[AD_EXE, "--tray"]),
    ]
    world.conns[11] = True                                 # tray/control keeps a relay connection open
    assert types(mon.poll()) == [("tool_started", "AnyDesk")]
    (state,) = mon.current()
    assert state.roles == {10: "service", 11: "tray", 12: "tray"}
    assert state.cmdlines[11] == [AD_EXE, "--control"]
    assert state.active_session is False                   # connections only checked on user-role pids
    assert not mon.is_remote_access_live()


def test_session_window_makes_anydesk_live(world_monitor):
    world, mon = world_monitor
    idle = [
        proc(10, "AnyDesk.exe", username="NT AUTHORITY\\SYSTEM", cmdline=[AD_EXE, "--service"]),
        proc(11, "AnyDesk.exe", cmdline=[AD_EXE, "--control"]),
    ]
    world.procs = idle
    mon.poll()
    assert not mon.is_remote_access_live()

    world.procs = idle + [proc(13, "AnyDesk.exe", cmdline=[AD_EXE])]
    world.conns[13] = True
    assert types(mon.poll()) == [("session_active", "AnyDesk"), ("live_started", "AnyDesk")]
    assert mon.is_remote_access_live()
    assert mon.current()[0].roles[13] == "user"

    world.procs = idle
    assert types(mon.poll()) == [("session_ended", "AnyDesk"), ("live_ended", "AnyDesk")]
    assert not mon.is_remote_access_live()


def test_live_events_follow_is_remote_access_live(world_monitor):
    world, mon = world_monitor
    tray = proc(11, "AnyDesk.exe", cmdline=[AD_EXE, "--control"])
    world.procs = [tray]
    assert types(mon.poll()) == [("tool_started", "AnyDesk")]      # tray only: not live, no live event

    # user-role process with no outside connection: live, but no session_active
    world.procs = [tray, proc(13, "AnyDesk.exe", cmdline=[AD_EXE])]
    evs = mon.poll()
    assert types(evs) == [("live_started", "AnyDesk")]
    assert evs[0].state.has_user_process and mon.is_remote_access_live()
    assert mon.poll() == []                                         # no repeat while still live

    # a second tool going live while already live: no second live_started
    world.procs.append(proc(20, "TeamViewer.exe"))
    assert types(mon.poll()) == [("tool_started", "TeamViewer")]

    # AnyDesk session window closes but TeamViewer still live: no live_ended
    world.procs = [tray, proc(20, "TeamViewer.exe")]
    assert mon.poll() == []
    assert mon.is_remote_access_live()

    # last user process gone: live_ended, even though AnyDesk tray keeps running
    world.procs = [tray]
    evs = mon.poll()
    assert types(evs) == [("tool_stopped", "TeamViewer"), ("live_ended", "TeamViewer")]
    assert not mon.is_remote_access_live()


def test_cmdline_shown_in_output():
    state = P.RemoteToolState("AnyDesk", [11, 12], {11: "tray", 12: "user"}, True, None, 0.0, 0.0,
                              cmdlines={11: [AD_EXE, "--control"], 12: None})
    out = P._fmt_state(state)
    assert r'"C:\Program Files (x86)\AnyDesk\AnyDesk.exe" --control' in out
    assert "<cmdline unreadable>" in out
    assert "live=True" in out
