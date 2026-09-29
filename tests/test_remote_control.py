"""act.remote_control with a fake psutil and a fake monitor: no real process is ever touched."""

from types import SimpleNamespace

from act.remote_control import RemoteControl


class NoSuchProcess(Exception):
    pass


class AccessDenied(Exception):
    pass


class FakePsutil:
    NoSuchProcess = NoSuchProcess
    AccessDenied = AccessDenied

    def __init__(self, procs):
        self.procs = procs                     # pid -> FakeProc
        self.calls = []                        # (action, pid)

    def Process(self, pid):
        if pid not in self.procs:
            raise NoSuchProcess(pid)
        return self.procs[pid]


class FakeProc:
    def __init__(self, ps, pid, name, deny=False):
        self.ps, self.pid, self._name, self.deny = ps, pid, name, deny

    def name(self):
        return self._name

    def suspend(self):
        if self.deny:
            raise AccessDenied(self.pid)
        self.ps.calls.append(("suspend", self.pid))

    def resume(self):
        self.ps.calls.append(("resume", self.pid))

    def kill(self):                            # must never be called
        raise AssertionError("kill() called")

    terminate = kill


class FakeMonitor:
    def __init__(self, states):
        self.index = {"anydesk.exe": "AnyDesk", "teamviewer.exe": "TeamViewer", "teamviewer_service.exe": "TeamViewer"}
        self.states = states

    def current(self):
        return list(self.states)

    def get_pids_for_tool(self, tool):
        s = next((s for s in self.states if s.tool == tool), None)
        return list(s.pids) if s else []


def state(tool, roles):
    return SimpleNamespace(tool=tool, pids=sorted(roles), roles=roles,
                           has_user_process=any(r == "user" for r in roles.values()))


def setup(deny=()):
    ps = FakePsutil({})
    for pid, name in {10: "AnyDesk.exe", 11: "AnyDesk.exe", 12: "AnyDesk.exe", 13: "AnyDesk.exe",
                      20: "TeamViewer_Service.exe", 21: "TeamViewer.exe", 99: "notepad.exe"}.items():
        ps.procs[pid] = FakeProc(ps, pid, name, deny=pid in deny)
    mon = FakeMonitor([state("AnyDesk", {10: "service", 11: "tray", 12: "user", 13: "other"}),
                       state("TeamViewer", {20: "service", 21: "user"})])
    return ps, mon, RemoteControl(mon, psutil_mod=ps, install_exit_hooks=False)


def test_suspends_only_user_role_pids_of_the_tool():
    ps, _, rc = setup()
    res = rc.suspend("AnyDesk")
    assert res.suspended == [12] and res.skipped == [10, 11, 13] and res.ok
    assert ps.calls == [("suspend", 12)]
    assert rc.suspended == {12: "AnyDesk"}


def test_never_touches_service_or_unknown_tools():
    ps, _, rc = setup()
    res = rc.suspend("TeamViewer")
    assert res.suspended == [21] and 20 in res.skipped
    bad = rc.suspend("notepad.exe")
    assert bad.error == "unknown tool" and not bad.suspended
    assert ("suspend", 20) not in ps.calls and ("suspend", 99) not in ps.calls


def test_pid_reused_by_another_program_is_skipped():
    ps, _, rc = setup()
    ps.procs[12]._name = "notepad.exe"          # AnyDesk exited, pid reused since the last poll
    res = rc.suspend("AnyDesk")
    assert res.gone == [12] and not res.suspended and not ps.calls
    assert res.could_not_pause


def test_access_denied_is_reported_not_raised():
    ps, _, rc = setup(deny={12})
    res = rc.suspend("AnyDesk")
    assert res.denied == [12] and not res.suspended and not res.ok and res.could_not_pause
    assert rc.suspended == {}


def test_resume_all_undoes_everything_and_forgets():
    ps, _, rc = setup()
    results = rc.suspend_live()
    assert [r.tool for r in results] == ["AnyDesk", "TeamViewer"]
    assert sorted(rc.resume_all()) == [12, 21]
    assert sorted(p for a, p in ps.calls if a == "resume") == [12, 21]
    assert rc.suspended == {} and rc.resume_all() == []


def test_resume_all_on_exit_hook_covers_registered_instances():
    import act.remote_control as RC

    ps, mon, _ = setup()
    rc = RemoteControl(mon, psutil_mod=ps, install_exit_hooks=False)
    RC._instances.append(rc)
    try:
        rc.suspend("AnyDesk")
        RC.resume_everything()                  # what atexit / SIGINT / SIGTERM run
        assert ("resume", 12) in ps.calls and rc.suspended == {}
    finally:
        RC._instances.remove(rc)
