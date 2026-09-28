import pytest

import kavach_config
import kavach_privacy as K

AD = r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe"


def test_title():
    assert K.redact_title("Inbox - me@example.com", show=False) == "<title hidden>"
    assert K.redact_title("Inbox - me@example.com", show=True) == "Inbox - me@example.com"
    assert K.redact_title(None, show=True) == ""


def test_text():
    assert K.redact_text("OTP 123456", show=False) == "<text hidden> (10 chars)"
    assert K.redact_text("OTP 123456", show=True) == "OTP 123456"


@pytest.mark.parametrize("args,expected", [
    ([AD, "--control"], "AnyDesk.exe --control"),
    ([AD, "123456789"], "AnyDesk.exe <1 arg hidden>"),
    ([AD, "--url=https://evil.example/x", "a", "b"], "AnyDesk.exe --url=<hidden> <2 args hidden>"),
    ([AD, "/S", "/home/user/file"], "AnyDesk.exe /S <1 arg hidden>"),
    ([AD, "--plain", r"-C:\Users\x"], "AnyDesk.exe --plain <1 arg hidden>"),
    ([AD], "AnyDesk.exe"),
    (None, "<cmdline unreadable>"),
    ([], ""),
])
def test_cmdline_hidden(args, expected):
    assert K.redact_cmdline(args, show=False) == expected


def test_cmdline_shown():
    assert K.redact_cmdline([AD, "123456789"], show=True) == f'"{AD}" 123456789'


def test_default_follows_config(monkeypatch):
    cfg = kavach_config.load_config(env={})
    monkeypatch.setattr(kavach_config, "get_config", lambda: cfg)
    cfg.privacy["debug_show_text"] = False
    assert K.redact_title("secret") == "<title hidden>"
    cfg.privacy["debug_show_text"] = True
    assert K.redact_title("secret") == "secret"
