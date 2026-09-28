"""Redaction for anything Kavach prints or logs.

Window titles, OCR text, transcripts and command lines can contain personal data (email subjects,
account numbers, remote-desktop IDs). They are shown only when config privacy.debug_show_text is
true; otherwise use these helpers wherever such text would be printed:

    from kavach_privacy import redact_title, redact_cmdline
    print(f"[{win.process_name}] {redact_title(win.title)}")

Self-test:  python -m kavach_privacy
"""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Iterable

TITLE_HIDDEN = "<title hidden>"
TEXT_HIDDEN = "<text hidden>"


def show_text(show: bool | None = None) -> bool:
    """Explicit `show` wins; otherwise config privacy.debug_show_text (default False)."""
    if show is not None:
        return bool(show)
    from kavach_config import get_config

    return get_config().privacy.get("debug_show_text", False) is True


def redact_title(title: str | None, show: bool | None = None) -> str:
    return (title or "") if show_text(show) else TITLE_HIDDEN


def redact_text(text: str | None, show: bool | None = None) -> str:
    """OCR text / transcripts: hidden entirely unless debug_show_text; the length is kept as a hint."""
    if show_text(show):
        return text or ""
    return f"{TEXT_HIDDEN} ({len(text or '')} chars)"


def redact_cmdline(args: Iterable[str] | None, show: bool | None = None) -> str:
    """Full cmdline if debug_show_text; otherwise exe basename + flag names only (values, paths, IDs hidden).

    ["C:\\...\\AnyDesk.exe", "--control"]          -> "AnyDesk.exe --control"
    ["C:\\...\\AnyDesk.exe", "123456789"]          -> "AnyDesk.exe <1 arg hidden>"
    ["C:\\...\\x.exe", "--url=https://a.b", "f"]   -> "x.exe --url=<hidden> <1 arg hidden>"
    """
    if args is None:
        return "<cmdline unreadable>"
    args = list(args)
    if not args:
        return ""
    if show_text(show):
        import subprocess

        return subprocess.list2cmdline(args)
    parts = [PureWindowsPath(args[0]).name]
    hidden = 0
    for arg in args[1:]:
        if arg.startswith(("-", "/")) and len(arg) > 1 and not _looks_like_path(arg):
            flag, sep, _ = arg.partition("=")
            parts.append(f"{flag}=<hidden>" if sep else flag)
        else:
            hidden += 1
    if hidden:
        parts.append(f"<{hidden} arg{'s' if hidden != 1 else ''} hidden>")
    return " ".join(parts)


def _looks_like_path(arg: str) -> bool:
    # "/foo" is a Windows switch; "/home/x/y", "//server/share" or "-C:\x" is a path.
    # Only the part before "=" is checked; the value is hidden anyway.
    flag = arg.partition("=")[0]
    return flag.startswith("//") or flag.count("/") > 1 or "\\" in flag or ":" in flag


if __name__ == "__main__":
    print(f"debug_show_text: {show_text()}")
    print(redact_title("Inbox (3) - someone@example.com"))
    print(redact_text("Your OTP is 123456"))
    print(redact_cmdline([r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe", "--control"]))
    print(redact_cmdline([r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe", "123456789"]))
