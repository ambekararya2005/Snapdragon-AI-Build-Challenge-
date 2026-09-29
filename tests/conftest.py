"""Shared pytest hooks. tests/test_risk.py appends its scenario results to TRUTH_TABLE, printed at the end."""

import pytest

TRUTH_TABLE: list[dict] = []


@pytest.fixture(scope="session")
def tk_root():
    """One Tk interpreter for all overlay tests: creating Tk() many times in one process fails
    intermittently here (Tcl init scripts not found, venv under OneDrive), so tests share one."""
    tk = pytest.importorskip("tkinter")
    root = None
    for _ in range(3):
        try:
            root = tk.Tk()
            break
        except tk.TclError as e:
            err = e
    if root is None:
        pytest.skip(f"no display: {err}")
    root.withdraw()
    yield root
    root.destroy()


_COLUMNS = (("scenario", 22), ("expect", 16), ("final", 5), ("max", 4), ("band", 7), ("alerts", 6),
            ("trig@s", 6), ("dec_ms", 6), ("ok", 3), ("reasons (final)", 0))


def pytest_terminal_summary(terminalreporter):
    if not TRUTH_TABLE:
        return
    tr = terminalreporter
    tr.write_sep("=", "fusion truth table (tests/test_risk.py)")
    tr.write_line("  ".join(f"{name:<{w}}" if w else name for name, w in _COLUMNS))
    for row in TRUTH_TABLE:
        cells = [row["scenario"], row["expect"], row["final"], row["max"], row["band"], row["alerts"], row["trig_s"],
                 row["dec_ms"], "yes" if row["ok"] else "NO", row["reasons"]]
        tr.write_line("  ".join(f"{str(c):<{w}}" if w else str(c) for c, (_, w) in zip(cells, _COLUMNS)))
