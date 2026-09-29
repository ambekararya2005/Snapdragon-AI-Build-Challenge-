"""Shared pytest hooks. tests/test_risk.py appends its scenario results to TRUTH_TABLE, printed at the end."""

TRUTH_TABLE: list[dict] = []

_COLUMNS = (("scenario", 22), ("expect", 16), ("final", 5), ("max", 4), ("band", 7), ("alerts", 6),
            ("trig@s", 6), ("t2a_ms", 6), ("ok", 3), ("reasons (final)", 0))


def pytest_terminal_summary(terminalreporter):
    if not TRUTH_TABLE:
        return
    tr = terminalreporter
    tr.write_sep("=", "fusion truth table (tests/test_risk.py)")
    tr.write_line("  ".join(f"{name:<{w}}" if w else name for name, w in _COLUMNS))
    for row in TRUTH_TABLE:
        cells = [row["scenario"], row["expect"], row["final"], row["max"], row["band"], row["alerts"], row["trig_s"],
                 row["t2a_ms"], "yes" if row["ok"] else "NO", row["reasons"]]
        tr.write_line("  ".join(f"{str(c):<{w}}" if w else str(c) for c, (_, w) in zip(cells, _COLUMNS)))
