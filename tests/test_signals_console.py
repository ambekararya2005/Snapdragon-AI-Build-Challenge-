"""Preview score of scripts/signals_console.py (config.fusion weights, no decay)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from kavach_config import load_config

_spec = importlib.util.spec_from_file_location(
    "signals_console", Path(__file__).resolve().parent.parent / "scripts" / "signals_console.py")
SC = importlib.util.module_from_spec(_spec)
sys.modules["signals_console"] = SC          # dataclasses need the module registered
_spec.loader.exec_module(SC)
SIGNALS = load_config(env={}).fusion.signals


def label(bank=0.0, upi=0.0, otp=0.0, fake=0.0, tactics=()):
    return SimpleNamespace(labels={"bank": bank, "upi_payment": upi, "otp_card": otp, "fake_alert": fake},
                           screen_tactics=frozenset(tactics))


def test_nothing_present():
    assert SC.preview_score(False, None, set(), SIGNALS) == (0, {})


def test_hero_scenario_caps_at_100():
    total, parts = SC.preview_score(True, label(bank=0.9, otp=0.8, tactics={"threat", "urgency", "authority"}),
                                    {"authority", "threat", "secrecy", "urgency", "money_move"}, SIGNALS)
    assert parts == {"remote_tool": 30, "money_screen": 25, "otp_card": 20, "screen_tactics": 30,
                     "call_tactics": 40, "combo_bonus": 20}
    assert total == 100


def test_combo_needs_remote_money_and_a_tactic():
    _, parts = SC.preview_score(True, label(upi=0.7), set(), SIGNALS)
    assert "combo_bonus" not in parts and parts == {"remote_tool": 30, "money_screen": 25}
    total, parts = SC.preview_score(True, label(upi=0.7), {"secrecy"}, SIGNALS)
    assert parts["combo_bonus"] == 20 and total == 30 + 25 + 10 + 20


def test_fake_alert_label_and_call_tactic_cap():
    total, parts = SC.preview_score(False, label(fake=1.0, tactics={"threat"}), {"a", "b", "c", "d", "e"}, SIGNALS)
    assert parts == {"fake_alert_label": 20, "screen_tactics": 15, "call_tactics": 40} and total == 75
