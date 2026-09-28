from pathlib import Path
from types import SimpleNamespace

import pytest

from detect import screen_classifier as C
from models.ocr import OcrLine, OcrResult

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "screen"


def load(name: str) -> tuple[str, dict]:
    title, _, body = (FIXTURES / f"{name}.txt").read_text(encoding="utf-8").partition("\n")
    return body, {"title": title.removeprefix("TITLE:").strip(), "process_name": "chrome.exe"}


def run(name: str, **cfg) -> C.ScreenLabel:
    body, window = load(name)
    return C.classify(body, window, {**C.DEFAULTS, **cfg} if cfg else None)


# ---------------------------------------------------------------- fixtures

def test_bank_transfer():
    r = run("bank_transfer")
    assert {"bank", "otp_card"} <= set(r.present)
    assert "fake_alert" not in r.present
    assert {"pattern:ifsc", "pattern:amount"} <= set(r.evidence)
    assert "money_move" in r.screen_tactics                     # "Refund Desk" beneficiary


def test_fake_kyc():
    r = run("fake_kyc")
    assert r.top_label == "bank"
    assert {"threat", "urgency", "authority", "secrecy"} <= r.screen_tactics
    assert "fake_alert" not in r.present


def test_fake_virus_alert():
    r = run("fake_virus_alert")
    assert r.top_label == "fake_alert" and r.labels["fake_alert"] == 1.0
    assert {"combo:brand_with_phone", "pattern:phone_tollfree"} <= set(r.evidence)   # 1-8OO-555-0100 fixed
    assert {"threat", "urgency", "money_move"} <= r.screen_tactics


def test_fake_electricity_bill():
    r = run("fake_electricity_bill")
    assert r.top_label == "upi_payment"
    assert {"threat", "urgency"} <= r.screen_tactics
    assert "pattern:upi_id" in r.evidence and "fake_alert" not in r.present


def test_gmail_inbox_is_normal():
    r = run("gmail_inbox")
    assert r.top_label == "normal" and r.present == [] and not r.screen_tactics
    assert "pattern:upi_id" not in r.evidence                   # an email address is not a UPI id


def test_amazon_checkout_is_payment_not_alert():
    r = run("amazon_checkout")
    assert {"upi_payment", "otp_card"} <= set(r.present)
    assert "fake_alert" not in r.present and r.labels["fake_alert"] == 0
    assert "pattern:card_number" in r.evidence                  # 4111 1111 1111 1111 passes Luhn
    assert not r.screen_tactics


def test_news_article_is_not_fake_alert():
    r = run("news_digital_arrest")
    assert r.news_context and "news_context" in r.evidence
    assert "fake_alert" not in r.present and r.top_label == "normal"
    assert not r.screen_tactics                                 # quoted threats/authorities damped
    assert r.tactic_scores["threat"] < 0.5


def test_news_guard_is_configurable():
    r = run("news_digital_arrest", news_factor=1.0)
    assert "threat" in r.screen_tactics                         # without damping the quotes would count


def test_evidence_never_contains_screen_text():
    for f in FIXTURES.glob("*.txt"):
        body, window = load(f.stem)
        ev = " ".join(C.classify(body, window).evidence).lower()
        for secret in ("rahul", "49,999", "demo0001234", "4111", "98765", "msedclbilling", "1-8oo-555", "priya"):
            assert secret not in ev, (f.name, secret)


# ---------------------------------------------------------------- normalisation / matching

def test_canon_folds_confusables_and_l():
    assert C.canon("Enter 0TP\n  NOW") == "enter otp now"
    assert C.canon("Benef1ciary") == C.canon("Beneficiary")
    assert C.canon("b1ocked") == C.canon("blocked")


def test_fix_digit_runs_only_inside_numbers():
    assert C.fix_digit_runs("₹ 2,48,35O.00") == "₹ 2,48,350.00"
    assert C.fix_digit_runs("1-8OO-555-0100") == "1-800-555-0100"
    assert C.fix_digit_runs("4l11 1111") == "4111 1111"
    assert C.fix_digit_runs("DEMO0001234 Oil Iron") == "DEMO0001234 Oil Iron"      # words untouched


def test_space_stripped_matching():
    r = C.classify("NETBANKING  FUNDTRANSFER  BENEFICIARYNAME  AVAILABLEBALANCE")
    assert r.labels["bank"] >= 0.5
    assert "bank:netbanking:fund_transfer" in r.evidence
    r = C.classify("hotpot recipe")                              # short terms need word boundaries
    assert r.labels["otp_card"] == 0


# ---------------------------------------------------------------- patterns

@pytest.mark.parametrize("text,expected", [
    ("IFSC  HDFC0001234", True),
    ("IFSC  SBIN0O12345", True),                  # O read for 0 in the 5th position... and in branch code
    ("IFSC: DEMOO001234", True),
    ("HELLO WORLD ABCDEFGHIJK", False),
    ("ABCD1234567", False),                       # 5th char must be 0
])
def test_ifsc(text, expected):
    assert ("ifsc" in C.detect_patterns(text)) is expected


@pytest.mark.parametrize("text,expected", [
    ("pay to rahul.v@oksbi", True), ("9876543210@ybl", True), ("msedcl@paytm now", True),
    ("mail me at someone@example.com", False), ("a@b", False),
])
def test_upi_id(text, expected):
    assert ("upi_id" in C.detect_patterns(text)) is expected


@pytest.mark.parametrize("text,expected", [
    ("4111 1111 1111 1111", True), ("5500-0000-0000-0004", True), ("4111 1111 1111 1112", False),
    ("0000 0000 0000 0000", False), ("Consumer No 170019284512", False),
])
def test_card_luhn(text, expected):
    assert ("card_number" in C.detect_patterns(text)) is expected


@pytest.mark.parametrize("text,found", [
    ("Toll free: 1800 425 3800", {"phone_tollfree", "phone_any"}),
    ("call 1-800-555-0100", {"phone_tollfree", "phone_any"}),
    ("+91 98765 43210", {"phone_mobile", "phone_any"}),
    ("09876543210", {"phone_mobile", "phone_any"}),
    ("pin 560001 order 123456", set()),
])
def test_phones(text, found):
    assert C.detect_patterns(text) & {"phone_tollfree", "phone_mobile", "phone_any"} == found


@pytest.mark.parametrize("text", ["₹ 1,250.00", "Rs. 499", "INR 10,000", "rs 5"])
def test_amount(text):
    assert "amount" in C.detect_patterns(text)


# ---------------------------------------------------------------- backend independence / inputs

def test_same_result_from_str_ocrresult_and_lines_only():
    body, window = load("fake_virus_alert")
    lines = [OcrLine(t, 0.9, [[0, i * 20], [100, i * 20], [100, i * 20 + 15], [0, i * 20 + 15]])
             for i, t in enumerate(body.splitlines())]
    as_result = OcrResult(lines, body, 1.0, 1.0, 2.0, "native", "CPUExecutionProvider")
    lines_only = SimpleNamespace(lines=lines)
    a, b, c = C.classify(body, window), C.classify(as_result, window), C.classify(lines_only, window)
    assert a.labels == b.labels == c.labels and a.screen_tactics == b.screen_tactics == c.screen_tactics


def test_window_title_and_process_are_used():
    r = C.classify("", SimpleNamespace(title="Windows Defender - Virus detected", process_name="msedge.exe"))
    assert r.labels["fake_alert"] > 0 and "ctx:browser" in r.evidence
    assert C.classify(None, None).top_label == "normal"


def test_format_label_has_no_text():
    body, window = load("bank_transfer")
    line = C.format_label(C.classify(body, window))
    assert line.startswith("top=") and "Rahul" not in line
