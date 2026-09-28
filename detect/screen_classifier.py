"""Classify OCR'd screen text into signals: money_screen, otp_card, fake_alert tactics.

F3: multi-label screen classifier (rule/lexicon baseline) over OCR text + window title + process name.

    label = classify(ocr_result, window_info)
    label.top_label        # bank | upi_payment | otp_card | fake_alert | normal
    label.labels           # {label: score 0-1}
    label.screen_tactics   # subset of {authority, threat, secrecy, urgency, money_move}
    label.evidence         # lexicon ids / pattern names only - never screen text

Backend-independent: only needs `.full_text` (or `.lines[*].text`) from the OCR result, so rapidocr
and native results work the same; a plain string also works.

Text is normalised twice:
  - keywords: fold_confusables() (0TP -> OTP) -> lowercase -> l->i, spaces collapsed; every term is
    matched on word boundaries, and long terms also against a space-stripped copy (OCR drops spaces).
  - digit regexes (IFSC, UPI id, card + Luhn, phones, amounts): the original text with only O->0 and
    l/I/| -> 1 inside digit runs.

Self-test:  python -m detect.screen_classifier          # classifies tests/fixtures/screen/*.txt
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from models.ocr import fold_confusables

ROOT = Path(__file__).resolve().parent.parent
LABELS = ("bank", "upi_payment", "otp_card", "fake_alert")
TACTICS = ("authority", "threat", "secrecy", "urgency", "money_move")
TOP_PRIORITY = ("fake_alert", "otp_card", "upi_payment", "bank")  # tie-break order
BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"}
DEFAULTS = {"lexicon": "detect/lexicons/screen.yaml", "label_threshold": 0.5, "tactic_threshold": 0.5,
            "news_min_hits": 2, "news_factor": 0.3, "compact_min_len": 6,
            "news_reduces": ["fake_alert", *TACTICS]}


@dataclass
class ScreenLabel:
    labels: dict[str, float]                 # bank, upi_payment, otp_card, fake_alert, normal -> 0..1
    top_label: str
    screen_tactics: frozenset[str]
    evidence: list[str]                      # ids only, e.g. "bank:netbanking:ifsc", "pattern:ifsc"
    ms: float
    present: list[str] = field(default_factory=list)          # labels >= label_threshold
    tactic_scores: dict[str, float] = field(default_factory=dict)
    news_context: bool = False


# ---------------------------------------------------------------- normalisation

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def canon(text: str) -> str:
    """Keyword form: confusables folded (0TP -> OTP), lowercase, 'l' -> 'i', whitespace collapsed."""
    folded = fold_confusables(_WS.sub(" ", text or "").strip())
    return folded.lower().replace("l", "i")


def compact(canon_text: str) -> str:
    return _NON_ALNUM.sub("", canon_text)


_DIGIT_RUN = re.compile(r"(?<![A-Za-z])[0-9OoIl|](?:[0-9OoIl|,.\-]*[0-9OoIl|])?(?![A-Za-z])")
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})


def fix_digit_runs(text: str) -> str:
    """O->0 and l/I/| -> 1, only inside runs that already contain >= 2 real digits (4O12 -> 4012)."""
    def repl(m: re.Match) -> str:
        run = m.group(0)
        return run.translate(_DIGIT_FIX) if sum(c.isdigit() for c in run) >= 2 else run
    return _DIGIT_RUN.sub(repl, text or "")


def slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")


# ---------------------------------------------------------------- digit/pattern detectors

_IFSC = re.compile(r"(?<![A-Z0-9])([A-Z]{4})[0O]([A-Z0-9]{6})(?![A-Z0-9])")
_UPI = re.compile(r"(?<![\w.@])[A-Za-z0-9][\w.\-]{1,255}@([A-Za-z]{2,64})(?![\w.@])")
_CARD = re.compile(r"(?<![\d])(?:\d[ \-]?){12,18}\d(?![\d])")
_TOLLFREE = re.compile(r"(?<![\d])1[\s\-]?800[\s\-]?\d{3}[\s\-]?\d{3,4}(?![\d])")
_MOBILE = re.compile(r"(?<![\d+])(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?![\d])")
_AMOUNT = re.compile(r"(?:₹|\brs\.?|\binr)\s?\d[\d,]*(?:\.\d{1,2})?", re.IGNORECASE)
_MASKED_ACCT = re.compile(r"(?<![A-Za-z0-9])[xX*]{2,}[\s\-]?\d{3,6}(?![\d])")


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def valid_ifsc(code: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", code)) and any(c.isdigit() for c in code[5:])


def detect_patterns(text: str) -> set[str]:
    """Pattern names found in (digit-fixed) text. Returns names only, never the matched values."""
    t = fix_digit_runs(text)
    found: set[str] = set()
    for m in _IFSC.finditer(t.upper()):
        if valid_ifsc(m.group(1) + "0" + m.group(2)):
            found.add("ifsc")
            break
    if _UPI.search(t):
        found.add("upi_id")
    for m in _CARD.finditer(t):
        if luhn_ok(re.sub(r"[ \-]", "", m.group(0))):
            found.add("card_number")
            break
    if _TOLLFREE.search(t):
        found.add("phone_tollfree")
    if _MOBILE.search(t):
        found.add("phone_mobile")
    if found & {"phone_tollfree", "phone_mobile"}:
        found.add("phone_any")
    if _AMOUNT.search(t):
        found.add("amount")
    if _MASKED_ACCT.search(t):
        found.add("account_number")
    return found


# ---------------------------------------------------------------- lexicon

@dataclass
class _Term:
    slug: str
    regex: re.Pattern
    compact: str | None                      # set when long enough for space-stripped matching
    key: str = ""                            # letters-only canonical form, for overlap de-duplication


@dataclass
class _Group:
    name: str
    weight: float
    max_hits: int
    terms: list[_Term]


@dataclass
class _Spec:
    groups: list[_Group]
    patterns: dict[str, float]
    combos: list[dict]


@dataclass
class Lexicon:
    labels: dict[str, _Spec]
    tactics: dict[str, _Spec]
    news_sites: list[_Term]
    news_words: list[_Term]


def _compile_term(term: str, compact_min_len: int) -> _Term:
    c = canon(term)
    words = [re.escape(w) for w in re.split(r"[^a-z0-9]+", c) if w]
    regex = re.compile(r"(?<![a-z0-9])" + r"[\W_]*".join(words) + r"(?![a-z0-9])")
    comp = compact(c)
    return _Term(slug(term), regex, comp if len(comp) >= compact_min_len else None, comp)


def _compile_spec(raw: Mapping, compact_min_len: int) -> _Spec:
    groups = [_Group(name, float(g.get("weight", 0.3)), int(g.get("max_hits", 99)),
                     [_compile_term(t, compact_min_len) for t in g.get("terms", [])])
              for name, g in (raw.get("groups") or {}).items()]
    return _Spec(groups, {k: float(v) for k, v in (raw.get("patterns") or {}).items()}, list(raw.get("combos") or []))


@lru_cache(maxsize=8)
def load_lexicon(path: str, compact_min_len: int) -> Lexicon:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    news = raw.get("news", {})
    return Lexicon(
        labels={k: _compile_spec(v, compact_min_len) for k, v in raw.get("labels", {}).items()},
        tactics={k: _compile_spec(v, compact_min_len) for k, v in raw.get("tactics", {}).items()},
        news_sites=[_compile_term(t, compact_min_len) for t in news.get("sites", [])],
        news_words=[_compile_term(t, compact_min_len) for t in news.get("words", [])],
    )


def _hits(terms: Iterable[_Term], canon_text: str, compact_text: str) -> list[str]:
    """Slugs of matching terms. A hit contained in a longer hit ("expiry" in "expiry date",
    "net banking" == "netbanking") is dropped so one piece of text is counted once."""
    hit = [t for t in terms if t.regex.search(canon_text) or (t.compact and t.compact in compact_text)]
    keep, seen = [], set()
    for t in sorted(hit, key=lambda t: -len(t.key)):
        if t.key in seen or any(t.key in k for k in seen):
            continue
        seen.add(t.key)
        keep.append(t.slug)
    return keep


def _score(prefix: str, spec: _Spec, canon_text: str, compact_text: str, patterns: set[str]) -> tuple[float, list[str]]:
    total, evidence, present = 0.0, [], set()
    for g in spec.groups:
        hits = _hits(g.terms, canon_text, compact_text)
        if hits:
            present.add(f"group:{g.name}")
            total += g.weight * min(len(hits), g.max_hits)
            evidence += [f"{prefix}:{g.name}:{h}" for h in hits]
    for name, w in spec.patterns.items():
        if name in patterns:
            total += w
            evidence.append(f"pattern:{name}")
    present |= {f"pattern:{p}" for p in patterns}
    for combo in spec.combos:
        if all(part in present for part in combo.get("all_of", [])):
            total += float(combo.get("weight", 0))
            evidence.append(f"combo:{combo['id']}")
    return min(1.0, total), evidence


# ---------------------------------------------------------------- classify

def _text_of(ocr: Any) -> str:
    if ocr is None:
        return ""
    if isinstance(ocr, str):
        return ocr
    text = getattr(ocr, "full_text", None)
    if text is not None:
        return text
    return "\n".join(getattr(l, "text", l[0] if isinstance(l, (tuple, list)) else str(l))
                     for l in getattr(ocr, "lines", []) or [])


def _window_fields(window: Any) -> tuple[str, str]:
    if window is None:
        return "", ""
    if isinstance(window, Mapping):
        return window.get("title") or "", (window.get("process_name") or "").lower()
    return getattr(window, "title", "") or "", (getattr(window, "process_name", "") or "").lower()


def _config() -> dict:
    try:
        from kavach_config import get_config
        return {**DEFAULTS, **dict(get_config().get("screen_classifier", {}))}
    except Exception:
        return dict(DEFAULTS)


def classify(ocr_result: Any, window_info: Any = None, config: Mapping | None = None) -> ScreenLabel:
    t0 = time.perf_counter()
    cfg = {**DEFAULTS, **(config or _config())}
    lex = load_lexicon(str(ROOT / cfg["lexicon"]), int(cfg["compact_min_len"]))
    title, process = _window_fields(window_info)
    raw = f"{title}\n{_text_of(ocr_result)}"
    c = canon(raw)
    cc = compact(c)
    patterns = detect_patterns(raw)

    evidence: list[str] = []
    labels: dict[str, float] = {}
    for name in LABELS:
        spec = lex.labels.get(name)
        score, ev = _score(name, spec, c, cc, patterns) if spec else (0.0, [])
        labels[name] = score
        evidence += ev
    tactic_scores: dict[str, float] = {}
    for name in TACTICS:
        spec = lex.tactics.get(name)
        score, ev = _score(f"tactic:{name}", spec, c, cc, set()) if spec else (0.0, [])
        tactic_scores[name] = score
        evidence += ev

    # News / article guard: an article *about* scams is not a scam screen.
    t_c = canon(title)
    t_cc = compact(t_c)
    site_hits = _hits(lex.news_sites, c, cc)
    word_hits = _hits(lex.news_words, c, cc)
    site_in_title = bool(_hits(lex.news_sites, t_c, t_cc))
    news = site_in_title or len(set(site_hits) | set(word_hits)) >= int(cfg["news_min_hits"])
    if news:
        f = float(cfg["news_factor"])
        for name in cfg["news_reduces"]:
            if name in labels:
                labels[name] *= f
            elif name in tactic_scores:
                tactic_scores[name] *= f
        evidence.append("news_context")
        evidence += [f"news:{h}" for h in sorted(set(site_hits) | set(word_hits))]

    if process in BROWSERS:
        evidence.append("ctx:browser")

    labels = {k: round(v, 3) for k, v in labels.items()}
    labels["normal"] = round(1.0 - max(labels.values()), 3)
    thr = float(cfg["label_threshold"])
    present = [k for k in LABELS if labels[k] >= thr]
    top = max(present, key=lambda k: (labels[k], -TOP_PRIORITY.index(k))) if present else "normal"
    tactics = frozenset(k for k, v in tactic_scores.items() if v >= float(cfg["tactic_threshold"]))
    return ScreenLabel(
        labels=labels, top_label=top, screen_tactics=tactics,
        evidence=sorted(set(evidence)), ms=(time.perf_counter() - t0) * 1000.0,
        present=present, tactic_scores={k: round(v, 3) for k, v in tactic_scores.items()}, news_context=news,
    )


def format_label(label: ScreenLabel, min_score: float = 0.1) -> str:
    """One-line summary with ids/scores only (safe to print/log)."""
    scores = " ".join(f"{k}={v:.2f}" for k, v in sorted(label.labels.items(), key=lambda kv: -kv[1]) if v >= min_score)
    tactics = ",".join(sorted(label.screen_tactics)) or "-"
    news = " [news]" if label.news_context else ""
    return f"top={label.top_label:<11} {scores}  tactics={tactics}{news}  ({label.ms:.1f} ms)"


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m detect.screen_classifier", description="Screen classifier self-test")
    p.add_argument("files", nargs="*", help="OCR-like text files (default: tests/fixtures/screen/*.txt)")
    p.add_argument("--evidence", action="store_true", help="also print evidence ids")
    args = p.parse_args(argv)
    files = [Path(f) for f in args.files] or sorted((ROOT / "tests" / "fixtures" / "screen").glob("*.txt"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        title, _, body = text.partition("\n") if text.startswith("TITLE:") else ("", "", text)
        label = classify(body, {"title": title.removeprefix("TITLE:").strip(), "process_name": "chrome.exe"})
        print(f"{f.name:<28} {format_label(label)}")
        if args.evidence:
            print("    " + ", ".join(label.evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
