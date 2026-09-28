"""Shared scam-tactic matcher over detect/lexicons/intent.yaml (one lexicon for calls and screens).

    lex = load_tactic_lexicon(path, channel="call", fuzzy_threshold=85)
    lex.match(canon_text)            # [TacticMatch(entry, key, fuzzy)], de-duplicated per tactic
    lex.has_any(lex.second_person, canon_text)
    lex.score_sum(canon_text, compact_text)   # screen scoring: {tactic: (score, [entry ids])}

detect/intent.py (calls: fuzzy matching + sentence context guard) and detect/screen_classifier.py
(screen tactics: exact + space-stripped matching) both use this, so the two cannot drift. The text passed
in must already be canonicalised with the same `canon` function the lexicon was loaded with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping

import yaml

TACTICS = ("authority", "threat", "secrecy", "urgency", "money_move")
CHANNELS = ("call", "screen")
LANGS = ("en", "hi_latn", "hi_deva")
_WCH = r"\wऀ-ॿ"                      # word characters incl. Devanagari letters and matras
_WORD = re.compile(rf"[{_WCH}]+")
_SPELLED = re.compile(r"\b(?:[a-z] ){1,5}[a-z]\b")      # "c b i" -> "cbi", "o t p" -> "otp"
_WS = re.compile(r"\s+")


def default_canon(text: str) -> str:
    """Speech/transcript canonical form: lowercase, curly quotes folded, spelled letters joined."""
    t = _WS.sub(" ", (text or "").lower().replace("’", "'").replace("‘", "'")).strip()
    return _SPELLED.sub(lambda m: m.group(0).replace(" ", ""), t)


def plain(canon_text: str) -> str:
    """Words only, single-spaced (for fuzzy alignment)."""
    return " ".join(_WORD.findall(canon_text))


@dataclass
class _Pattern:
    key: str                          # letters-only form, for containment de-duplication
    regex: re.Pattern
    compact: str | None = None        # space-stripped form (screen OCR), when long enough
    fuzzy: str | None = None          # single-spaced words, when long enough for fuzzy matching


@dataclass
class TacticEntry:
    id: str
    tactic: str
    weight: float
    where: frozenset[str]
    patterns: list[_Pattern] = field(repr=False)


@dataclass
class TacticMatch:
    entry: TacticEntry
    key: str
    fuzzy: bool = False


def _compile(text: str, canon: Callable[[str], str], compact_min_len: int | None,
             fuzzy_min_len: int | None) -> _Pattern | None:
    c = canon(text)
    words = _WORD.findall(c)
    if not words:
        return None
    rx = re.compile(rf"(?<![{_WCH}])" + r"[\W_]*".join(re.escape(w) for w in words) + rf"(?![{_WCH}])")
    key = "".join(words)
    return _Pattern(key, rx,
                    key if compact_min_len and len(key) >= compact_min_len else None,
                    " ".join(words) if fuzzy_min_len and len(key) >= fuzzy_min_len and not any(
                        ch.isdigit() for ch in key) else None)             # "2 hours" must not match "24 hours"


def _lang_terms(block: Mapping | Iterable | None) -> list[str]:
    if not block:
        return []
    if isinstance(block, Mapping):
        return [t for lang in LANGS for t in (block.get(lang) or [])]
    return list(block)


class TacticLexicon:
    def __init__(self, raw: Mapping, channel: str, canon: Callable[[str], str] = default_canon,
                 compact_min_len: int | None = None, fuzzy_threshold: float | None = None, fuzzy_min_len: int = 8):
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        self.channel = channel
        self.canon = canon
        self.fuzzy_threshold = fuzzy_threshold
        fz = fuzzy_min_len if fuzzy_threshold else None
        self.entries: list[TacticEntry] = []
        seen: set[str] = set()
        for e in raw.get("entries") or []:
            if e["id"] in seen:
                raise ValueError(f"duplicate tactic entry id {e['id']!r}")
            seen.add(e["id"])
            if e["tactic"] not in TACTICS:
                raise ValueError(f"{e['id']}: unknown tactic {e['tactic']!r}")
            where = frozenset(e.get("where") or CHANNELS)
            if channel not in where:
                continue
            pats = [p for p in (_compile(t, canon, compact_min_len, fz) for t in _lang_terms(e.get("patterns"))) if p]
            pats += [_Pattern(f"re:{e['id']}:{i}", re.compile(r)) for i, r in enumerate(e.get("regex") or [])]
            self.entries.append(TacticEntry(e["id"], e["tactic"], float(e.get("weight", 0.5)), where, pats))
        self.reporting = [p for p in (_compile(t, canon, None, None) for t in _lang_terms(raw.get("reporting"))) if p]
        self.second_person = [p for p in (_compile(t, canon, None, None)
                                          for t in _lang_terms(raw.get("second_person"))) if p]

    # -- matching
    def _fuzzy_hit(self, p: _Pattern, text_plain: str, text_words: list[str]) -> bool:
        """partial_ratio >= threshold against the sentence (cheap prefilter), confirmed by fuzz.ratio
        against word windows of n-1..n+1 words, so the match starts and ends on whole words
        ("be officers" is not "ed officers"; "legal action" is not inside "illegal action")."""
        from rapidfuzz import fuzz

        thr = self.fuzzy_threshold
        if fuzz.partial_ratio(p.fuzzy, text_plain, score_cutoff=thr) == 0:
            return False
        pw = p.fuzzy.split()
        n = len(pw)
        short = [w for w in pw if len(w) <= 3]        # short words must be exact: "ed" is not "be"
        sizes = (n, n + 1, n - 1) if n >= 3 else (n, n + 1)   # n-1: a word split/merged by ASR
        for size in sizes:
            if size < 1 or size > len(text_words):
                continue
            for i in range(len(text_words) - size + 1):
                win = text_words[i:i + size]
                if short and not all(w in win for w in short):
                    continue
                if fuzz.ratio(p.fuzzy, " ".join(win), score_cutoff=thr):
                    return True
        return False

    def match(self, canon_text: str, compact_text: str | None = None) -> list[TacticMatch]:
        """Entries found in canon_text. Within a tactic, an entry whose matched text is contained in a
        longer match is dropped ("arrest" inside "digital arrest")."""
        text_plain = plain(canon_text) if self.fuzzy_threshold else ""
        text_words = text_plain.split()
        found: list[TacticMatch] = []
        for e in self.entries:
            best: TacticMatch | None = None
            for p in e.patterns:
                hit = bool(p.regex.search(canon_text)) or bool(p.compact and compact_text and p.compact in compact_text)
                fuzzy = False
                if not hit and p.fuzzy and text_plain:
                    hit = fuzzy = self._fuzzy_hit(p, text_plain, text_words)
                # exact beats fuzzy, then the longer pattern wins
                if hit and (best is None or (best.fuzzy, len(p.key)) > (fuzzy, len(best.key))):
                    best = TacticMatch(e, p.key, fuzzy)
            if best:
                found.append(best)
        keep: list[TacticMatch] = []
        for m in sorted(found, key=lambda m: -len(m.key)):
            if any(k.entry.tactic == m.entry.tactic and m.key != k.key and m.key in k.key for k in keep):
                continue
            keep.append(m)
        return keep

    @staticmethod
    def has_any(patterns: Iterable[_Pattern], canon_text: str) -> list[str]:
        return [p.key for p in patterns if p.regex.search(canon_text)]

    def score_sum(self, canon_text: str, compact_text: str | None = None) -> dict[str, tuple[float, list[str]]]:
        """Screen scoring: tactic score = min(1, sum of matched entry weights)."""
        out: dict[str, tuple[float, list[str]]] = {t: (0.0, []) for t in TACTICS}
        for m in self.match(canon_text, compact_text):
            s, ids = out[m.entry.tactic]
            out[m.entry.tactic] = (s + m.entry.weight, ids + [m.entry.id])
        return {t: (min(1.0, s), sorted(ids)) for t, (s, ids) in out.items()}


@lru_cache(maxsize=16)
def load_tactic_lexicon(path: str, channel: str, canon: Callable[[str], str] = default_canon,
                        compact_min_len: int | None = None, fuzzy_threshold: float | None = None,
                        fuzzy_min_len: int = 8) -> TacticLexicon:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TacticLexicon(raw, channel, canon, compact_min_len, fuzzy_threshold, fuzzy_min_len)
