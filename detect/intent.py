"""Scam tactics in the rolling call transcript (F5): authority, threat, secrecy, urgency, money_move.

    res = detect(asr.rolling)                 # or a string, or [(ts_start, ts_end, text), ...]
    res.tactics                               # {tactic: TacticHit(score, evidence_ids, first_ts, last_ts)}
    res.distinct_tactics                      # number of tactics >= intent.tactic_threshold (fusion call_tactic)

Lexicon: detect/lexicons/intent.yaml (shared with the screen classifier through detect/tactic_lexicon.py).
English + romanised Hinglish patterns; Devanagari entries can be added under hi_deva.

ASR robustness: exact word-boundary regex for every pattern, plus rapidfuzz partial_ratio >=
intent.fuzzy_threshold for patterns with >= intent.fuzzy_min_len letters ("diggital arrest"), spelled
letters joined ("c b i" -> cbi) and explicit ASR spellings in the lexicon ("see bee eye").

Context guard, per sentence: second-person address ("you", "aapka") multiplies the matched entries by
intent.second_person_boost; reporting/news/advisory language ("scammers", "police said", "beware")
multiplies them by intent.reporting_factor. A sentence with neither cue inherits a reporting cue that ended
within intent.reporting_carry_words words before it (0 = off): ASR splits news sentences and chunk
boundaries cut them before their cue arrives; a second-person sentence without a cue ends the context. Decisions are recorded in evidence as guard:second_person:<id> /
guard:reporting:<id> (+ guard:reporting_context:<id> when inherited; fuzzy hits as fuzzy:<id>).

Scoring: each entry counts once, at its best sentence. Tactic score = sum of its entries' values in
descending order, each further entry multiplied by intent.repeat_decay (1, 0.5, 0.25, ...), capped at
1, so many down-weighted mentions in a news report cannot add up to a tactic. Evidence holds lexicon
ids only, never transcript text.

Self-test:  python -m detect.intent [files...]      # default: demo/scripts/*.txt + tests/fixtures/intent/*.txt
"""

from __future__ import annotations

import argparse
import bisect
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from detect.tactic_lexicon import TACTICS, default_canon, load_tactic_lexicon

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {"lexicon": "detect/lexicons/intent.yaml", "tactic_threshold": 0.5, "fuzzy_threshold": 85,
            "fuzzy_min_len": 8, "second_person_boost": 1.2, "reporting_factor": 0.25, "repeat_decay": 0.5,
            "max_sentence_words": 30, "reporting_carry_words": 30}
_SENT_END = re.compile(r"(?<=[.!?।])\s+|\n+")


@dataclass
class TacticHit:
    score: float
    evidence_ids: list[str]                 # lexicon entry ids
    first_ts: float | None
    last_ts: float | None


@dataclass
class IntentResult:
    tactics: dict[str, TacticHit]           # only tactics >= threshold
    distinct_tactics: int
    ms: float
    scores: dict[str, float] = field(default_factory=dict)     # every tactic, for logs/tables
    evidence: list[str] = field(default_factory=list)          # entry ids + guard:/fuzzy: decisions


def _config() -> dict:
    try:
        from kavach_config import get_config
        return {**DEFAULTS, **dict(get_config().get("intent", {}))}
    except Exception:
        return dict(DEFAULTS)


def segments_of(window: Any) -> list[tuple[float | None, float | None, str]]:
    """str | RollingTranscript (.segments()) | [Transcript-like | (ts_start, ts_end, text) | dict]."""
    if window is None:
        return []
    if isinstance(window, str):
        return [(None, None, window)]
    if hasattr(window, "segments"):
        return [(a, b, t) for a, b, _src, t in window.segments()]
    out = []
    for s in window:
        if isinstance(s, str):
            out.append((None, None, s))
        elif isinstance(s, Mapping):
            out.append((s.get("ts_start"), s.get("ts_end"), s.get("text", "")))
        elif isinstance(s, (tuple, list)):
            out.append((s[0], s[1], s[-1]))
        else:
            out.append((getattr(s, "ts_start", None), getattr(s, "ts_end", None), getattr(s, "text", "")))
    return out


def sentences(segments: Sequence[tuple[float | None, float | None, str]],
              max_words: int = 30) -> list[tuple[str, float | None, float | None]]:
    """Split the joined transcript into sentences (which may span chunk joins) with their time span.
    Unpunctuated ASR text is cut every max_words words so the context guard stays local."""
    parts, starts, pos = [], [], 0
    for _a, _b, text in segments:
        starts.append(pos)
        parts.append(text)
        pos += len(text) + 1
    joined = " ".join(parts)
    out = []
    begin = 0
    for m in list(_SENT_END.finditer(joined)) + [None]:
        end = m.start() if m else len(joined)
        piece = joined[begin:end]
        if piece.strip():
            words = list(re.finditer(r"\S+", piece))
            for i in range(0, len(words), max_words):
                ws = words[i:i + max_words]
                s0, s1 = begin + ws[0].start(), begin + ws[-1].end()
                i0 = max(0, bisect.bisect_right(starts, s0) - 1)
                i1 = max(0, bisect.bisect_right(starts, s1 - 1) - 1)
                out.append((joined[s0:s1], segments[i0][0], segments[i1][1]))
        begin = m.end() if m else len(joined)
    return out


def detect(transcript_window: Any, config: Mapping | None = None) -> IntentResult:
    t0 = time.perf_counter()
    cfg = {**DEFAULTS, **(config if config is not None else _config())}
    lex = load_tactic_lexicon(str(ROOT / cfg["lexicon"]), "call", default_canon, None,
                              float(cfg["fuzzy_threshold"]), int(cfg["fuzzy_min_len"]))
    boost, reduce_ = float(cfg["second_person_boost"]), float(cfg["reporting_factor"])

    best: dict[str, float] = {}                         # entry id -> best weighted value
    tactic_of: dict[str, str] = {}
    ts: dict[str, list[float]] = {}                     # entry id -> timestamps of its sentences
    evidence: set[str] = set()
    carry_words = int(cfg["reporting_carry_words"])
    since_cue: float = math.inf                         # words since the end of the last sentence with a cue
    for text, a, b in sentences(segments_of(transcript_window), int(cfg["max_sentence_words"])):
        c = default_canon(text)
        second = bool(lex.has_any(lex.second_person, c))
        own_reporting = bool(lex.has_any(lex.reporting, c))
        # Reporting context: ASR splits news sentences ("... police said ... safe account. For verification.")
        # and a chunk boundary cuts one before its cue ("officials say") is transcribed. A sentence with no
        # cue of its own and no second-person address inherits a cue that ended within carry_words words
        # before it. A sentence addressing "you" without a cue ends the context (a caller talking to the listener).
        inherited = carry_words > 0 and not own_reporting and not second and since_cue <= carry_words
        n_words = len(c.split())
        if own_reporting:
            since_cue = 0
        elif second:
            since_cue = math.inf
        else:
            since_cue += n_words
        matches = lex.match(c)
        if not matches:
            continue
        reporting = own_reporting or inherited
        factor = (boost if second else 1.0) * (reduce_ if reporting else 1.0)
        for m in matches:
            eid = m.entry.id
            tactic_of[eid] = m.entry.tactic
            best[eid] = max(best.get(eid, 0.0), m.entry.weight * factor)
            ts.setdefault(eid, []).extend(x for x in (a, b) if x is not None)
            evidence.add(eid)
            if second:
                evidence.add(f"guard:second_person:{eid}")
            if reporting:
                evidence.add(f"guard:reporting:{eid}")
            if inherited:
                evidence.add(f"guard:reporting_context:{eid}")
            if m.fuzzy:
                evidence.add(f"fuzzy:{eid}")

    decay, thr = float(cfg["repeat_decay"]), float(cfg["tactic_threshold"])
    scores: dict[str, float] = {}
    hits: dict[str, TacticHit] = {}
    for tactic in TACTICS:
        ids = sorted((e for e in best if tactic_of[e] == tactic), key=lambda e: -best[e])
        score = min(1.0, sum(best[e] * decay ** i for i, e in enumerate(ids)))
        scores[tactic] = round(score, 3)
        if ids and score >= thr:
            times = [x for e in ids for x in ts.get(e, [])]
            hits[tactic] = TacticHit(round(score, 3), ids, min(times) if times else None,
                                     max(times) if times else None)
    return IntentResult(hits, len(hits), (time.perf_counter() - t0) * 1000.0, scores, sorted(evidence))


def format_result(res: IntentResult) -> str:
    """One line, scores and ids only (safe to print/log)."""
    cells = " ".join(f"{t}={res.scores.get(t, 0):.2f}{'*' if t in res.tactics else ''}" for t in TACTICS)
    return f"{res.distinct_tactics} tactics  {cells}  ({res.ms:.1f} ms)"


def read_script(path: Path) -> str:
    """Demo/fixture text files: '#' lines are comments."""
    return "\n".join(l for l in path.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m detect.intent", description="Call-transcript tactic detector self-test")
    p.add_argument("files", nargs="*", help="transcript text files (default: demo scripts + intent fixtures)")
    p.add_argument("--evidence", action="store_true", help="also print evidence ids")
    args = p.parse_args(argv)
    files = [Path(f) for f in args.files] or [*sorted((ROOT / "demo" / "scripts").glob("*.txt")),
                                             *sorted((ROOT / "tests" / "fixtures" / "intent").glob("*.txt"))]
    for f in files:
        res = detect(read_script(f))
        print(f"{f.name:<24} {format_result(res)}")
        if args.evidence:
            print("    " + ", ".join(res.evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
