"""detect.intent on the demo scripts, a noisy-ASR hero call, the real TTS->ASR news transcript, and units."""

import re

import pytest

from detect import intent as I
from detect.tactic_lexicon import TACTICS, default_canon, load_tactic_lexicon

SCRIPTS = I.ROOT / "demo" / "scripts"
FIXTURES = I.ROOT / "tests" / "fixtures" / "intent"
CFG = dict(I.DEFAULTS)                     # tests pin the defaults, independent of config.yaml edits


def run(text, **over):
    return I.detect(text, {**CFG, **over})


def script(name):
    return I.read_script(SCRIPTS / f"{name}.txt")


# Typical Whisper mistakes on a phone-quality call.
ASR_NOISE = [
    ("CBI", "see bee eye"), ("digital arrest", "diggital arrest"), ("safe account", "safe acount"),
    ("Do not disconnect the call", "do not this connect the call"), ("money laundering", "money laundring"),
    ("confidential investigation", "confidencial investigation"), ("Kisi ko mat batana", "kisi ko matt batana"),
    ("arrest warrant", "a rest warrant"), ("OTP", "o t p"), ("AnyDesk", "any desk"),
    ("within 30 minutes", "within thirty minutes"), ("fully refundable", "fully refundible"),
    ("customs", "customes"), ("frozen", "frozan"), ("Don't tell anyone", "dont tell any one"),
]


def noisy(text):
    for a, b in ASR_NOISE:
        text = re.sub(re.escape(a), b, text, flags=re.IGNORECASE)
    return text.replace(".", "").replace(",", "")          # ASR often drops punctuation too


# ---------------------------------------------------------------- fixtures

def test_hero_call_hits_all_five_tactics():
    r = run(script("hero_call"))
    assert set(r.tactics) == set(TACTICS) and r.distinct_tactics == 5
    assert all(h.score >= 0.5 for h in r.tactics.values())


def test_normal_it_call_has_no_tactics():
    r = run(script("normal_it_call"))
    assert r.distinct_tactics == 0
    assert r.scores["money_move"] < 0.5                    # AnyDesk alone stays under threshold
    assert "money_remote_access" in r.evidence
    assert all(v == 0 for k, v in r.scores.items() if k != "money_move")


def test_news_clip_at_most_one_tactic():
    r = run(script("news_clip"))
    assert r.distinct_tactics <= 1
    assert any(e.startswith("guard:reporting:") for e in r.evidence)


def test_news_tts_asr_transcript_at_most_one_tactic():
    r = run(I.read_script(FIXTURES / "news_tts_asr.txt"))
    assert r.distinct_tactics <= 1
    assert "guard:reporting:threat_digital_arrest" in r.evidence


def test_noisy_asr_hero_call_still_finds_four_tactics():
    text = noisy(script("hero_call"))
    assert "see bee eye" in text and "diggital arrest" in text and "safe acount" in text
    r = run(text)
    assert r.distinct_tactics >= 4, r.scores
    assert any(e.startswith("fuzzy:") for e in r.evidence)
    assert "auth_cbi" in r.evidence and "threat_digital_arrest" in r.evidence


# ---------------------------------------------------------------- guard / scoring units

def test_second_person_boost_and_reporting_reduction():
    lone = run("This is the CBI calling.")
    to_you = run("This is the CBI calling you.")
    reported = run("Scammers pretend to be from the CBI.")
    assert to_you.scores["authority"] > lone.scores["authority"] > reported.scores["authority"]
    assert "guard:second_person:auth_cbi" in to_you.evidence
    assert "guard:reporting:auth_cbi" in reported.evidence and "authority" not in reported.tactics


def test_guard_is_per_sentence():
    r = run("Police said scammers are active. You are under digital arrest.")
    assert "threat" in r.tactics and "guard:reporting:threat_digital_arrest" not in r.evidence


def test_hinglish_phrases():
    r = run("Aapka aadhaar misuse hua hai, arrest ho jayega. Call mat kaatna, kisi ko mat batana. "
            "Turant transfer karo, safe account mein.")
    assert {"threat", "secrecy", "money_move"} <= set(r.tactics)


def test_urgency_needs_two_cues():
    assert "urgency" not in run("Please do it immediately.").tactics
    assert "urgency" in run("Do it immediately, this is your last warning.").tactics


def test_repeat_decay_limits_many_weak_mentions():
    text = " ".join(f"Scammers used the {w} name." for w in ["CBI", "customs", "police", "RBI", "narcotics"])
    assert "authority" not in run(text).tactics


def test_containment_counts_once():
    r = run("You are under digital arrest.")
    assert "threat_digital_arrest" in r.evidence and "threat_arrest" not in r.evidence


def test_fuzzy_respects_word_boundaries_and_short_words():
    assert "money_safe_account" not in run("Your unsafe accounting was fixed.").evidence   # whole words only
    assert "auth_ed_short" not in run("Callers pretend to be officers.").evidence
    assert "urg_within_2h" not in run("Pay within 24 hours.").evidence       # digits never fuzzy
    assert "fuzzy:threat_digital_arrest" in run("You are in diggital arest now.").evidence


def test_fuzzy_threshold_is_configurable():
    assert "fuzzy:threat_digital_arrest" in run("you are under diggital arrest").evidence
    strict = run("you are under diggital arrest", fuzzy_threshold=99)
    assert not any(e.startswith("fuzzy:") for e in strict.evidence)
    assert strict.evidence == ["guard:second_person:threat_arrest", "threat_arrest"]    # exact "arrest" only


def test_timestamps_from_segments():
    segs = [(100.0, 105.0, "Hello, this is the CBI. You are"), (104.0, 109.0, "under digital arrest."),
            (120.0, 125.0, "Transfer the amount to the safe account now.")]
    r = run(segs)
    assert r.tactics["threat"].first_ts == 100.0 and r.tactics["threat"].last_ts == 109.0   # spans the join
    assert r.tactics["money_move"].first_ts == 120.0
    assert r.tactics["authority"].evidence_ids == ["auth_cbi"]


def test_accepts_rolling_transcript():
    from models.asr import RollingTranscript
    rt = RollingTranscript()
    rt.add("This is the CBI. You are under digital arrest.", 0, 5, "loopback")
    assert I.detect(rt, CFG).distinct_tactics == 2


def test_empty_and_normal_text():
    assert run("").distinct_tactics == 0
    assert run([]).distinct_tactics == 0
    r = run("Hi mom, I'll be home by eight. Can you buy some milk on the way?")
    assert r.distinct_tactics == 0 and r.evidence == []


def test_evidence_never_contains_transcript_text():
    words = set(re.findall(r"[a-z]{4,}", script("hero_call").lower()))
    lex = load_tactic_lexicon(str(I.ROOT / CFG["lexicon"]), "call", default_canon, None, 85.0, 8)
    ids = {e.id for e in lex.entries}
    for e in run(script("hero_call")).evidence:
        assert e.split(":")[-1] in ids, e               # every evidence item ends in a lexicon id
    assert not any(w in " ".join(run(script("hero_call")).evidence).split(":") for w in words)


# ---------------------------------------------------------------- lexicon structure / sharing

def test_lexicon_entries_well_formed():
    import yaml
    raw = yaml.safe_load((I.ROOT / CFG["lexicon"]).read_text(encoding="utf-8"))
    ids = [e["id"] for e in raw["entries"]]
    assert len(ids) == len(set(ids))
    for e in raw["entries"]:
        assert e["tactic"] in TACTICS and 0 < e["weight"] <= 1
        assert set(e["patterns"]) == {"en", "hi_latn", "hi_deva"}      # Devanagari slot ready for P1
        assert set(e.get("where", ["call", "screen"])) <= {"call", "screen"}
    assert {"en", "hi_latn", "hi_deva"} <= set(raw["reporting"]) and "hi_deva" in raw["second_person"]


def test_screen_uses_the_same_lexicon():
    import yaml
    screen = yaml.safe_load((I.ROOT / "detect" / "lexicons" / "screen.yaml").read_text(encoding="utf-8"))
    assert screen["tactics_from"] == CFG["lexicon"] and "tactics" not in screen
    from detect.screen_classifier import classify
    r = classify("You are under DIGITAL ARREST. Do not disconnect the call.")
    assert {"threat", "secrecy"} <= set(r.screen_tactics)
    assert "tactic:threat:threat_digital_arrest" in r.evidence


def test_devanagari_patterns_are_matchable():
    from detect.tactic_lexicon import TacticLexicon
    lex = TacticLexicon({"entries": [{"id": "hi_arrest", "tactic": "threat", "weight": 0.5,
                                      "patterns": {"en": [], "hi_latn": [], "hi_deva": ["गिरफ्तार"]}}]}, "call")
    assert [m.entry.id for m in lex.match(default_canon("आपको गिरफ्तार किया जाएगा"))] == ["hi_arrest"]
