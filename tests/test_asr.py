"""models.asr: numpy pre/post-processing, hallucination guard, overlap dedup, and an end-to-end
transcription of synthetic Windows SAPI speech (skipped without weights or off Windows)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from models import asr as A

WEIGHTS = A.ROOT / "weights" / "asr" / "whisper_base"


# ---------------------------------------------------------------- log-mel / tokens

def test_hann_is_periodic():
    w = A.hann_window(400)
    assert w[0] == 0 and w[200] == pytest.approx(1.0) and w[-1] > 0     # periodic, not symmetric


def test_log_mel_shape_and_range():
    filt = np.abs(np.random.default_rng(0).standard_normal((201, 80))).astype(np.float32) * 1e-3
    t = np.arange(5 * 16000) / 16000
    mel = A.log_mel(np.sin(2 * np.pi * 440 * t).astype(np.float32), filt)
    assert mel.shape == (80, 3000) and mel.dtype == np.float32
    assert mel.max() - mel.min() <= 2.0 + 1e-6                            # clamp to max-8, /4
    silent = A.log_mel(np.zeros(16000, np.float32), filt)
    assert np.allclose(silent, (np.log10(1e-10) + 4) / 4)


def test_log_mel_trims_long_audio():
    filt = np.ones((201, 80), np.float32)
    assert A.log_mel(np.ones(40 * 16000, np.float32), filt).shape == (80, 3000)


def test_token_decoder_bytes_and_specials():
    table = {"first_special_id": 4, "special": {"<|endoftext|>": 4},
             "bytes_hex": [b"Hi".hex(), b" there".hex(), "c3".encode().decode(), "a9"]}   # "é" split across 2 ids
    dec = A.TokenDecoder(table)
    assert dec.decode([0, 1, 4, 99]) == "Hi there"                       # special + out of range skipped
    assert dec.decode([0, 2, 3]) == "Hié"                                # multi-token UTF-8 joins


# ---------------------------------------------------------------- hallucination guard

@pytest.mark.parametrize("text,reason", [
    ("", "empty"), ("   ", "empty"), ("...", "punctuation"), (" - ! ?", "punctuation"),
    ("Thank you for watching!", "phrase"), ("thanks for watching.", "phrase"), ("You", "phrase"),
    ("Subscribe", "phrase"), ("Please subscribe to my channel.", "phrase"),
    ("the the the the the", "repetition"), ("I am sure. I am sure. I am sure. I am sure.", "repetition"),
    ("Do not disconnect the call, you are under digital arrest.", None),
    ("Thank you, sir. Please share the OTP now.", None),
    ("You must pay the fine today", None),
    ("yes yes yes", None),
])
def test_hallucination_reason(text, reason):
    assert A.hallucination_reason(text) == reason


def test_max_ngram_repeats():
    assert A.max_ngram_repeats("a b a b a b c".split()) == 3
    assert A.max_ngram_repeats("one two three".split()) == 1


# ---------------------------------------------------------------- overlap dedup / rolling transcript

def test_dedup_overlap_removes_join_words():
    prev = "this is the CBI calling about your".split()
    assert A.dedup_overlap(prev, "about your account, do not".split()) == (0, 2)
    assert A.dedup_overlap(prev, "About, your account".split()) == (0, 2)      # case/punctuation ignored
    assert A.dedup_overlap(prev, "ut about your account".split()) == (0, 3)    # cut-word fragment in new
    assert A.dedup_overlap(prev, "completely new words".split()) == (0, 0)


@pytest.mark.parametrize("prev,new,expected", [
    # seen live: the previous chunk ends on a cut-off word that the next chunk has whole
    ("callers pretend to be officers from the CB", "from the CBI or customs.", (1, 2)),
    ("Victims are told that a parcel is", "that a parcel in their name", (1, 3)),
    ("Officials say no government agent", "government agency will ever ask you", (1, 1)),
    # one matching word + unrelated last word: not a fragment, keep everything
    ("please do not disconnect the call", "the bank will call you", (0, 0)),
])
def test_dedup_overlap_prev_fragment(prev, new, expected):
    assert A.dedup_overlap(prev.split(), new.split()) == expected


def test_rolling_transcript_replaces_cut_word():
    r = A.RollingTranscript()
    r.add("callers pretend to be officers from the CB", 0, 5, "loopback")
    assert r.add("from the CBI or customs.", 4, 9, "loopback") == "CBI or customs."
    assert r.text() == "callers pretend to be officers from the CBI or customs."


def test_rolling_transcript_dedups_only_overlapping_same_source():
    r = A.RollingTranscript(max_s=180)
    assert r.add("this is the CBI calling about your", 0, 5, "loopback") == "this is the CBI calling about your"
    assert r.add("about your account do not disconnect", 4, 9, "loopback") == "account do not disconnect"
    assert r.add("about your account", 4, 9, "mic") == "about your account"                     # other source
    assert r.add("disconnect the call", 20, 25, "loopback") == "disconnect the call"            # no overlap
    assert r.text("loopback") == "this is the CBI calling about your account do not disconnect disconnect the call"


def test_rolling_transcript_keeps_max_seconds():
    r = A.RollingTranscript(max_s=10)
    for i in range(6):
        r.add(f"word{i}", i * 4, i * 4 + 5, "loopback")
    assert r.text().split() == ["word3", "word4", "word5"]


# ---------------------------------------------------------------- ASR facade with a fake backend

class FakeBackend:
    provider = "fake"

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio):
        return self.texts.pop(0), 5, 1.0, 2.0


def _chunk(ts, speech=True, source="loopback"):
    return SimpleNamespace(samples=np.zeros(5 * 16000, np.float32), ts_start=ts, ts_end=ts + 5,
                           is_speech=speech, source=source)


def test_asr_skips_non_speech_and_guards():
    asr = A.ASR({"asr": {}, "audio": {"sample_rate": 16000}},
                backend=FakeBackend(["This is the CBI calling about your", "about your account now", "Thanks for watching!"]))
    assert asr.transcribe(_chunk(0, speech=False)) is None
    t1 = asr.transcribe(_chunk(0))
    t2 = asr.transcribe(_chunk(4))
    t3 = asr.transcribe(_chunk(8))
    assert t1.text == "This is the CBI calling about your" and t1.provider == "fake" and t1.n_tokens == 5
    assert t2.text == "account now" and t2.dropped is None
    assert t3.text == "" and t3.dropped == "phrase"
    assert asr.rolling.text() == "This is the CBI calling about your account now"


def test_asr_rejects_unknown_backend():
    with pytest.raises(ValueError):
        A.ASR({"asr": {"backend": "whisper_cpp"}})


# ---------------------------------------------------------------- end to end (real weights)

@pytest.mark.skipif(sys.platform != "win32", reason="SAPI TTS needs Windows")
@pytest.mark.skipif(not (WEIGHTS / "whisper.json").is_file(), reason="weights/asr/whisper_base missing")
def test_transcribes_sapi_speech():
    audio = A.sapi_speech("This is the CBI, do not disconnect the call.")
    if audio is None:
        pytest.skip("SAPI voice unavailable")
    padded = np.zeros(5 * 16000, np.float32)                  # one 5 s AudioChunk, like capture.audio emits
    padded[:min(audio.size, padded.size)] = audio[:padded.size]
    chunk = SimpleNamespace(samples=padded, ts_start=0.0, ts_end=5.0, is_speech=True, source="loopback")
    asr = A.ASR({"asr": {"backend": "aihub_whisper", "model": "whisper_base"}, "audio": {"sample_rate": 16000},
                 "runtime": {"provider": "cpu", "fallback_to_cpu": True}})
    t = asr.transcribe(chunk)
    assert t is not None and t.dropped is None
    assert "cbi" in t.text.lower() or "disconnect" in t.text.lower()
    assert t.encoder_ms > 0 and t.decoder_ms > 0 and t.provider == "CPUExecutionProvider"
