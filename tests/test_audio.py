"""capture.audio signal path on synthetic arrays (no audio device, no pyaudiowpatch needed)."""

import numpy as np
import pytest

from capture import audio as A

SR = 16000


def sine(freq, seconds, sr=SR, amp=0.1):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds, sr=SR, amp=0.1, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(int(seconds * sr))).astype(np.float32)


def dominant_freq(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return np.fft.rfftfreq(x.size, 1 / sr)[np.argmax(spec)]


# ---------------------------------------------------------------- downmix / resample

def test_downmix_interleaved_stereo():
    left, right = np.full(4, 0.2, np.float32), np.full(4, 0.6, np.float32)
    inter = np.stack([left, right], axis=1).ravel()           # L R L R ...
    out = A.downmix(inter, channels=2)
    assert out.dtype == np.float32 and out.shape == (4,)
    assert np.allclose(out, 0.4)
    assert np.allclose(A.downmix(np.stack([left, right], axis=1)), 0.4)   # 2-D frames x channels
    assert A.downmix(left, channels=1) is not None and np.allclose(A.downmix(left, 1), left)


def test_downmix_drops_partial_frame():
    assert A.downmix(np.arange(7, dtype=np.float32), channels=2).shape == (3,)


def test_resample_48k_to_16k_keeps_pitch_and_length():
    x = sine(440, 1.0, sr=48000)
    y = A.resample(x, 48000, 16000)
    assert y.dtype == np.float32
    assert abs(y.size - 16000) <= 1
    assert abs(dominant_freq(y, 16000) - 440) < 2
    assert abs(A.rms(y) - A.rms(x)) < 0.005
    assert A.resample(x, 16000, 16000) is not None and A.resample(y, 16000, 16000).size == y.size


def test_streaming_resampler_matches_one_shot():
    x = sine(300, 2.0, sr=48000)
    rs = A.Resampler(48000, 16000)
    blocks = [rs(b) for b in np.array_split(x, 20)] + [rs(np.zeros(0, np.float32), last=True)]
    y = np.concatenate(blocks)
    ref = A.resample(x, 48000, 16000)
    assert abs(y.size - ref.size) <= 2
    n = min(y.size, ref.size) - 200
    assert np.max(np.abs(y[200:n] - ref[200:n])) < 1e-3       # identical away from the edges


# ---------------------------------------------------------------- speech gate

def narrowband_speech(seconds, rms_target, sr=SR, seed=1):
    """Call-like audio: 1800-3400 Hz band noise (upper telephone band) with a 4 Hz syllable envelope.
    Its ZCR is above the voiced range, like fricative-heavy narrowband speech on a WhatsApp call."""
    x = np.random.default_rng(seed).standard_normal(int(seconds * sr))
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1 / sr)
    spec[(f < 1800) | (f > 3400)] = 0
    x = np.fft.irfft(spec, x.size) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * np.arange(x.size) / sr))
    return (x * rms_target / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def test_gate_tone_is_speech_silence_and_low_noise_are_not():
    gate = A.SpeechGate(silence_rms=0.005, loud_rms=0.015)
    level, voiced, speech = gate(sine(200, 5), SR)             # voiced-like: low ZCR, enough energy
    assert speech and voiced > 0.9 and level == pytest.approx(0.1 / np.sqrt(2), rel=0.01)
    assert gate(np.zeros(5 * SR, np.float32), SR) == (0.0, 0.0, False)
    assert gate(sine(200, 5, amp=0.002), SR)[2] is False      # below silence_rms
    level, voiced, speech = gate(noise(5, amp=0.008), SR)     # above silence, below loud, ZCR ~0.5
    assert A.SpeechGate().silence_rms < level < 0.015 and voiced == 0.0 and speech is False


def test_gate_loud_narrowband_speech_passes():
    gate = A.SpeechGate(silence_rms=0.005, loud_rms=0.015)
    loud = narrowband_speech(5, rms_target=10 ** (-30 / 20))  # -30 dBFS, like the real call
    level, voiced, speech = gate(loud, SR)
    assert voiced < gate.min_voiced                            # voiced check alone would miss it
    assert speech and level == pytest.approx(0.0316, rel=0.01)
    quiet = narrowband_speech(5, rms_target=0.008)             # same signal below loud_rms
    assert gate(quiet, SR)[2] is False


def test_gate_needs_min_voiced_fraction():
    gate = A.SpeechGate(silence_rms=0.005, loud_rms=0.015, min_voiced=0.03)
    short = np.concatenate([sine(200, 0.1), np.zeros(int(4.9 * SR), np.float32)])   # 2% voiced, rms 0.01
    longer = np.concatenate([sine(200, 0.3), np.zeros(int(4.7 * SR), np.float32)])  # 6% voiced
    assert gate(short, SR)[2] is False
    assert gate(longer, SR)[2] is True


def test_gate_from_config():
    g = A.SpeechGate.from_config({"silence_rms": 0.02, "loud_rms": 0.05, "vad_zcr": [0.05, 0.3],
                                  "vad_min_voiced": 0.2})
    assert (g.silence_rms, g.loud_rms, g.zcr_min, g.zcr_max, g.min_voiced) == (0.02, 0.05, 0.05, 0.3, 0.2)
    d = A.SpeechGate.from_config({})
    assert (d.loud_rms, d.min_voiced) == (0.015, 0.03)
    assert A.zero_crossing_rate(noise(1)) > 0.4 and A.zero_crossing_rate(sine(200, 1)) < 0.03


# ---------------------------------------------------------------- chunking with overlap

def test_chunks_overlap_and_timestamps():
    ck = A.Chunker(SR, chunk_s=5, overlap_s=1.0)
    x = np.arange(14 * SR, dtype=np.float32)                   # ramp: every sample identifies its position
    chunks = []
    for i, block in enumerate(np.array_split(x, 140)):         # 100 ms blocks, contiguous timestamps
        chunks += ck.push(block, ts_end=1000.0 + (i + 1) * 0.1)
    # first chunk after 5 s, then one every 4 s (hop = chunk - overlap): 5, 9, 13 s
    assert len(chunks) == 3
    assert [c.samples.size for c in chunks] == [5 * SR] * 3
    for c, start in zip(chunks, (0, 4 * SR, 8 * SR)):
        assert c.samples[0] == start and c.samples[-1] == start + 5 * SR - 1
        assert c.ts_start == pytest.approx(1000.0 + start / SR, abs=1e-6)
        assert c.ts_end - c.ts_start == pytest.approx(5.0)
        assert c.source == "loopback" and c.samples.dtype == np.float32
    assert np.array_equal(chunks[0].samples[-SR:], chunks[1].samples[:SR])   # 1 s overlap
    assert ck.pending_s == pytest.approx(1.0)                  # 13..14 s waiting for the next hop


def test_single_big_push_yields_all_chunks():
    ck = A.Chunker(SR, chunk_s=2, overlap_s=0.5)
    chunks = ck.push(np.zeros(8 * SR, np.float32), ts_end=10.0)
    assert len(chunks) == 5                                    # 2, 3.5, 5, 6.5, 8 s
    assert chunks[-1].ts_end == pytest.approx(10.0) and chunks[0].ts_start == pytest.approx(2.0)


def test_zero_overlap_and_invalid_overlap():
    ck = A.Chunker(SR, chunk_s=1, overlap_s=0)
    assert len(ck.push(np.zeros(3 * SR, np.float32), ts_end=3.0)) == 3
    with pytest.raises(ValueError):
        A.Chunker(SR, chunk_s=1, overlap_s=1)


def test_chunk_speech_flags():
    ck = A.Chunker(SR, chunk_s=5, overlap_s=1.0, gate=A.SpeechGate(silence_rms=0.005))
    x = np.concatenate([sine(200, 5), np.zeros(8 * SR, np.float32), noise(4, amp=0.008)])
    chunks = ck.push(x, ts_end=17.0)
    # tone | 1 s tone overlap + silence (still speech: phrase tail) | silence | 1 s silence + low noise
    assert [c.is_speech for c in chunks] == [True, True, False, False]
    assert chunks[2].rms == 0.0 and 0.005 < chunks[3].rms < 0.015
    assert chunks[0].voiced_fraction > 0.9 and chunks[3].voiced_fraction == 0.0


def test_idle_tick_flushes_tail_padded_with_silence():
    ck = A.Chunker(SR, chunk_s=5, overlap_s=1.0, idle_flush_s=1.0)
    assert ck.push(sine(200, 2), ts_end=102.0) == []
    assert ck.tick(now=102.5) == []                            # not idle long enough
    (c,) = ck.tick(now=103.5)
    assert c.ts_start == pytest.approx(100.0) and c.samples.size == 5 * SR
    assert c.is_speech and not c.samples[2 * SR:].any()        # tone then zero padding
    assert ck.tick(now=110.0) == [] and ck.pending_s == 0


def test_gap_in_input_flushes_before_new_audio():
    ck = A.Chunker(SR, chunk_s=5, overlap_s=1.0, gap_s=0.25)
    ck.push(sine(200, 2), ts_end=2.0)
    chunks = ck.push(sine(200, 1), ts_end=10.0)                # 7 s gap (loopback idle)
    assert len(chunks) == 1 and chunks[0].ts_start == pytest.approx(0.0)
    assert ck.pending_s == pytest.approx(1.0)                  # new audio starts a fresh chunk


def test_capture_config_without_device():
    cap = A.AudioCapture({"audio": {"sample_rate": 16000, "chunk_s": 4, "overlap_s": 1.0, "capture_mic": True}})
    assert set(cap.sources) == {"loopback", "mic"}
    assert cap.sources["mic"].chunker.source == "mic"
    assert cap.sources["loopback"].chunker.n_chunk == 4 * SR
    got = []
    cap.on_chunk = got.append
    cap._dispatch(cap.sources["loopback"].chunker.push(sine(200, 4), ts_end=4.0))
    assert len(got) == 1 and cap.counts == {"chunks": 1, "speech_chunks": 1, "reopens": 0, "device_switches": 0}
