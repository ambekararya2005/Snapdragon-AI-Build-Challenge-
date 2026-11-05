"""Kavach brag soundtrack: 38 s, 120 BPM, A minor. Music + effects on one beat grid from frame 0.

python3 music.py out.wav
"""
import sys
import wave

import numpy as np

SR = 48000
DUR = 38.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(7)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------- buffers ----------------
class Bus:
    def __init__(self):
        self.L = np.zeros(N)
        self.R = np.zeros(N)

    def add(self, sig, t0, gain=1.0, pan=0.0):
        i0 = int(round(t0 * SR))
        if i0 >= N:
            return
        if i0 < 0:
            sig = sig[-i0:]
            i0 = 0
        n = min(len(sig), N - i0)
        gl, gr = gain * np.sqrt(0.5 * (1 - pan)), gain * np.sqrt(0.5 * (1 + pan))
        self.L[i0:i0 + n] += sig[:n] * gl
        self.R[i0:i0 + n] += sig[:n] * gr

    def add_st(self, l, r, t0, gain=1.0):
        i0 = int(round(t0 * SR))
        n = min(len(l), N - i0)
        self.L[i0:i0 + n] += l[:n] * gain
        self.R[i0:i0 + n] += r[:n] * gain


drums, bass, pad, keys, sfx, verb_send = Bus(), Bus(), Bus(), Bus(), Bus(), Bus()

# ---------------- wavetables ----------------
TW = 4096
_tabs = {}


def table(nh, bright):
    key = (nh, round(bright, 2))
    if key not in _tabs:
        ph = np.arange(TW) / TW
        w = np.zeros(TW)
        for k in range(1, nh + 1):
            w += np.sin(2 * np.pi * k * ph) / k * np.exp(-(k - 1) / bright)
        _tabs[key] = w / np.max(np.abs(w))
    return _tabs[key]


def osc(f, n, nh=24, bright=8.0, phase0=0.0):
    nh = max(1, min(nh, int(19000 / f)))
    tab = table(nh, bright)
    ph = (phase0 + f * np.arange(n) / SR) % 1.0
    idx = ph * TW
    i0 = idx.astype(np.int64)
    fr = idx - i0
    return tab[i0] * (1 - fr) + tab[(i0 + 1) % TW] * fr


def sine(f, n, phase0=0.0):
    return np.sin(2 * np.pi * (phase0 + f * np.arange(n) / SR))


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4)
    if hi:
        g *= 1 / np.sqrt(1 + (f / hi) ** 4)
    return np.fft.irfft(X * g, len(x))


def env_ad(n, a, d):
    t = np.arange(n) / SR
    e = np.exp(-np.maximum(t - a, 0) / d)
    if a > 0:
        e *= np.clip(t / a, 0, 1)
    return e


def env_asr(n, a, r):
    t = np.arange(n) / SR
    e = np.clip(t / max(a, 1e-4), 0, 1)
    tail = np.clip((len(t) / SR - t) / max(r, 1e-4), 0, 1)
    return e * tail


# ---------------- instruments ----------------
def kick(gain=1.0):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 44 + 110 * np.exp(-t / 0.035)
    ph = np.cumsum(f) / SR
    body = np.sin(2 * np.pi * ph) * np.exp(-t / 0.26)
    click = fft_filter(rng.standard_normal(n), lo=2000, hi=9000) * np.exp(-t / 0.004) * 0.25
    return (body + click) * gain


NOISE = rng.standard_normal(SR)
CLAP_N = fft_filter(NOISE[: int(0.4 * SR)], lo=900, hi=5000)
HAT_N = fft_filter(NOISE[: int(0.25 * SR)], lo=7000, hi=16000)


def clap():
    n = len(CLAP_N)
    t = np.arange(n) / SR
    e = np.zeros(n)
    for k, off in enumerate((0.0, 0.011, 0.022)):
        e += np.where(t >= off, np.exp(-(t - off) / (0.006 if k < 2 else 0.11)), 0)
    return CLAP_N * e / 2.2


def hat(decay=0.028):
    n = len(HAT_N)
    t = np.arange(n) / SR
    return HAT_N * np.exp(-t / decay) * 0.8


def pluck(m, dur=0.6, bright=14.0, decay=0.22):
    n = int(dur * SR)
    f = mtof(m)
    b = osc(f, n, nh=30, bright=bright) * np.exp(-np.arange(n) / SR / 0.06)
    d = osc(f, n, nh=30, bright=2.5)
    x = b + d * 0.8
    return x * env_ad(n, 0.003, decay)


def pad_note(m, dur, gain=1.0, bright=5.0):
    n = int(dur * SR)
    l = np.zeros(n)
    r = np.zeros(n)
    for c, (gl, gr) in zip((-9, 0, 9), ((1.0, 0.35), (0.7, 0.7), (0.35, 1.0))):
        f = mtof(m) * 2 ** (c / 1200)
        v = osc(f, n, nh=28, bright=bright, phase0=rng.random())
        l += v * gl
        r += v * gr
    e = env_asr(n, 0.25, 0.45)
    return l * e * gain / 3, r * e * gain / 3


def bass_note(m, dur, gain=1.0):
    n = int(dur * SR)
    f = mtof(m)
    x = osc(f, n, nh=14, bright=3.0) * 0.7 + sine(f, n) * 0.6
    return x * env_ad(n, 0.004, 0.16) * env_asr(n, 0.001, 0.03) * gain


def stab(ms, gain=1.0, decay=0.22):
    n = int(0.7 * SR)
    x = sum(osc(mtof(m), n, nh=26, bright=10.0, phase0=rng.random()) for m in ms) / len(ms)
    return x * env_ad(n, 0.002, decay) * gain


def impact(gain=1.0):
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    f = 32 + 40 * np.exp(-t / 0.12)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    nz = fft_filter(rng.standard_normal(n), hi=1800) * np.exp(-t / 0.25) * 0.5
    return (sub + nz) * gain


def riser(dur, gain=1.0, m0=57, m1=81):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = t / dur
    # noise swell, brightening: crossfade dark -> bright filtered noise
    base = rng.standard_normal(n)
    dark, bright = fft_filter(base, lo=200, hi=1200), fft_filter(base, lo=1500, hi=9000)
    nz = dark * (1 - k) + bright * k
    f = mtof(m0) * (mtof(m1) / mtof(m0)) ** k
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    return (nz * 0.6 + tone) * k ** 2.2 * gain


def whoosh(dur=0.5, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = fft_filter(rng.standard_normal(n), lo=500, hi=6000)
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return x * e * gain


def ring(dur=0.6, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    trem = (np.sin(2 * np.pi * 20 * t) > 0).astype(float)
    trem = np.convolve(trem, np.ones(96) / 96, mode="same")
    x = sine(mtof(76), n) * 0.6 + sine(mtof(81), n) * 0.4  # E5 + A5
    return x * trem * env_asr(n, 0.01, 0.08) * gain


def click(gain=1.0):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    x = fft_filter(rng.standard_normal(n), lo=1500, hi=7000) * np.exp(-t / 0.0025)
    x += sine(mtof(93), n) * np.exp(-t / 0.008) * 0.35  # A6 blip
    return x * gain


def shimmer(gain=1.0):
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in (93, 100, 96, 105, 98):  # A6 E7 C7 A7 D7, staggered
        off = rng.random() * 0.06
        x += np.where(t >= off, sine(mtof(m) * (1 + 0.002 * rng.standard_normal()), n) * np.exp(-(t - off) / 0.5), 0)
    return x / 5 * env_asr(n, 0.004, 0.2) * gain


def tick(gain=1.0):
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    return fft_filter(rng.standard_normal(n), lo=3000, hi=12000) * np.exp(-t / 0.003) * gain


def heartbeat(gain=1.0):
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    f = 40 + 30 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12) * gain


# ---------------- arrangement ----------------
AM, F, C, G = (57, 60, 64), (53, 57, 60), (55, 60, 64), (55, 59, 62)
ROOT = {AM: 45, F: 41, C: 48, G: 43}
ARP = {AM: (69, 72, 76, 81), F: (65, 69, 72, 77), C: (67, 72, 76, 79), G: (67, 71, 74, 79)}

# chord per bar (2 s bars), None = no chord
BARS = [None, None, AM, F, AM, F, C, G, AM, F, C, G, AM, F, F, C, G, F, C]
assert len(BARS) == 19

kicks = []


def put_kick(t, g=0.95):
    drums.add(kick(), t, g)
    kicks.append(t)


# --- hook 0-4: drone, heartbeat, ring, riser
n = int(4.2 * SR)
tt = np.arange(n) / SR
drone = (osc(mtof(33), n, nh=10, bright=2.0) * 0.6 + sine(mtof(45), n) * 0.4) * np.clip(tt / 1.2, 0, 1) * np.clip((4.2 - tt) / 0.3, 0, 1)
bass.add(drone, 0.0, 0.32)
pad.add_st(*pad_note(69, 4.1, gain=0.55, bright=2.5), 0.0)  # high A pad, dark
for b in range(4):
    drums.add(heartbeat(), b * 1.0, 0.75)
    drums.add(heartbeat(), b * 1.0 + 0.2, 0.45)
sfx.add(ring(0.55), 0.0, 0.10, pan=-0.2)
sfx.add(ring(0.55), 2.0, 0.10, pan=0.2)
sfx.add(riser(1.5, m0=57, m1=81), 2.5, 0.22)
sfx.add(whoosh(0.5), 3.75, 0.10)

# --- impacts on big moments
for t0, g in ((4.0, 0.75), (16.0, 0.85), (34.0, 0.7)):
    sfx.add(impact(), t0, g)
    verb_send.add(impact(), t0, 0.25)

# --- transitions
for t0 in (7.75, 23.75, 27.75):
    sfx.add(whoosh(0.5), t0 - 0.25, 0.07)

# --- per bar music
for bi, ch in enumerate(BARS):
    t0 = bi * 2.0
    if ch is None:
        continue
    sec = ("reveal" if t0 < 8 else "demo" if t0 < 16 else "alert" if t0 < 20 else "relief" if t0 < 24
           else "sprint" if t0 < 28 else "results" if t0 < 34 else "outro")
    # pad
    padg = {"reveal": 0.6, "demo": 0.5, "alert": 0.35, "relief": 0.7, "sprint": 0.45, "results": 0.5, "outro": 0.8}[sec]
    padlen = 4.4 if sec == "outro" and t0 >= 36 else 2.3
    for m in ch:
        l, r = pad_note(m, padlen, gain=padg, bright=7.0 if sec in ("results", "sprint") else 5.0)
        pad.add_st(l, r, t0)
    # bass 8ths
    if sec not in ("outro",):
        for k in range(4 if sec == "relief" else 8):
            st = t0 + k * (0.5 if sec == "relief" else 0.25)
            if sec == "reveal" and k % 2:
                continue
            if 19.0 <= st < 20.0:  # the freeze
                continue
            bass.add(bass_note(ROOT[ch], 0.24 if sec != "relief" else 0.45), st, 0.55)
    # drums
    for beat in range(4):
        bt = t0 + beat * BEAT
        if 19.0 <= bt < 20.0:
            continue
        if sec == "reveal":
            if t0 >= 4 and beat % 2 == 0:
                put_kick(bt, 0.7)
            if t0 >= 6:
                drums.add(hat(), bt + 0.25, 0.10, pan=0.25)
            continue
        if sec == "relief":
            if beat == 0 or beat == 3 and t0 == 22:
                put_kick(bt, 0.8)
            if beat == 2:
                drums.add(clap(), bt, 0.30, pan=0.0)
            drums.add(hat(), bt + 0.25, 0.08, pan=0.25)
            continue
        if sec == "outro":
            continue
        put_kick(bt)
        if beat in (1, 3):
            drums.add(clap(), bt, 0.42)
            verb_send.add(clap(), bt, 0.12)
        drums.add(hat(), bt + 0.25, 0.13, pan=0.25)
        if sec in ("sprint", "results"):
            drums.add(hat(0.012), bt + 0.125, 0.06, pan=-0.3)
            drums.add(hat(0.012), bt + 0.375, 0.06, pan=-0.3)
    # arps
    if sec in ("demo", "sprint", "results"):
        step = 0.25 if sec == "demo" else 0.125
        pat = ARP[ch]
        order = (0, 1, 2, 3, 2, 1, 2, 3) if step == 0.25 else (0, 1, 2, 3, 1, 2, 3, 2, 0, 1, 2, 3, 1, 2, 3, 2)
        for k in range(int(2.0 / step)):
            st = t0 + k * step
            m = pat[order[k % len(order)]]
            g = 0.10 if sec == "demo" else 0.075
            keys.add(pluck(m, 0.4, bright=10.0, decay=0.12), st, g, pan=(0.35 if k % 2 else -0.35))
            verb_send.add(pluck(m, 0.4, bright=10.0, decay=0.12), st, g * 0.6)
    if sec == "reveal" and t0 >= 6:
        for k, m in enumerate(ARP[ch]):
            keys.add(pluck(m, 0.7), t0 + k * 0.5, 0.10, pan=-0.2 + 0.13 * k)
            verb_send.add(pluck(m, 0.7), t0 + k * 0.5, 0.08)
    if sec == "relief":
        for k, m in enumerate(ARP[ch][:3]):
            keys.add(pluck(m + 12, 0.9, decay=0.3), t0 + 0.5 + k * 0.5, 0.06)
            verb_send.add(pluck(m + 12, 0.9, decay=0.3), t0 + 0.5 + k * 0.5, 0.08)
    if sec == "alert":
        for k in range(4):
            bt = t0 + k * BEAT
            if 19.0 <= bt < 20.0:
                continue
            ms = tuple(m + 12 for m in ch)
            keys.add(stab(ms), bt, 0.22)
            verb_send.add(stab(ms), bt, 0.12)

# demo riser into the alert
sfx.add(riser(2.0, m0=57, m1=81), 14.0, 0.20)
# relief -> sprint riser
sfx.add(riser(1.0, m0=64, m1=76), 23.0, 0.10)

# signal plucks (in key, rising) when each dashboard signal lights
for t0, m in ((8.5, 81), (10.5, 84), (13.5, 88)):
    keys.add(pluck(m, 0.9, bright=18, decay=0.3), t0, 0.16)
    verb_send.add(pluck(m, 0.9, bright=18, decay=0.3), t0, 0.14)
sfx.add(tick(), 11.0, 0.05)          # toast
# the click + freeze
sfx.add(click(), 19.0, 0.35)
sfx.add(shimmer(), 19.0, 0.16)
verb_send.add(shimmer(), 19.0, 0.22)
# sprint: counters + keyboard texture
for t0, m in ((24.0, 76), (24.5, 81), (25.0, 84), (25.5, 88), (26.0, 93)):
    keys.add(pluck(m, 0.6, bright=16, decay=0.18), t0, 0.07)
for k in range(32):
    if rng.random() < 0.6:
        sfx.add(tick(), 24.0 + k * 0.125, 0.025, pan=rng.uniform(-0.4, 0.4))
# results: one tick per card
for i, t0 in enumerate((28.5, 29.0, 29.5, 30.0)):
    keys.add(pluck((81, 84, 88, 93)[i], 0.8, bright=18, decay=0.25), t0, 0.11)
    verb_send.add(pluck((81, 84, 88, 93)[i], 0.8, bright=18, decay=0.25), t0, 0.12)
# outro bells
for k, m in enumerate((77, 81, 84, 88)):
    keys.add(pluck(m, 1.6, bright=12, decay=0.6), 34.5 + k * 0.5, 0.08)
    verb_send.add(pluck(m, 1.6, bright=12, decay=0.6), 34.5 + k * 0.5, 0.10)
for m in (72, 76, 79, 84):  # final C chord shimmer at 36
    keys.add(pluck(m, 2.0, bright=10, decay=0.9), 36.0, 0.05)
    verb_send.add(pluck(m, 2.0, bright=10, decay=0.9), 36.0, 0.08)

# ---------------- sidechain ----------------
t = np.arange(N) / SR
duck = np.ones(N)
for kt in kicks:
    i0 = int(kt * SR)
    n = min(int(0.35 * SR), N - i0)
    d = 1 - 0.55 * np.exp(-np.arange(n) / SR / 0.09)
    duck[i0:i0 + n] = np.minimum(duck[i0:i0 + n], d)
for b in (bass, pad, keys):
    b.L *= duck
    b.R *= duck

# ---------------- reverb ----------------
def reverb(L, R, secs=1.8):
    n = int(secs * SR)
    tt = np.arange(n) / SR
    out = []
    for x, seed in ((L, 1), (R, 2)):
        ir = np.random.default_rng(seed).standard_normal(n) * np.exp(-tt / 0.45)
        ir = fft_filter(ir, lo=250, hi=6000)
        ir[: int(0.012 * SR)] = 0  # pre-delay
        m = 1 << int(np.ceil(np.log2(len(x) + n)))
        y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[: len(x)]
        out.append(y)
    return out

vL, vR = reverb(verb_send.L + 0.25 * (pad.L + keys.L), verb_send.R + 0.25 * (pad.R + keys.R))
vn = max(np.max(np.abs(vL)), np.max(np.abs(vR)))

mixL = drums.L * 1.0 + bass.L * 0.9 + pad.L * 0.75 + keys.L * 0.9 + sfx.L * 0.9 + vL / vn * 0.22
mixR = drums.R * 1.0 + bass.R * 0.9 + pad.R * 0.75 + keys.R * 0.9 + sfx.R * 0.9 + vR / vn * 0.22

# fade in / out
fade = np.ones(N)
fi = int(0.01 * SR)
fade[:fi] = np.linspace(0, 1, fi)
fo0 = int(36.6 * SR)
fade[fo0:] = np.cos(np.linspace(0, np.pi / 2, N - fo0)) ** 2
mixL *= fade
mixR *= fade

# master: gentle glue + soft clip
pk = max(np.max(np.abs(mixL)), np.max(np.abs(mixR)))
mixL, mixR = mixL / pk * 1.15, mixR / pk * 1.15
mixL, mixR = np.tanh(mixL * 1.1) / np.tanh(1.1), np.tanh(mixR * 1.1) / np.tanh(1.1)
pk = max(np.max(np.abs(mixL)), np.max(np.abs(mixR)))
st = np.stack([mixL, mixR], axis=1) / pk * 0.89
pcm = (st * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", sys.argv[1], f"{N / SR:.2f}s", "kicks", len(kicks))
