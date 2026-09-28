"""System audio capture (WASAPI loopback) in overlapping 16 kHz mono chunks for ASR (F4).

The default output device is captured through its WASAPI loopback endpoint (pyaudiowpatch) at its
native rate/channels (usually 48 kHz stereo), downmixed to mono float32, resampled with soxr to
audio.sample_rate and cut into audio.chunk_s chunks that repeat the last audio.overlap_s of the
previous chunk, so phrases are not cut at chunk boundaries. Each chunk gets an RMS level and a cheap
speech flag (energy + zero-crossing rate per frame, or simply loud); is_speech=False chunks must not go to ASR
(saves power, and Whisper hallucinates text on silence).

Audio stays in RAM only: nothing is ever written to disk, not even in debug mode.

The default render (and capture) endpoint is checked every audio.device_check_s through Core Audio;
when it changes (e.g. Bluetooth headphones connect) the streams are reopened on the new default.
Device errors are logged and retried every audio.retry_s; the capture threads never crash.

    cap = AudioCapture(get_config(), on_chunk=lambda c: asr.submit(c) if c.is_speech else None)
    cap.start()
    ...
    cap.stop()

Self-test / CLI:
    python -m capture.audio --devices
    python -m capture.audio --meter --seconds 30 [--mic] [--chunk-s 1]
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import soxr

log = logging.getLogger("capture.audio")

SOURCE_LOOPBACK, SOURCE_MIC = "loopback", "mic"
QUEUE_BLOCKS = 100                 # ~10 s of 100 ms callback blocks per source before dropping


@dataclass
class AudioChunk:
    ts_start: float                    # wall clock (time.time()) of the first sample
    ts_end: float
    samples: np.ndarray = field(repr=False)   # float32 mono at audio.sample_rate
    rms: float
    is_speech: bool
    source: str = SOURCE_LOOPBACK      # loopback | mic
    voiced_fraction: float = 0.0       # fraction of frames that passed the voiced check

    @property
    def duration_s(self) -> float:
        return self.ts_end - self.ts_start


# ---------------------------------------------------------------- signal processing (pure, tested)

def downmix(x: np.ndarray, channels: int = 1) -> np.ndarray:
    """Interleaved 1-D samples (or a 2-D frames x channels array) -> mono float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x.mean(axis=1, dtype=np.float32) if x.shape[1] > 1 else x[:, 0].copy()
    if channels <= 1:
        return x
    n = x.size - x.size % channels
    return x[:n].reshape(-1, channels).mean(axis=1, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """One-shot resampling (whole array)."""
    x = np.asarray(x, dtype=np.float32)
    if int(sr_in) == int(sr_out):
        return x
    return soxr.resample(x, int(sr_in), int(sr_out), quality="HQ").astype(np.float32, copy=False)


class Resampler:
    """Streaming resampler: feed callback blocks in order; filter state is kept between blocks."""

    def __init__(self, sr_in: int, sr_out: int):
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self._rs = None if self.sr_in == self.sr_out else soxr.ResampleStream(
            self.sr_in, self.sr_out, 1, dtype="float32", quality="HQ")

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x if self._rs is None else self._rs.resample_chunk(x, last=last)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def zero_crossing_rate(x: np.ndarray) -> float:
    """Sign changes per sample (DC removed). Voiced speech ~0.02-0.15 at 16 kHz, white noise ~0.5."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < 2:
        return 0.0
    s = np.signbit(x - x.mean())
    return float(np.count_nonzero(s[1:] != s[:-1]) / (x.size - 1))


@dataclass
class SpeechGate:
    """Chunk is speech if RMS >= silence_rms AND (voiced fraction >= min_voiced OR RMS >= loud_rms).
    A frame is voiced if its RMS >= silence_rms and its ZCR is within [zcr_min, zcr_max]. Narrowband
    call audio (e.g. WhatsApp at -30 dBFS) can fail the voiced check, so loud chunks pass anyway."""
    silence_rms: float = 0.005
    loud_rms: float = 0.015            # ~ -36 dBFS
    zcr_min: float = 0.01
    zcr_max: float = 0.25
    min_voiced: float = 0.03           # fraction of voiced frames
    frame_ms: int = 30

    @classmethod
    def from_config(cls, audio: Mapping[str, Any]) -> "SpeechGate":
        zcr = audio.get("vad_zcr") or (cls.zcr_min, cls.zcr_max)
        return cls(silence_rms=float(audio.get("silence_rms", cls.silence_rms)),
                   loud_rms=float(audio.get("loud_rms", cls.loud_rms)),
                   zcr_min=float(zcr[0]), zcr_max=float(zcr[1]),
                   min_voiced=float(audio.get("vad_min_voiced", cls.min_voiced)),
                   frame_ms=int(audio.get("vad_frame_ms", cls.frame_ms)))

    def voiced_fraction(self, x: np.ndarray, sample_rate: int) -> float:
        n = max(2, int(sample_rate * self.frame_ms / 1000))
        frames = np.asarray(x, dtype=np.float32)[: x.size - x.size % n].reshape(-1, n)
        if not len(frames):
            return 0.0
        e = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        centered = frames - frames.mean(axis=1, keepdims=True)
        s = np.signbit(centered)
        zcr = np.count_nonzero(s[:, 1:] != s[:, :-1], axis=1) / (n - 1)
        voiced = (e >= self.silence_rms) & (zcr >= self.zcr_min) & (zcr <= self.zcr_max)
        return float(voiced.mean())

    def __call__(self, x: np.ndarray, sample_rate: int) -> tuple[float, float, bool]:
        """-> (rms, voiced_fraction, is_speech)"""
        level = rms(x)
        voiced = self.voiced_fraction(x, sample_rate)
        speech = level >= self.silence_rms and (voiced >= self.min_voiced or level >= self.loud_rms)
        return level, voiced, speech


class Chunker:
    """RAM buffer that cuts a mono stream into chunk_s chunks overlapping by overlap_s.

    push() takes resampled samples plus the wall-clock time of their last sample and returns the
    finished chunks. A gap in the input (> gap_s; WASAPI loopback delivers nothing while no app is
    playing) or tick() after idle_flush_s without input pads the pending tail with silence and emits
    it, so the end of a phrase is not held back until audio resumes.
    """

    def __init__(self, sample_rate: int, chunk_s: float, overlap_s: float = 1.0,
                 gate: SpeechGate | None = None, source: str = SOURCE_LOOPBACK,
                 gap_s: float = 0.25, idle_flush_s: float = 1.0):
        self.sample_rate = int(sample_rate)
        self.n_chunk = int(round(chunk_s * self.sample_rate))
        self.n_keep = int(round(overlap_s * self.sample_rate))
        if self.n_chunk <= 0 or not 0 <= self.n_keep < self.n_chunk:
            raise ValueError(f"need chunk_s > 0 and 0 <= overlap_s < chunk_s (got {chunk_s}, {overlap_s})")
        self.gate = gate or SpeechGate()
        self.source = source
        self.gap_s, self.idle_flush_s = gap_s, idle_flush_s
        self._buf = np.zeros(self.n_chunk, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._n = 0            # samples in buffer (incl. overlap carried from the last chunk)
        self._new = 0          # samples not yet part of an emitted chunk
        self._ts_end: float | None = None

    @property
    def pending_s(self) -> float:
        return self._new / self.sample_rate

    def push(self, samples: np.ndarray, ts_end: float | None = None) -> list[AudioChunk]:
        x = np.asarray(samples, dtype=np.float32).ravel()
        if not x.size:
            return []
        ts_end = time.time() if ts_end is None else float(ts_end)
        out: list[AudioChunk] = []
        if self._ts_end is not None and ts_end - x.size / self.sample_rate - self._ts_end > self.gap_s:
            out += self.flush()
        pos = 0
        while pos < x.size:
            take = min(x.size - pos, self.n_chunk - self._n)
            self._buf[self._n:self._n + take] = x[pos:pos + take]
            self._n += take
            self._new += take
            pos += take
            if self._n == self.n_chunk:
                out.append(self._emit(ts_end - (x.size - pos) / self.sample_rate))
        self._ts_end = ts_end
        return out

    def tick(self, now: float | None = None) -> list[AudioChunk]:
        """Call periodically; flushes the pending tail after idle_flush_s without input."""
        now = time.time() if now is None else now
        if self._new and self._ts_end is not None and now - self._ts_end >= self.idle_flush_s:
            return self.flush()
        return []

    def flush(self) -> list[AudioChunk]:
        """Pad the pending tail with silence to a full chunk, emit it and start over (no overlap)."""
        out = []
        if self._new and self._ts_end is not None:
            pad = self.n_chunk - self._n
            self._buf[self._n:] = 0.0
            self._n = self.n_chunk
            out.append(self._emit(self._ts_end + pad / self.sample_rate))
        self.reset()
        return out

    def _emit(self, ts_end: float) -> AudioChunk:
        samples = self._buf.copy()
        level, voiced, speech = self.gate(samples, self.sample_rate)
        chunk = AudioChunk(ts_end - self.n_chunk / self.sample_rate, ts_end, samples, level, speech,
                           self.source, voiced)
        if self.n_keep:
            self._buf[:self.n_keep] = self._buf[self.n_chunk - self.n_keep:]
        self._n = self.n_keep
        self._new = 0
        return chunk


# ---------------------------------------------------------------- Core Audio default-device probe

class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


_CLSID_MMDeviceEnumerator = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMMDeviceEnumerator = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_E_RENDER, _E_CAPTURE, _E_CONSOLE = 0, 1, 0


def _vcall(obj: ctypes.c_void_p, index: int, *args: Any, argtypes: tuple = (), restype: Any = ctypes.HRESULT) -> Any:
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])
    return fn(obj, *args)


def default_endpoint_ids(include_capture: bool = False) -> tuple[str | None, ...] | None:
    """Core Audio ids of the default render (and capture) endpoints, or None if the probe is unavailable.
    PortAudio caches its device list until re-initialised, so this is how a default switch is noticed."""
    if sys.platform != "win32":
        return None
    ole32 = ctypes.windll.ole32
    hr = ole32.CoInitializeEx(None, 0)          # MTA; S_FALSE / RPC_E_CHANGED_MODE also leave COM usable
    try:
        clsid, iid, enum = _GUID(), _GUID(), ctypes.c_void_p()
        ole32.CLSIDFromString(ctypes.c_wchar_p(_CLSID_MMDeviceEnumerator), ctypes.byref(clsid))
        ole32.CLSIDFromString(ctypes.c_wchar_p(_IID_IMMDeviceEnumerator), ctypes.byref(iid))
        if ole32.CoCreateInstance(ctypes.byref(clsid), None, 0x17, ctypes.byref(iid), ctypes.byref(enum)) != 0:
            return None
        try:
            flows = (_E_RENDER, _E_CAPTURE) if include_capture else (_E_RENDER,)
            return tuple(_endpoint_id(enum, flow) for flow in flows)
        finally:
            _vcall(enum, 2, restype=ctypes.c_ulong)                 # Release
    except Exception as e:  # noqa: BLE001 - probe is best effort
        log.debug("default endpoint probe failed: %s", e)
        return None
    finally:
        if hr in (0, 1):
            ole32.CoUninitialize()


def _endpoint_id(enum: ctypes.c_void_p, flow: int) -> str | None:
    dev = ctypes.c_void_p()
    try:     # IMMDeviceEnumerator::GetDefaultAudioEndpoint
        _vcall(enum, 4, flow, _E_CONSOLE, ctypes.byref(dev),
               argtypes=(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)))
    except OSError:
        return None                                                 # no device for this flow
    try:
        pid = ctypes.c_void_p()                                     # IMMDevice::GetId
        _vcall(dev, 5, ctypes.byref(pid), argtypes=(ctypes.POINTER(ctypes.c_void_p),))
        try:
            return ctypes.wstring_at(pid.value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(pid)
    finally:
        _vcall(dev, 2, restype=ctypes.c_ulong)


# ---------------------------------------------------------------- device selection (pyaudiowpatch)

def _pyaudio():
    import pyaudiowpatch as pyaudio  # lazy: tests and non-Windows machines import this module without it
    return pyaudio


def find_loopback(pa: Any) -> dict:
    """Loopback analogue of the current default WASAPI output device."""
    return pa.get_default_wasapi_loopback()


def find_mic(pa: Any) -> dict:
    pyaudio = _pyaudio()
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    idx = wasapi.get("defaultInputDevice", -1)
    if idx is None or idx < 0:
        raise LookupError("no default WASAPI input device")
    return pa.get_device_info_by_index(idx)


def _describe(dev: Mapping) -> str:
    ch = int(dev["maxInputChannels"]) or int(dev["maxOutputChannels"])
    return f"[{dev['index']}] {dev['name']} ({int(dev['defaultSampleRate'])} Hz, {ch} ch)"


# ---------------------------------------------------------------- capture

class _Source:
    """One input stream (loopback or mic): callback -> queue -> worker (downmix, resample, chunk)."""

    def __init__(self, name: str, chunker: Chunker):
        self.name = name
        self.chunker = chunker
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_BLOCKS)
        self.stream: Any = None
        self.device: str | None = None
        self.gen = 0                   # bumped on every (re)open; the worker resets on a new generation
        self.dropped = 0
        self.next_retry = 0.0


class AudioCapture:
    def __init__(self, config: Mapping | None = None, on_chunk: Callable[[AudioChunk], None] | None = None):
        if config is None:
            from kavach_config import get_config
            config = get_config()
        a = config.get("audio", {}) or {}
        self.sample_rate = int(a.get("sample_rate", 16000))
        self.chunk_s = float(a.get("chunk_s", 5))
        self.overlap_s = float(a.get("overlap_s", 1.0))
        self.capture_mic = bool(a.get("capture_mic", False))
        self.device_check_s = float(a.get("device_check_s", 10))
        self.retry_s = float(a.get("retry_s", 5))
        self.gate = SpeechGate.from_config(a)
        self.on_chunk = on_chunk or (lambda chunk: None)
        names = [SOURCE_LOOPBACK] + ([SOURCE_MIC] if self.capture_mic else [])
        self.sources = {n: _Source(n, Chunker(self.sample_rate, self.chunk_s, self.overlap_s, self.gate, n))
                        for n in names}
        self._pa: Any = None
        self._endpoint_ids: tuple | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.counts = {"chunks": 0, "speech_chunks": 0, "reopens": 0, "device_switches": 0}

    # -- lifecycle
    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._threads = [threading.Thread(target=self._supervise, name="audio-supervisor", daemon=True)]
        self._threads += [threading.Thread(target=self._work, args=(s,), name=f"audio-{s.name}", daemon=True)
                          for s in self.sources.values()]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []
        self._close_all()

    def stats(self) -> dict[str, Any]:
        return {**self.counts, "devices": {n: s.device for n, s in self.sources.items()},
                "dropped_blocks": {n: s.dropped for n, s in self.sources.items()}}

    # -- supervisor: open streams, watch default-device changes, retry on errors
    def _supervise(self) -> None:
        next_check = 0.0
        while not self._stop.is_set():
            try:
                loop = self.sources[SOURCE_LOOPBACK]
                if loop.stream is None:
                    try:
                        self._open_all()
                    except Exception as e:  # noqa: BLE001 - device errors must never kill capture
                        log.warning("audio: cannot open loopback capture (%s); retrying in %.0f s", e, self.retry_s)
                        self._close_all()
                        self._stop.wait(self.retry_s)
                        continue
                    next_check = time.monotonic() + self.device_check_s
                if self._stop.wait(1.0):
                    break
                mic = self.sources.get(SOURCE_MIC)
                if mic is not None and mic.stream is None and time.monotonic() >= mic.next_retry:
                    self._open_source(mic, find_mic)
                dead = [s.name for s in self.sources.values() if s.stream is not None and not _is_active(s.stream)]
                if dead:
                    log.warning("audio: %s stream stopped; reopening", ",".join(dead))
                    self._close_all()
                    continue
                if time.monotonic() >= next_check:
                    next_check = time.monotonic() + self.device_check_s
                    ids = default_endpoint_ids(self.capture_mic)
                    if ids is not None and self._endpoint_ids is not None and ids != self._endpoint_ids:
                        log.info("audio: default audio device changed; reopening")
                        self.counts["device_switches"] += 1
                        self._close_all()
            except Exception:  # noqa: BLE001
                log.exception("audio: supervisor error; retrying in %.0f s", self.retry_s)
                self._close_all()
                self._stop.wait(self.retry_s)
        self._close_all()

    def _open_all(self) -> None:
        pyaudio = _pyaudio()
        with self._lock:
            self._endpoint_ids = default_endpoint_ids(self.capture_mic)   # before opening: a later switch differs
            self._pa = pyaudio.PyAudio()                                  # fresh init = fresh device list
        self._open_source(self.sources[SOURCE_LOOPBACK], find_loopback, required=True)
        if SOURCE_MIC in self.sources:
            self._open_source(self.sources[SOURCE_MIC], find_mic)
        self.counts["reopens"] += 1

    def _open_source(self, src: _Source, finder: Callable[[Any], dict], required: bool = False) -> None:
        pyaudio = _pyaudio()
        try:
            dev = finder(self._pa)
            rate, channels = int(dev["defaultSampleRate"]), max(1, int(dev["maxInputChannels"]))
            src.gen += 1
            gen, q = src.gen, src.q

            def callback(in_data, frame_count, time_info, status):  # PortAudio thread: keep it tiny
                try:
                    q.put_nowait((gen, rate, channels, in_data, time.time()))
                except queue.Full:
                    src.dropped += 1
                return None, pyaudio.paContinue

            stream = self._pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                                   input_device_index=int(dev["index"]), frames_per_buffer=max(256, rate // 10),
                                   stream_callback=callback)
        except Exception as e:  # noqa: BLE001
            if required:
                raise
            src.next_retry = time.monotonic() + self.retry_s
            log.warning("audio: cannot open %s capture (%s); retrying in %.0f s", src.name, e, self.retry_s)
            return
        name = _describe(dev)
        if src.device is not None and src.device != name:
            log.info("audio: %s switched %s -> %s", src.name, src.device, name)
        else:
            log.info("audio: %s <- %s -> mono %d Hz", src.name, name, self.sample_rate)
        src.stream, src.device = stream, name

    def _close_all(self) -> None:
        with self._lock:
            for s in self.sources.values():
                if s.stream is not None:
                    try:
                        s.stream.stop_stream()
                        s.stream.close()
                    except Exception as e:  # noqa: BLE001
                        log.debug("audio: closing %s stream: %s", s.name, e)
                    s.stream = None
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception as e:  # noqa: BLE001
                    log.debug("audio: terminate: %s", e)
                self._pa = None

    # -- worker: downmix + resample + chunk + dispatch (off the PortAudio callback thread)
    def _work(self, src: _Source) -> None:
        gen, resampler = None, None
        while not self._stop.is_set():
            try:
                try:
                    g, rate, channels, data, ts = src.q.get(timeout=0.25)
                except queue.Empty:
                    self._dispatch(src.chunker.tick())
                    continue
                if g != gen:                         # new device/stream: new rate, discontinuous audio
                    self._dispatch(src.chunker.flush())
                    gen, resampler = g, Resampler(rate, self.sample_rate)
                mono = downmix(np.frombuffer(data, dtype=np.float32), channels)
                self._dispatch(src.chunker.push(resampler(mono), ts))
            except Exception:  # noqa: BLE001
                log.exception("audio: %s worker error", src.name)
        self._dispatch(src.chunker.flush())

    def _dispatch(self, chunks: list[AudioChunk]) -> None:
        for c in chunks:
            self.counts["chunks"] += 1
            self.counts["speech_chunks"] += c.is_speech
            try:
                self.on_chunk(c)
            except Exception:  # noqa: BLE001
                log.exception("audio: on_chunk callback failed")


def _is_active(stream: Any) -> bool:
    try:
        return bool(stream.is_active())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- CLI

def list_devices() -> int:
    pyaudio = _pyaudio()
    with pyaudio.PyAudio() as pa:
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("WASAPI host API not available")
            return 1
        out_idx, in_idx = wasapi.get("defaultOutputDevice", -1), wasapi.get("defaultInputDevice", -1)
        print(f"WASAPI devices ({wasapi['name']}, {wasapi['deviceCount']} endpoints):")
        for d in pa.get_device_info_generator_by_host_api(host_api_index=wasapi["index"]):
            tags = [t for t, on in (("loopback", d.get("isLoopbackDevice")), ("default output", d["index"] == out_idx),
                                    ("default input", d["index"] == in_idx)) if on]
            print(f"  [{d['index']:>2}] {d['name'][:52]:<52} in={d['maxInputChannels']:<2} "
                  f"out={d['maxOutputChannels']:<2} {int(d['defaultSampleRate']):>6} Hz  {' | '.join(tags)}")
        try:
            print(f"default output : {_describe(pa.get_device_info_by_index(out_idx))}")
        except Exception as e:  # noqa: BLE001
            print(f"default output : none ({e})")
        try:
            lb = find_loopback(pa)
            print(f"chosen loopback: {_describe(lb)} -> mono 16000 Hz")
        except Exception as e:  # noqa: BLE001
            print(f"chosen loopback: none ({e})")
        try:
            print(f"default mic    : {_describe(find_mic(pa))}")
        except Exception as e:  # noqa: BLE001
            print(f"default mic    : none ({e})")
    ids = default_endpoint_ids(include_capture=True)
    print("device-change probe (Core Audio): " + ("ok" if ids and ids[0] else "unavailable"))
    return 0


def _bar(level: float, width: int = 30) -> str:
    db = 20 * math.log10(max(level, 1e-6))
    n = int(round(width * min(1.0, max(0.0, (db + 60) / 60))))
    return f"{db:6.1f} dBFS |{'#' * n}{'.' * (width - n)}|"


def meter(seconds: float | None, mic: bool, chunk_s: float | None) -> int:
    from kavach_config import get_config
    cfg = get_config().to_dict()
    cfg["audio"]["capture_mic"] = mic or cfg["audio"].get("capture_mic", False)
    if chunk_s:
        cfg["audio"]["chunk_s"] = chunk_s
        cfg["audio"]["overlap_s"] = min(float(cfg["audio"].get("overlap_s", 1.0)), chunk_s / 2)

    def on_chunk(c: AudioChunk) -> None:
        print(f"{time.strftime('%H:%M:%S', time.localtime(c.ts_start))} {c.source:<8} "
              f"{c.duration_s:4.1f}s rms={c.rms:.4f} {_bar(c.rms)} voiced={c.voiced_fraction:4.0%} "
              f"{'SPEECH' if c.is_speech else 'silent'}",
              flush=True)

    cap = AudioCapture(cfg, on_chunk)
    print(f"metering (chunk {cap.chunk_s:g} s, overlap {cap.overlap_s:g} s, silence_rms {cap.gate.silence_rms:g}, "
          f"loud_rms {cap.gate.loud_rms:g}, min voiced {cap.gate.min_voiced:.0%}); "
          "Ctrl+C to stop", flush=True)
    cap.start()
    try:
        end = None if seconds is None else time.monotonic() + seconds
        while end is None or time.monotonic() < end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
    print(f"stats: {cap.stats()}")
    return 0


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m capture.audio", description="System audio capture self-test")
    p.add_argument("--devices", action="store_true", help="list WASAPI devices and the chosen loopback (default)")
    p.add_argument("--meter", action="store_true", help="print RMS bar and speech flag per chunk")
    p.add_argument("--seconds", type=float, default=None, help="stop the meter after N seconds")
    p.add_argument("--mic", action="store_true", help="also capture the default microphone")
    p.add_argument("--chunk-s", type=float, default=None, help="override audio.chunk_s for the meter")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose or args.meter else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.meter:
        return meter(args.seconds, args.mic, args.chunk_s)
    return list_devices()


if __name__ == "__main__":
    raise SystemExit(_main())
