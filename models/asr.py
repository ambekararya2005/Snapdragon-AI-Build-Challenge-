"""Speech-to-text for call audio chunks (F4): Whisper encoder/decoder ONNX via models.runtime.

Backends (config asr.backend):
  aihub_whisper   primary. The qai_hub_models Whisper (encoder -> cross-attention KV cache; decoder with
                  a fixed 200-slot self-attention KV cache), exported by scripts/export_whisper.py to
                  weights/asr/<asr.model>/. Sessions come from runtime.create_session, so the provider
                  switch (qnn | dml | cpu) applies. Pre/post-processing is pure numpy/python (no torch,
                  transformers or tokenizer packages): log-mel identical to WhisperFeatureExtractor and
                  byte-level BPE decoding from tokens.json. The loop mirrors qai_hub_models'
                  HfWhisperApp._transcribe_single_chunk, with the English/transcribe/no-timestamps prefix
                  forced and Whisper's suppress-token rules applied.
  faster_whisper  dev-only CPU fallback (CTranslate2 int8). NOT the NPU path.

Audio and text stay in RAM. Logs carry only counts/latencies unless privacy.debug_show_text is true.

    asr = ASR()                           # config asr.*
    t = asr.transcribe(chunk)             # AudioChunk -> Transcript | None (None if not speech)
    asr.rolling.text()                    # last asr.rolling_s seconds, overlap-deduplicated

Self-test:  python -m models.asr [--wav-free] [--provider cpu]    (synthetic SAPI speech if available)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

log = logging.getLogger("models.asr")

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("aihub_whisper", "faster_whisper")
DEFAULT_DROP_PHRASES = ("thank you for watching", "thanks for watching", "subscribe", "you")


@dataclass
class Transcript:
    text: str                       # after hallucination guard + overlap dedup ("" if dropped)
    ts_start: float
    ts_end: float
    source: str
    encoder_ms: float
    decoder_ms: float
    total_ms: float
    provider: str
    n_tokens: int = 0
    dropped: str | None = None      # empty | punctuation | phrase | repetition
    raw_text: str = field(default="", repr=False)   # before guard/dedup; RAM only

    @property
    def words(self) -> int:
        return len(self.text.split())


# ---------------------------------------------------------------- log-mel (WhisperFeatureExtractor, numpy)

def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (torch.hann_window / transformers window_function default)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def log_mel(audio: np.ndarray, mel_filters: np.ndarray, n_fft: int = 400, hop: int = 160,
            n_samples: int = 480000) -> np.ndarray:
    """(n_mels, n_samples // hop) log-mel. Pads/trims to n_samples with zeros, centred STFT with
    reflect padding, power spectrum, Slaney mel, log10 floor 1e-10, drop last frame,
    clamp to max-8, (x + 4) / 4 -- exactly WhisperFeatureExtractor._np_extract_fbank_features."""
    x = np.zeros(n_samples, dtype=np.float64)
    a = np.asarray(audio, dtype=np.float64).ravel()[:n_samples]
    x[:a.size] = a
    x = np.pad(x, n_fft // 2, mode="reflect")
    n_frames = 1 + (x.size - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(x, (n_frames, n_fft), (x.strides[0] * hop, x.strides[0]))
    power = np.abs(np.fft.rfft(frames * hann_window(n_fft), axis=1)) ** 2       # (frames, n_fft//2+1)
    mel = power @ mel_filters.astype(np.float64)                                 # (frames, n_mels)
    logm = np.log10(np.maximum(mel, 1e-10)).T[:, :-1]
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).astype(np.float32)


# ---------------------------------------------------------------- byte-level BPE decoding

class TokenDecoder:
    def __init__(self, table: Mapping[str, Any]):
        self.first_special = int(table["first_special_id"])
        self._bytes = [bytes.fromhex(h) for h in table["bytes_hex"]]
        self.special = dict(table.get("special", {}))

    @classmethod
    def load(cls, path: Path) -> "TokenDecoder":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def decode(self, ids: Sequence[int]) -> str:
        """Special tokens (>= first_special_id) are skipped, like skip_special_tokens=True."""
        data = b"".join(self._bytes[i] for i in ids if 0 <= i < self.first_special)
        return data.decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------- hallucination guard

_PUNCT = re.compile(rf"[{re.escape(string.punctuation)}‘’“”…¿¡\s]+")


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def max_ngram_repeats(words: Sequence[str], max_n: int = 4) -> int:
    """Longest run of the same n-gram repeated back to back (n = 1..max_n)."""
    best = 1
    for n in range(1, max_n + 1):
        i = 0
        while i + 2 * n <= len(words):
            run = 1
            while i + (run + 1) * n <= len(words) and words[i + run * n:i + (run + 1) * n] == words[i:i + n]:
                run += 1
            best = max(best, run)
            i += max(1, (run - 1) * n) if run > 1 else 1
    return best


def hallucination_reason(text: str, drop_phrases: Sequence[str] = DEFAULT_DROP_PHRASES,
                         max_repeats: int = 4) -> str | None:
    """Why a Whisper output should be dropped, or None to keep it."""
    if not text.strip():
        return "empty"
    norm = normalize(text)
    if not norm or not any(c.isalnum() for c in norm):
        return "punctuation"
    phrases = {normalize(p) for p in drop_phrases}
    if norm in phrases:
        return "phrase"
    # short outputs that contain a multi-word stock phrase ("Thanks for watching, see you!"); single
    # words like "you" only count as the whole output, except "subscribe"
    contained = [p for p in phrases if len(p.split()) > 1 or p == "subscribe"]
    if len(norm.split()) <= 8 and any(re.search(rf"\b{re.escape(p)}\b", norm) for p in contained):
        return "phrase"
    if max_ngram_repeats(norm.split()) >= max_repeats:
        return "repetition"
    return None


# ---------------------------------------------------------------- rolling transcript + overlap dedup

def _fragment_of(frag: str, word: str) -> bool:
    """frag looks like the start of word, cut at a chunk boundary (CB -> CBI, agent -> agency)."""
    common = len(os.path.commonprefix([frag, word]))
    return common >= max(2, (len(frag) + 1) // 2)


def dedup_overlap(prev_words: Sequence[str], new_words: Sequence[str], max_k: int = 12,
                  max_skip: int = 1) -> tuple[int, int]:
    """Words repeated at a chunk join -> (words to drop from the END of prev, words to drop from the
    START of new). Compared normalized. Either side may carry up to max_skip fragment words of a word
    cut at the boundary ("from the CB" | "from the CBI or ..."). With only one matching word, a dropped
    prev fragment must look like the start of the following new word, so "the call" | "the bank" stays."""
    p = [normalize(w) for w in prev_words][-(max_k + max_skip):]
    n = [normalize(w) for w in new_words]
    for k in range(min(len(p), len(n), max_k), 0, -1):
        for ps in range(max_skip + 1):                     # fragment at the end of prev
            for ns in range(max_skip + 1):                 # fragment at the start of new
                if len(p) < ps + k or len(n) < ns + k:
                    continue
                tail = p[len(p) - ps - k:len(p) - ps]
                if n[ns:ns + k] != tail or not all(tail):
                    continue
                if k < 2 and (ps or ns):
                    if ns or ns + k >= len(n) or not _fragment_of(p[-1], n[k]):
                        continue
                return ps, ns + k
    return 0, 0


class RollingTranscript:
    """RAM-only transcript of the last max_s seconds per source; removes words repeated at chunk joins."""

    def __init__(self, max_s: float = 180.0):
        self.max_s = max_s
        self._items: deque[tuple[float, float, str, list[str]]] = deque()   # ts_start, ts_end, source, words

    def add(self, text: str, ts_start: float, ts_end: float, source: str) -> str:
        words = text.split()
        prev = next((it for it in reversed(self._items) if it[2] == source), None)
        if prev is not None and ts_start < prev[1] and words:
            drop_prev, cut = dedup_overlap(prev[3], words)
            if drop_prev:
                del prev[3][-drop_prev:]                   # cut-off word; the new chunk has it whole
            words = words[cut:]
        if words:
            self._items.append((ts_start, ts_end, source, words))
        while self._items and self._items[-1][1] - self._items[0][1] > self.max_s:
            self._items.popleft()
        return " ".join(words)

    def segments(self, source: str | None = None) -> list[tuple[float, float, str, str]]:
        """[(ts_start, ts_end, source, text)] oldest first (input for detect.intent.detect)."""
        return [(a, b, s, " ".join(w)) for a, b, s, w in self._items if w and (source is None or s == source)]

    def text(self, source: str | None = None) -> str:
        return " ".join(" ".join(w) for _, _, s, w in self._items if source is None or s == source)

    def clear(self) -> None:
        self._items.clear()


# ---------------------------------------------------------------- aihub_whisper backend

class WhisperOnnx:
    """Encoder once per chunk, then greedy decoding with the fixed-size self-attention KV cache."""

    def __init__(self, model_dir: Path, provider: str | None = None, runtime_cfg: Mapping | None = None,
                 language: str = "en", max_tokens: int = 96):
        from models import runtime

        meta_path = model_dir / "whisper.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} missing: run scripts/export_whisper.py in .venv-aihub")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        m = self.meta
        self.mel_filters = np.load(model_dir / "mel_filters.npy")
        self.tokens = TokenDecoder.load(model_dir / "tokens.json")
        name = model_dir.name
        self.encoder = runtime.create_session(model_dir / m["io"]["encoder"]["file"], f"asr_{name}_encoder",
                                              provider, runtime_cfg)
        self.decoder = runtime.create_session(model_dir / m["io"]["decoder"]["file"], f"asr_{name}_decoder",
                                              provider, runtime_cfg)
        self.provider = self.decoder.actual_provider
        sp = m["special"]
        lang = sp.get(f"<|{language}|>")
        if lang is None:
            raise ValueError(f"language {language!r} not in exported special tokens")
        self.prefix = [sp["<|startoftranscript|>"], lang, sp["<|transcribe|>"], sp["<|notimestamps|>"]]
        self.eot = sp["<|endoftext|>"]
        self.L = int(m["decode_len"])
        self.max_tokens = max(1, min(int(max_tokens), self.L - 1 - len(self.prefix)))
        # Whisper greedy rules without timestamps: never emit special/timestamp tokens (except EOT),
        # suppress the generation-config non-speech symbols, and no leading blank/EOT.
        V = int(m["vocab_size"])
        self._suppress = np.zeros(V, dtype=bool)
        self._suppress[self.tokens.first_special:] = True
        self._suppress[self.eot] = False
        self._suppress[[t for t in m["suppress_tokens"] if t < V]] = True
        self._begin_suppress = [t for t in m["begin_suppress_tokens"] if t < V]
        self.enc_out = m["io"]["encoder"]["outputs"]                        # k_cache_cross_i / v_cache_cross_i
        self.dec_out = m["io"]["decoder"]["outputs"]                        # logits, k/v_cache_self_i_out
        H, D, nl = m["decoder_heads"], m["d_model"], m["decoder_layers"]
        self._self_shapes = {f"k_cache_self_{i}_in": (H, 1, D // H, self.L - 1) for i in range(nl)} | \
                            {f"v_cache_self_{i}_in": (H, 1, self.L - 1, D // H) for i in range(nl)}
        # The AI Hub compiled models (weights/qnn/) take/return float16 and list outputs in their own
        # order: outputs are requested by name, feeds cast to each session's input types (no-op on float32).
        self._enc_types = self.encoder.input_dtypes()
        self._dec_types = self.decoder.input_dtypes()

    @staticmethod
    def _cast(feeds: dict[str, np.ndarray], types: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return {k: v if k not in types or v.dtype == types[k] else v.astype(types[k]) for k, v in feeds.items()}

    def features(self, audio: np.ndarray) -> np.ndarray:
        m = self.meta
        return log_mel(audio, self.mel_filters, m["n_fft"], m["hop_length"], m["n_samples"])[None]

    def transcribe_tokens(self, audio: np.ndarray) -> tuple[list[int], float, float]:
        """-> (content token ids, encoder_ms, decoder_ms). audio: float32 mono 16 kHz, <= 30 s."""
        t0 = time.perf_counter()
        mel = self.features(audio)
        cross = dict(zip(self.enc_out, self.encoder.run(self._cast({"input_features": mel}, self._enc_types),
                                                        self.enc_out)))
        t1 = time.perf_counter()

        feeds: dict[str, np.ndarray] = {k: np.zeros(s, np.float32) for k, s in self._self_shapes.items()}
        feeds.update(cross)
        feeds = self._cast(feeds, self._dec_types)
        mask = np.full((1, 1, 1, self.L), self.meta["mask_neg"], dtype=np.float32)
        tokens = list(self.prefix)
        out: list[int] = []
        for n in range(self.L - 1):
            mask[..., self.L - n - 1] = 0.0
            feeds["input_ids"] = np.array([[tokens[n]]], dtype=np.int32)
            feeds["attention_mask"] = mask.astype(self._dec_types.get("attention_mask", np.float32), copy=False)
            feeds["position_ids"] = np.array([n], dtype=np.int32)
            res = self.decoder.run(feeds, self.dec_out)
            for name, val in zip(self.dec_out[1:], res[1:]):
                feeds[name.replace("_out", "_in")] = val
            if n < len(tokens) - 1:
                continue                                      # still feeding the forced prefix
            logits = res[0].reshape(-1).astype(np.float32, copy=True)
            logits[self._suppress] = -np.inf
            if not out:
                logits[self._begin_suppress] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == self.eot or len(out) >= self.max_tokens:
                break
            out.append(nxt)
            tokens.append(nxt)
        return out, (t1 - t0) * 1000.0, (time.perf_counter() - t1) * 1000.0

    def transcribe(self, audio: np.ndarray) -> tuple[str, int, float, float]:
        ids, enc_ms, dec_ms = self.transcribe_tokens(audio)
        return self.tokens.decode(ids), len(ids), enc_ms, dec_ms


# ---------------------------------------------------------------- faster_whisper backend (dev only)

class FasterWhisper:
    """Loads only from weights/asr/faster_whisper/<model>/ (never downloads at runtime). Fetch it once with
    scripts/export_whisper.py --faster-whisper base.en."""

    def __init__(self, model: str = "base.en", compute_type: str = "int8", language: str = "en"):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no network from the runtime
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("faster_whisper not installed: pip install --no-deps faster-whisper, then "
                               "pip install ctranslate2 tokenizers av huggingface_hub") from e
        model_dir = ROOT / "weights" / "asr" / "faster_whisper" / model
        if not (model_dir / "model.bin").is_file():
            raise FileNotFoundError(f"{model_dir} missing: run scripts/export_whisper.py --faster-whisper {model}")
        log.warning("ASR backend faster_whisper (%s, %s, CPU): dev fallback, NOT the NPU path", model, compute_type)
        self.model = WhisperModel(str(model_dir), device="cpu", compute_type=compute_type, local_files_only=True)
        self.language = language
        self.provider = f"faster_whisper-cpu-{compute_type}"

    def transcribe(self, audio: np.ndarray) -> tuple[str, int, float, float]:
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(np.asarray(audio, np.float32), language=self.language, beam_size=1,
                                            vad_filter=False, without_timestamps=True,
                                            condition_on_previous_text=False)
        segs = list(segments)
        ms = (time.perf_counter() - t0) * 1000.0
        return " ".join(s.text.strip() for s in segs).strip(), sum(len(s.tokens) for s in segs), 0.0, ms


# ---------------------------------------------------------------- ASR facade

class ASR:
    def __init__(self, config: Mapping | None = None, provider: str | None = None, backend: Any = None):
        """backend: optional object with .provider and .transcribe(audio) -> (text, n_tokens, enc_ms, dec_ms)
        (tests); otherwise built from asr.backend."""
        if config is None:
            from kavach_config import get_config
            config = get_config()
        a = config.get("asr", {}) or {}
        self.backend_name = a.get("backend", "aihub_whisper")
        if self.backend_name not in BACKENDS:
            raise ValueError(f"asr.backend must be one of {BACKENDS}, got {self.backend_name!r}")
        self.sample_rate = int((config.get("audio", {}) or {}).get("sample_rate", 16000))
        self.drop_phrases = tuple(a.get("drop_phrases", DEFAULT_DROP_PHRASES))
        self.max_repeats = int(a.get("max_ngram_repeats", 4))
        self.rolling = RollingTranscript(float(a.get("rolling_s", 180)))
        language = a.get("language", "en")
        if backend is not None:
            self.backend = backend
        elif self.backend_name == "aihub_whisper":
            # asr.model_dir overrides weights/asr/<asr.model> (e.g. the NPU-compiled weights/qnn/whisper_base)
            model_dir = ROOT / (a.get("model_dir") or Path("weights") / "asr" / a.get("model", "whisper_base"))
            self.backend: Any = WhisperOnnx(model_dir, provider, config.get("runtime"), language,
                                            int(a.get("max_tokens", 96)))
        else:
            self.backend = FasterWhisper(a.get("faster_whisper_model", "base.en"),
                                         a.get("faster_whisper_compute", "int8"), language)
        self.provider = self.backend.provider

    def transcribe(self, chunk: Any) -> Transcript | None:
        """AudioChunk (capture.audio) -> Transcript; None for chunks with is_speech False."""
        if not getattr(chunk, "is_speech", True):
            return None
        return self.transcribe_array(chunk.samples, chunk.ts_start, chunk.ts_end, getattr(chunk, "source", "loopback"))

    def transcribe_array(self, samples: np.ndarray, ts_start: float = 0.0, ts_end: float | None = None,
                         source: str = "loopback") -> Transcript:
        t0 = time.perf_counter()
        raw, n_tok, enc_ms, dec_ms = self.backend.transcribe(np.asarray(samples, np.float32))
        ts_end = ts_start + len(samples) / self.sample_rate if ts_end is None else ts_end
        reason = hallucination_reason(raw, self.drop_phrases, self.max_repeats)
        text = "" if reason else self.rolling.add(raw, ts_start, ts_end, source)
        total = (time.perf_counter() - t0) * 1000.0
        log.debug("asr %s: %d tokens, %d words, enc %.0f ms, dec %.0f ms%s", source, n_tok, len(text.split()),
                  enc_ms, dec_ms, f", dropped ({reason})" if reason else "")
        return Transcript(text, ts_start, ts_end, source, enc_ms, dec_ms, total, self.provider, n_tok, reason, raw)


# ---------------------------------------------------------------- synthetic speech (tests / self-test)

def sapi_speech(text: str, sample_rate: int = 16000) -> np.ndarray | None:
    """Speak `text` with Windows SAPI into an in-memory stream -> float32 mono. None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return None
    pythoncom.CoInitialize()
    try:
        fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = {8000: 8, 11025: 12, 16000: 18, 22050: 22, 44100: 34, 48000: 38}[sample_rate]  # 16-bit mono
        stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
        stream.Format = fmt
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.AudioOutputStream = stream
        voice.Speak(text)
        data = bytes(stream.GetData())
    except Exception as e:  # noqa: BLE001
        log.debug("SAPI unavailable: %s", e)
        return None
    finally:
        pythoncom.CoUninitialize()
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _main(argv: list[str] | None = None) -> int:
    from kavach_privacy import redact_text

    p = argparse.ArgumentParser(prog="python -m models.asr", description="ASR self-test on synthetic SAPI speech")
    p.add_argument("--provider", default=None, help="override runtime.provider (qnn | dml | cpu)")
    p.add_argument("--text", default="This is the CBI, do not disconnect the call.")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    audio = sapi_speech(args.text)
    if audio is None:
        print("SAPI speech unavailable (Windows + pywin32 needed)")
        return 1
    asr = ASR(provider=args.provider)
    print(f"backend {asr.backend_name}, provider {asr.provider}, audio {audio.size / 16000:.2f} s")
    for i in range(args.runs):
        asr.rolling.clear()
        t = asr.transcribe_array(audio)
        print(f"run {i}: total {t.total_ms:.0f} ms (enc {t.encoder_ms:.0f}, dec {t.decoder_ms:.0f}), "
              f"{t.n_tokens} tokens, RTF {t.total_ms / 1000 / (audio.size / 16000):.3f}: {redact_text(t.text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
