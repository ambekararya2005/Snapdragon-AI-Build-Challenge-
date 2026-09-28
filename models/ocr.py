"""OCR wrapper. Backend 'rapidocr' (third-party, creates its own onnxruntime sessions — the one allowed exception) or 'native' (static det/rec ONNX models via models.runtime; see models/ocr_native.py).

    ocr = create_backend()                 # config ocr.backend
    result = ocr(bgr_image)                # OcrResult; text stays in memory
    result.full_text, result.total_ms, result.provider

OCR text is screen content: never log or print it except through kavach_privacy.redact_text.

Self-test:
    python -m models.ocr                   # synthetic "Enter OTP 482913" image
    python -m models.ocr --image some.png  # text shown only if privacy.debug_show_text
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Protocol, Sequence

import numpy as np

from models import runtime

log = logging.getLogger("models.ocr")
ROOT = Path(__file__).resolve().parent.parent

Box = list[list[float]]  # 4 points [[x, y], ...], clockwise from top-left, in input-image pixels


class OcrLine(NamedTuple):
    text: str
    confidence: float
    box: Box


@dataclass
class OcrResult:
    lines: list[OcrLine]
    full_text: str
    det_ms: float
    rec_ms: float
    total_ms: float
    backend: str
    provider: str
    cls_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class OcrBackend(Protocol):
    name: str
    provider: str

    def __call__(self, image: np.ndarray) -> OcrResult: ...


# ---------------------------------------------------------------- pure helpers (unit-tested)

def _box_metrics(box: Box) -> tuple[float, float, float, float]:
    pts = np.asarray(box, dtype=np.float32)
    ys, xs = pts[:, 1], pts[:, 0]
    return float(ys.mean()), float(ys.max() - ys.min()), float(xs.min()), float(ys.min())


def order_lines(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    """Group lines into visual rows (top-to-bottom), each row sorted left-to-right.

    Two boxes share a row when their vertical centres differ by less than half the smaller height.
    """
    items = sorted(lines, key=lambda l: _box_metrics(l.box)[0])
    rows: list[list[OcrLine]] = []
    row_center = row_height = 0.0
    for line in items:
        cy, h, _, _ = _box_metrics(line.box)
        if rows and abs(cy - row_center) < 0.5 * max(1.0, min(h, row_height)):
            rows[-1].append(line)
            n = len(rows[-1])
            row_center += (cy - row_center) / n
            row_height = max(row_height, h)
        else:
            rows.append([line])
            row_center, row_height = cy, h
    return [sorted(r, key=lambda l: _box_metrics(l.box)[2]) for r in rows]


def join_text(lines: Sequence[OcrLine]) -> str:
    """Rows joined by newlines; boxes within a row joined by two spaces."""
    return "\n".join("  ".join(l.text for l in row) for row in order_lines(lines))


_CONFUSABLE_IN_WORD = str.maketrans({"0": "O", "1": "I", "5": "S", "|": "I"})


def fold_confusables(text: str) -> str:
    """Upper-case text with digit look-alikes inside letter words folded (0TP -> OTP, L0GIN -> LOGIN).

    OCR (and scammers) swap O/0, I/1, S/5. Only tokens that already contain a letter are folded, so
    pure numbers like 482913 stay intact. For keyword matching only; never for display.
    """
    out = []
    for token in text.upper().split(" "):
        if any(c.isalpha() for c in token):
            token = token.translate(_CONFUSABLE_IN_WORD)
        out.append(token)
    return " ".join(out)


# ---------------------------------------------------------------- rapidocr backend

def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return p if p.is_file() else None


class RapidOcrBackend:
    """rapidocr_onnxruntime 1.4.x. It builds its own InferenceSessions (allowed exception, see CLAUDE.md).

    Provider: rapidocr only knows use_dml / use_cuda flags per stage. We map runtime.provider onto
    them, then read the providers its sessions actually got. QNN is not supported by rapidocr, so
    it runs on CPU (the native backend is the NPU path).
    """

    name = "rapidocr"

    def __init__(self, config: Mapping[str, Any] | None = None):
        if config is None:
            from kavach_config import get_config
            config = get_config()
        ocfg, rt = config["ocr"], config["runtime"]
        self.min_confidence = float(ocfg.get("min_confidence", 0.5))
        self.use_cls = bool(ocfg.get("use_cls", False))
        self.requested_key = str(rt.get("provider", "cpu")).lower()
        self.latency = runtime.LatencyStats()
        self._model_paths = {
            "det_model_path": _resolve(ocfg.get("det_model")),
            "rec_model_path": _resolve(ocfg.get("rec_model")),
            "cls_model_path": _resolve(ocfg.get("cls_model")),
        }
        self.model_source = "weights/ocr" if all(self._model_paths.values()) else "rapidocr package"

        flags = self._provider_flags(self.requested_key)
        try:
            self.engine = self._build(flags)
            if flags:
                self._smoke_test()          # DML/CUDA can fail at first run, not at session creation
        except Exception as e:
            if not flags or not rt.get("fallback_to_cpu", True):
                raise
            log.warning("rapidocr on %s failed (%s); retrying on CPU", self.requested_key, type(e).__name__)
            self.engine = self._build({})
        self.providers = self._session_providers()
        self.provider = self._describe_provider()
        if self.requested_key != "cpu" and "CPU" in self.provider.split("=")[-1] and self.requested_key != "qnn":
            log.warning("rapidocr: requested %s but running on %s", self.requested_key, self.provider)
        runtime.register_external_stats("ocr.rapidocr", self.stats)

    def _provider_flags(self, key: str) -> dict[str, bool]:
        if key == "qnn":
            log.warning("rapidocr has no QNN support; OCR runs on CPU (use ocr.backend: native for the NPU)")
            return {}
        if key in ("dml", "cuda"):
            return {f"{stage}_use_{key}": True for stage in ("det", "cls", "rec")}
        return {}

    @staticmethod
    def _quiet_rapidocr_loggers() -> None:
        # rapidocr's get_logger() (lru_cached) forces DEBUG + its own handler; fetch the cached
        # loggers once and raise them to WARNING so our output is not flooded with INFO lines.
        from rapidocr_onnxruntime.utils.logger import get_logger

        for name in ("OrtInferSession", "RapidOCR"):
            lg = get_logger(name)
            lg.setLevel(logging.WARNING)
            lg.propagate = False

    def _build(self, flags: Mapping[str, bool]):
        from rapidocr_onnxruntime import RapidOCR

        self._quiet_rapidocr_loggers()
        kwargs: dict[str, Any] = {"text_score": self.min_confidence, "use_cls": self.use_cls, **flags}
        for k, p in self._model_paths.items():
            if p is not None:
                kwargs[k] = str(p)
        if "rec_model_path" in kwargs:
            keys = _resolve("weights/ocr/keys.txt")
            if keys:
                kwargs["rec_keys_path"] = str(keys)
        return RapidOCR(**kwargs)

    def _smoke_test(self) -> None:
        import cv2

        img = np.full((64, 256, 3), 255, np.uint8)
        cv2.putText(img, "Kavach 123", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        self.engine(img, use_cls=self.use_cls)

    def _session_providers(self) -> dict[str, str]:
        found: dict[str, str] = {}
        stages = {"det": ("text_det", "infer"), "rec": ("text_rec", "session"), "cls": ("text_cls", "infer")}
        for stage, (a, b) in stages.items():
            try:
                sess = getattr(getattr(self.engine, a), b).session
                found[stage] = sess.get_providers()[0]
            except (AttributeError, IndexError):
                pass
        return found

    def _describe_provider(self) -> str:
        used = {k: v for k, v in self.providers.items() if k != "cls" or self.use_cls}
        if not used:
            return "cpu (rapidocr)"
        vals = set(used.values())
        if len(vals) == 1:
            return vals.pop()
        return ", ".join(f"{k}={v}" for k, v in used.items())

    def __call__(self, image: np.ndarray) -> OcrResult:
        t0 = time.perf_counter()
        raw, elapse = self.engine(image, use_cls=self.use_cls)
        total_ms = (time.perf_counter() - t0) * 1000.0
        self.latency.record(total_ms)

        lines = []
        for item in raw or []:
            box, text, score = item[0], item[1], float(item[2])
            if score >= self.min_confidence and text.strip():
                lines.append(OcrLine(text, score, [[float(x), float(y)] for x, y in box]))
        det_s, cls_s, rec_s = (list(elapse or []) + [0.0, 0.0, 0.0])[:3]
        return OcrResult(
            lines=lines, full_text=join_text(lines),
            det_ms=det_s * 1000.0, rec_ms=rec_s * 1000.0, cls_ms=cls_s * 1000.0, total_ms=total_ms,
            backend=self.name, provider=self.provider,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "requested_provider": runtime.PROVIDER_NAMES.get(self.requested_key, self.requested_key),
            "actual_provider": self.provider,
            **self.latency.summary(),
        }


def create_backend(config: Mapping[str, Any] | None = None) -> OcrBackend:
    if config is None:
        from kavach_config import get_config
        config = get_config()
    backend = config["ocr"].get("backend", "rapidocr")
    if backend == "rapidocr":
        return RapidOcrBackend(config)
    if backend == "native":
        from models.ocr_native import NativeOcrBackend  # static det/rec via models.runtime

        return NativeOcrBackend(config)
    raise ValueError(f"unknown ocr.backend {backend!r}")


# ---------------------------------------------------------------- CLI

def synthetic_image(text: str = "Enter OTP 482913") -> np.ndarray:
    import cv2

    img = np.full((160, 900, 3), 255, np.uint8)
    cv2.putText(img, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    return img


def _main(argv: list[str] | None = None) -> int:
    import cv2

    from kavach_config import get_config
    from kavach_privacy import redact_text

    p = argparse.ArgumentParser(prog="python -m models.ocr", description="OCR self-test")
    p.add_argument("--image", help="image file to OCR (read-only; text shown only if debug_show_text)")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    t0 = time.perf_counter()
    ocr = create_backend(get_config())
    print(f"backend: {ocr.name}  provider: {ocr.provider}  models: {getattr(ocr, 'model_source', '?')}  "
          f"init: {(time.perf_counter() - t0) * 1000:.0f} ms")

    image = cv2.imread(args.image) if args.image else synthetic_image()
    if image is None:
        print(f"cannot read {args.image}")
        return 1
    for _ in range(max(1, args.runs)):
        result = ocr(image)
    print(f"lines: {len(result.lines)}  det: {result.det_ms:.1f} ms  rec: {result.rec_ms:.1f} ms  "
          f"total: {result.total_ms:.1f} ms")
    if args.image:
        print(redact_text(result.full_text))
    else:
        folded = fold_confusables(result.full_text)  # Hershey "O" reads as "0"
        print(f"synthetic text recognised: {'OTP' in folded and '482913' in folded}")
    print(f"stats: {ocr.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
