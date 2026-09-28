from pathlib import Path

import cv2
import numpy as np
import pytest

from models import ocr as O
from models import runtime

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = [ROOT / "weights" / "ocr" / n for n in ("det.onnx", "rec.onnx")]
needs_weights = pytest.mark.skipif(
    not all(p.is_file() for p in WEIGHTS), reason="run scripts/fetch_ocr_models.py first"
)


def line(text, x, y, w=100, h=20, conf=0.9):
    return O.OcrLine(text, conf, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


# ---------------------------------------------------------------- pure helpers

def test_order_lines_rows_and_columns():
    lines = [
        line("IFSC", 10, 100), line("DEMO0001234", 300, 102),      # same row, slight y offset
        line("Beneficiary", 10, 50), line("Rahul", 300, 48),
        line("Transfer funds", 10, 0, h=30),
    ]
    assert O.join_text(lines) == "Transfer funds\nBeneficiary  Rahul\nIFSC  DEMO0001234"


def test_order_lines_empty():
    assert O.order_lines([]) == [] and O.join_text([]) == ""


def test_fold_confusables():
    assert O.fold_confusables("Enter 0TP 482913") == "ENTER OTP 482913"
    assert O.fold_confusables("L0GIN to acc0unt") == "LOGIN TO ACCOUNT"
    assert O.fold_confusables("Pay 5000 now") == "PAY 5000 NOW"      # numbers untouched


def test_unknown_backend():
    with pytest.raises(ValueError):
        O.create_backend({"ocr": {"backend": "tesseract"}, "runtime": {"provider": "cpu"}})


# ---------------------------------------------------------------- real OCR (rapidocr, CPU)

@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("rapidocr_onnxruntime")
    cfg = {
        "ocr": {"backend": "rapidocr", "min_confidence": 0.5, "use_cls": False,
                "det_model": "weights/ocr/det.onnx", "rec_model": "weights/ocr/rec.onnx",
                "cls_model": "weights/ocr/cls.onnx"},
        "runtime": {"provider": "cpu", "fallback_to_cpu": True},
    }
    b = O.RapidOcrBackend(cfg)
    yield b
    runtime.unregister_external_stats("ocr.rapidocr")


@needs_weights
def test_ocr_finds_otp_puttext(backend):
    img = np.full((160, 900, 3), 255, np.uint8)
    cv2.putText(img, "Enter OTP 482913", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    result = backend(img)
    # Hershey fonts draw "O" almost like "0"; PP-OCR reads "0TP", so match on folded text.
    assert "OTP" in O.fold_confusables(result.full_text)
    assert "482913" in result.full_text
    assert result.lines and all(l.confidence >= 0.5 for l in result.lines)
    assert result.backend == "rapidocr" and result.provider == "CPUExecutionProvider"
    assert result.total_ms > 0 and result.det_ms > 0


@needs_weights
def test_ocr_finds_otp_ui_font(backend):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("calibri.ttf", 56)
    except OSError:
        pytest.skip("calibri.ttf not available")
    img = Image.new("RGB", (900, 160), "white")
    ImageDraw.Draw(img).text((30, 50), "Enter OTP 482913", fill="black", font=font)
    result = backend(np.ascontiguousarray(np.array(img)[:, :, ::-1]))
    assert "OTP" in result.full_text                                   # real UI font: exact match


@needs_weights
def test_blank_image_and_stats_registered(backend):
    result = backend(np.full((200, 400, 3), 255, np.uint8))
    assert result.lines == [] and result.full_text == ""
    stats = {s["name"]: s for s in runtime.all_stats()}
    assert stats["ocr.rapidocr"]["n"] >= 1
    assert stats["ocr.rapidocr"]["actual_provider"] == "CPUExecutionProvider"


# ---------------------------------------------------------------- rec width measurement / static shapes

def test_line_widths_and_pick_width():
    import importlib
    m = importlib.import_module("scripts.measure_rec_widths")
    boxes = [[[0, 0], [96, 0], [96, 24], [0, 24]],          # 4:1 -> 192 at h=48
             [[0, 0], [300, 0], [300, 20], [0, 20]]]        # 15:1 -> 720
    assert m.line_widths(boxes).tolist() == [192.0, 720.0]
    assert m.pick_width(np.array([100.0] * 19 + [1144.0])) == 320      # p95 small -> floor 320
    assert m.pick_width(np.array([1144.0] * 20)) == 1152               # rounded up to 32, no cap
    assert m.pick_width(np.array([])) == 320


@needs_weights
def test_fix_ocr_shapes_static(tmp_path):
    import onnx
    fix = pytest.importorskip("scripts.fix_ocr_shapes")
    det = fix.fix_model(ROOT / "weights" / "ocr" / "det.onnx", tmp_path / "det_s.onnx", [1, 3, 96, 160])
    assert fix._dims(det.graph.input[0]) == [1, 3, 96, 160]
    rec_path = tmp_path / "rec_s.onnx"
    rec = fix.fix_model(ROOT / "weights" / "ocr" / "rec.onnx", rec_path, [2, 3, 48, 320])
    out_shapes, _ = fix.cpu_run(rec_path, "test.rec_static")
    assert out_shapes == [(2, 40, 6625)]
    fix.pin_output_shapes(rec, rec_path, out_shapes)
    reloaded = onnx.load(str(rec_path))
    assert fix._dims(reloaded.graph.output[0]) == [2, 40, 6625]
    runtime.clear_registry()


def test_bucket_batches():
    fix = pytest.importorskip("scripts.fix_ocr_shapes")
    assert fix.bucket_batches([320, 640], 8) == {320: 8, 640: 8}
    assert fix.bucket_batches([320, 640, 1280], {320: 8, 640: 8, 1280: 4}) == {320: 8, 640: 8, 1280: 4}
    with pytest.raises(SystemExit):
        fix.bucket_batches([320, 960], {320: 8})
