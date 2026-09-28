from pathlib import Path

import cv2
import numpy as np
import pytest

from models import ocr as O
from models import ocr_native as N
from models import runtime

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = [320, 640, 1280]
STATIC = [ROOT / "weights" / "ocr" / n for n in
          ("det_static.onnx", *(f"rec_static_{w}.onnx" for w in BUCKETS), "keys.txt")]
needs_static = pytest.mark.skipif(
    not all(p.is_file() for p in STATIC), reason="run scripts/fetch_ocr_models.py and scripts/fix_ocr_shapes.py"
)


# ---------------------------------------------------------------- pure pieces

def test_letterbox_det_keeps_aspect_and_pads():
    img = np.full((1020, 1920, 3), 255, np.uint8)
    tensor, (vh, vw) = N.letterbox_det(img, 736, 1280)
    assert tensor.shape == (1, 3, 736, 1280) and tensor.dtype == np.float32
    assert (vh, vw) == (680, 1280)                                # scale 2/3
    assert tensor[0, :, :vh, :vw].max() == pytest.approx(1.0)     # white -> +1
    assert tensor[0, :, vh:, :].min() == pytest.approx(-1.0)      # black padding -> -1

    small = np.full((160, 900, 3), 255, np.uint8)                 # upscaled to fit width
    _, (vh, vw) = N.letterbox_det(small, 736, 1280)
    assert vw == 1280 and vh == round(160 * 1280 / 900)


def test_global_preprocess_letterboxes_wide_strips():
    img, op = N.global_preprocess(np.zeros((60, 1920, 3), np.uint8))   # taskbar-like, w/h > 8
    assert op["padding_1"]["top"] > 0 and img.shape[0] > 60
    img, op = N.global_preprocess(np.zeros((1020, 1920, 3), np.uint8))
    assert op == {"preprocess": {"ratio_h": 1.0, "ratio_w": 1.0}, "padding_1": {"top": 0, "left": 0}}


def test_get_origin_points_undoes_padding():
    op = {"preprocess": {"ratio_h": 1.0, "ratio_w": 1.0}, "padding_1": {"top": 90, "left": 0}}
    box = np.array([[[10, 100], [50, 100], [50, 120], [10, 120]]], np.float32)
    out = N.get_origin_points(box, op, raw_h=60, raw_w=1920)
    assert out[0].tolist() == [[10, 10], [50, 10], [50, 30], [10, 30]]


@pytest.mark.parametrize("width,expected", [(100, (640, False)), (640, (640, False)),
                                            (641, (1280, False)), (1280, (1280, False)), (1913, (1280, True))])
def test_assign_bucket(width, expected):
    assert N.assign_bucket(width, [640, 1280]) == expected


def test_normalize_rec_pads_and_squashes():
    crop = np.full((24, 100, 3), 255, np.uint8)                   # -> width 200 at h=48
    out = N.normalize_rec(crop, 640)
    assert out.shape == (3, 48, 640)
    assert out[:, :, :200].min() == pytest.approx(1.0) and np.all(out[:, :, 200:] == 0)
    wide = np.full((20, 1000, 3), 255, np.uint8)                  # 2400 wide -> squashed to 1280
    assert N.normalize_rec(wide, 1280).min() == pytest.approx(1.0)


def test_ctc_decode_merges_repeats_and_drops_blank():
    chars = ["blank", "O", "T", "P", " "]
    steps = [1, 1, 0, 2, 2, 0, 0, 3, 4]                          # O O _ T T _ _ P ' '
    preds = np.zeros((1, len(steps), len(chars)), np.float32)
    preds[0, np.arange(len(steps)), steps] = 0.9
    [(text, conf)] = N.ctc_decode(preds, chars)
    assert text == "OTP " and conf == pytest.approx(0.9)
    [(text, conf)] = N.ctc_decode(np.zeros((1, 3, 5), np.float32), chars)    # all blank
    assert text == "" and conf == 0


def test_sorted_boxes_row_then_x():
    def b(x, y):
        return np.array([[x, y], [x + 10, y], [x + 10, y + 5], [x, y + 5]], np.float32)
    out = N.sorted_boxes(np.array([b(100, 3), b(0, 50), b(0, 0)]))
    assert [tuple(o[0]) for o in out] == [(0, 0), (100, 3), (0, 50)]    # same row within 10 px


def test_rotate_crop_turns_tall_crops():
    img = np.zeros((200, 200, 3), np.uint8)
    pts = np.array([[10, 10], [30, 10], [30, 110], [10, 110]], np.float32)   # 20 wide, 100 tall
    assert N.get_rotate_crop_image(img, pts).shape[:2] == (20, 100)


# ---------------------------------------------------------------- end to end (CPU)

@pytest.fixture(scope="module")
def native():
    cfg = {"ocr": {"backend": "native", "min_confidence": 0.5, "rec_height": 48,
                   "rec_buckets": BUCKETS, "keys": "weights/ocr/keys.txt"},
           "runtime": {"provider": "cpu", "fallback_to_cpu": True}}
    b = O.create_backend(cfg)
    yield b
    runtime.clear_registry()


@needs_static
def test_native_finds_otp(native):
    img = np.full((160, 900, 3), 255, np.uint8)
    cv2.putText(img, "Enter OTP 482913", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    r = native(img)
    assert "OTP" in O.fold_confusables(r.full_text) and "482913" in r.full_text
    assert r.backend == "native" and r.provider == "CPUExecutionProvider"
    counts = r.extra["buckets"]
    assert sum(counts[str(w)] for w in BUCKETS) == len(r.lines) == 1 and counts["squashed"] == 0
    x0, y0 = r.lines[0].box[0]
    assert 0 <= x0 < 120 and 40 <= y0 < 110                       # box mapped back to input pixels


@needs_static
def test_native_sessions_in_registry(native):
    native(np.full((100, 300, 3), 255, np.uint8))
    names = {s["name"]: s for s in runtime.all_stats()}
    assert {"ocr.det", *(f"ocr.rec_{w}" for w in BUCKETS), "ocr.native"} <= set(names)
    assert native.rec_batch == {320: 8, 640: 8, 1280: 4}           # read from the static models
    assert names["ocr.det"]["actual_provider"] == "CPUExecutionProvider"
