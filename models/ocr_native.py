"""Native OCR backend: static-shape det/rec ONNX models loaded through models.runtime (qnn | dml | cuda | cpu).

Pre/post-processing mirrors rapidocr_onnxruntime 1.4.4 (Global preprocess + letterbox, DetPreProcess
normalisation, DBPostProcess, filter_tag_det_res, sorted_boxes, get_rotate_crop_image,
TextRecognizer.resize_norm_img, CTCLabelDecode). Differences, forced by static shapes:
  - det: the frame is letterboxed (scaled to fit, black padding bottom/right) into ocr.det_input_hw
    instead of rapidocr's "min side >= 736, round to 32" resize; boxes are mapped back by the scale.
  - rec: each line goes to the smallest ocr.rec_buckets width that fits it at height 48 (wider lines
    are squashed to the largest bucket); the batch size per bucket is read from each static model
    (ocr.rec_batch at export time) and partial batches are zero-padded.
Models: weights/ocr/det_static.onnx, weights/ocr/rec_static_<W>.onnx (scripts/fix_ocr_shapes.py),
keys: weights/ocr/keys.txt (scripts/fetch_ocr_models.py).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from models import runtime
from models.ocr import OcrLine, OcrResult, join_text

log = logging.getLogger("models.ocr_native")
ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "weights" / "ocr"

# rapidocr 1.4.4 config.yaml defaults (Global / Det)
MAX_SIDE_LEN, MIN_SIDE_LEN, MIN_HEIGHT, WIDTH_HEIGHT_RATIO = 2000, 30, 30, 8
DB_THRESH, DB_BOX_THRESH, DB_MAX_CANDIDATES, DB_UNCLIP_RATIO = 0.3, 0.5, 1000, 1.6
DB_MIN_SIZE = 3


# ---------------------------------------------------------------- global preprocess (rapidocr main.py)

def _resize_round32(img: np.ndarray, ratio: float) -> tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    rh = int(round(int(h * ratio) / 32) * 32)
    rw = int(round(int(w * ratio) / 32) * 32)
    if rh <= 0 or rw <= 0:
        raise ValueError("resize to zero size")
    return cv2.resize(img, (rw, rh)), h / rh, w / rw


def global_preprocess(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """rapidocr RapidOCR.preprocess + maybe_add_letterbox. Returns (image, op_record)."""
    ratio_h = ratio_w = 1.0
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE_LEN:
        img, ratio_h, ratio_w = _resize_round32(img, MAX_SIDE_LEN / (h if h > w else w))
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE_LEN:
        img, ratio_h, ratio_w = _resize_round32(img, MIN_SIDE_LEN / (h if h < w else w))
    op: dict = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}

    h, w = img.shape[:2]
    if h <= MIN_HEIGHT or w / h > WIDTH_HEIGHT_RATIO:
        new_h = max(int(w / WIDTH_HEIGHT_RATIO), MIN_HEIGHT) * 2
        pad = int(abs(new_h - h) / 2)
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        op["padding_1"] = {"top": pad, "left": 0}
    else:
        op["padding_1"] = {"top": 0, "left": 0}
    return img, op


def get_origin_points(boxes: np.ndarray, op: dict, raw_h: int, raw_w: int) -> np.ndarray:
    """rapidocr RapidOCR._get_origin_points: undo letterbox padding and preprocess resize."""
    arr = np.array(boxes).astype(np.float32)
    for name in reversed(list(op.keys())):
        v = op[name]
        if "padding" in name:
            arr[:, :, 0] -= v["left"]
            arr[:, :, 1] -= v["top"]
        elif "preprocess" in name:
            arr[:, :, 0] *= v["ratio_w"]
            arr[:, :, 1] *= v["ratio_h"]
    arr = np.where(arr < 0, 0, arr)
    arr[..., 0] = np.where(arr[..., 0] > raw_w, raw_w, arr[..., 0])
    arr[..., 1] = np.where(arr[..., 1] > raw_h, raw_h, arr[..., 1])
    return arr


# ---------------------------------------------------------------- det

def normalize_det(img: np.ndarray) -> np.ndarray:
    """DetPreProcess.normalize + permute: (x/255 - 0.5) / 0.5, HWC -> CHW."""
    return ((img.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)


def letterbox_det(img: np.ndarray, out_h: int, out_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Scale img to fit (out_h, out_w) keeping aspect, pad bottom/right with black.

    Returns (tensor [1,3,out_h,out_w] float32, (valid_h, valid_w)). The valid region is what
    DBPostProcess maps back to the source size, like rapidocr's resized det input.
    """
    h, w = img.shape[:2]
    s = min(out_h / h, out_w / w)
    vh, vw = min(out_h, max(1, int(round(h * s)))), min(out_w, max(1, int(round(w * s))))
    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    canvas[:vh, :vw] = cv2.resize(img, (vw, vh))
    return normalize_det(canvas)[np.newaxis], (vh, vw)


class DBPostProcess:
    """rapidocr 1.4.4 DBPostProcess (score_mode fast, dilation on)."""

    def __init__(self, thresh=DB_THRESH, box_thresh=DB_BOX_THRESH, max_candidates=DB_MAX_CANDIDATES,
                 unclip_ratio=DB_UNCLIP_RATIO, use_dilation=True):
        self.thresh, self.box_thresh = thresh, box_thresh
        self.max_candidates, self.unclip_ratio = max_candidates, unclip_ratio
        self.min_size = DB_MIN_SIZE
        self.dilation_kernel = np.array([[1, 1], [1, 1]]) if use_dilation else None

    def __call__(self, pred: np.ndarray, ori_shape: tuple[int, int]) -> tuple[np.ndarray, list[float]]:
        src_h, src_w = ori_shape
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        mask = segmentation[0]
        if self.dilation_kernel is not None:
            mask = cv2.dilate(np.array(segmentation[0]).astype(np.uint8), self.dilation_kernel)
        return self.boxes_from_bitmap(pred[0], mask, src_w, src_h)

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[1] if len(outs) == 3 else outs[0]
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            try:
                box = self.unclip(points)
            except ValueError:  # pyclipper returned several/no polygons (rapidocr would raise)
                continue
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32), scores

    @staticmethod
    def get_mini_boxes(contour: np.ndarray) -> tuple[np.ndarray, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        i1, i4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
        i2, i3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
        return np.array([points[i1], points[i2], points[i3], points[i4]]), min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin: ymax + 1, xmin: xmax + 1], mask)[0]

    def unclip(self, box: np.ndarray) -> np.ndarray:
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance)).reshape((-1, 1, 2))


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left, right = x_sorted[:2, :], x_sorted[2:, :]
    tl, bl = left[np.argsort(left[:, 1]), :]
    tr, br = right[np.argsort(right[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")


def filter_tag_det_res(dt_boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    img_h, img_w = image_shape
    out = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        for p in range(box.shape[0]):
            box[p, 0] = int(min(max(box[p, 0], 0), img_w - 1))
            box[p, 1] = int(min(max(box[p, 1], 0), img_h - 1))
        if int(np.linalg.norm(box[0] - box[1])) <= 3 or int(np.linalg.norm(box[0] - box[3])) <= 3:
            continue
        out.append(box)
    return np.array(out)


def sorted_boxes(dt_boxes: np.ndarray) -> list[np.ndarray]:
    """rapidocr RapidOCR.sorted_boxes: top-to-bottom, then left-to-right within 10 px."""
    boxes = list(sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0])))
    for i in range(len(boxes) - 1):
        for j in range(i, -1, -1):
            if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and boxes[j + 1][0][0] < boxes[j][0][0]:
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return dst


# ---------------------------------------------------------------- rec

def rec_width(crop: np.ndarray, rec_h: int = 48) -> int:
    """Width after resizing to rec_h keeping aspect (TextRecognizer.resize_norm_img: ceil)."""
    h, w = crop.shape[:2]
    return int(math.ceil(rec_h * w / float(h)))


def assign_bucket(width: int, buckets: Sequence[int]) -> tuple[int, bool]:
    """(bucket, squashed): smallest bucket >= width, else the largest (squashed)."""
    for b in buckets:
        if width <= b:
            return b, False
    return buckets[-1], True


def normalize_rec(crop: np.ndarray, bucket_w: int, rec_h: int = 48) -> np.ndarray:
    """resize_norm_img into a fixed bucket: resize to height rec_h, (x/255-0.5)/0.5, zero-pad right."""
    resized_w = min(rec_width(crop, rec_h), bucket_w)
    img = cv2.resize(crop, (resized_w, rec_h)).astype(np.float32).transpose(2, 0, 1) / 255
    img = (img - 0.5) / 0.5
    out = np.zeros((3, rec_h, bucket_w), dtype=np.float32)
    out[:, :, :resized_w] = img
    return out


def load_keys(path: str | Path) -> list[str]:
    """CTCLabelDecode.read_character_file + blank at 0 and space at the end."""
    chars = []
    with open(path, "rb") as f:
        for line in f.readlines():
            chars.append(line.decode("utf-8").strip("\n").strip("\r\n"))
    return ["blank", *chars, " "]


def ctc_decode(preds: np.ndarray, characters: Sequence[str]) -> list[tuple[str, float]]:
    """Greedy CTC: argmax, merge repeats, drop blank (0); confidence = mean prob of kept steps."""
    idx, prob = preds.argmax(axis=2), preds.max(axis=2)
    out = []
    for b in range(len(idx)):
        sel = np.ones(len(idx[b]), dtype=bool)
        sel[1:] = idx[b][1:] != idx[b][:-1]
        sel &= idx[b] != 0
        conf = prob[b][sel].tolist() or [0]
        out.append(("".join(characters[i] for i in idx[b][sel]), float(np.mean(conf))))
    return out


# ---------------------------------------------------------------- backend

class NativeOcrBackend:
    name = "native"

    def __init__(self, config: Mapping[str, Any] | None = None):
        if config is None:
            from kavach_config import get_config
            config = get_config()
        ocfg, rt = config["ocr"], config["runtime"]
        self.min_confidence = float(ocfg.get("min_confidence", 0.5))
        self.rec_h = int(ocfg.get("rec_height", 48))
        self.buckets = sorted(int(w) for w in ocfg.get("rec_buckets", [640, 1280]))
        self.characters = load_keys(ROOT / ocfg.get("keys", "weights/ocr/keys.txt"))
        self.db = DBPostProcess()
        self.latency = runtime.LatencyStats()

        def load(path: Path, name: str) -> runtime.KavachSession:
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing; run scripts/fix_ocr_shapes.py "
                                        f"(or scripts/fetch_qnn_models.py for weights/qnn/)")
            s = runtime.create_session(path, name, runtime_cfg=rt)
            log.info("%s: requested %s, running on %s", name, s.requested_provider, s.actual_provider)
            return s

        # config ocr.native_det / ocr.native_rec ({w} = bucket width); default: the plain static exports
        det_path = ROOT / ocfg.get("native_det", "weights/ocr/det_static.onnx")
        rec_path = str(ocfg.get("native_rec", "weights/ocr/rec_static_{w}.onnx"))
        self.det = load(det_path, "ocr.det")
        self.det_input = self.det.input_specs()[0]["name"]
        _, _, self.det_h, self.det_w = self.det.input_specs()[0]["shape"]
        self.rec = {w: load(ROOT / rec_path.format(w=w), f"ocr.rec_{w}") for w in self.buckets}
        self.rec_input = {w: s.input_specs()[0]["name"] for w, s in self.rec.items()}
        self.rec_batch = {w: int(s.input_specs()[0]["shape"][0]) for w, s in self.rec.items()}

        providers = {"det": self.det.actual_provider, **{f"rec_{w}": s.actual_provider for w, s in self.rec.items()}}
        vals = set(providers.values())
        self.provider = vals.pop() if len(vals) == 1 else ", ".join(f"{k}={v}" for k, v in providers.items())
        self.requested_provider = self.det.requested_provider
        runtime.register_external_stats("ocr.native", self.stats)

    # -- stages

    def detect(self, img: np.ndarray) -> np.ndarray:
        tensor, (vh, vw) = letterbox_det(img, self.det_h, self.det_w)
        pred = self.det.run({self.det_input: tensor})[0][:, :, :vh, :vw]
        boxes, _ = self.db(pred, img.shape[:2])
        return filter_tag_det_res(boxes, img.shape[:2])

    def recognize(self, crops: Sequence[np.ndarray]) -> tuple[list[tuple[str, float]], dict]:
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        groups: dict[int, list[int]] = {w: [] for w in self.buckets}
        squashed = 0
        widths = [rec_width(c, self.rec_h) for c in crops]
        for i, w in enumerate(widths):
            b, sq = assign_bucket(w, self.buckets)
            groups[b].append(i)
            squashed += sq
        for b, idxs in groups.items():
            idxs.sort(key=lambda i: widths[i])
            n = self.rec_batch[b]
            for start in range(0, len(idxs), n):
                chunk = idxs[start: start + n]
                batch = np.zeros((n, 3, self.rec_h, b), dtype=np.float32)  # partial batch: zero-padded
                for k, i in enumerate(chunk):
                    batch[k] = normalize_rec(crops[i], b, self.rec_h)
                preds = self.rec[b].run({self.rec_input[b]: batch})[0]
                for i, res in zip(chunk, ctc_decode(preds[: len(chunk)], self.characters)):
                    results[i] = res
        counts = {str(b): len(v) for b, v in groups.items()}
        counts["squashed"] = squashed
        return results, counts

    def __call__(self, image: np.ndarray) -> OcrResult:
        t0 = time.perf_counter()
        raw_h, raw_w = image.shape[:2]
        img, op = global_preprocess(image)
        boxes = self.detect(img)
        t1 = time.perf_counter()
        lines: list[OcrLine] = []
        counts = {str(b): 0 for b in self.buckets} | {"squashed": 0}
        if len(boxes):
            boxes = sorted_boxes(boxes)
            crops = [get_rotate_crop_image(img, b.copy()) for b in boxes]
            rec, counts = self.recognize(crops)
            keep = [(b, r) for b, r in zip(boxes, rec) if r[1] >= self.min_confidence and r[0].strip()]
            if keep:
                origin = get_origin_points([b for b, _ in keep], op, raw_h, raw_w)
                lines = [OcrLine(text, conf, box.tolist()) for box, (text, conf) in zip(origin, [r for _, r in keep])]
        t2 = time.perf_counter()
        total_ms = (t2 - t0) * 1000.0
        self.latency.record(total_ms)
        return OcrResult(
            lines=lines, full_text=join_text(lines),
            det_ms=(t1 - t0) * 1000.0, rec_ms=(t2 - t1) * 1000.0, total_ms=total_ms,
            backend=self.name, provider=self.provider, extra={"buckets": counts},
        )

    def stats(self) -> dict[str, Any]:
        return {"requested_provider": self.requested_provider, "actual_provider": self.provider,
                **self.latency.summary()}
