"""OCR wrapper. Backend 'rapidocr' (third-party, creates its own onnxruntime sessions — the one allowed exception) or 'native' (det/rec ONNX models via models.runtime)."""

# TODO: implement rapidocr backend and native det/rec pipeline; filter by ocr.min_confidence.


if __name__ == "__main__":
    print("models.ocr: stub, not implemented yet")
