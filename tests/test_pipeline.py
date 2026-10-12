"""Exercise the prepare/infer/evaluate boundary without GPU or company data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_pipeline.common import read_jsonl, write_jsonl
from ocr_pipeline.evaluation import evaluate
from ocr_pipeline.inference import run_inference
from ocr_pipeline.preprocess import prepare_images


class FakeOCR:
    def __init__(self, run_dir: Path, max_tokens: int) -> None:
        pass

    def predict(self, image_path: Path, sample_id: str) -> str:
        assert image_path.is_file()
        return f"text for {sample_id}"


class PipelineTest(unittest.TestCase):
    def test_prepare_infer_resume_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            for name in ("a", "b"):
                Image.new("RGB", (112, 84), color="white").save(raw / f"{name}.jpg")

            manifest = prepare_images(raw, root / "prepared", 3136)
            prepared = read_jsonl(manifest)
            self.assertEqual([row["id"] for row in prepared], ["a", "b"])
            self.assertTrue((manifest.parent / prepared[0]["image"]).is_file())

            run = root / "run"
            model = "tests.test_pipeline:FakeOCR"
            self.assertEqual(run_inference(manifest, run, model, "0", 0, 2, 100), (1, 0))
            self.assertEqual(run_inference(manifest, run, model, "1", 1, 2, 100), (1, 0))
            self.assertEqual(run_inference(manifest, run, model, "0", 0, 2, 100), (0, 0))

            labels = root / "labels.jsonl"
            write_jsonl(labels, [{"id": name, "text": f"text for {name}"} for name in ("a", "b")])
            metrics = evaluate(run, labels)
            self.assertEqual(metrics["scored"], 2)
            self.assertEqual(metrics["cer"], 0)
            self.assertEqual(metrics["wer"], 0)
            self.assertEqual(metrics["exact_match"], 1)


if __name__ == "__main__":
    unittest.main()
