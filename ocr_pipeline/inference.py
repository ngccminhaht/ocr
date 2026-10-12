"""Phase 2: run any OCR adapter over a prepared manifest."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

from ocr_pipeline.common import read_jsonl


def load_adapter(spec: str, run_dir: Path, max_tokens: int):
    if ":" not in spec:
        raise ValueError("Model must be a Python class path such as ocr_pipeline.models.dots_mocr:DotsMOCR")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(run_dir=run_dir, max_tokens=max_tokens)


def run_inference(
    manifest_path: Path,
    run_dir: Path,
    model_spec: str,
    gpu: str,
    shard: int,
    shards: int,
    max_tokens: int,
    limit: int | None = None,
) -> tuple[int, int]:
    if not 0 <= shard < shards:
        raise ValueError("shard must be between 0 and shards-1")
    manifest_path = manifest_path.resolve()
    run_dir = run_dir.resolve()
    rows = read_jsonl(manifest_path)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {manifest_path}")
    selected = rows[shard::shards]
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / f"predictions_shard{shard}.jsonl"
    previous = read_jsonl(predictions_path) if predictions_path.exists() else []
    if any(row.get("model") != model_spec for row in previous):
        raise ValueError(f"Existing predictions in {predictions_path} use another model; choose another run directory")
    completed = {row["id"] for row in previous if row.get("status") == "ok"}
    pending = [row for row in selected if row["id"] not in completed]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        pending = pending[:limit]
    print(f"GPU {gpu}, shard {shard}/{shards}: {len(selected)} assigned, {len(pending)} pending", flush=True)
    if not pending:
        return 0, 0

    # Set CUDA visibility before importing the adapter or PyTorch.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    adapter = load_adapter(model_spec, run_dir, max_tokens)
    successes = 0
    failures = 0
    for index, row in enumerate(pending, start=1):
        image_path = (manifest_path.parent / row["image"]).resolve()
        start = time.monotonic()
        try:
            prediction = adapter.predict(image_path, row["id"])
            status = "ok"
            error = None
            successes += 1
        except Exception as exc:
            prediction = None
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            failures += 1
        result = {
            "id": row["id"],
            "model": model_spec,
            "image": row["image"],
            "prediction": prediction,
            "status": status,
            "error": error,
            "seconds": round(time.monotonic() - start, 2),
            "gpu": gpu,
        }
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(pending)}] {row['id']}: {status} ({result['seconds']}s)", flush=True)
    return successes, failures
