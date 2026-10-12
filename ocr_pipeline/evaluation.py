"""Phase 3: score OCR text against JSONL ground truth."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from ocr_pipeline.common import read_jsonl


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def evaluate(run_dir: Path, labels_path: Path, output_path: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    labels = read_jsonl(labels_path.resolve())
    label_ids = [row["id"] for row in labels]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError(f"Duplicate IDs in labels: {labels_path}")
    predictions: dict[str, dict] = {}
    files = sorted(run_dir.glob("predictions_shard*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {run_dir}")
    for path in files:
        for row in read_jsonl(path):
            predictions[row["id"]] = row

    total_char_edits = total_chars = total_word_edits = total_words = exact = scored = 0
    missing = []
    for label in labels:
        row = predictions.get(label["id"])
        if row is None or row.get("status") != "ok":
            missing.append(label["id"])
            continue
        truth = normalize(label["text"])
        guess = normalize(row["prediction"])
        total_char_edits += edit_distance(truth, guess)
        total_chars += len(truth)
        total_word_edits += edit_distance(truth.split(), guess.split())
        total_words += len(truth.split())
        exact += truth == guess
        scored += 1
    summary = {
        "labels": len(labels),
        "scored": scored,
        "missing_or_failed": missing,
        "cer": total_char_edits / max(total_chars, 1),
        "wer": total_word_edits / max(total_words, 1),
        "exact_match": exact / max(scored, 1),
        "normalization": "Unicode NFC; collapse whitespace; case and punctuation preserved",
    }
    output_path = output_path or run_dir / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
