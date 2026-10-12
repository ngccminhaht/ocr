"""Run OCR pipeline phases independently: prepare, infer, evaluate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_pipeline.common import REPO_ROOT


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)

    prepare = phases.add_parser("prepare", help="CPU: normalize images and write manifest")
    prepare.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "raw" / "images")
    prepare.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "processed" / "v1")
    prepare.add_argument("--max-pixels", type=int, default=262144)

    infer = phases.add_parser("infer", help="GPU: run one persistent model worker")
    infer.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "processed" / "v1" / "manifest.jsonl")
    infer.add_argument("--run", type=Path, default=REPO_ROOT / "data" / "runs" / "dots_mocr_v1")
    infer.add_argument("--model", default="ocr_pipeline.models.dots_mocr:DotsMOCR")
    infer.add_argument("--gpu", choices=("0", "1"), required=True)
    infer.add_argument("--shard", type=int, required=True)
    infer.add_argument("--shards", type=int, default=2)
    infer.add_argument("--max-tokens", type=int, default=4096)
    infer.add_argument("--limit", type=int, help="Process at most N pending images (smoke test)")

    score = phases.add_parser("evaluate", help="CPU: calculate CER/WER from ground truth")
    score.add_argument("--run", type=Path, default=REPO_ROOT / "data" / "runs" / "dots_mocr_v1")
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--output", type=Path)

    args = cli.parse_args()
    if args.phase == "prepare":
        from ocr_pipeline.preprocess import prepare_images

        print(prepare_images(args.input, args.output, args.max_pixels))
    elif args.phase == "infer":
        from ocr_pipeline.inference import run_inference

        ok, failed = run_inference(args.manifest, args.run, args.model, args.gpu, args.shard, args.shards, args.max_tokens, args.limit)
        print(f"Done: {ok} successful, {failed} failed")
    else:
        from ocr_pipeline.evaluation import evaluate

        print(json.dumps(evaluate(args.run, args.labels, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
