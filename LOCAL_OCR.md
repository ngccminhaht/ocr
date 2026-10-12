# OCR pipeline on two Tesla T4 GPUs

Work from `/root/ocr-arabic`, the generic project root. The current upstream
model clone is `models/dots_mocr/`. Reusable code in `ocr_pipeline/` has three
independent phases:

```text
data/raw/images → prepare (CPU) → data/processed/v1/manifest.jsonl
                                 → infer (CPU + GPU, one worker per GPU)
                                 → data/runs/dots_mocr_v1/predictions_shard*.jsonl
                                 → evaluate (CPU + ground truth)
                                 → data/runs/dots_mocr_v1/metrics.json
```

Activate the environment with `source .venv/bin/activate`.

## Model setup (once per download)

For a new machine, clone the model into `models/dots_mocr/` and install its
dependencies. On this machine the clone and weights are already present.

```bash
git clone https://github.com/rednote-hilab/dots.mocr.git models/dots_mocr
python -m pip install -r models/dots_mocr/requirements.txt
cd models/dots_mocr
python tools/download_model.py
cd ../..
python tools/prepare_t4.py
```

`prepare_t4.py` changes the downloaded model's Python code for T4/SDPA. It
does not change checkpoint tensors. Run it again after redownloading weights.

## Phase 1: prepare images

Put JPG/PNG files in `data/raw/images/`, then run:

```bash
python -m ocr_pipeline prepare
```

This writes normalized PNGs and a manifest to `data/processed/v1/`. Images
larger than the pixel limit are downsampled, which can affect small text.
The default maximum is 262144 pixels per image for T4 memory. For a different
resolution, use `--max-pixels N` and a new `--output data/processed/v2` path.
Preparation is done once and reused by every model on the same dataset.

## Phase 2: inference on two GPUs

Open two terminals. Run these commands at the same time:

```bash
python -m ocr_pipeline infer --gpu 0 --shard 0 --shards 2
```

```bash
python -m ocr_pipeline infer --gpu 1 --shard 1 --shards 2
```

Each worker loads one model copy and handles half of the manifest. Predictions
and per-image errors are in `data/runs/dots_mocr_v1/predictions_shard0.jsonl`
and `predictions_shard1.jsonl`. Re-running skips successful IDs. Default
generation limit is 4096 tokens per image; use `--max-tokens N` for dense pages.
For a short smoke test, add `--limit 1 --max-tokens 64` and use a separate
`--run data/runs/smoke_pipeline` directory.

## Phase 3: evaluation

Create `data/labels/ground_truth.jsonl` with one row per image:

```json
{"id":"img1","text":"Correct text on img1"}
```

Then run:

```bash
python -m ocr_pipeline evaluate --labels data/labels/ground_truth.jsonl
```

This writes `metrics.json` with CER, WER, exact match and missing/failed IDs.
Text is normalized to Unicode NFC and whitespace is collapsed; case and
punctuation still count. This phase needs ground truth and does not train a model.

## Adding another OCR model

Create a class in `ocr_pipeline/models/` with:

```python
class MyOCR:
    def __init__(self, run_dir, max_tokens):
        # Load the model once on the visible GPU.
        ...

    def predict(self, image_path, sample_id) -> str:
        # Return OCR text for one prepared image.
        ...
```

Run it with a new run directory:

```bash
python -m ocr_pipeline infer --model ocr_pipeline.models.my_ocr:MyOCR \
  --run data/runs/my_ocr_v1 --gpu 0 --shard 0 --shards 2
```

Start a second worker with GPU/shard `1`. The prepare and evaluate code needs
no model-specific changes for text OCR. Use the same manifest and labels for a
fair comparison. Keep a separate run directory for each model/configuration.
If a model outputs layout, tables or fields instead of plain text, add metrics
for that output type. The adapter can keep model-specific processing inside
its own file without changing the common prepare phase.
