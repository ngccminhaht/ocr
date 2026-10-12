# OCR evaluation workspace

This is the **model-neutral project root**. Each OCR model lives under `models/`
as a local dependency; reusable preparation, inference and evaluation code is
in `ocr_pipeline/`.

```text
ocr-arabic/
├── ocr_pipeline/       Shared prepare, infer and evaluate phases
│   └── models/         One small adapter per OCR model
├── models/             Local model repositories/weights (ignored by Git)
│   └── dots_mocr/      Current upstream dots.mocr clone
├── data/               Images, ground truth, processed data and runs
├── tools/              Model-specific setup helpers
└── tests/              Tests using generated data; no company documents
```

Read [LOCAL_OCR.md](LOCAL_OCR.md) for exact commands. The `dots_mocr` directory
name identifies one model; it is no longer the name of the whole project.

Adding a text OCR model requires a new adapter in `ocr_pipeline/models/` with
`predict(image_path, sample_id) -> str`. The common prepare and evaluate phases
can use the same images and labels for all text OCR models.
