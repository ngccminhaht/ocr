# Local OCR data

Place company images in `data/raw/images/`. Keep filenames unique, including
across extensions (`page1.jpg` and `page1.png` would collide in the parser).

`data/labels/` holds ground truth JSONL (`id`, `text`). `data/processed/`
contains normalized images and a manifest shared by all models. OCR predictions,
native outputs, metrics and logs go in `data/runs/`, one subdirectory per model.

These four directories are ignored by Git. They may contain private documents,
large files, or generated results. Only this README is intended for the repo.

The upstream model's `demo/` and `assets/` are under `models/dots_mocr/`;
they are not company data.
