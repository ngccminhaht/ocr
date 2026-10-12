"""Phase 1: prepare images once and record a reusable manifest."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageOps

from ocr_pipeline.common import IMAGE_SUFFIXES, write_jsonl


def resized_dimensions(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Fit an image within a pixel budget without model-specific patch rules."""
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    while new_width * new_height > max_pixels:
        if new_width >= new_height and new_width > 1:
            new_width -= 1
        elif new_height > 1:
            new_height -= 1
        else:
            raise ValueError("Image cannot fit within max_pixels")
    return new_width, new_height


def prepare_images(source_dir: Path, output_dir: Path, max_pixels: int) -> Path:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {source_dir}")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")

    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No JPG/PNG images in {source_dir}")
    stems = [p.stem for p in files]
    if len(stems) != len(set(stems)):
        raise ValueError("Image names must have unique stems across extensions")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in files:
        destination = images_dir / f"{source.stem}.png"
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = image.size
            new_width, new_height = resized_dimensions(width, height, max_pixels)
            if image.size != (new_width, new_height):
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image.save(destination, format="PNG")
        manifest.append({
            "id": source.stem,
            "image": f"images/{destination.name}",
            "source": str(source),
            "width": new_width,
            "height": new_height,
        })

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    return manifest_path
