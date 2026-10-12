"""Make downloaded DotsMOCR model code usable with PyTorch SDPA on T4 GPUs.

Run after downloading weights. The script changes only Python files beside the
downloaded weights; it does not modify checkpoint tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> bool:
    content = path.read_text(encoding="utf-8")
    if new in content:
        return False
    if old not in content:
        raise RuntimeError(f"Unexpected model source format: {path}")
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    return True


def prepare(model_dir: Path) -> None:
    vision = model_dir / "modeling_dots_vision.py"
    if not vision.is_file():
        raise FileNotFoundError(f"Download model weights first: {vision}")

    changes = [
        replace_once(
            vision,
            "from flash_attn import flash_attn_varlen_func",
            "try:\n    from flash_attn import flash_attn_varlen_func\nexcept ImportError:\n    flash_attn_varlen_func = None",
        ),
        replace_once(
            vision,
            "        x = self.proj(x).view(-1, self.embed_dim)",
            "        x = x.to(dtype=self.proj.weight.dtype)\n        x = self.proj(x).view(-1, self.embed_dim)",
        ),
        replace_once(
            vision,
            '    "flash_attention_2": VisionFlashAttention2,',
            '    "flash_attention_2": VisionSdpaAttention,',
        ),
    ]
    print(f"T4 model code ready: {vision} ({sum(changes)} changes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models" / "dots_mocr" / "weights" / "DotsMOCR")
    args = parser.parse_args()
    prepare(args.model_dir)
