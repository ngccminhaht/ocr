"""dots.mocr adapter using one persistent Hugging Face model per worker."""

from __future__ import annotations

from pathlib import Path

from ocr_pipeline.common import REPO_ROOT


class DotsMOCR:
    def __init__(self, run_dir: Path, max_tokens: int) -> None:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_dir = REPO_ROOT / "models" / "dots_mocr" / "weights" / "DotsMOCR"
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Download dots.mocr weights first: {model_dir}")
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.max_tokens = max_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            attn_implementation="sdpa",
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)

    def predict(self, image_path: Path, sample_id: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "Extract the text content from this image."},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = {
            key: value.to(device="cuda", dtype=self.torch.float16)
            if self.torch.is_tensor(value) and value.is_floating_point()
            else value.to(device="cuda") if self.torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        trimmed = [out[len(source):] for source, out in zip(inputs["input_ids"], generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
