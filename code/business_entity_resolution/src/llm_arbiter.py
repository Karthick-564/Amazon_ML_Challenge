"""LLM Edge-Case Arbiter using Qwen2.5-7B-Instruct to protect Macro F0.5 on razor-edge borderline pairs."""

from __future__ import annotations

import os
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .normalize import NormalizedEntity

os.environ["HF_HOME"] = r"E:\ML_challange\.cache\huggingface"


class LLMEdgeCaseArbiter:
    """Evaluates high-risk borderline candidate pairs (0.50 - 0.75) to prevent catastrophic false merges on singletons."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled and torch.cuda.is_available()
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        if self.enabled and self.model is None:
            print(f"Loading {self.model_name} onto RTX 4090...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()

    def arbitrate_singleton_pair(
        self, s1: NormalizedEntity, cand: NormalizedEntity, score: float
    ) -> bool:
        """Return True if candidate is a genuine business match, False if it is a distractor/singleton."""
        if not self.enabled:
            # Fallback to score threshold if LLM is disabled
            return score >= 0.65

        self.load_model()

        prompt = f"""You are an expert commercial business entity resolution arbiter.
Determine if Reference Business (Source 1) and Candidate Business (Source 2/3) refer to the EXACT SAME physical business entity or different businesses/branches.

Reference Business:
Name: {s1.clean_name}
Address: {s1.clean_address}
Country: {s1.country}

Candidate Business:
Name: {cand.clean_name}
Address: {cand.clean_address}
Country: {cand.country}

Preliminary Match Score: {score:.2f}

Guidelines:
1. Different building numbers, unit numbers, or different cities indicate DIFFERENT branches/locations (NOT a match).
2. Minor spelling differences, legal suffix differences (e.g. Pvt Ltd vs Limited), or Indic transliterations ARE matches if the location agrees.
3. Answer with EXACTLY 'MATCH' or 'DIFFERENT' on the first line.
"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=10,
                temperature=0.01,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return "MATCH" in response.upper()
