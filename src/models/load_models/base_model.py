import json
import re
import torch
from transformers import pipeline, AutoTokenizer


# MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
_generator = None

def _get_generator():
    global _generator
    if _generator:
        return _generator
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    _generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return _generator

def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))

def generate_json(system_prompt: str, user_prompt: str, max_new_tokens: int = 700) -> dict:

    generator = _get_generator()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = generator(messages, max_new_tokens=max_new_tokens, do_sample=False)
    response = output[0]["generated_text"][-1]["content"]
    return _extract_json(response)