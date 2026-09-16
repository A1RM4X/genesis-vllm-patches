import torch
import os
from vllm import LLM, SamplingParams

# Test single-token decode with vLLM Python API
llm = LLM(
    model="orcarouter/Qwen3.8-27B-Uncensored-FP8",
    tensor_parallel_size=2,
    dtype="float16",
    kv_cache_dtype="fp8_e4m3",
    mamba_ssm_cache_dtype="float16",
    max_num_seqs=4,
    max_model_len=2048,
    trust_remote_code=True,
    enforce_eager=True,
)

prompts = ["Por favor escribe un breve resumen en español sobre cómo funciona la energía solar fotovoltaica."]
sampling_params = SamplingParams(temperature=0.0, max_tokens=120)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print("PROMPT:", output.prompt)
    print("GENERATED TEXT:", output.outputs[0].text)

