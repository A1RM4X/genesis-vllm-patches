import os
from vllm._genesis.patches.apply_all import main as apply_all
apply_all()
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    llm = LLM(
        model="orcarouter/Qwen3.8-27B-Uncensored-FP8",
        tensor_parallel_size=2,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.75,
    )

    prompts = ["¿Por qué el cielo es azul? Explicación física breve en dos frases."]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print("OUTPUT:", output.outputs[0].text)
