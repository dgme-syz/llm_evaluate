from vllm import LLM, SamplingParams
import textwrap

model_path = "/home/nfs05/shenyz/models/Seed-X-PPO-7B"

model = LLM(model=model_path,
            tensor_parallel_size=1,
            enable_prefix_caching=False, 
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.95)

src_text = "\"She had a real fear of food waste,\" Mr. Coe said."
pred_text = "\"Hänelle oli todellinen huoli ruoan tarjoamisesta,\" sanoi herra Coe."
messages = [
    # without CoT
    textwrap.dedent(
        f"""
        Given the source text: 
        
        {src_text}
        
        Improve the following draft Finnish translation into a high-quality Finnish version, without explanations:

        {pred_text}
        """
    ).strip(),
]

# Beam search (We recommend using beam search decoding)
# decoding_params = BeamSearchParams(beam_width=4,
#                                    max_tokens=512)
# Greedy decoding
decoding_params = SamplingParams(temperature=0,
                                 max_tokens=512,
                                 skip_special_tokens=True)

results = model.generate(messages, decoding_params)
responses = [res.outputs[0].text.strip() for res in results]

print(responses)
