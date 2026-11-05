import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)

model.eval()

cfg = model.config
N_Q = cfg.num_attention_heads
N_KV = cfg.num_key_value_heads
HDIM = cfg.hidden_size // N_Q
GROUP = N_Q // N_KV
THETA = cfg.rope_parameters["rope_theta"]
SCALE = HDIM ** -0.5


if __name__ == "__main__":
    texts = ["To find the outliers, we must first create a sequence length.", "What is capital of france", "Why quantization is important in inference"]
    for text in texts: 
        inputs = tok(text, return_tensors="pt").to(device)

        actual_seq_len = inputs.input_ids.shape[1]
        print(f"Our text is {actual_seq_len} tokens long. This is our Seq_Len.\n")

        with torch.no_grad():
            # use_cache=True forces the model to return the KV cache
            outputs = model(**inputs, use_cache=True)

        cache = outputs.past_key_values

        for layer in range(0, 24):


            layer_cache = cache.layers[layer]

            K_cache = layer_cache.keys
            V_cache = layer_cache.values

            # print(f"--- Layer {layer} KV Cache ---")
            # print("K Cache Shape:", K_cache.shape) 
            # print("V Cache Shape:", V_cache.shape)

            k_abs = torch.abs(K_cache)

            channel_maxes = torch.amax(k_abs, dim=(0, 1, 2))
            max_val = torch.max(channel_maxes)
            median_val = torch.median(channel_maxes)
            outlier_ratio = max_val / median_val

            print(f"K Outlier Ratio (Max/Median): {outlier_ratio.item():.2f}")
            print()