import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
model.eval()

cfg = model.config
LAYERS = 24
N_KV   = cfg.num_key_value_heads
HDIM   = cfg.hidden_size // cfg.num_attention_heads


PRECISIONS = {
    "fp32 (baseline, unquantized)": 4.0,
    "int8":                          1.0,
    "int4 (packed)":                 0.5,
}

BYTES_PER_WEIGHT = 4.0  # model loaded as float32
num_params = sum(p.numel() for p in model.parameters())
weight_bytes = num_params * BYTES_PER_WEIGHT

if torch.cuda.is_available():
    _, total_bytes = torch.cuda.mem_get_info()
else:
    total_bytes = 16 * 1024**3  # RTX 5060 Ti 16GB, hardcoded fallback while driver is down

SAFETY_MARGIN = int(0.10 * total_bytes)      # e.g. 10% headroom
budget = total_bytes - weight_bytes - SAFETY_MARGIN


def cache_bytes(seq_len, batch, bytes_per_value):
    return LAYERS * 2 * N_KV * HDIM * seq_len * batch * bytes_per_value


BATCH_SIZES = [1, 2, 4, 8, 16]

results = {}
for label, bpv in PRECISIONS.items():
    for batch in BATCH_SIZES:
        max_seq_len = budget // (LAYERS * 2 * N_KV * HDIM * batch * bpv)
        results[(label, batch)] = max_seq_len


# print(f"Model weights: {num_params:,} params -> {weight_bytes/1e9:.2f} GB (fixed cost)")
# print(f"Total VRAM: {total_bytes/1e9:.2f} GB, budget after weights + margin: {budget/1e9:.2f} GB\n")

for (label, batch), max_seq_len in results.items():
    print(f"{label:30s} batch={batch:3d} -> max context ≈ {max_seq_len} tokens")


REFERENCE_SEQ_LENS = [2048, 4096, 8192]

print("\nCache size at fixed context lengths (comparable across precisions):")
for seq_len in REFERENCE_SEQ_LENS:
    for label, bpv in PRECISIONS.items():
        for batch in BATCH_SIZES:
            cb = cache_bytes(seq_len, batch=batch, bytes_per_value=bpv)
            print(f" batch={batch:3d}  seq_len={seq_len:>5}  {label:30s} -> {cb/1e9:.3f} GB  ({cb/total_bytes:.2%} of total VRAM)")
