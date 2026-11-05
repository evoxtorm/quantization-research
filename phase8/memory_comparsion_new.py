import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
model.eval()

cfg = model.config
LAYERS = 24
N_KV   = cfg.num_key_value_heads
HDIM   = cfg.hidden_size // cfg.num_attention_heads

PRECISIONS = {
    "fp32": 4.0,
    "int8": 1.0,
    "int4": 0.5,
}

BYTES_PER_WEIGHT = 4.0  # model loaded as float32
num_params = sum(p.numel() for p in model.parameters())
weight_bytes = num_params * BYTES_PER_WEIGHT

if torch.cuda.is_available():
    _, total_bytes = torch.cuda.mem_get_info()
else:
    total_bytes = 16 * 1024**3  # RTX 5060 Ti 16GB fallback

SAFETY_MARGIN = int(0.10 * total_bytes)
budget = total_bytes - weight_bytes - SAFETY_MARGIN


def cache_bytes(seq_len, batch, bytes_per_value):
    return LAYERS * 2 * N_KV * HDIM * seq_len * batch * bytes_per_value


BATCH_SIZES   = [1, 8, 32, 64, 128, 256]
CONTEXT_LENS  = [2048, 8192, 32768]

print(f"Model weights: {weight_bytes/1e9:.2f} GB (fixed cost)")
print(f"Total VRAM: {total_bytes/1e9:.2f} GB | cache budget after weights+margin: {budget/1e9:.2f} GB\n")

# Cache as % of total VRAM, across batch x context x precision.
# The story is in the fp32 column: where does it cross from "rounding error" to "dominant"?
for ctx in CONTEXT_LENS:
    print(f"--- context = {ctx} tokens ---")
    print(f"{'batch':>6} | {'fp32':>18} | {'int8':>10} | {'int4':>10} | note")
    for batch in BATCH_SIZES:
        cb_fp32 = cache_bytes(ctx, batch, 4.0)
        cb_int8 = cache_bytes(ctx, batch, 1.0)
        cb_int4 = cache_bytes(ctx, batch, 0.5)

        pct_fp32 = cb_fp32 / total_bytes
        pct_int8 = cb_int8 / total_bytes
        pct_int4 = cb_int4 / total_bytes

        # does the fp32 cache still fit in the budget?
        fp32_fits = cb_fp32 <= budget
        int4_fits = cb_int4 <= budget

        if not int4_fits:
            note = "even int4 OOMs -- need weight quant too"
        elif not fp32_fits:               # int4 fits, fp32 doesn't
            note = "fp32 OOMs, int4 fits <-- quant pays"
        elif pct_fp32 >= 0.40:            # both fit, cache large
            note = "cache dominant <-- quant matters"
        elif pct_fp32 < 0.05:
            note = "cache negligible"
        else:
            note = ""

        fp32_str = f"{cb_fp32/1e9:.2f}GB {pct_fp32:>5.1%}"
        print(f"{batch:>6} | {fp32_str:>18} | {pct_int8:>10.1%} | {pct_int4:>10.1%} | {note}")
    print()