# Quantization Research

KV-cache quantization for LLM inference, built from scratch: a hand-written Qwen2.5-0.5B-Instruct forward
pass with an INT4 KV cache, then Triton kernels to make it fast.

This repo is the quantization stage of an 18-month, self-directed journey into LLM inference. Phases 1–7
covered general inference and live in a separate repository. From phase 8 on, the work narrows to one
specialization: **quantization**.

---

## Where things stand

- **INT4 KV cache, KIVI-style.** K is quantized per channel in sealed 32-token groups, with an unquantized
  residual for the newest tokens. V is quantized per token. The cache is about **6–7× smaller** than fp32.
- **Exact oracles.** The from-scratch fp32 path matches `model.generate` token for token, and so does a
  cache-free recompute. The Triton dequant output is bit-identical to PyTorch's (`torch.equal`).
- **Triton kernels.** The INT4 dequant kernels are **~3.6× faster** than the PyTorch version. A fused
  INT4 K-scores kernel unpacks the nibbles in registers and handles grouped-query attention by indexing,
  without `repeat_kv`.
- **Throughput.** At batch 128, INT4 fused decode reaches **2075 tok/s** vs **1347 tok/s** for plain fp32
  (1.54×). Most of that gain comes from fusion (1.45×). INT4 itself adds 1.06× on top. The fp32 twin kernel
  exists to keep that split honest.
- **No CPU↔GPU syncs** in the decode step, enforced by a `torch.cuda.set_sync_debug_mode("error")` gate.
- **Quality cost.** WikiText-2 perplexity (first 2048 test tokens) goes from **10.12 → 11.45 (+13.2%)**.

---

## The journey

| Phase | Focus | Where |
|---|---|---|
| 1–7 | LLM inference fundamentals | separate repository |
| 8 | KV-cache quantization in PyTorch: INT8, then INT4 | [`phase8/`](phase8/) |
| 9 | A correct INT4 KV cache, then Triton kernels | [`phase9/`](phase9/) |

### Phase 8: Is KV-cache quantization worth it, and how far can it go?

Everything starts from a hand-written decoder block (RoPE, grouped-query attention, a manual KV cache),
so quantize/dequantize can be inserted between the K/V projections and attention. Each script is a
standalone snapshot of one experiment.

| Script | Question it answers |
|---|---|
| `first.py` | Does K have outlier channels? Measures the per-layer max/median ratio of per-channel K magnitudes. |
| `memory_comparision.py`, `memory_comparsion_new.py` | When does the KV cache dominate VRAM? Cache size across batch × context × precision on a 16 GB card. |
| `kv_quantization.py` | Baseline INT8 KV cache. Validated against `model.generate` and a cache-free recompute. |
| `kv_cache_comparision.py` | Same INT8 scheme, scored by perplexity (fp32 vs INT8) on a longer text. |
| `lower_kv_cache.py`, `main_kv_cache.py` | Keeps recent K in an fp32 "residual window" and measures reconstruction error against the true K. |
| `per_token_lower_kv_cache.py` | V in INT4 with nibble packing (two values per byte), plus a pack/unpack self-test. |
| `block2_kv_cache.py` | K and V both in INT4: the most aggressive variant. |

For scale: in fp32, this model's KV cache costs **24 KiB per token** (24 layers × K,V × 2 KV heads × 64
dims × 4 bytes). At batch 64 and an 8K context, that's 12 GiB, more than the model's ~2 GB of fp32
weights several times over.

### Phase 9: Make INT4 correct, then make it fast

Phase 9 began by fixing what phase 8 got wrong, with a measurement behind each fix:

1. **The phase 8 residual window was unbounded.** It grew forever, so decode-time K was never quantized.
2. **A frozen prefill scale saturates.** With a 1-token prefill, 78.5% of decode K values were clamped,
   and perplexity jumped to 2114 (fp32: 44.04). With a 512-token prefill it was 0.04%. This was
   saturation, not rounding error.
3. **The fix was sealed groups plus a bounded residual.** K tokens collect in an fp32 residual (at most
   31 tokens). Once 32 have arrived, they're quantized together as one group with its own per-channel
   scales, so nothing is clamped. Scales are also per KV head now, not shared.

The scheme that the kernels reproduce:

| | K | V |
|---|---|---|
| Granularity | per channel, per head, per 32-token group | per token, per head |
| Quantized when | a 32-token group fills ("seals") | immediately |
| Unquantized part | fp32 residual, 0–31 tokens | none |
| Format | symmetric INT4, `scale = amax / 7`, clamp `[-8, 7]`, two nibbles per `uint8` | same |

Then came the kernels, each step checked against a frozen PyTorch oracle:

| File | What it adds |
|---|---|
| `kv_cache_reference.py` | Frozen PyTorch INT4 reference, snapshotted before any Triton work. This is the oracle. |
| `triton_test.py` | Small learning kernels (per-token and per-group max) on a 4×4 matrix. |
| `kv_cache_triton.py` | Triton dequant kernels for K and V, plus a residual-merge kernel, a correctness check and benchmarks. |
| `fused_kernel.py` | Fused INT4 K-scores kernel (`fused`), its fp32 twin (`fused_fp32`) and a `torch.profiler` breakdown. |
| `fused_nosync.py` | The same, with every CPU↔GPU sync removed from decode and a `check_no_syncs()` gate. **Latest.** |
| `TRITON_NOTES.md` | Detailed design notes: worker/grid design, strides, bugs hit, profiler reading. |
| `*.txt` | Raw output of benchmark and profiling runs. |

---

## Results

Measured on an RTX 5060 Ti (16 GB), with fp32 weights, a 512-token prompt and greedy decode. Numbers are
from `phase9/fused_nosync_run3.txt`. Run-to-run spread in the ratios is about ±0.01 (checked over 3 runs).

**Full decode throughput (tok/s):**

| batch | fp32 raw | fp32 fused | int4 PyTorch | int4 Triton | int4 fused | int4 fused / fp32 fused |
|---|---|---|---|---|---|---|
| 1 | 97.4 | 87.2 (0.90×) | 67.7 (0.70×) | 73.3 (0.75×) | 72.0 (0.74×) | 0.83× |
| 4 | 392.5 | 350.5 (0.89×) | 270.0 (0.69×) | 292.5 (0.75×) | 286.0 (0.73×) | 0.82× |
| 16 | 902.3 | 1064.2 (1.18×) | 800.2 (0.89×) | 884.0 (0.98×) | 1059.7 (1.17×) | 1.00× |
| 64 | 1312.9 | 1861.4 (1.42×) | 1095.1 (0.83×) | 1336.3 (1.02×) | 1993.4 (1.52×) | 1.07× |
| 128 | 1346.8 | 1950.5 (1.45×) | 1096.5 (0.81×) | 1369.3 (1.02×) | **2075.2 (1.54×)** | 1.06× |

Ratios in parentheses are relative to fp32 raw. *fp32 fused* is the twin kernel: fusion without
quantization. Comparing it with *int4 fused* isolates what INT4 itself contributes.

**Dequant only** (one layer, 536 tokens): Triton is 0.026 ms vs PyTorch's 0.094 ms at batch 1–16
(~3.6×), and 0.147 ms vs 0.546 ms at batch 64.

**Correctness gates** (all passing in the latest run):

- fp32 raw, fp32 bypass and fp32 fused all match `model.generate` exactly, including runs that cross a
  group-seal boundary.
- INT4 Triton and INT4 fused match the INT4 PyTorch reference token for token (23/23). The fused-scores
  max difference is ≤ 6e-7.
- No CPU↔GPU sync in any of the six decode modes.

---

## Lessons learned

- **Short passages flatter INT4.** On a 117-token slice, the perplexity cost looked like +5%, but 41% of
  the attended K was still the fp32 residual. On 2048 tokens of WikiText-2 the honest number is +13.2%.
- **Separate fusion from quantization.** The early "INT4 fused is 1.39× faster" result was almost
  entirely fusion. An fp32 twin with the same fusion showed INT4 itself contributes ~1.01–1.07×. Any
  non-quantization speedup has to go into both paths, or INT4 gets the credit.
- **INT4's biggest saving was the cache append, not attention.** `torch.cat` rewrites the whole cache
  every step, and the INT4 cache is 8× smaller. That's an artifact of the append, which a preallocated
  cache removes for both paths.
- **After fusing K, V is the bottleneck.** The `repeat_kv` copy of V plus probs @ V take about half of
  the GPU time. K scores are about 5%.
- **The fused INT4 kernel isn't memory-bound yet.** It reaches ~43 GB/s on a ~448 GB/s card: the tiles
  are tiny and there's too little work per program.
- **Profilers need care.** If Self CPU time ≈ GPU time, that doesn't prove the CPU is the bottleneck.
  It includes time the CPU spends blocked waiting on the GPU.
- **One hidden sync per layer per step:** `torch.tensor([pos], device=...)` copied a CPU value to the GPU
  in every layer. Building positions on the GPU and collecting tokens with a single `.tolist()` at the
  end removed every sync.

[`phase9/TRITON_NOTES.md`](phase9/TRITON_NOTES.md) has the full write-up, including a recipe for
designing a kernel's grid from its scale shape and a table of bugs with their symptoms.

---

## Running it

Requires a CUDA GPU for the Triton kernels; phase 8 scripts also run on CPU. Developed with Python 3.14,
PyTorch 2.13, Transformers 5.14.1 and Triton.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers triton datasets "huggingface-hub>=1.5.0,<2.0"

python phase8/kv_quantization.py     # any phase 8 experiment
python phase9/fused_nosync.py        # latest: checks, generation oracle, benchmark, profile
```

- Each script is standalone. It downloads `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face on first run
  and has no CLI arguments.
- `phase9/fused_nosync.py` is configured by flags near the top (`QUANTIZE`, `USE_TRITON`, `RAW`,
  `USE_FUSED`, `RAW_FUSED`) and in its `__main__` block (`RUN_BENCHMARK`, `RUN_PROFILE`, `RUN_DUMP`).
  `RUN_DUMP` regenerates `phase9/dumps/qk_oracle.pt`, which isn't committed.
- `huggingface-hub` is pinned below 2.0 because 2.x breaks Transformers 5.14.1.
- Function names use `qunatize` / `dequnatize`. The typo is kept on purpose so every file stays
  greppable with the same name.

---

## Next steps

1. **Fused dequant + attention for V:** read INT4 tiles, dequant in registers, online softmax, multiply
   by V and accumulate, with no fp32 cache ever written. This targets the ~50% of GPU time now spent on V.
2. **Preallocated cache** in both paths: no `torch.cat` per step, fixed shapes, then CUDA graphs.
3. **More work per Triton program:** all 7 query heads of a KV head, or 64+ tokens per tile.
4. **Quality:** asymmetric INT4 (scale + zero-point), smaller K groups, WikiText-2 at 2048 tokens as the
   reference metric.

---

## References

- Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750):
  source of the per-channel K / per-token V split and the residual-group idea.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135): the online-softmax approach planned for
  the fused V kernel.
- [Triton](https://triton-lang.org/) and [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
