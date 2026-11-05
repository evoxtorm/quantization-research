# Triton KV-cache notes (phase9)

Notes from building the INT4 KV-cache reference and the first Triton kernels for Qwen2.5-0.5B.
Last updated: 2026-11-04 (fused K-scores kernel, fp32 twin, profiling: section 11).

---

## 0. Where things are

| File | What it is |
|---|---|
| `phase9/kv_cache_reference.py` | Frozen PyTorch reference (sealed K groups + residual, per-head scales). Snapshot taken **before** any Triton work. This is the oracle. |
| `phase9/kv_cache_triton.py` | Working file: the same scheme plus the Triton kernels `dequnatize_v_cache_kernel`, `dequnatize_k_cache_kernel` and `merge_cache_k_residual`, `dequantize_triton()`, the `USE_TRITON` flag, `check_dequant_triton()`, `bench_dequant()`, and the benchmark in `__main__`. |
| `phase9/triton_test.py` | Small learning kernels (`per_token_max`, `per_group_channel_max`) on a 4×4 matrix. |
| `phase9/fused_kernel.py` | Copy of `kv_cache_triton.py` plus the fused INT4 scores kernel `fused`, its fp32 twin `fused_fp32`, the `USE_FUSED` / `RAW_FUSED` flags, `check_fused_scores*()`, `profile_step()` and the `RUN_PROFILE` flag. |
| `phase9/fused_run2.txt`, `phase9/fused_run3.txt` | Full output: batch up to 64 with a batch-64 profile, then up to 128 with a batch-128 profile. |

Run: `source .venv/bin/activate && python phase9/kv_cache_triton.py`
(prints the correctness check, then dequant-only timing, then full decode tok/s for fp32, INT4 PyTorch and INT4 Triton)

---

## 1. The quantization scheme (what the kernels must reproduce)

| | K | V |
|---|---|---|
| Granularity | **per-channel, per-head, per group of `K_GROUP = 32` tokens** | **per-token, per-head** |
| Scale shape | `k_scale [B, N_KV, n_groups, 64]` | `v_scale [B, N_KV, S, 1]` |
| When quantized | when 32 tokens have collected ("sealed") | immediately, every token |
| Unquantized part | `k_residual [B, N_KV, R, 64]` fp32, R = 0…31 | none |

- INT4 symmetric: `scale = amax(|x|) / 7`, `q = clamp(round(x / scale), -8, 7)`.
- **Packing:** `pack_int4` adds 8 (so values are 0…15) and stores two per `uint8`. **High nibble = even channel, low nibble = odd channel.** `byte j = [channel 2j | channel 2j+1]`.
- Per-layer cache tuple: `(cache_k, k_scale, cache_v, v_scale, k_residual)`.
- `QUANTIZE = False` is the **bypass/baseline**: `x / s * s`, float32, last dim 64 (no packing). `QUANTIZE = True` gives `uint8`, last dim 32.
- `K_GROUP` is **our** choice (KIVI uses 32), not something from the model config. Smaller is more accurate but needs more scales. Keep it a power of 2 for `tl.arange`.

Example shapes (batch 4, after a 512-token prefill + 7 decode steps, INT4):
```
cache_k    [4, 2, 512, 32] uint8   strides (32768, 32, 64, 1)   <- NOT contiguous
k_scale    [4, 2,  16, 64]
k_residual [4, 2,   7, 64]
cache_v    [4, 2, 519, 32] uint8
v_scale    [4, 2, 519,  1]
```
`cache_k` covers only the sealed tokens; K length = sealed + residual = V length.

---

## 2. How we got here (decisions and evidence)

1. **Removed the old unbounded `residual_k`.** It grew forever and decode K was never quantized.
2. **Tried a frozen prefill K scale (no residual).** A clamp probe showed it saturates:

   | prefill tokens | decode K clamped |
   |---|---|
   | 1 | 78.5% |
   | 8 | 7.2% |
   | 32 | 1.7% |
   | 512 | 0.04% |

   Perplexity with a 1-token prefill: **2114** (fp32: 44.04). That was saturation, not rounding error.
3. **Per-head scales:** dropped `dim=1` from the reductions, so the 2 KV heads no longer share a scale.
4. **Sealed groups + bounded residual (KIVI-style):** the residual is at most 31 tokens, and **0 values get clamped** (each group's scale comes from its own max).
5. **Oracles (all passed):**
   - fp32 path: exact greedy match with `model.generate` and `generate_nocache`, including runs that cross a seal boundary (5-token prompt + 40 new; 117-token prompt + 20 new).
   - WikiText-2 test, first 2048 tokens: fp32 **10.1213** (= HF exactly), INT4 **11.4529 (+13.2%)**.
   - 117-token slice: +5.0%, but 41% of attended K was still fp32 residual. **Short passages flatter INT4**; the 2048-token number is the honest one.
6. Benchmark run-to-run spread is about **±0.01** in the ratio (3 runs).

---

## 3. Results with the Triton dequant (2026-10-16, RTX 5060 Ti)

Correctness: Triton == PyTorch (`torch.equal`) for K and V at seq 5 (no sealed group), 64 (no residual), 101 (both).

Dequant only (one layer, 536 tokens):

| batch | PyTorch | Triton | speedup |
|---|---|---|---|
| 1 | 0.093 ms | 0.028 ms | 3.4× |
| 16 | 0.091 ms | 0.027 ms | 3.4× |
| 64 | 0.531 ms | 0.147 ms | 3.6× |

Full decode (24 layers, 512-token prompt), tok/s:

| batch | fp32 | INT4 PyTorch | INT4 Triton |
|---|---|---|---|
| 1 | 72.2 | 59.2 (0.82×) | 63.6 (0.88×) |
| 4 | 271.7 | 224.4 (0.83×) | 241.2 (0.89×) |
| 16 | 682.8 | 607.1 (0.89×) | 648.3 (0.95×) |
| 64 | 1073.5 | 994.9 (0.93×) | **1179.9 (1.10×)** |

What this means:
- Batch 1–16 dequant time is flat, so it's **launch overhead**: PyTorch's dequant is many small ops, Triton is 3 kernels.
- INT4 is still below fp32 at small batch, because the full fp32 `cache_k_` / `cache_v_` is written to memory and read back by the matmuls. **The fused dequant + attention kernel is what removes that.**
- The "fp32" baseline also runs the PyTorch bypass dequant on an 8× bigger cache, so it isn't pure fp32 attention.

---

## 4. Concepts: my questions and the answers

**Why sealed groups? Is it like AWQ?**
K has outlier **channels** (big in every token), so K needs a scale per channel. A per-channel scale needs many tokens, but decode delivers them one at a time, so we wait for 32 (in the residual), then quantize them together. AWQ is different: it quantizes **weights** offline, using activation statistics only to protect salient weight channels. Here we quantize the **activations** (K/V) online. Shared idea: outlier channels plus group-wise scales.

**Does a big new value get clamped? What if more than 32 tokens arrive?**
No. A group's scale is computed **after** all 32 tokens are present, from their own max, so nothing in the group can exceed ±7. At 32 tokens the residual is quantized (not clamped or dropped) into a new sealed group and emptied. Every token is kept forever.

**PyTorch `dim` vs Triton `axis`**
- `dim` (PyTorch) = the dimension of the **full tensor** that disappears in the reduction.
- `tl.program_id(axis)` = which dimension of the **grid** (which worker you are).
- `tl.max(x, axis)` = a dimension of the **block `x` this worker loaded**.
- So `amax(dim=3)` on V becomes `axis=0` in Triton (the block is just `[64]`).

**What is a program / grid / block?**
A program is a **worker**. The grid is **how many workers** (up to 3 dimensions; total = product). The block is the data **one worker loads at once** (`tl.arange`). Memory is one long **tape**; a kernel only gets a pointer to its start.

**Why 4 workers in one example and 2 in another?**
The worker count is a **choice about how to split the work**, not part of the maths. Rule: dimensions you keep go on the grid, dimensions you reduce go inside the block. Prefer splits where each worker reads **contiguous rows**.

**What is stride?**
Not the shape. **Shape = how many; stride = how far to jump on the tape** for +1 in that dimension. `position = Σ index × stride`. A contiguous tensor has `stride[i] = product of sizes to the right`.
- `transpose`/`permute` only **reorder** the strides (each dimension keeps its stride, nothing is copied). That's why `cache_k` is `(32768, 32, 64, 1)`: `decode_project` does `.view(B, S, N_KV, HDIM).transpose(1, 2)`, so the heads are interleaved inside each token on the tape.
- Strides have no "product stays the same" rule; only the **shape's** product (element count) is preserved.

**Is the block size the stride?**
No. **Block size = how many values a worker processes** (`BYTES = 32`). **Stride = how far apart** things are (token stride 64). The V worker reads 32 bytes, but the next token is 64 away (the other head sits in between).

**How do I decide the number of workers?**
`workers = total scales ÷ scales per worker`.
- V: 4,152 scales, 1 per worker, so 4,152 workers (one per (b, h, token) row).
- K: 8,192 scales, 64 per worker (a 32-token × 64-channel tile), so **128** workers. One worker per K scale (8,192) would read strided columns and load every byte twice, because each byte holds 2 channels.

**How do I know K needs a 2-D tile and groups, but V doesn't?** (It's NOT about contiguity.)
Compare the **scale shape** with the **cache value shape**, dimension by dimension:

| scale vs cache | meaning | in one worker |
|---|---|---|
| equal | each index has its own scale | fixed (from the grid) |
| 1 | whole dimension shares one scale | varying (`tl.arange`, take all of it) |
| smaller (e.g. 16 vs 512) | groups: 512/16 = 32 share a scale | take one group (`tl.arange(0, 32)`); the group index comes from the grid |

V: token equal, channel 1, so a 1-D row. K: token smaller (grouped), channel equal, so a 2-D `[32 tokens, 32 bytes]` tile with 64 scales. **The scale shape decides the block shape; the strides decide the addresses.**

**What do I pass to a kernel?**
A kernel knows only its `program_id` and its arguments (no shapes, no Python variables). Write the read and write position formulas; **every symbol that isn't a `program_id` must be passed**: pointers, strides, `tl.constexpr` sizes for `tl.arange` (`BYTES`, `GROUP`, `D`), and plain numbers like offsets.
- With a 3-D grid `(S, H, B)` you don't pass `S`/`n_rows`; the grid carries it.
- Each tensor needs **its own** strides (input bytes, scales, output): same `(b, h, t)`, different strides.
- Only 3 strides per tensor are passed because the 4th (last) dimension is walked with `tl.arange`, which assumes stride 1. **Assert `x.stride(-1) == 1` in the wrapper.**

**Why `tok_offset` in the merge kernel?**
When the **input and output coordinates differ** for the same worker, you need an offset. Residual token `t` goes to output token `S_sealed + t`. In the dequant kernels input token = output token, so no offset is needed.

**How does the residual end up next to the dequantized K without combining anything?**
Allocate **one** output `cache_k_` with `S_sealed + R` tokens. The dequant kernel writes rows `0…S_sealed-1`, the merge kernel writes rows `S_sealed…`. Neither knows about the other; they write to different places in the same tensor ("two people writing different pages of one notebook"). Kernels can only touch the tensors passed to them, so V's output is never involved (and V has no residual).

**Even/odd stores vs one 64-wide store**
Unpacking gives `high` (even channels) and `low` (odd). Either store `high` to `o_start + 2*arange` and `low` to `+1` (what I use), or `tl.interleave(high, low)` and store to `o_start + arange(0, 64)`. Don't store `high` then `low` into 64 consecutive slots: wrong order, and no error. (There's also a "64 positions, each loads its byte `c // 2` and shifts by `(1 - c % 2) * 4`" trick; skip it, it's confusing.)

**Dequant is element-wise, not a matmul.**
Each output = one nibble × one scale. `tl.dot` only appears later, in fused attention.

**Should `repeat_kv` become a kernel?**
No. It **copies** K and V 7× (the `reshape` after `expand` can't be a view). Better: group the **queries** instead: `q.view(B, N_KV, GROUP, HDIM) @ cache_k_.transpose(-2, -1)`, then `out.view(B, N_Q, 1, HDIM)`. Same grouping (query head `i` uses KV head `i // 7`). In a fused kernel it's just indexing.

**Transposes?**
Free (only strides change). The matmul reads transposed inputs directly. In Triton, "transpose" just means choosing which stride goes with which index.

**Softmax in Triton?**
Not as a separate kernel (same reads and writes as PyTorch's, no gain). It belongs **inside the fused dequant + attention kernel**, using **online softmax** (running max and sum across K tiles, the FlashAttention trick).

**Why did `out` print zeros but `out2` print numbers before the kernel ran?**
`torch.empty` doesn't initialise memory. `out` got fresh (zeroed) memory; `out2` got a block recycled by PyTorch's caching allocator that still held the freed temporary `A.abs()`. The kernel then overwrote everything. **Fill test outputs with NaN** so unwritten slots stand out.

---

## 5. Recipe for writing a kernel

1. **Worker design:** compare scale shape vs cache shape (equal / 1 / smaller). Decide what one worker covers.
2. **Grid:** one worker per combination of the dimensions it doesn't cover (`(S, H, B)` for V, `(n_groups, H, B)` for K). Largest dimension on axis 0 (axes 1 and 2 are limited to 65,535).
3. **Positions:** `Σ coordinate × stride` per tensor. Fixed coordinates come from `program_id` (the "start"); varying ones from `tl.arange`. Two varying dimensions: `rows[:, None] * stride + cols[None, :]`.
4. **Arguments:** pointers, 3 strides per tensor, constexpr sizes, offsets.
5. **Body:** load, compute, store. Keep **position names** (`q`, `even`, `odd`) separate from **value names** (`q_bytes`, `high`, `high_val`).
6. **Wrapper:** read shapes at launch time (S changes every step), allocate the output with the right size, assert `stride(-1) == 1`, skip launches with a zero-size grid (`n_groups == 0`, `R == 0`).
7. **Test against PyTorch with `torch.equal`** (same maths, so bit-identical), with NaN-filled outputs, on non-contiguous inputs, and on edge cases (no sealed group, no residual, both).

---

## 6. The three kernels (design summary)

| Kernel | Grid | One worker | Block | Notes |
|---|---|---|---|---|
| `dequnatize_v_cache_kernel` | `(S_V, H, B)` | one token row | 32 bytes, 1 scale | `q_start + arange(32)`; scale = one number; store high to even, low to odd |
| `dequnatize_k_cache_kernel` | `(n_groups, H, B)` | one 32-token group | `[32, 32]` bytes, 64 scales | `tok = g*GROUP + rows`; scale start uses **`g * s_st`** (group dimension, not token); `scale_even = s_start + 2*cols`, `scale_odd = +1`; multiply by `scale[None, :]` |
| `merge_cache_k_residual` | `(R, H, B)` | one residual row | 64 floats | plain copy; output token = `tok_offset + t` with `tok_offset = S_sealed` |

Unpack: `high = ((b >> 4) & 0xF).to(tl.int32) - 8`, `low = (b & 0xF).to(tl.int32) - 8`. Convert to int32 **before** `- 8` (uint8 would wrap 3 − 8 to 251), then `.to(tl.float32) * scale`.

---

## 7. Triton syntax cheat sheet

```python
@triton.jit
def kernel(x_ptr, out_ptr, n, stride_r, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)                       # which worker
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # my positions (BLOCK must be a power of 2)
    mask = offs < n                               # never touch past the end
    x = tl.load(x_ptr + offs, mask=mask, other=0) # pointer + positions -> values
    tl.store(out_ptr + offs, x * 2, mask=mask)

kernel[(triton.cdiv(n, BLOCK),)](x, out, n, x.stride(0), BLOCK=256)   # kernel[grid](args)
```

| Need | Write |
|---|---|
| Load one number | `tl.load(ptr + i)` |
| 2-D block | `r[:, None] * stride + c[None, :]`, mask `(r[:, None] < R) & (c[None, :] < C)` |
| Broadcast a row vector | `v[None, :]` (or column `v[:, None]`) |
| Cast | `x.to(tl.float32)` |
| Bits | `>>`, `<<`, `&`, `\|` |
| Select | `tl.where(cond, a, b)` |
| Reduce | `tl.max(x, axis=0)`, `tl.sum(x, axis=0)` |
| Rounding identical to `torch.round` | `libdevice.rint` (round-half-even), not `floor(x + 0.5)` |
| Sizes | `triton.cdiv`, `triton.next_power_of_2` |

Argument kinds: tensor becomes a pointer; Python int/float becomes a runtime scalar; `tl.constexpr` is compile-time (a new compile for each value).

---

## 8. Bugs I hit (and how to recognise them)

| Symptom | Cause | Fix |
|---|---|---|
| `cannot reshape tensor of 0 elements ... -1` | prefill < 32 tokens, so 0 groups | use the explicit last dim, not `-1` |
| `attn_w @ vf`: expected 514 got 513 | residual concatenated **twice** (old `torch.cat` left active) | keep exactly one residual join |
| expected 512 got 513 | `cache_k_` allocated for `S_sealed` only; the merge kernel wrote **past the end** (silent out-of-bounds write!) | allocate `S_sealed + R`; fresh run afterwards |
| fp32 baseline broken after adding Triton | Triton or merge code running outside `if QUANTIZE` | Triton only in the INT4 path; fp32 keeps `torch.cat` |
| only token 0 dequantized | grid `(S, H, B)` used `S` from `hidden.shape` (= 1 in decode) | use the cache's own shape (`S_V`) |
| `program_id(3)` | the grid has only 3 dimensions | one worker per tile, channels via `tl.arange` |
| K scales wrong / out of range | scale position used `tok` (token index) on `k_scale`'s **group** dimension | `g * s_st` |
| `out_k` wrong size | last dim set to `n_groups` | last dim is `HDIM` (64) |
| positions lost | reused `even` (positions) as a value name | separate position and value names |
| 3 − 8 = 251 | subtraction in uint8 | `.to(tl.int32)` first |
| shapes print 64 then 32 | `benchmark` runs fp32 (`QUANTIZE=False`) then INT4 per batch | print `QUANTIZE` and `dtype` with shapes |

---

## 9. Debugging and testing tools

- **Oracles first, benchmark second.** fp32 must match HF and `generate_nocache` exactly (lossless bypass). INT4 perplexity must be **identical** before and after swapping in kernels.
- **`check_dequant_triton()`** in the file: Triton vs PyTorch with `torch.equal`.
- **NaN-filled outputs:** `torch.full(shape, float("nan"))`, so missed slots show as NaN.
- **`tl.device_print("name", x)`** inside a kernel (tiny grid, then `torch.cuda.synchronize()`). Output order is mixed.
- **`TRITON_INTERPRET=1 python ...`** runs kernels on the CPU in Python; `breakpoint()` works inside the kernel. Slow, tiny shapes only.
- **Debug output pointer:** store an intermediate (e.g. `high`) and compare that stage alone with PyTorch.
- **Self-labelling prints:** `layer == 0`, one or two steps only, include `QUANTIZE`, batch, `dtype`, `.stride()`.
- **`assert cache_k_.shape[2] == cache_v_.shape[2]`** before attention.
- Benchmarks: warm up, `torch.cuda.synchronize()` around timing, run 3× and check the spread before trusting a difference.

---

## 10. Environment gotchas

- **NVIDIA driver vanished after a kernel update** (`nvidia-smi` failed, PyTorch silently fell back to CPU). The `linux-modules-nvidia-595-open-<kernel>` package wasn't installed for the new kernel. Fix: `sudo apt install linux-modules-nvidia-595-open-$(uname -r) linux-modules-nvidia-595-open-generic-hwe-26.04`, then reboot (or boot the previous kernel from GRUB). Always check `torch.cuda.is_available()` before benchmarking.
- **`pip install datasets`** pulled `huggingface_hub` 2.x, which breaks `transformers` 5.14.1. Pinned `huggingface-hub>=1.5.0,<2.0` (now 1.33.0).
- WikiText-2 needs the namespaced id: `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")`.
- `/tmp` (and the scratchpad) is wiped on reboot. Keep test scripts in `phase9/`.

---

## 11. Fused K-scores kernel, fp32 twin and profiling (2026-11-04)

### What was built

- **`fused` (INT4):** grid `(n_groups, N_Q, B)`, one worker = one query head × one 32-token group. `kv = h // Q_PER_KV` (GQA without `repeat_kv`). It folds the scales into q (`qe = q_even * scale_even * SCALE`), unpacks the nibbles in registers and does `scores = Σ high·qe + low·qo`. No interleave is needed, because a sum doesn't depend on order. The residual scores are done in PyTorch and joined sealed-first.
- **`fused_fp32` (twin):** same grid and split, plain fp32 `[32, 64]` K tile, `tl.sum(k * q[None, :], axis=1) * SCALE`. The first `(S // 32) * 32` tokens go through the kernel, the tail through PyTorch.
- **Why the twin exists:** INT4 fused vs RAW mixes two gains: **fusion** (no `repeat_kv` copy of K, no separate matmul) and **INT4** (fewer bytes). The twin has the fusion but not the INT4, so `fp32 fused / fp32 raw` = fusion and `int4 fused / fp32 fused` = INT4. **The twin is a measuring stick, not the product.** Any non-quantization speed-up (fusion, preallocated cache, CUDA graphs) must go into both, or INT4 gets credit for it.
- Checks: max diff ≤ 6e-7 for both kernels. `fp32 fused` matches HF (23/23); `int4 fused` matches INT4 PyTorch (23/23).

### Results (tok/s, prompt 512)

| batch | fp32 raw | fp32 fused / raw (fusion) | int4 fused / fp32 fused (INT4) | int4 fused / raw |
|---|---|---|---|---|
| 1 | 77.9 | 0.91× | 0.87× | 0.79× |
| 16 | 775.8 | 1.07× | 0.90× | 0.96× |
| 64 | 1239.6 | 1.38× | 1.01× | 1.39× |
| 128 | 1303.6 | 1.43× | 1.03× | 1.48× |

`int4 fused / raw` = fusion × INT4 (1.43 × 1.03 ≈ 1.48). Almost all of the earlier "INT4 fused 1.39×" was fusion.

### Predictions I made, and what happened

| Prediction | Result |
|---|---|
| Fusion is the bigger factor at batch 64 | ✔ 1.38× vs 1.01× |
| My kernel loads each K value **7 times** (one per query head); DRAM sees it once if the other 6 loads hit **L2** (registers are per worker, not shared) | consistent: `fused_fp32` reads at ~240 GB/s, i.e. ~once from DRAM |
| The extra INT4 time is the **V side + per-step quantize** (K nibbles unpack in registers, nearly free) | ✔ `dequnatize_v_cache_kernel` 15.6 ms (b64) / 46.7 ms (b128) per 10 steps |

### How to read a `torch.profiler` table

- **Self CUDA** = GPU time inside that op: the column to read. CPU columns = launching **and blocking**; see the correction below.
- Rows come in two kinds: the PyTorch op (`aten::mm`) **and** the GPU kernel it launched (`cutlass...sgemm`, `gemvx`). It's the same time listed twice, so the percentages add up to > 100%. Triton kernels appear only under their own name.
- **Identify rows by `# of Calls`:** 240 = 24 layers × 10 steps (once per layer per step), 480 = twice per layer, 1200 = 5 per layer (weight matmuls), 10 = once per step (`lm_head`).
- **Check a guess with bytes ÷ bandwidth (~448 GB/s):** the 7× V at batch 64 = 64 × 14 × ~530 × 64 × 4 B ≈ 121 MB, giving ≈ 270 µs; `gemvx` measured 288 µs/call, so it is probs @ V.

### Where the GPU time goes (10 decode steps)

| Part | b64 fp32 fused | b64 int4 fused | b128 fp32 fused | b128 int4 fused |
|---|---|---|---|---|
| weight matmuls (`aten::mm`) | 100.6 ms | 100.5 | 161.1 | 160.9 |
| `repeat_kv` copy of V | 81.4 | 80.7 | 170.3 | 170.3 |
| probs @ V (`gemvx`) | 69.2 | 69.0 | 145.1 | 145.1 |
| cache append (`cat`) | 37.3 | 4.9 | 84.0 | 7.6 |
| K scores kernel | 17.3 | 13.4 | 32.7 | 27.0 |
| V dequant kernel | — | 15.6 | — | 46.7 |
| **Total GPU** | **322.7** | **303.8** | **617.1** | **584.0** |

Lessons:
1. **V is now the main cost:** `repeat_kv` V + probs @ V ≈ 46% of GPU time (b64), and ≈ 52% of INT4 time at b128 including the V dequant. After fusion, K scores are only ~5%.
2. **INT4's biggest saving is the cache append, not attention.** `torch.cat` rewrites the whole cache every step, and the INT4 cache is 8× smaller. That's an artifact of how we append; a preallocated cache removes it for both paths.
3. **The INT4 kernel is only ~1.3× faster than the fp32 twin, not 8×.** It reads 2.4 MB in 56 µs (~43 GB/s), so it is not memory-bound: tiny 32×32-byte tiles, load latency, and 7 workers per tile. Fix: more work per worker (all 7 query heads of a KV head, or more tokens).
4. **V dequant grew 3× from b64 to b128** (everything else ~2×). Likely its 17 MB fp32 output fit in L2 at b64 but 34 MB doesn't at b128 (~300 → ~175 GB/s).
5. **Batch 128 test** (reviewer: "same CPU work, ~2× cache GPU work"): call counts are identical at 64 and 128, and cache rows grew ~2×. The INT4 GPU saving (6.2% → 5.4%) showed up in tok/s as 1% → 3%. So more of the saving becomes visible when the GPU is busier.
6. **Correction:** "Self CPU total ≈ GPU time" does **not** prove the CPU is the bottleneck. It jumped 324 → 616 ms at b128 with the same kernels launched, so it includes time the CPU is blocked waiting for the GPU.

### Where INT4 stands

- **Better:** memory (KV cache ~6–7× smaller: 4 bits + scales vs 32 bits), which allows bigger batches or longer context on a 16 GB card; cache append; K scores bytes; +3% tok/s at batch 128.
- **Worse:** small batches (0.87× at b1), the V dequant kernel (it also writes a full fp32 V temp, so peak memory isn't fully saved yet), the per-step quantize ops, and quality (WikiText-2 +13.2%).
- **Goal:** make **INT4 fused** fast. Keep improving the fp32 twin with the same non-quantization tricks, so the comparison stays fair.

---

## 12. Next steps

1. ~~Replace `repeat_kv` with query grouping~~: done for **K** inside `fused` (`kv = h // Q_PER_KV`); still to do for **V**.
2. **Fused dequant + attention kernel**: read INT4 tiles, dequant in registers, `q·k`, online softmax, `× V`, accumulate. No fp32 cache written. K scores are done (`fused`); **V + online softmax is next**. It removes the `repeat_kv` V copy, the 7× V read and the V dequant kernel (~50% of GPU time).
3. **Preallocated cache** (write the new token into a slot, no `torch.cat`) in both paths: removes the append cost and makes the INT4 vs fp32 comparison fair. It also gives fixed shapes, which allows **CUDA graphs** (one launch per decode step) for the CPU side.
4. More work per worker in `fused`: all 7 query heads of a KV head per program, or 64+ tokens per tile.
5. Quality: asymmetric INT4 (scale + zero-point), smaller `K_GROUP`, and WikiText-2 at 2048 as the honest metric.
6. Optional practice: a standalone softmax kernel (row max, exp, sum, divide), tested against `torch.softmax`.
