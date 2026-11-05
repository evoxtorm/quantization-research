import os
import time
import traceback

import torch
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity

# Clean copy of fused_kernel.py with the CPU<->GPU syncs removed from the decode path.
# Found with torch.cuda.set_sync_debug_mode: the only sync in decode_step was
#   rope_cos_sin(torch.tensor([pos], device=device))   -> CPU tensor copied to the GPU, once per layer per step
# Fixes: build the position on the GPU (torch.arange), and generate_* collect tokens on the GPU with one
# .tolist() at the end instead of an .item() per step. check_no_syncs() guards against new syncs.

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EOS_IDS = (151645, 151643)

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

print(f"heads: {N_Q}q / {N_KV}kv, head_dim {HDIM}, group {GROUP}, theta {THETA}")

cache = [None] * 24

K_GROUP = 32        # tokens per sealed K group; newer K tokens wait unquantized in k_residual until a group fills
QUANTIZE = True     # False -> lossless bypass (x/s*s), same plumbing as INT4
USE_TRITON = True   # QUANTIZE=True only: True -> Triton dequant kernels, False -> PyTorch dequnatize_kv_cache
RAW = False         # True -> plain fp32 KV cache (no scales/residual/dequant): the real fp32 baseline
USE_FUSED = False   # QUANTIZE=True only: True -> fused INT4 scores kernel for K (takes priority over USE_TRITON)
RAW_FUSED = False   # RAW=True only: True -> K scores via the fp32 twin kernel (fair fp32 baseline for USE_FUSED)
DUMP_QK = False     # set only inside dump_qk(): prefill records each layer's post-RoPE q and k into qk_dump

qk_dump = []
DUMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps", "qk_oracle.pt")   # works from any cwd

ORACLE_TEXT = "Large language models require significant memory and computational bandwidth during inference. Quantization reduces this burden by compressing floating-point weights and activation tensors into low-bit integer formats such as INT8 or INT4. By scaling and rounding floating-point values, quantization preserves model accuracy while drastically lowering VRAM usage and accelerating matrix multiplication on modern hardware. Key-Value cache quantization specifically targets attention memory bottlenecks during auto-regressive decoding, enabling long-context generation on resource-constrained edge devices. Measuring perplexity over a longer text block ensures that tiny fluctuations in probability do not skew the evaluation."


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions, head_dim=HDIM, theta=THETA, device=device, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = positions[:, None].float() * inv_freq[None, :]
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep=GROUP):
    B, nkv, S, d = x.shape
    x = x[:, :, None, :, :].expand(B, nkv, n_rep, S, d)
    return x.reshape(B, nkv * n_rep, S, d)


def init_layer_cache(ck, cv):
    # seed one layer's cache from prefill K/V
    if RAW:
        return (ck, None, cv, None, ck[:, :, :0])   # empty residual, so pos = cache length (correct RoPE position)
    return qunatize_kv_cache(ck, cv)


def my_layer_forward(layer, h, is_prefill=True, cache_k=None, cache_v=None, k_scale=None, v_scale=None, k_residual=None):
    lyr = model.model.layers[layer]
    with torch.no_grad():
        residual = h
        h_norm = lyr.input_layernorm(h)
        if is_prefill:
            attn_out, ck, cv = prefill(h_norm, layer)
        else:
            attn_out, ck, cv, k_scale, v_scale, k_residual = decode(layer, h_norm, cache_k, cache_v, k_scale, v_scale, k_residual)

        h = residual + attn_out

        residual = h
        h_norm = lyr.post_attention_layernorm(h)
        h = residual + lyr.mlp(h_norm)
    return h, ck, cv, k_scale, v_scale, k_residual


def decode_project(layer, hidden):
    attn = model.model.layers[layer].self_attn
    B, S, _ = hidden.shape
    q = (hidden @ attn.q_proj.weight.T + attn.q_proj.bias).view(B, S, N_Q, HDIM).transpose(1, 2)
    k = (hidden @ attn.k_proj.weight.T + attn.k_proj.bias).view(B, S, N_KV, HDIM).transpose(1, 2)
    v = (hidden @ attn.v_proj.weight.T + attn.v_proj.bias).view(B, S, N_KV, HDIM).transpose(1, 2)
    return q, k, v


def Wo_of(layer):
    return model.model.layers[layer].self_attn.o_proj.weight.T


@triton.jit
def fused(
    k_quant, k_scales, q_proj, out,
    k_sb, k_sh, k_st,
    s_sb, s_sh, s_st,
    o_sb, o_sh, o_st,
    q_sb, q_sh, q_st,
    BYTES: tl.constexpr, GROUP: tl.constexpr, SCALE: tl.constexpr, Q_PER_KV: tl.constexpr,
):
    g = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rows = tl.arange(0, GROUP)
    cols = tl.arange(0, BYTES)

    toks = g * GROUP + rows

    kv = h // Q_PER_KV

    k = b * k_sb + kv * k_sh + toks[:, None] * k_st + cols[None, :]
    k_bytes = tl.load(k_quant + k)
    start = b * s_sb + kv * s_sh + g * s_st
    scale_even = start + 2 * cols
    scale_odd = start + 2 * cols + 1

    scale_val_even = tl.load(k_scales + scale_even)
    scale_val_odd = tl.load(k_scales + scale_odd)

    q_start = b * q_sb + h * q_sh
    q_even = q_start + 2 * cols
    q_odd = q_start + 2 * cols + 1
    q_val_even = tl.load(q_proj + q_even)
    q_val_odd = tl.load(q_proj + q_odd)

    qe = q_val_even * scale_val_even * SCALE
    qo = q_val_odd * scale_val_odd * SCALE

    high = ((k_bytes >> 4) & 0xf).to(tl.int32) - 8
    low = (k_bytes & 0xf).to(tl.int32) - 8

    high_f = high.to(tl.float32)
    low_f = low.to(tl.float32)

    scores = tl.sum(high_f * qe[None, :] + low_f * qo[None, :], axis=1)

    o_pos = b * o_sb + h * o_sh + toks * o_st

    tl.store(out + o_pos, scores)


@triton.jit
def fused_fp32(
    k_ptr, q_ptr, out,
    k_sb, k_sh, k_st,
    o_sb, o_sh, o_st,
    q_sb, q_sh,
    D: tl.constexpr, GROUP: tl.constexpr, SCALE: tl.constexpr, Q_PER_KV: tl.constexpr,
):
    # fp32 twin of `fused`: same grid and same GQA indexing, but K is plain fp32 (no nibbles, no scales)
    g = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rows = tl.arange(0, GROUP)        # 32 tokens of this group
    cols = tl.arange(0, D)            # all 64 channels, already in order
    toks = g * GROUP + rows
    kv = h // Q_PER_KV

    k_pos = b * k_sb + kv * k_sh + toks[:, None] * k_st + cols[None, :]
    k_vals = tl.load(k_ptr + k_pos)                          # [32 tokens, 64 channels]
    q_vals = tl.load(q_ptr + b * q_sb + h * q_sh + cols)     # [64]

    scores = tl.sum(k_vals * q_vals[None, :], axis=1) * SCALE

    o_pos = b * o_sb + h * o_sh + toks * o_st
    tl.store(out + o_pos, scores)


def decode(layer, hidden, cache_k, cache_v, k_scale, v_scale, k_residual):
    B, S, _ = hidden.shape
    q, k, v = decode_project(layer, hidden)

    pos = cache_k.shape[2] + k_residual.shape[2]     # Python int (shapes live on the CPU): no sync
    # was rope_cos_sin(torch.tensor([pos], device=device)): a CPU tensor copied to the GPU = 1 sync per layer per step
    cos, sin = rope_cos_sin(torch.arange(pos, pos + 1, device=device))   # built directly on the GPU
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scores = None   # set by the fused paths; every other path computes it from cache_k_ below
    if RAW:
        # plain fp32 cache: no scales, no residual, nothing to dequantize
        cache_k = torch.cat([cache_k, k], dim=2)
        cache_v = torch.cat([cache_v, v], dim=2)
        cache_k_, cache_v_ = cache_k, cache_v
        if RAW_FUSED:
            scores = fused_scores_fp32(q, cache_k)     # K scores via the fp32 twin kernel; V unchanged
    else:
        # V: per-token, per-head -- each new token gets its own scale ([B,N_KV,1,1]) appended to v_scale
        v_new_scale = torch.amax(torch.abs(v), dim=3, keepdim=True) / 7.0

        if QUANTIZE:
            v_q = pack_int4(torch.clamp(torch.round(v / v_new_scale), min=-8, max=7))
        else:
            v_q = v / v_new_scale

        v_scale = torch.cat([v_scale, v_new_scale], dim=2)
        cache_v = torch.cat([cache_v, v_q], dim=2)

        # K: per-channel, per-head over sealed groups -- new K waits in k_residual,
        # and once K_GROUP tokens have collected they are quantized together with their own scale
        k_residual = torch.cat([k_residual, k], dim=2)
        if k_residual.shape[2] == K_GROUP:          # shape check, not a tensor value: no sync
            k_q, k_new_scale = quantize_k_groups(k_residual)
            cache_k = torch.cat([cache_k, k_q], dim=2)
            k_scale = torch.cat([k_scale, k_new_scale], dim=2)
            k_residual = k_residual[:, :, :0]

        # the cache now holds every token (the new k is in k_residual, or just sealed into cache_k)
        if QUANTIZE and USE_FUSED:
            scores = fused_scores(q, cache_k, k_scale, k_residual)     # K: scores straight from INT4, no dequant/repeat_kv
            cache_v_ = dequantize_v_triton(cache_v, v_scale)            # V: still dequantized
        elif QUANTIZE and USE_TRITON:
            cache_k_, cache_v_ = dequantize_triton(cache_k, k_scale, cache_v, v_scale, k_residual)
        else:
            # PyTorch path: the lossless bypass (QUANTIZE=False) or int4 without Triton (USE_TRITON=False)
            cache_k_, cache_v_ = dequnatize_kv_cache((cache_k, k_scale, cache_v, v_scale))
            cache_k_ = torch.cat([cache_k_, k_residual], dim=2)

    if scores is None:   # every non-fused path
        scores = (q @ repeat_kv(cache_k_).transpose(-2, -1)) * SCALE
    assert scores.shape[-1] == cache_v_.shape[2], (scores.shape, cache_v_.shape)   # one score per V token

    vf = repeat_kv(cache_v_)
    attn_w = scores.softmax(dim=-1)
    out = attn_w @ vf
    out = out.transpose(1, 2).reshape(B, S, N_Q * HDIM) @ Wo_of(layer)

    return out, cache_k, cache_v, k_scale, v_scale, k_residual


@triton.jit
def merge_cache_k_residual(
    k_residual, out_k,
    r_sb, r_sh, r_st,
    o_sb, o_sh, o_st,
    tok_offset,
    D: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    cols = tl.arange(0, D)

    r_pos = b * r_sb + h * r_sh + t * r_st + cols
    o_pos = b * o_sb + h * o_sh + (t + tok_offset) * o_st + cols

    r_vals = tl.load(k_residual + r_pos)

    tl.store(out_k + o_pos, r_vals)


def prefill(hidden, layer):
    B, P, _ = hidden.shape
    q, k, v = decode_project(layer, hidden)
    cos, sin = rope_cos_sin(torch.arange(P, device=device))
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if DUMP_QK:   # dump_qk() only (.cpu() syncs, but this is prefill, never the decode path)
        qk_dump.append((q.contiguous().cpu(), k.contiguous().cpu()))
    cache_k, cache_v = k, v

    kf, vf = repeat_kv(k), repeat_kv(v)
    scores = (q @ kf.transpose(-2, -1)) * SCALE

    mask = torch.triu(torch.full((P, P), float("-inf"), device=device), diagonal=1)
    scores = scores + mask

    attn_w = scores.softmax(dim=-1)
    out = attn_w @ vf
    out = out.transpose(1, 2).reshape(B, P, N_Q * HDIM) @ Wo_of(layer)
    return out, cache_k, cache_v


def cut_at_eos(ids):
    """Tokens up to and including the first EOS, so generating a few extra steps changes nothing."""
    for i, t in enumerate(ids):
        if t in EOS_IDS:
            return ids[:i + 1]
    return ids


def generate_nocache(ids, max_new_tokens=20):
    # was: .item() every step (2 syncs per step) for the list and the EOS check.
    # Now: tokens stay on the GPU, one .tolist() at the end, EOS handled afterwards by cut_at_eos.
    seq = ids.clone()
    generated = []
    for _ in range(max_new_tokens):
        h = model.model.embed_tokens(seq)
        for layer in range(24):
            h, _, _, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
        h = model.model.norm(h)
        logits = model.lm_head(h)
        next_id = logits[:, -1, :].argmax(-1)
        generated.append(next_id)
        seq = torch.cat([seq, next_id[:, None]], dim=1)
    return cut_at_eos(torch.cat(generated).tolist())   # the only sync


def pack_int4(x):
    # x: values in [-8, 7] -> offset to 0..15, two nibbles per uint8 along the last dim
    u = (x + 8).to(torch.uint8)
    high = u[..., 0::2]
    low = u[..., 1::2]
    return (high << 4) | low


def unpack_int4(packed):
    high = (packed >> 4) & 0xF
    low = packed & 0xF
    # interleave back to the original order and remove the offset -> -8..7
    return torch.stack([high, low], dim=-1).flatten(start_dim=-2).to(torch.int8) - 8


def quantize_k_groups(k):
    # k: [B, N_KV, n*K_GROUP, HDIM] -> one scale per (head, group, channel): [B, N_KV, n, HDIM]
    # each group's scale comes from its own amax, so nothing in the group can clamp
    B, H, S, D = k.shape
    kg = k.reshape(B, H, S // K_GROUP, K_GROUP, D)
    scale = torch.amax(torch.abs(kg), dim=3, keepdim=True) / 7.0                 # [B,H,n,1,D]
    if QUANTIZE:
        k_q = pack_int4(torch.clamp(torch.round(kg / scale), min=-8, max=7))
    else:
        k_q = kg / scale
    return k_q.reshape(B, H, S, k_q.shape[-1]), scale.squeeze(3)   # explicit last dim: S can be 0 (prefill < K_GROUP)


def qunatize_kv_cache(k, v):
    # K: seal every full group of K_GROUP prefill tokens, leftover tail stays unquantized in k_residual
    P = k.shape[2]
    n_sealed = (P // K_GROUP) * K_GROUP
    quantized_layer_k, layer_scales = quantize_k_groups(k[:, :, :n_sealed])      # K per-channel per-head [B,N_KV,n,HDIM]
    k_residual = k[:, :, n_sealed:]

    per_token_scale = torch.amax(torch.abs(v), dim=3, keepdim=True) / 7.0        # V per-token per-head [B,N_KV,S,1]
    if QUANTIZE:
        quantized_v = pack_int4(torch.clamp(torch.round(v / per_token_scale), min=-8, max=7))
    else:
        quantized_v = v / per_token_scale
    return quantized_layer_k, layer_scales, quantized_v, per_token_scale, k_residual


@triton.jit
def dequnatize_v_cache_kernel(
    v_quant, v_scales, v_dequantized,
    q_sb, q_sh, q_st,
    s_sb, s_sh, s_st,
    o_sb, o_sh, o_st,
    BYTES: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    q_start = b * q_sb + h * q_sh + t * q_st
    s_pos = b * s_sb + h * s_sh + t * s_st
    o_start = b * o_sb + h * o_sh + t * o_st

    q = q_start + tl.arange(0, BYTES)
    even = o_start + 2 * tl.arange(0, BYTES)
    odd = o_start + 2 * tl.arange(0, BYTES) + 1

    q_bytes = tl.load(v_quant + q)
    scale = tl.load(v_scales + s_pos)

    high = ((q_bytes >> 4) & 0xf).to(tl.int32) - 8
    low = (q_bytes & 0xf).to(tl.int32) - 8
    high_val = high.to(tl.float32) * scale
    low_val = low.to(tl.float32) * scale

    tl.store(v_dequantized + even, high_val)
    tl.store(v_dequantized + odd, low_val)


@triton.jit
def dequnatize_k_cache_kernel(
    k_quant, k_scales, k_dequantized,
    q_sb, q_sh, q_st,
    s_sb, s_sh, s_st,
    o_sb, o_sh, o_st,
    BYTES: tl.constexpr, GROUP: tl.constexpr,
):
    g = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rows = tl.arange(0, GROUP)
    cols = tl.arange(0, BYTES)
    tok = g * GROUP + rows

    q = b * q_sb + h * q_sh + tok[:, None] * q_st + cols[None, :]
    s_start = b * s_sb + h * s_sh + g * s_st
    scale_even = s_start + 2 * cols
    scale_odd = scale_even + 1
    even = b * o_sb + h * o_sh + tok[:, None] * o_st + 2 * cols[None, :]
    odd = even + 1

    q_bytes = tl.load(k_quant + q)
    scale_val_even = tl.load(k_scales + scale_even)
    scale_val_odd = tl.load(k_scales + scale_odd)

    high = ((q_bytes >> 4) & 0xf).to(tl.int32) - 8
    low = (q_bytes & 0xf).to(tl.int32) - 8
    high_val = high.to(tl.float32) * scale_val_even[None, :]
    low_val = low.to(tl.float32) * scale_val_odd[None, :]

    tl.store(k_dequantized + even, high_val)
    tl.store(k_dequantized + odd, low_val)


def dequnatize_kv_cache(cache_val, dtype=torch.float32):
    k_quant, k_scales, v_quant, v_scales = cache_val
    if QUANTIZE:
        k_quant = unpack_int4(k_quant)
        v_quant = unpack_int4(v_quant)

    k_float = k_quant.to(dtype)
    v_float = v_quant.to(dtype)

    # each group scale covers K_GROUP consecutive tokens
    k_dequnatized = k_float * k_scales.repeat_interleave(K_GROUP, dim=2)
    v_dequnatized = v_float * v_scales

    return k_dequnatized, v_dequnatized


def dequantize_triton(cache_k, k_scale, cache_v, v_scale, k_residual):
    # same result as dequnatize_kv_cache + torch.cat([cache_k_, k_residual]), but with the Triton kernels
    cache_v_ = dequantize_v_triton(cache_v, v_scale)

    B_K, H_K, S_sealed, _ = cache_k.shape
    R = k_residual.shape[2]
    n_groups = S_sealed // K_GROUP
    cache_k_ = torch.empty(B_K, H_K, S_sealed + R, HDIM,                  # <- room for sealed + residual
                           dtype=torch.float32, device=cache_k.device)
    if n_groups > 0:
        assert cache_k.stride(-1) == 1 and k_scale.stride(-1) == 1
        dequnatize_k_cache_kernel[(n_groups, H_K, B_K)](                  # fills tokens 0 .. S_sealed-1
            cache_k, k_scale, cache_k_,
            *cache_k.stride()[:3], *k_scale.stride()[:3], *cache_k_.stride()[:3],
            BYTES=HDIM // 2, GROUP=K_GROUP,
        )
    if R > 0:
        assert k_residual.stride(-1) == 1
        merge_cache_k_residual[(R, H_K, B_K)](                            # fills tokens S_sealed .. S_sealed+R-1
            k_residual, cache_k_,
            *k_residual.stride()[:3], *cache_k_.stride()[:3],
            S_sealed, D=HDIM,
        )
    return cache_k_, cache_v_


def fused_scores(q, cache_k, k_scale, k_residual):
    """Scores for every K token, [B, N_Q, 1, S_sealed + R]: sealed from INT4 via Triton, residual in PyTorch."""
    B, _, S_sealed, _ = cache_k.shape
    n_groups = S_sealed // K_GROUP
    sealed = torch.empty(B, N_Q, S_sealed, dtype=torch.float32, device=q.device)
    if n_groups > 0:
        assert q.stride(-1) == 1 and cache_k.stride(-1) == 1 and k_scale.stride(-1) == 1
        fused[(n_groups, N_Q, B)](                                     # one worker per (group, QUERY head, batch)
            cache_k, k_scale, q, sealed,
            *cache_k.stride()[:3], *k_scale.stride()[:3], *sealed.stride()[:3], *q.stride()[:3],
            BYTES=HDIM // 2, GROUP=K_GROUP, SCALE=SCALE, Q_PER_KV=GROUP,   # global GROUP = 7 query heads per KV head
        )
    resid = (q @ repeat_kv(k_residual).transpose(-2, -1)) * SCALE        # [B, N_Q, 1, R], at most 31 tokens
    return torch.cat([sealed.unsqueeze(2), resid], dim=-1)               # sealed first, then residual: same order as V


def dequantize_v_triton(cache_v, v_scale):
    B_V, H_V, S_V, _ = cache_v.shape
    assert cache_v.stride(-1) == 1 and v_scale.stride(-1) == 1
    cache_v_ = torch.empty(B_V, H_V, S_V, HDIM, dtype=torch.float32, device=cache_v.device)
    dequnatize_v_cache_kernel[(S_V, H_V, B_V)](
        cache_v, v_scale, cache_v_,
        *cache_v.stride()[:3], *v_scale.stride()[:3], *cache_v_.stride()[:3],
        BYTES=HDIM // 2,
    )
    return cache_v_


def fused_scores_fp32(q, cache_k):
    """RAW_FUSED scores, [B, N_Q, 1, S]. Split like the INT4 fused path: whole 32-token groups go through
    the fp32 twin kernel, the leftover tail (like the INT4 residual) through PyTorch."""
    B, _, S, _ = cache_k.shape
    n_groups = S // K_GROUP
    n_main = n_groups * K_GROUP
    main = torch.empty(B, N_Q, n_main, dtype=torch.float32, device=q.device)
    if n_groups > 0:
        assert q.stride(-1) == 1 and cache_k.stride(-1) == 1
        fused_fp32[(n_groups, N_Q, B)](
            cache_k, q, main,
            *cache_k.stride()[:3], *main.stride()[:3], *q.stride()[:2],
            D=HDIM, GROUP=K_GROUP, SCALE=SCALE, Q_PER_KV=GROUP,     # global GROUP = 7 query heads per KV head
        )
    tail = (q @ repeat_kv(cache_k[:, :, n_main:]).transpose(-2, -1)) * SCALE     # [B, N_Q, 1, S % 32]
    return torch.cat([main.unsqueeze(2), tail], dim=-1)


def make_test_cache(batch, seq):
    # random K/V with the same non-contiguous layout decode_project produces, quantized like a prefill
    k = torch.randn(batch, seq, N_KV, HDIM, device=device).transpose(1, 2)
    v = torch.randn(batch, seq, N_KV, HDIM, device=device).transpose(1, 2)
    return qunatize_kv_cache(k, v)


def check_dequant_triton(batch=2, seq=3 * K_GROUP + 5):
    # Triton must give bit-identical numbers to the PyTorch dequant (QUANTIZE must be True)
    torch.manual_seed(0)
    ck, k_scale, cv, v_scale, k_res = make_test_cache(batch, seq)
    ref_k, ref_v = dequnatize_kv_cache((ck, k_scale, cv, v_scale))
    ref_k = torch.cat([ref_k, k_res], dim=2)
    out_k, out_v = dequantize_triton(ck, k_scale, cv, v_scale, k_res)
    return torch.equal(out_k, ref_k), torch.equal(out_v, ref_v)


def check_fused_scores(batch=2, seq=3 * K_GROUP + 5):
    # fused scores vs the old path (dequant K + residual, repeat_kv, q @ K.T); allclose, not equal: folding the
    # scales into q rounds slightly differently (QUANTIZE must be True)
    torch.manual_seed(0)
    ck, k_scale, cv, v_scale, k_res = make_test_cache(batch, seq)
    q = torch.randn(batch, 1, N_Q, HDIM, device=device).transpose(1, 2)    # [B, N_Q, 1, HDIM], same layout as decode_project
    ref_k, _ = dequnatize_kv_cache((ck, k_scale, cv, v_scale))
    ref_k = torch.cat([ref_k, k_res], dim=2)
    ref = (q @ repeat_kv(ref_k).transpose(-2, -1)) * SCALE
    mine = fused_scores(q, ck, k_scale, k_res)
    return torch.allclose(mine, ref, atol=1e-4, rtol=1e-4), (mine - ref).abs().max().item()


def check_fused_scores_fp32(batch=2, seq=3 * K_GROUP + 5):
    # fp32 twin vs plain PyTorch attention scores on an fp32 cache (no quantization involved)
    torch.manual_seed(0)
    k = torch.randn(batch, seq, N_KV, HDIM, device=device).transpose(1, 2)   # same layout as decode_project
    q = torch.randn(batch, 1, N_Q, HDIM, device=device).transpose(1, 2)
    ref = (q @ repeat_kv(k).transpose(-2, -1)) * SCALE
    mine = fused_scores_fp32(q, k)
    return torch.allclose(mine, ref, atol=1e-4, rtol=1e-4), (mine - ref).abs().max().item()


def check_no_syncs(batch=4, prompt_len=100, steps=40):
    """Decode steps under set_sync_debug_mode("error") in the current mode: any CPU<->GPU sync raises.
    prompt_len=100 leaves 4 residual tokens, so 40 steps also cross a K seal (at step 28).
    Returns None, or "line N: <code>" for the first syncing line in this file."""
    if not torch.cuda.is_available():
        return None
    seed_cache(batch, prompt_len)
    nxt = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)
    decode_step(nxt)                                  # first call compiles the Triton kernels, outside the check
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(steps):
            nxt = decode_step(nxt)
    except RuntimeError as e:
        mine = [f for f in traceback.extract_tb(e.__traceback__) if os.path.basename(f.filename) == os.path.basename(__file__)]
        return f"line {mine[-1].lineno}: {mine[-1].line}" if mine else str(e)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return None


def time_ms(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1000


def bench_dequant(batch, seq=512 + 24):
    # dequant only, one layer: PyTorch path vs Triton path (seq = 16 sealed groups + 24 residual tokens)
    ck, k_scale, cv, v_scale, k_res = make_test_cache(batch, seq)

    def torch_path():
        k_, v_ = dequnatize_kv_cache((ck, k_scale, cv, v_scale))
        return torch.cat([k_, k_res], dim=2), v_

    def triton_path():
        return dequantize_triton(ck, k_scale, cv, v_scale, k_res)

    return time_ms(torch_path), time_ms(triton_path)


def decode_step(token_ids):
    """[B,1] token ids -> [B,1] next ids. Mutates the global `cache`. No CPU<->GPU sync (see check_no_syncs)."""
    h = model.model.embed_tokens(token_ids)
    for layer in range(24):
        ck, k_scale, cv, v_scale, k_residual = cache[layer]
        h, ck, cv, k_scale, v_scale, k_residual = my_layer_forward(
            layer, h, is_prefill=False,
            cache_k=ck, cache_v=cv, k_scale=k_scale, v_scale=v_scale, k_residual=k_residual)
        cache[layer] = (ck, k_scale, cv, v_scale, k_residual)
    h = model.model.norm(h)
    logits = model.lm_head(h)[:, -1, :]
    return logits.argmax(-1, keepdim=True)


def prefill_cache(ids):
    """Prefill through our own layers and seed the global `cache`. Returns the last hidden states [B, P, hidden].
    No lm_head here: at batch 64 x 512 tokens its logits alone would be ~20 GB."""
    h = model.model.embed_tokens(ids)
    for layer in range(24):
        h, ck, cv, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
        cache[layer] = init_layer_cache(ck, cv)
    return h


def seed_cache(batch, prompt_len):
    """Random prompt -> populated cache (decode_step needs one)."""
    ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)
    with torch.no_grad():
        prefill_cache(ids)


def dump_qk(ids, path=DUMP_PATH):
    """One prefill on `ids`, saving every layer's post-RoPE q and k: fp32, before any quantization
    (k is exactly what the cache quantizes). Prefill is plain fp32 in every mode, so the mode doesn't matter.
    Saved: q [24, B, N_Q, P, HDIM], k [24, B, N_KV, P, HDIM], ids [B, P]."""
    global DUMP_QK
    qk_dump.clear()
    DUMP_QK = True
    try:
        with torch.no_grad():
            prefill_cache(ids)
    finally:
        DUMP_QK = False
    q = torch.stack([q for q, _ in qk_dump])
    k = torch.stack([k for _, k in qk_dump])
    qk_dump.clear()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"q": q, "k": k, "ids": ids.cpu(),
                "layout": "q [layer, B, N_Q, P, HDIM], k [layer, B, N_KV, P, HDIM]; post-RoPE fp32, before quantization"},
               path)
    return q, k


@torch.no_grad()
def generate_cached(ids, max_new_tokens):
    """Greedy generation through our own prefill + quantized cache + decode. Uses the global mode flags.
    was: .item() every step for the list and the EOS check. Now: one .tolist() at the end + cut_at_eos."""
    h = prefill_cache(ids)
    next_id = model.lm_head(model.model.norm(h))[:, -1, :].argmax(-1, keepdim=True)   # first token comes from prefill
    generated = [next_id]
    for _ in range(max_new_tokens - 1):
        next_id = decode_step(next_id)
        generated.append(next_id)
    return cut_at_eos(torch.cat(generated, dim=1)[0].tolist())   # the only sync


def first_diff(a, b):
    """Index of the first token where two generations differ, or None if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(batch, prompt_len=512, warmup=10, iters=100):
    seed_cache(batch, prompt_len)                     # 1. seed the cache with a prefill
    nxt = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)

    for _ in range(warmup):                           # 2. warmup -- not timed
        nxt = decode_step(nxt)
    sync()

    t0 = time.perf_counter()                          # 3. timed
    for _ in range(iters):
        nxt = decode_step(nxt)
    sync()
    t1 = time.perf_counter()

    return (iters * batch) / (t1 - t0)   # tokens/sec


def profile_step(batch=64, steps=10):
    """Per-kernel GPU time for `steps` decode steps in the current mode (RAW / QUANTIZE / USE_FUSED flags)."""
    benchmark(batch, iters=1)                 # seeds the cache + warms up
    nxt = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            nxt = decode_step(nxt)
        sync()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))


MODES = {   # name -> flags; every mode sets all five so nothing leaks from the previous one
    "fp32 raw":     dict(RAW=True,  RAW_FUSED=False, QUANTIZE=True,  USE_TRITON=False, USE_FUSED=False),
    "fp32 fused":   dict(RAW=True,  RAW_FUSED=True,  QUANTIZE=True,  USE_TRITON=False, USE_FUSED=False),
    "fp32 bypass":  dict(RAW=False, RAW_FUSED=False, QUANTIZE=False, USE_TRITON=False, USE_FUSED=False),
    "int4 pytorch": dict(RAW=False, RAW_FUSED=False, QUANTIZE=True,  USE_TRITON=False, USE_FUSED=False),
    "int4 triton":  dict(RAW=False, RAW_FUSED=False, QUANTIZE=True,  USE_TRITON=True,  USE_FUSED=False),
    "int4 fused":   dict(RAW=False, RAW_FUSED=False, QUANTIZE=True,  USE_TRITON=True,  USE_FUSED=True),
}


def set_mode(name):
    globals().update(MODES[name])


if __name__ == "__main__":

    text = ORACLE_TEXT

    RUN_DUMP = True          # save the oracle prompt's per-layer q and k to phase9/dumps/qk_oracle.pt
    RUN_BENCHMARK = True     # set False to run only the checks and the generation test
    RUN_PROFILE = True       # per-kernel GPU time table: fp32 fused vs int4 fused
    PROFILE_BATCH = 128

    # 1. correctness gates: Triton dequant == PyTorch dequant, fused scores == PyTorch scores,
    #    incl. no sealed group (5), no residual (64), both (101)
    set_mode("int4 triton")
    for seq in (5, 2 * K_GROUP, 3 * K_GROUP + 5):
        k_ok, v_ok = check_dequant_triton(seq=seq)
        print(f"dequant check seq={seq:3d}:  K {'OK' if k_ok else 'MISMATCH'}   V {'OK' if v_ok else 'MISMATCH'}")
    for seq in (5, 2 * K_GROUP, 3 * K_GROUP + 5):
        ok, max_diff = check_fused_scores(seq=seq)
        print(f"fused scores check seq={seq:3d}:  {'OK' if ok else 'MISMATCH'}   max diff {max_diff:.2e}")
    for seq in (5, 2 * K_GROUP, 3 * K_GROUP + 5):
        ok, max_diff = check_fused_scores_fp32(seq=seq)
        print(f"fp32 fused scores check seq={seq:3d}:  {'OK' if ok else 'MISMATCH'}   max diff {max_diff:.2e}")

    # 2. sync gate: decode_step must never make the CPU wait for the GPU, in any mode
    for name in MODES:
        set_mode(name)
        where = check_no_syncs()
        print(f"sync check {name:13s}: {'no sync' if where is None else 'SYNC at ' + where}")

    # 3. generation oracle: a long prompt, so prefill seals groups AND a seal fires during decode
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    P = ids.shape[1]
    first_seal = K_GROUP - P % K_GROUP          # decode steps until the residual reaches K_GROUP
    N_NEW = first_seal + 12                     # cross the first decode seal with room to spare
    print(f"\n--- generation oracle ---")
    print(f"P = {P}. Prefill seals {P // K_GROUP} groups. Residual after prefill: {P % K_GROUP} tokens. "
          f"First decode seal after {first_seal} decode steps. Generating {N_NEW} new tokens.")

    with torch.no_grad():
        hf = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, repetition_penalty=1.0,
                            temperature=1.0, top_p=1.0)[0, P:].tolist()
    nocache = generate_nocache(ids, max_new_tokens=N_NEW)

    out = {}
    for name in MODES:
        set_mode(name)
        out[name] = generate_cached(ids, N_NEW)

    print(f"raw fp32 vs HF:               {'MATCH' if out['fp32 raw'] == hf else 'MISMATCH'}")
    print(f"raw fp32 vs nocache:          {'MATCH' if out['fp32 raw'] == nocache else 'MISMATCH'}")
    print(f"fp32 fused vs HF:             {'MATCH' if out['fp32 fused'] == hf else 'DIFFERS'}"
          f"  (summation order differs from PyTorch, so a near-tie could flip)")
    print(f"fp32 bypass vs HF:            {'MATCH' if out['fp32 bypass'] == hf else 'MISMATCH'}")
    print(f"fp32 bypass vs nocache:       {'MATCH' if out['fp32 bypass'] == nocache else 'MISMATCH'}")
    print(f"int4 triton vs int4 pytorch:  {'MATCH' if out['int4 triton'] == out['int4 pytorch'] else 'MISMATCH'}")
    fused_agree = sum(a == b for a, b in zip(out['int4 fused'], out['int4 pytorch']))
    print(f"int4 fused vs int4 pytorch:   {'MATCH' if out['int4 fused'] == out['int4 pytorch'] else 'DIFFERS'}"
          f"  ({fused_agree}/{len(out['int4 pytorch'])} tokens agree; tiny rounding differences can flip a near-tie)")
    agree = sum(a == b for a, b in zip(out['int4 triton'], hf))
    print(f"int4 vs HF (info only, not expected to match): {agree}/{len(hf)} tokens agree")

    runs = [("hf", hf), ("nocache", nocache)] + list(out.items())
    print("\n--- generated tokens ---")
    for name, out_ids in runs:
        print(f"{name:13s} ids : {out_ids}")
    print("\n--- generated text ---")
    for name, out_ids in runs:
        print(f"{name:13s} text: {tok.decode(out_ids)!r}")
    print("\n--- first differing position (None = identical) ---")
    for name, a, b in [("raw fp32 vs hf", out['fp32 raw'], hf), ("raw fp32 vs nocache", out['fp32 raw'], nocache),
                       ("fp32 fused vs hf", out['fp32 fused'], hf),
                       ("fp32 bypass vs hf", out['fp32 bypass'], hf), ("fp32 bypass vs nocache", out['fp32 bypass'], nocache),
                       ("int4 triton vs int4 pytorch", out['int4 triton'], out['int4 pytorch']),
                       ("int4 fused vs int4 pytorch", out['int4 fused'], out['int4 pytorch']),
                       ("int4 triton vs hf", out['int4 triton'], hf)]:
        i = first_diff(a, b)
        detail = "" if i is None else f"   ({a[i] if i < len(a) else '-'} vs {b[i] if i < len(b) else '-'}, decode step {i})"
        print(f"{name:28s}: {i}{detail}")

    if RUN_DUMP:
        q_d, k_d = dump_qk(ids)
        print(f"\nQ/K dump: q {tuple(q_d.shape)}, k {tuple(k_d.shape)} -> {DUMP_PATH}")

    if RUN_BENCHMARK:
        # 4. dequant only, one layer, 536 tokens
        set_mode("int4 triton")
        print("\n--- dequant only (one layer, 536 tokens) ---")
        for batch in [1, 4, 16, 64]:
            t_torch, t_triton = bench_dequant(batch)
            print(f"batch={batch:3d}  pytorch: {t_torch:7.3f} ms   triton: {t_triton:7.3f} ms   speedup: {t_torch/t_triton:5.2f}×")

        # 5. full decode step (24 layers), prompt 512
        print("\n--- full decode (tok/s) ---")
        for batch in [1, 4, 16, 64, 128]:
            tps = {}
            for name in ("fp32 raw", "fp32 fused", "int4 pytorch", "int4 triton", "int4 fused"):
                set_mode(name)
                tps[name] = benchmark(batch)
            fp, fp_fused = tps["fp32 raw"], tps["fp32 fused"]
            print(f"batch={batch:3d}  fp32 raw: {fp:8.1f}   fp32 fused: {fp_fused:8.1f} ({fp_fused/fp:.2f}×)   "
                  f"int4 pytorch: {tps['int4 pytorch']:8.1f} ({tps['int4 pytorch']/fp:.2f}×)   "
                  f"int4 triton: {tps['int4 triton']:8.1f} ({tps['int4 triton']/fp:.2f}×)   "
                  f"int4 fused: {tps['int4 fused']:8.1f} ({tps['int4 fused']/fp:.2f}×)   "
                  f"int4 fused / fp32 fused: {tps['int4 fused']/fp_fused:.2f}×")

    if RUN_PROFILE:
        # 6. where the GPU time goes, same 512-token prompt as the benchmark
        for name in ("fp32 fused", "int4 fused"):
            set_mode(name)
            print(f"\n--- profile: {name}, batch {PROFILE_BATCH}, 10 decode steps ---")
            profile_step(PROFILE_BATCH)
