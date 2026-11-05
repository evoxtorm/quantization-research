import triton
import torch
import triton.language as tl

import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

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

print(cfg)
print(f"heads: {N_Q}q / {N_KV}kv, head_dim {HDIM}, group {GROUP}, theta {THETA}")

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

LAYER = 0
attn = model.model.layers[LAYER].self_attn
Wq, bq = attn.q_proj.weight, attn.q_proj.bias
Wk, bk = attn.k_proj.weight, attn.k_proj.bias
Wv, bv = attn.v_proj.weight, attn.v_proj.bias

Wo = attn.o_proj.weight

cache = [None] * 24
cache_k_raw = [None] * 24

K_GROUP = 32   # tokens per sealed K group; newer K tokens wait unquantized in k_residual until a group fills
USE_TRITON = True   # QUANTIZE=True only: True -> Triton dequant kernels, False -> PyTorch dequnatize_kv_cache
RAW = False         # True -> plain fp32 KV cache (no scales/residual/dequant): the real fp32 baseline


def init_layer_cache(ck, cv):
    # seed one layer's cache from prefill K/V
    if RAW:
        return (ck, None, cv, None, ck[:, :, :0])   # empty residual, so pos = cache length (correct RoPE position)
    return qunatize_kv_cache(ck, cv)

def project(hidden):
    B, S, _ = hidden.shape
    q = (hidden @ Wq.T + bq).view(B, S, N_Q, HDIM).transpose(1, 2)
    k = (hidden @ Wk.T + bk).view(B, S, N_KV, HDIM).transpose(1, 2)
    v = (hidden @ Wv.T + bv).view(B, S, N_KV, HDIM).transpose(1, 2)
    return q, k, v

def my_layer_forward(layer, h, is_prefill=True, cache_k=None, cache_v=None, k_scale=None, v_scale=None, k_residual=None):
    lyr = model.model.layers[layer]
    with torch.no_grad():
        residual = h
        h_norm = lyr.input_layernorm(h)
        if is_prefill:
            attn_out, ck, cv = prefill(h_norm,layer)
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


def decode(layer, hidden, cache_k, cache_v, k_scale, v_scale, k_residual):
    B, S, _ = hidden.shape
    q, k, v = decode_project(layer, hidden)

    pos = cache_k.shape[2] + k_residual.shape[2]
    cos, sin = rope_cos_sin(torch.tensor([pos], device=device))
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if RAW:
        # plain fp32 cache: no scales, no residual, nothing to dequantize
        cache_k = torch.cat([cache_k, k], dim=2)
        cache_v = torch.cat([cache_v, v], dim=2)
        cache_k_, cache_v_ = cache_k, cache_v
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
        if k_residual.shape[2] == K_GROUP:
            k_q, k_new_scale = quantize_k_groups(k_residual)
            cache_k = torch.cat([cache_k, k_q], dim=2)
            k_scale = torch.cat([k_scale, k_new_scale], dim=2)
            k_residual = k_residual[:, :, :0]


        # print("cache_k", cache_k.shape, "cache_v", cache_v.shape)
        # print("cache_k", cache_k.shape, cache_k.dtype, "| k_scale", k_scale.shape,
        #   "| k_residual", k_residual.shape)
        # print("cache_v", cache_v.shape, cache_v.dtype, "| v_scale", v_scale.shape)
        # print("cache_k contiguous:", cache_k.is_contiguous(), "strides:", cache_k.stride())

        # B_V, H_V, S_V, _ = cache_v.shape
        # out_v = torch.empty(B_V, H_V, S_V, 64, dtype=torch.float32, device=cache_v.device)
        # dequnatize_v_cache_kernel[(S_V, H_V, B_V)](
        #     cache_v, v_scale, out_v,
        #     *cache_v.stride()[:3],
        #     *v_scale.stride()[:3],
        #     *out_v.stride()[:3],
        #     BYTES=32,
        # )

        # B, H, S_sealed, _ = cache_k.shape
        # n_groups = S_sealed // K_GROUP
        # out_k = torch.empty(B, H, S_sealed, 64, dtype=torch.float32, device=cache_v.device)
        # dequnatize_k_cache_kernel[(n_groups, H, B)](
        #     cache_k, k_scale, out_k,
        #     *cache_k.stride()[:3],
        #     *k_scale.stride()[:3],
        #     *out_k.stride()[:3],
        #     BYTES=32, GROUP=K_GROUP,
        # )
        if QUANTIZE and USE_TRITON:
            cache_k_, cache_v_ = dequantize_triton(cache_k, k_scale, cache_v, v_scale, k_residual)
        else:
            # PyTorch path: the lossless bypass (QUANTIZE=False) or int4 without Triton (USE_TRITON=False)
            cache_k_, cache_v_ = dequnatize_kv_cache((cache_k, k_scale, cache_v, v_scale))
            cache_k_ = torch.cat([cache_k_, k_residual], dim=2)

    assert cache_k_.shape[2] == cache_v_.shape[2], (cache_k_.shape, cache_v_.shape)

    # cache_k_ = torch.cat([cache_k_, k_residual], dim=2)      # keep this: the residual still joins after dequant

    # R = k_residual.shape[2]
    # if R > 0:
    #     B_K, H_K, S_sealed, _ = cache_k.shape
    #     assert k_residual.stride(-1) == 1
    #     merge_cache_k_residual[(R, H_K, B_K)](
    #         k_residual, cache_k_,
    #         *k_residual.stride()[:3], *cache_k_.stride()[:3],
    #         S_sealed, D=HDIM,
    #     )


    # cache_k_, cache_v_ = dequnatize_kv_cache((cache_k, k_scale, cache_v, v_scale))
    # cache_k_ = torch.cat([cache_k_, k_residual], dim=2)
    # prefill_len = cache_k_raw[layer].shape[2]
    # recon_err = (cache_k_[:, :, :prefill_len, :] - cache_k_raw[layer]).abs().max()
    # print(f"layer {layer} prefill K recon error: {recon_err.item():.6f}")

    # print("k_scale", k_scale.shape, "cache_k", cache_k.shape)
    # print("v_scale", v_scale.shape, "cache_v", cache_v.shape)
    kf, vf = repeat_kv(cache_k_), repeat_kv(cache_v_)

    # print(cache_k_.shape, cache_v_.shape, " this is the shape")
    # print("cache_k_ dtype:", cache_k_.dtype)
    # print("cache_k_ abs max:", cache_k_.abs().max().item())
    scores = (q @ kf.transpose(-2, -1)) * SCALE
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

    r_pos = b *r_sb + h * r_sh + t * r_st + cols
    o_pos = b * o_sb  + h * o_sh + (t + tok_offset) * o_st + cols

    r_vals = tl.load(k_residual + r_pos)

    tl.store(out_k + o_pos, r_vals)



def prefill(hidden, layer):
    B, P, _ =  hidden.shape
    q, k, v = decode_project(layer, hidden)
    cos, sin = rope_cos_sin(torch.arange(P, device=device))
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    cache_k, cache_v = k, v

    kf, vf = repeat_kv(k), repeat_kv(v)
    scores = (q @ kf.transpose(-2, -1)) * SCALE

    mask = torch.triu(torch.full((P,P), float("-inf"), device=device), diagonal=1)

    scores = scores + mask

    attn_w = scores.softmax(dim=-1)
    out = attn_w @ vf
    out = out.transpose(1, 2).reshape(B, P, N_Q * HDIM) @ Wo_of(layer)
    return out, cache_k, cache_v


def generate_nocache(ids, max_new_tokens=20):
    seq = ids.clone()
    generated = []
    for _ in range(max_new_tokens):
        h = model.model.embed_tokens(seq)
        for layer in range(24):
            h, _, _, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
        h  = model.model.norm(h)
        logits = model.lm_head(h)
        next_id = logits[:, -1, :].argmax(-1)
        generated.append(next_id.item())
        seq = torch.cat([seq, next_id[:, None]], dim=1)
        if next_id.item() in (151645, 151643):
            break
    return generated

# def qunatize_kv_cache(k, v):
#     layer_maxes = torch.amax(torch.abs(k), dim=(0, 1, 3), keepdim=True)
#     layer_scales = layer_maxes / 127
#     quantized_layer_k = torch.round(k / layer_scales).to(torch.int8)
#     # quantized_layer_k = k / layer_scales


#     per_token_max = torch.amax(torch.abs(v), dim=(0,1, 3), keepdim=True)
#     per_token_scale = per_token_max / 127
#     quantized_v = torch.round(v / per_token_scale).to(torch.int8)
#     # quantized_v = v / per_token_scale
#     return quantized_layer_k, layer_scales, quantized_v, per_token_scale

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
    # layer_scales = torch.amax(torch.abs(k), dim=(0,1,2), keepdim=True) / 7.0
    # per_token_scale = torch.amax(torch.abs(v), dim=(0,1,3), keepdim=True) / 7.0
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

    q = q_start  + tl.arange(0, BYTES)
    # s = s_pos
    even = o_start + 2 * tl.arange(0, BYTES )
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

    # q_start = b * q_sb + h * q_sh + t * q_st
    # s_pos = b * s_sb + h * s_sh + t * s_st  
    # o_start = b * o_sb + h * o_sh + t * o_st


    q = b * q_sb + h * q_sh + tok[:, None] * q_st + cols[None, :]
    # s = s_pos
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
    B_V, H_V, S_V, _ = cache_v.shape
    assert cache_v.stride(-1) == 1 and v_scale.stride(-1) == 1
    cache_v_ = torch.empty(B_V, H_V, S_V, HDIM, dtype=torch.float32, device=cache_v.device)
    dequnatize_v_cache_kernel[(S_V, H_V, B_V)](
        cache_v, v_scale, cache_v_,
        *cache_v.stride()[:3], *v_scale.stride()[:3], *cache_v_.stride()[:3],
        BYTES=HDIM // 2,
    )

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

def calculate_for_hf(ids):
    with torch.no_grad():
            outputs = model(ids)
    
    # print(f"Logits shape: {outputs.logits.shape}")

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()

    print(f"Shifted logits shape: {shift_logits.shape}")

    shift_labels = ids[:, 1:].contiguous()

    # print(f"Shifted labels shape: {shift_labels.shape}")

    loss_fct = CrossEntropyLoss()

    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    perplexity = torch.exp(loss)

    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")


def score_perplexity(ids):
    token_ids = ids[0]
    B, N = ids.shape

    n = token_ids.shape[0]
    log_probs = []

    first = ids[:, :1]
    # first = token_ids[:1].unsqueeze(0)
    h = model.model.embed_tokens(first)
    for layer in range(24):
        h, ck, cv, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
        cache_k_raw[layer] = ck.clone()
        cache[layer] = init_layer_cache(ck, cv)

    h = model.model.norm(h)
    logits = model.lm_head(h)[:, -1, :]

    # lp = torch.log_softmax(logits, dim=-1)[0, token_ids[1]]
    lp = torch.log_softmax(logits, dim=-1)[torch.arange(B), ids[:, 1]]
    log_probs.append(lp)

    for i in range(1, n-1):
        # cur = token_ids[i:i+1].unsqueeze(0)
        cur = ids[:, i:i+1]
        h = model.model.embed_tokens(cur)
        for layer in range(24):
            ck, k_scale, cv, v_scale, k_residual = cache[layer]
            h, ck, cv, k_scale, v_scale, k_residual = my_layer_forward(
                layer, h, is_prefill=False,
                cache_k=ck, cache_v=cv, k_scale=k_scale, v_scale=v_scale, k_residual=k_residual)
            cache[layer] = (ck, k_scale, cv, v_scale, k_residual)

        h = model.model.norm(h)
        logits = model.lm_head(h)[:, -1, :]

        # lp = torch.log_softmax(logits, dim=-1)[0, token_ids[i+1]]
        lp = torch.log_softmax(logits, dim=-1)[torch.arange(B), ids[:, i+1]]
        log_probs.append(lp)

    avg_neg_logprob = -torch.stack(log_probs).mean()
    perplexity = torch.exp(avg_neg_logprob)
    return perplexity.item()


def standalone_test():
    # start with known signed int4 values, -8..7

    x = torch.randint(-8, 8, (64,))          # 64 values

    # --- pack (your logic, inline) ---
    u = x + 8                                 # offset to 0..15
    high = u[0::2]                            # even positions -> 32 values
    low  = u[1::2]                            # odd positions  -> 32 values
    packed = (high << 4) | low                # 32 bytes, two nibbles each

    # --- unpack (your logic, inline) ---
    hi = (packed >> 4) & 0xF                  # top nibble  -> 32 values
    lo = packed & 0xF                         # bottom nibble -> 32 values
    # interleave hi and lo back into 64
    back = torch.empty(64, dtype=torch.long)
    back[0::2] = hi
    back[1::2] = lo
    back = back - 8                           # remove offset -> back to -8..7

    # --- check ---
    print((x - back).abs().max().item())      # should be 0




def decode_step(token_ids):
    """[B,1] token ids -> [B,1] next ids. Mutates the global `cache`."""
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


@torch.no_grad()
def generate_cached(ids, max_new_tokens):
    """Greedy generation through our own prefill + quantized cache + decode. Uses the global QUANTIZE / USE_TRITON."""
    h = model.model.embed_tokens(ids)
    for layer in range(24):
        h, ck, cv, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
        cache_k_raw[layer] = ck.clone()
        cache[layer] = init_layer_cache(ck, cv)
    next_id = model.lm_head(model.model.norm(h))[:, -1, :].argmax(-1, keepdim=True)   # first token comes from prefill
    generated = [next_id.item()]
    for _ in range(max_new_tokens - 1):
        next_id = decode_step(next_id)
        generated.append(next_id.item())
        if next_id.item() in (151645, 151643):
            break
    return generated


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
    # 1. seed the cache with a prefill — decode_step needs a populated cache
    ids = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)
    h = model.model.embed_tokens(ids)
    for layer in range(24):
        h, ck, cv, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
        cache_k_raw[layer] = ck.clone()
        cache[layer] = init_layer_cache(ck, cv)

    nxt =torch.randint(0, cfg.vocab_size, (batch, 1), device=device)

    # 2. warmup — not timed
    for _ in range(warmup):
        nxt = decode_step(nxt)
    sync()

    # 3. timed
    t0 = time.perf_counter()
    for _ in range(iters):
        nxt = decode_step(nxt)
    sync()
    t1 = time.perf_counter()

    return (iters * batch) / (t1 - t0)   # tokens/sec



if __name__ == "__main__":

    

    text = "Large language models require significant memory and computational bandwidth during inference. Quantization reduces this burden by compressing floating-point weights and activation tensors into low-bit integer formats such as INT8 or INT4. By scaling and rounding floating-point values, quantization preserves model accuracy while drastically lowering VRAM usage and accelerating matrix multiplication on modern hardware. Key-Value cache quantization specifically targets attention memory bottlenecks during auto-regressive decoding, enabling long-context generation on resource-constrained edge devices. Measuring perplexity over a longer text block ensures that tiny fluctuations in probability do not skew the evaluation."
    # ids = tok(text, return_tensors="pt").input_ids.to(device)

    # calculate_for_hf(ids)
    global QUANTIZE

    # 1. correctness gate: Triton dequant == PyTorch dequant, incl. no sealed group (5), no residual (64), both (101)
    QUANTIZE = True
    for seq in (5, 2 * K_GROUP, 3 * K_GROUP + 5):
        k_ok, v_ok = check_dequant_triton(seq=seq)
        print(f"dequant check seq={seq:3d}:  K {'OK' if k_ok else 'MISMATCH'}   V {'OK' if v_ok else 'MISMATCH'}")

    # 2. generation oracle: a long prompt, so prefill seals groups AND a seal fires during decode
    RUN_BENCHMARK = True     # set False to run only the checks above and this generation test

    ids = tok(text, return_tensors="pt").input_ids.to(device)
    P = ids.shape[1]
    first_seal = K_GROUP - P % K_GROUP          # decode steps until the residual reaches K_GROUP
    N_NEW = first_seal + 12                     # cross the first decode seal with room to spare
    print(f"\n--- generation oracle ---")
    print(f"P = {P}. Prefill seals {P // K_GROUP} groups. Residual after prefill: {P % K_GROUP} tokens. "
          f"First decode seal after {first_seal} decode steps. Generating {N_NEW} new tokens.")

    with torch.no_grad():
        hf = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, repetition_penalty=1.0,
                            temperature=1.0, top_p=1.0, top_k=0)[0, P:].tolist()
    nocache = generate_nocache(ids, max_new_tokens=N_NEW)

    RAW = True
    raw_ids = generate_cached(ids, N_NEW)
    RAW = False
    QUANTIZE, USE_TRITON = False, False
    fp32_ids = generate_cached(ids, N_NEW)
    QUANTIZE, USE_TRITON = True, False
    int4_torch_ids = generate_cached(ids, N_NEW)
    QUANTIZE, USE_TRITON = True, True
    int4_triton_ids = generate_cached(ids, N_NEW)
    ck, _, cv, _, k_res = cache[0]
    print(f"layer 0 after triton run: sealed K {ck.shape[2]}, residual {k_res.shape[2]}, V {cv.shape[2]}")

    print(f"raw fp32 vs HF:               {'MATCH' if raw_ids == hf else 'MISMATCH'}")
    print(f"raw fp32 vs nocache:          {'MATCH' if raw_ids == nocache else 'MISMATCH'}")
    print(f"fp32 bypass vs HF:            {'MATCH' if fp32_ids == hf else 'MISMATCH'}")
    print(f"fp32 bypass vs nocache:       {'MATCH' if fp32_ids == nocache else 'MISMATCH'}")
    print(f"int4 triton vs int4 pytorch:  {'MATCH' if int4_triton_ids == int4_torch_ids else 'MISMATCH'}")
    agree = sum(a == b for a, b in zip(int4_triton_ids, hf))
    print(f"int4 vs HF (info only, not expected to match): {agree}/{len(hf)} tokens agree")

    # full outputs, so a mismatch can be compared token by token
    runs = [("hf", hf), ("nocache", nocache), ("raw fp32", raw_ids), ("fp32 bypass", fp32_ids),
            ("int4 pytorch", int4_torch_ids), ("int4 triton", int4_triton_ids)]
    print("\n--- generated tokens ---")
    for name, out_ids in runs:
        print(f"{name:13s} ids : {out_ids}")
    print("\n--- generated text ---")
    for name, out_ids in runs:
        print(f"{name:13s} text: {tok.decode(out_ids)!r}")
    print("\n--- first differing position (None = identical) ---")
    for name, a, b in [("raw fp32 vs hf", raw_ids, hf), ("raw fp32 vs nocache", raw_ids, nocache),
                       ("fp32 bypass vs hf", fp32_ids, hf), ("fp32 bypass vs nocache", fp32_ids, nocache),
                       ("int4 triton vs int4 pytorch", int4_triton_ids, int4_torch_ids),
                       ("int4 triton vs hf", int4_triton_ids, hf)]:
        i = first_diff(a, b)
        detail = "" if i is None else f"   ({a[i] if i < len(a) else '-'} vs {b[i] if i < len(b) else '-'}, decode step {i})"
        print(f"{name:28s}: {i}{detail}")

    if RUN_BENCHMARK:
        # 3. dequant only, one layer, 536 tokens
        print("\n--- dequant only (one layer, 536 tokens) ---")
        for batch in [1, 4, 16, 64]:
            t_torch, t_triton = bench_dequant(batch)
            print(f"batch={batch:3d}  pytorch: {t_torch:7.3f} ms   triton: {t_triton:7.3f} ms   speedup: {t_torch/t_triton:5.2f}×")

        # 4. full decode step (24 layers), prompt 512
        print("\n--- full decode (tok/s) ---")
        for batch in [1, 4, 16, 64]:
            RAW = True                      # real fp32 baseline: plain cache, no quant plumbing
            fp = benchmark(batch)
            RAW = False
            QUANTIZE, USE_TRITON = True, False
            q_torch = benchmark(batch)
            QUANTIZE, USE_TRITON = True, True
            q_triton = benchmark(batch)
            print(f"batch={batch:3d}  fp32 raw: {fp:8.1f}   int4 pytorch: {q_torch:8.1f} ({q_torch/fp:.2f}×)   "
                  f"int4 triton: {q_triton:8.1f} ({q_triton/fp:.2f}×)")


    # QUANTIZE = False
    # print(f"Bypass (fp32) perplexity: {score_perplexity(ids):.4f}")

    # QUANTIZE = True
    # print(f"Int8 quantized perplexity: {score_perplexity(ids):.4f}")

    # standalone_test()

    # texts = [
    #     "Large language models require significant memory ...",
    #     "Quantization reduces this burden by compressing ...",
    #     "Key-Value cache quantization specifically targets ...",
    #     "Measuring perplexity over a longer text block ...",
    # ]
    # enc     = [tok(t, return_tensors="pt").input_ids for t in texts]
    # min_len = min(e.shape[1] for e in enc)                       # truncate to shortest
    # ids     = torch.cat([e[:, :min_len] for e in enc], dim=0).to(device)   # [B, min_len]

    # QUANTIZE = False
    # print(f"Bypass (fp32) perplexity: {score_perplexity(ids):.4f}")
    
    # QUANTIZE = True
    # print(f"Int8 quantized perplexity: {score_perplexity(ids):.4f}")



    
    # prompt = "The capital of France is"
    # ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    # global QUANTIZE
    # QUANTIZE = True
    # # get the exact input to layer-0 attention: embed -> input_layernorm
    # layers = 24
    # h = model.model.embed_tokens(ids)
    # for layer in range(layers):
    #     h, ck, cv, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
    #     cache_k_raw[layer] = ck.clone()
    #     cache[layer] = qunatize_kv_cache(ck, cv)

    # h = model.model.norm(h)                      # final norm
    # logits = model.lm_head(h)                    # [B, P, vocab]
    # next_id = logits[:, -1, :].argmax(-1)        # last position -> first new token
    # generated = [next_id.item()]

    # for step in range(19):
    #     h = model.model.embed_tokens(next_id).unsqueeze(1)
    #     for layer in range(layers):
    #         ck, k_scale, cv, v_scale, k_residual  = cache[layer]
    #         h, ck, cv, k_scale, v_scale, k_residual = my_layer_forward(
    #             layer, h, is_prefill=False, cache_k=ck, cache_v=cv, k_scale=k_scale, v_scale=v_scale, k_residual=k_residual
    #         )
    #         cache[layer] = (ck, k_scale, cv, v_scale, k_residual)
    #     h = model.model.norm(h)
    #     logits = model.lm_head(h)
    #     next_id = logits[:, -1, :].argmax(-1)
    #     generated.append(next_id.item())
    #     if next_id.item() == tok.eos_token_id:
    #         break

    # # YOUR output
    # mine = generated

    # # HF reference — same prompt, greedy, same length
    # # hf_out = model.generate(ids, max_new_tokens=20, do_sample=False)
    # # hf_ids = hf_out[0, ids.shape[1]:].tolist()   # strip the prompt, keep generated

    # hf_out = model.generate(
    #     ids,
    #     max_new_tokens=20,
    #     do_sample=False,
    #     repetition_penalty=1.0,
    #     temperature=1.0,
    #     top_p=1.0,
    #     top_k=0,
    # )
    # hf_ids = hf_out[0, ids.shape[1]:].tolist()

    # print("mine:", mine)
    # print("hf:  ", hf_ids)
    # print("MATCH" if mine == hf_ids else "MISMATCH")

    # nocache = generate_nocache(ids, max_new_tokens=20)

    # print("cached:   ", mine)
    # print("nocache:  ", nocache)
    # print("MATCH" if mine == nocache else "MISMATCH")