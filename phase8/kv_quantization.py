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

def project(hidden):
    B, S, _ = hidden.shape
    q = (hidden @ Wq.T + bq).view(B, S, N_Q, HDIM).transpose(1, 2)
    k = (hidden @ Wk.T + bk).view(B, S, N_KV, HDIM).transpose(1, 2)
    v = (hidden @ Wv.T + bv).view(B, S, N_KV, HDIM).transpose(1, 2)
    return q, k, v

def my_layer_forward(layer, h, is_prefill=True, cache_k=None, cache_v=None, k_scale=None, v_scale=None):
    lyr = model.model.layers[layer]
    with torch.no_grad():
        residual = h
        h_norm = lyr.input_layernorm(h)
        if is_prefill:
            attn_out, ck, cv = prefill(h_norm,layer)
        else:
            attn_out, ck, cv, k_scale, v_scale = decode(layer, h_norm, cache_k, cache_v, k_scale, v_scale)

        h = residual + attn_out

        residual = h
        h_norm = lyr.post_attention_layernorm(h)
        h = residual + lyr.mlp(h_norm)
    return h, ck, cv, k_scale, v_scale


def decode_project(layer, hidden):
    attn = model.model.layers[layer].self_attn
    B, S, _ = hidden.shape
    q = (hidden @ attn.q_proj.weight.T + attn.q_proj.bias).view(B, S, N_Q, HDIM).transpose(1, 2)
    k = (hidden @ attn.k_proj.weight.T + attn.k_proj.bias).view(B, S, N_KV, HDIM).transpose(1, 2)
    v = (hidden @ attn.v_proj.weight.T + attn.v_proj.bias).view(B, S, N_KV, HDIM).transpose(1, 2)
    return q, k, v


def Wo_of(layer):
    return model.model.layers[layer].self_attn.o_proj.weight.T


def decode(layer, hidden, cache_k, cache_v, k_scale, v_scale):
    B, S, _ = hidden.shape
    q, k, v = decode_project(layer, hidden)

    pos = cache_k.shape[2]
    cos, sin = rope_cos_sin(torch.tensor([pos], device=device))
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    k_new_scale = torch.amax(torch.abs(k), dim=(0, 1, 3), keepdim=True) / 127
    v_new_scale = torch.amax(torch.abs(v), dim=(0, 1, 3), keepdim=True) / 127

    k_q = torch.round(k / k_new_scale).to(torch.int8)
    # k_q = k / k_new_scale
    k_scale = torch.cat([k_scale, k_new_scale], dim=2)
    v_q = torch.round(v / v_new_scale).to(torch.int8)
    # v_q = v / v_new_scale
    v_scale = torch.cat([v_scale, v_new_scale], dim=2)

    cache_k = torch.cat([cache_k, k_q], dim=2)
    cache_v = torch.cat([cache_v, v_q], dim=2)

    cache_k_, cache_v_ = dequnatize_kv_cache((cache_k, k_scale, cache_v, v_scale))

    # print("k_scale", k_scale.shape, "cache_k", cache_k.shape)
    # print("v_scale", v_scale.shape, "cache_v", cache_v.shape)
    kf, vf = repeat_kv(cache_k_), repeat_kv(cache_v_)

    scores = (q @ kf.transpose(-2, -1)) * SCALE
    attn_w = scores.softmax(dim=-1)
    out = attn_w @ vf
    out = out.transpose(1, 2).reshape(B, S, N_Q * HDIM) @ Wo_of(layer)

    return out, cache_k, cache_v, k_scale, v_scale


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
            h, _, _, _, _ = my_layer_forward(layer, h, is_prefill=True)
        h  = model.model.norm(h)
        logits = model.lm_head(h)
        next_id = logits[:, -1, :].argmax(-1)
        generated.append(next_id.item())
        seq = torch.cat([seq, next_id[:, None]], dim=1)
        if next_id.item() in (151645, 151643):
            break
    return generated

def qunatize_kv_cache(k, v):
    layer_maxes = torch.amax(torch.abs(k), dim=(0, 1, 3), keepdim=True)
    layer_scales = layer_maxes / 127
    quantized_layer_k = torch.round(k / layer_scales).to(torch.int8)
    # quantized_layer_k = k / layer_scales


    per_token_max = torch.amax(torch.abs(v), dim=(0,1, 3), keepdim=True)
    per_token_scale = per_token_max / 127
    quantized_v = torch.round(v / per_token_scale).to(torch.int8)
    # quantized_v = v / per_token_scale
    return quantized_layer_k, layer_scales, quantized_v, per_token_scale

def dequnatize_kv_cache(cache_val, dtype=torch.float16):
    k_quant, k_scales, v_quant, v_scales = cache_val

    k_float = k_quant.to(dtype)
    v_float = v_quant.to(dtype)

    k_dequnatized = k_float * k_scales

    v_dequnatized = v_float * v_scales

    return k_dequnatized, v_dequnatized


if __name__ == "__main__":
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    print(model.generation_config)
 
    # get the exact input to layer-0 attention: embed -> input_layernorm
    layers = 24
    h = model.model.embed_tokens(ids)
    for layer in range(layers):
        h, ck, cv, _, _ = my_layer_forward(layer, h, is_prefill=True)
        cache[layer] = qunatize_kv_cache(ck, cv)

    h = model.model.norm(h)                      # final norm
    logits = model.lm_head(h)                    # [B, P, vocab]
    next_id = logits[:, -1, :].argmax(-1)        # last position -> first new token
    generated = [next_id.item()]

    for step in range(19):
        h = model.model.embed_tokens(next_id).unsqueeze(1)
        for layer in range(layers):
            ck, k_scale, cv, v_scale  = cache[layer]
            h, ck, cv, k_scale, v_scale = my_layer_forward(layer, h, is_prefill=False, cache_k=ck, cache_v=cv, k_scale=k_scale, v_scale=v_scale)
            cache[layer] = (ck, k_scale, cv, v_scale)
        h = model.model.norm(h)
        logits = model.lm_head(h)
        next_id = logits[:, -1, :].argmax(-1)
        generated.append(next_id.item())
        if next_id.item() == tok.eos_token_id:
            break

    # YOUR output
    mine = generated

    # HF reference — same prompt, greedy, same length
    # hf_out = model.generate(ids, max_new_tokens=20, do_sample=False)
    # hf_ids = hf_out[0, ids.shape[1]:].tolist()   # strip the prompt, keep generated

    hf_out = model.generate(
        ids,
        max_new_tokens=20,
        do_sample=False,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )
    hf_ids = hf_out[0, ids.shape[1]:].tolist()

    print("mine:", mine)
    print("hf:  ", hf_ids)
    print("MATCH" if mine == hf_ids else "MISMATCH")

    nocache = generate_nocache(ids, max_new_tokens=20)

    print("cached:   ", mine)
    print("nocache:  ", nocache)
    print("MATCH" if mine == nocache else "MISMATCH")