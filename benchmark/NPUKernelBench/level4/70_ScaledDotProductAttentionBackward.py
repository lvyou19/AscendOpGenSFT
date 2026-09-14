import torch
import math
import json
import os


import torch.nn as nn


class Model(nn.Module):
    """
    Scaled Dot-Product Attention Backward.

    Computes gradients w.r.t. query, key, and value using the logsumexp
    from forward pass for numerical stability.
    Supports optional causal masking.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, grad_output, query, key, value, attn_output, logsumexp, is_causal=False):
        B, H, S_q, D = query.shape
        S_k = key.shape[2]
        scale = 1.0 / math.sqrt(D)
        dtype = query.dtype
        query_f = query.float()
        key_f = key.float()
        value_f = value.float()
        grad_output_f = grad_output.float()

        attn_weight = torch.matmul(query_f, key_f.transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.triu(torch.ones(S_q, S_k, device=query.device, dtype=torch.bool), diagonal=S_k - S_q + 1)
            attn_weight.masked_fill_(causal_mask, float('-inf'))

        attn_probs = torch.exp(attn_weight - logsumexp.unsqueeze(-1).float())
        attn_probs_in = attn_probs if dtype == torch.float32 else attn_probs.to(dtype).float()

        grad_value = torch.matmul(attn_probs_in.transpose(-2, -1), grad_output_f)
        grad_attn_probs = torch.matmul(grad_output_f, value_f.transpose(-2, -1))
        D_term = (grad_output_f * attn_output.float()).sum(dim=-1, keepdim=True)
        grad_attn_weight = attn_probs * (grad_attn_probs - D_term)
        grad_attn_weight_in = grad_attn_weight if dtype == torch.float32 else grad_attn_weight.to(dtype).float()

        grad_query = torch.matmul(grad_attn_weight_in, key_f) * scale
        grad_key = torch.matmul(grad_attn_weight_in.transpose(-2, -1), query_f) * scale

        return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "70_ScaledDotProductAttentionBackward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype_, zero_mean=False):
        mu = 0.0 if zero_mean else float(torch.empty(1).uniform_(0.5, 1.2).item())
        sigma = float(torch.empty(1).uniform_(0.5, 1.5).item())
        if dtype_ is torch.bfloat16:
            return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype_)
        return torch.normal(mu, sigma, shape, dtype=dtype_)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

        grad_info = next(inp for inp in inputs if inp["name"] == "grad_output")
        q_info = next(inp for inp in inputs if inp["name"] == "query")
        k_info = next(inp for inp in inputs if inp["name"] == "key")
        v_info = next(inp for inp in inputs if inp["name"] == "value")
        attn_info = next(inp for inp in inputs if inp["name"] == "attn_output")
        lse_info = next(inp for inp in inputs if inp["name"] == "logsumexp")
        dtype = dtype_map[q_info["dtype"]]

        grad_output = random_tensor(grad_info["shape"], dtype) * 0.2
        query = random_tensor(q_info["shape"], dtype)
        key = random_tensor(k_info["shape"], dtype, zero_mean=True)
        value = random_tensor(v_info["shape"], dtype) * 0.2
        is_causal = next(inp for inp in inputs if inp["name"] == "is_causal")["value"]
        with torch.no_grad():
            q_f = query.float()
            k_f = key.float()
            v_f = value.float()
            B, H, S_q, D = q_f.shape
            S_k = k_f.shape[2]
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(S_q, S_k, device=q_f.device, dtype=torch.bool),
                    diagonal=S_k - S_q + 1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            logsumexp = torch.logsumexp(scores, dim=-1)
            attn_probs = torch.exp(scores - logsumexp.unsqueeze(-1))
            attn_output = torch.matmul(attn_probs, v_f).to(dtype)

        input_groups.append([grad_output, query, key, value, attn_output, logsumexp, is_causal])
    return input_groups


def get_init_inputs():
    return []
