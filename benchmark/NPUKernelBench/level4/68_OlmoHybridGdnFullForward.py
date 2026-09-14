import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import json
import os


def rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    dims = tuple(range(-len(normalized_shape), 0))
    variance = x.pow(2).mean(dims, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight
    return out


def silu(x):
    return x * torch.sigmoid(x)


class Model(nn.Module):
    """
    OLMo Hybrid Gated Delta Net (GDN) Full Forward.

    Semantics aligned with the spec reference (vLLM
    OlmoHybridGatedDeltaNetAttention._full_forward):
      (1) Fused QKVG projection (w_qkvg): [Q(key_dim) | K(key_dim) | V(value_dim) | Gate(value_dim)]
      (2) Separate B and A projections (w_b, w_a)
      (3) Causal depthwise 1D convolution + SiLU activation
      (4) GQA expansion of Q and K from num_heads_k to num_heads_v
      (5) GDN gating (fp32): g = -exp(A_log) * softplus(a + dt_bias),
          beta = sigmoid(b) * 2; A_log/dt_bias clamped to [-3, 3]
      (6) Gated delta rule recurrence (fp32):
          S[t] = decay * S[t-1] + outer(v[t] - S·k_norm, k_norm * beta[t])
      (7) Per-head gated RMSNorm (fp32, eps=1e-5) with SiLU(gate):
          out = rms_norm(x, w_o_norm) * gate * sigmoid(gate), gate clamped to [-10, 10]
      (8) Output projection in fp32, cast back to input dtype

    The three input projections (1)(2) are precomputed in get_input_groups()
    and passed in as projected_qkvg / proj_b / proj_a, so forward() starts
    from the conv1d step.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, hidden_states, output_buffer, projected_qkvg, proj_b, proj_a, w_conv1d,
                dt_bias, A_log, w_o_norm, w_o_proj, num_heads_k, num_heads_v,
                head_k_dim, head_v_dim, conv_kernel_size):
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Clamp decay parameters to avoid fp16 overflow/NaN in exp/softplus
        A_log = A_log.clamp(-3.0, 3.0)
        dt_bias = dt_bias.clamp(-3.0, 3.0)

        key_dim = num_heads_k * head_k_dim
        value_dim = num_heads_v * head_v_dim

        # (1)(2) projections precomputed by get_input_groups
        mixed_qkv = projected_qkvg[:, :key_dim * 2 + value_dim]
        gate = projected_qkvg[:, key_dim * 2 + value_dim:]
        b = proj_b
        a = proj_a

        # (3) Causal depthwise conv1d + SiLU
        conv_dim = key_dim * 2 + value_dim
        mixed_qkv_t = mixed_qkv.unsqueeze(0).transpose(1, 2)
        conv_w = w_conv1d.view(conv_dim, 1, conv_kernel_size)
        pad_size = conv_kernel_size - 1
        mixed_qkv_padded = F.pad(mixed_qkv_t, (pad_size, 0))
        mixed_qkv_conv = F.conv1d(mixed_qkv_padded, conv_w, groups=conv_dim)
        mixed_qkv_conv = mixed_qkv_conv.squeeze(0).transpose(0, 1)
        mixed_qkv_conv = silu(mixed_qkv_conv)

        query = mixed_qkv_conv[:, :key_dim].view(num_tokens, num_heads_k, head_k_dim)
        key = mixed_qkv_conv[:, key_dim:2 * key_dim].view(num_tokens, num_heads_k, head_k_dim)
        value = mixed_qkv_conv[:, 2 * key_dim:].view(num_tokens, num_heads_v, head_v_dim)

        # (4) GQA expansion of Q and K
        if num_heads_v > num_heads_k:
            expand_ratio = num_heads_v // num_heads_k
            query = query.unsqueeze(2).expand(-1, -1, expand_ratio, -1).reshape(num_tokens, num_heads_v, head_k_dim)
            key = key.unsqueeze(2).expand(-1, -1, expand_ratio, -1).reshape(num_tokens, num_heads_v, head_k_dim)

        # (5) GDN gate (fp32)
        g = -A_log.float().exp().unsqueeze(0) * F.softplus(a.float() + dt_bias.unsqueeze(0))
        beta = torch.sigmoid(b.float()) * 2.0

        q_f = query.float()
        k_f = key.float()
        v_f = value.float()

        # (6) Gated delta rule recurrence (fp32)
        core_attn_out = torch.zeros(num_tokens, num_heads_v, head_v_dim, dtype=torch.float32, device=device)
        S = torch.zeros(num_heads_v, head_v_dim, head_k_dim, dtype=torch.float32, device=device)
        scale = 1.0 / math.sqrt(head_k_dim)

        for t in range(num_tokens):
            decay = torch.exp(g[t]).unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[t].unsqueeze(-1)
            S = decay * S

            k_t = k_f[t]
            q_t = q_f[t]
            v_t = v_f[t]

            k_norm = k_t / (torch.norm(k_t, dim=-1, keepdim=True) + 1e-6)
            q_norm = q_t / (torch.norm(q_t, dim=-1, keepdim=True) + 1e-6) * scale

            Sk = (k_norm.unsqueeze(1) * S).sum(dim=-1)
            residual = v_t - Sk
            S = S + beta_t.unsqueeze(-1) * residual.unsqueeze(-1) * k_norm.unsqueeze(1)
            core_attn_out[t] = (q_norm.unsqueeze(1) * S).sum(dim=-1)

        core_attn_out = core_attn_out.to(dtype)

        # (7) Per-head gated RMSNorm with SiLU(gate)
        gate = gate.view(num_tokens, num_heads_v, head_v_dim).clamp(-10.0, 10.0)
        core_attn_out_flat = core_attn_out.reshape(-1, head_v_dim)
        gate_flat = gate.reshape(-1, head_v_dim)
        normed = rms_norm(core_attn_out_flat.float(), (head_v_dim,), weight=w_o_norm)
        core_attn_out_normed = normed * gate_flat.float() * torch.sigmoid(gate_flat.float())

        # (8) Output projection (fp32 -> dtype)
        core_attn_out_reshaped = core_attn_out_normed.view(num_tokens, value_dim)
        output = F.linear(core_attn_out_reshaped, w_o_proj.float()).to(dtype)
        output_buffer[:num_tokens] = output
        return output_buffer


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "68_OlmoHybridGdnFullForward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype_):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype_ is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype_)
            return torch.normal(mu, sigma, shape, dtype=dtype_)
        else:
            return torch.empty(shape, dtype=dtype_).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

        def tensor_input(name):
            info = next(inp for inp in inputs if inp["name"] == name)
            return random_tensor(info["shape"], dtype_map[info["dtype"]])

        def attr_input(name):
            return next(inp for inp in inputs if inp["name"] == name)["value"]

        hs_info = next(inp for inp in inputs if inp["name"] == "hidden_states")
        ob_info = next(inp for inp in inputs if inp["name"] == "output_buffer")
        dtype = dtype_map[hs_info["dtype"]]
        hidden_states = tensor_input("hidden_states")
        output_buffer = torch.empty(ob_info["shape"], dtype=dtype)

        w_qkvg = tensor_input("w_qkvg")
        w_b = tensor_input("w_b")
        w_a = tensor_input("w_a")
        with torch.no_grad():
            projected_qkvg = F.linear(hidden_states, w_qkvg)
            proj_b = F.linear(hidden_states, w_b)
            proj_a = F.linear(hidden_states, w_a)

        input_groups.append([
            hidden_states,
            output_buffer,
            projected_qkvg,
            proj_b,
            proj_a,
            tensor_input("w_conv1d"),
            tensor_input("dt_bias"),
            tensor_input("A_log"),
            tensor_input("w_o_norm"),
            tensor_input("w_o_proj"),
            attr_input("num_heads_k"),
            attr_input("num_heads_v"),
            attr_input("head_k_dim"),
            attr_input("head_v_dim"),
            attr_input("conv_kernel_size"),
        ])
    return input_groups


def get_init_inputs():
    return []