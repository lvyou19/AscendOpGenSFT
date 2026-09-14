import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        raw_g,
        beta,
        A_log,
        g_bias,
        initial_state,
        scale,
        lower_bound,
        use_bias,
        use_lower_bound,
        use_initial_state,
        output_final_state,
        use_qk_l2norm,
    ):
        torch.manual_seed(42)
        batch, tokens, heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
        if value_heads % heads != 0:
            raise ValueError("value heads must be divisible by query heads")
        query = q.float()
        key = k.float()
        if use_qk_l2norm:
            query = F.normalize(query, p=2.0, dim=-1, eps=1e-6)
            key = F.normalize(key, p=2.0, dim=-1, eps=1e-6)
        query = query.repeat_interleave(value_heads // heads, dim=2)
        key = key.repeat_interleave(value_heads // heads, dim=2)

        gate_input = raw_g.float()
        if use_bias:
            gate_input = gate_input + g_bias.float().reshape(
                value_heads, key_dim
            )
        decay_rate = torch.exp(A_log.float())[:, None]
        if use_lower_bound:
            gate = lower_bound * torch.sigmoid(decay_rate * gate_input)
        else:
            gate = -decay_rate * F.softplus(gate_input)

        if use_initial_state:
            state = initial_state.float().clone()
        else:
            state = torch.zeros(
                batch,
                value_heads,
                key_dim,
                value_dim,
                device=q.device,
                dtype=torch.float32,
            )
        output = torch.empty_like(v)
        beta_value = beta.float()
        for token in range(tokens):
            state = state * torch.exp(gate[:, token].float())[..., None]
            key_token = key[:, token]
            value_token = v[:, token].float()
            residual = value_token - torch.einsum(
                "bhk,bhkv->bhv", key_token, state
            )
            residual = residual * beta_value[:, token, :, None]
            state = state + torch.einsum(
                "bhk,bhv->bhkv", key_token, residual
            )
            output[:, token] = (
                torch.einsum(
                    "bhk,bhkv->bhv", query[:, token] * scale, state
                ).to(v.dtype)
            )
        final_state = state if output_final_state else None
        return output, final_state


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, scale=0.15):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    ) * scale
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _normalized_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    tensor = F.normalize(tensor, p=2.0, dim=-1, eps=1e-6)
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _beta_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    logits = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    return torch.sigmoid(logits).to(
        dtype=_DTYPE_MAP[spec["dtype"]]
    ).npu()


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_input_groups():
    torch.manual_seed(42)
    groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = {item["name"]: item for item in case["inputs"]}
        groups.append([
            _normalized_tensor(specs["q"], 42 + case_index * 8),
            _normalized_tensor(specs["k"], 43 + case_index * 8),
            _random_tensor(specs["v"], 44 + case_index * 8, 0.2),
            _random_tensor(specs["raw_g"], 45 + case_index * 8, 0.8),
            _beta_tensor(specs["beta"], 46 + case_index * 8),
            _random_tensor(specs["A_log"], 47 + case_index * 8, 0.4),
            _random_tensor(specs["g_bias"], 48 + case_index * 8, 0.3),
            _random_tensor(
                specs["initial_state"], 49 + case_index * 8, 0.04
            ),
            specs["scale"]["value"],
            specs["lower_bound"]["value"],
            specs["use_bias"]["value"],
            specs["use_lower_bound"]["value"],
            specs["use_initial_state"]["value"],
            specs["output_final_state"]["value"],
            specs["use_qk_l2norm"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
