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
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        scale,
        use_gate_in_kernel,
        use_bias,
        use_initial_state,
        output_final_state,
        use_qk_l2norm,
        use_beta_sigmoid,
        allow_neg_eigval,
        lower_bound,
        use_lower_bound,
        state_v_first,
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

        if use_gate_in_kernel:
            gate_input = g.float()
            if use_bias:
                gate_input = gate_input + dt_bias.float().reshape(
                    value_heads, key_dim
                )
            rate = torch.exp(A_log.float())[:, None]
            if use_lower_bound:
                gate = lower_bound * torch.sigmoid(rate * gate_input)
            else:
                gate = -rate * F.softplus(gate_input)
        else:
            gate = g.float()

        beta_value = beta.float()
        if use_beta_sigmoid:
            beta_value = torch.sigmoid(beta_value)
            if allow_neg_eigval:
                beta_value = beta_value * 2.0
        if use_initial_state:
            state = initial_state.float().clone()
            if state_v_first:
                state = state.transpose(-2, -1).contiguous()
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
        for token in range(tokens):
            state = state * torch.exp(gate[:, token])[..., None]
            key_token = key[:, token]
            residual = v[:, token].float() - torch.einsum(
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
        if state_v_first:
            state = state.transpose(-2, -1).contiguous()
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


def _gate_tensor(spec, seed, raw_gate):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    if raw_gate:
        tensor = tensor * 0.8
    else:
        tensor = -F.softplus(tensor) * 0.08
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _beta_tensor(spec, seed, raw_beta):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    if not raw_beta:
        tensor = torch.sigmoid(tensor)
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_input_groups():
    torch.manual_seed(42)
    groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = {item["name"]: item for item in case["inputs"]}
        raw_gate = specs["use_gate_in_kernel"]["value"]
        raw_beta = specs["use_beta_sigmoid"]["value"]
        groups.append([
            _normalized_tensor(specs["q"], 42 + case_index * 8),
            _normalized_tensor(specs["k"], 43 + case_index * 8),
            _random_tensor(specs["v"], 44 + case_index * 8, 0.2),
            _gate_tensor(specs["g"], 45 + case_index * 8, raw_gate),
            _beta_tensor(specs["beta"], 46 + case_index * 8, raw_beta),
            _random_tensor(specs["A_log"], 47 + case_index * 8, 0.4),
            _random_tensor(specs["dt_bias"], 48 + case_index * 8, 0.3),
            _random_tensor(
                specs["initial_state"], 49 + case_index * 8, 0.04
            ),
            specs["scale"]["value"],
            raw_gate,
            specs["use_bias"]["value"],
            specs["use_initial_state"]["value"],
            specs["output_final_state"]["value"],
            specs["use_qk_l2norm"]["value"],
            raw_beta,
            specs["allow_neg_eigval"]["value"],
            specs["lower_bound"]["value"],
            specs["use_lower_bound"]["value"],
            specs["state_v_first"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
