import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        k,
        w,
        u,
        g,
        gk,
        initial_state,
        chunk_size,
        gate_mode,
        use_initial_state,
        output_final_state,
        state_v_first,
    ):
        torch.manual_seed(42)
        batch, tokens, heads, key_dim = k.shape
        value_heads, value_dim = u.shape[2], u.shape[3]
        if value_heads % heads != 0:
            raise ValueError("value heads must be divisible by key heads")
        key = k.repeat_interleave(value_heads // heads, dim=2)
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
                device=k.device,
                dtype=torch.float32,
            )

        num_chunks = (tokens + chunk_size - 1) // chunk_size
        states = torch.empty(
            batch,
            num_chunks,
            value_heads,
            key_dim,
            value_dim,
            device=k.device,
            dtype=k.dtype,
        )
        v_new = torch.empty_like(u)
        for chunk in range(num_chunks):
            start = chunk * chunk_size
            end = min(tokens, start + chunk_size)
            states[:, chunk] = state.to(dtype=k.dtype)

            residual = u[:, start:end].float() - torch.einsum(
                "bthk,bhkv->bthv", w[:, start:end].float(),
                state.to(dtype=w.dtype).float()
            )
            v_new[:, start:end] = residual.to(dtype=u.dtype)
            update_value = residual

            if gate_mode == 1:
                chunk_gate = g[:, start:end].float()
                final_gate = chunk_gate[:, -1]
                update_value = update_value * torch.exp2(
                    final_gate[:, None, :, None]
                    - chunk_gate[:, :, :, None]
                )
                state = state * torch.exp2(
                    final_gate[:, :, None, None]
                )
            elif gate_mode == 2:
                final_gate = gk[:, end - 1].float()
                state = state * torch.exp2(final_gate)[..., None]
            elif gate_mode != 0:
                raise ValueError("gate_mode must be 0, 1, or 2")
            state = state + torch.einsum(
                "bthk,bthv->bhkv",
                key[:, start:end].float(),
                update_value.to(dtype=k.dtype).float(),
            )

        if state_v_first:
            states = states.transpose(-2, -1).contiguous()
            final_state = state.transpose(-2, -1).contiguous()
        else:
            final_state = state
        if not output_final_state:
            final_state = None
        return states, v_new, final_state


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


def _cumulative_gate(spec, seed, chunk_size):
    generator = torch.Generator()
    generator.manual_seed(seed)
    increments = -torch.rand(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    ) * 0.08
    output = torch.empty_like(increments)
    tokens = increments.shape[1]
    for start in range(0, tokens, chunk_size):
        end = min(tokens, start + chunk_size)
        output[:, start:end] = increments[:, start:end].cumsum(dim=1)
    return output.npu()


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_input_groups():
    torch.manual_seed(42)
    groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = {item["name"]: item for item in case["inputs"]}
        chunk_size = specs["chunk_size"]["value"]
        groups.append([
            _random_tensor(specs["k"], 42 + case_index * 6, 0.08),
            _random_tensor(specs["w"], 43 + case_index * 6, 0.06),
            _random_tensor(specs["u"], 44 + case_index * 6, 0.18),
            _cumulative_gate(specs["g"], 45 + case_index * 6, chunk_size),
            _cumulative_gate(specs["gk"], 46 + case_index * 6, chunk_size),
            _random_tensor(
                specs["initial_state"], 47 + case_index * 6, 0.04
            ),
            chunk_size,
            specs["gate_mode"]["value"],
            specs["use_initial_state"]["value"],
            specs["output_final_state"]["value"],
            specs["state_v_first"]["value"],
        ])
    return groups


def get_init_inputs():
    return []