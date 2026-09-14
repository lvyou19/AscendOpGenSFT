import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        h,
        g,
        g_gamma,
        scale,
        chunk_size,
        gate_mode,
        state_v_first,
    ):
        torch.manual_seed(42)
        batch, tokens, heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
        if value_heads % heads != 0:
            raise ValueError("value heads must be divisible by query heads")
        group_size = value_heads // heads
        query = q.repeat_interleave(group_size, dim=2).float()
        key = k.repeat_interleave(group_size, dim=2).float()
        output = torch.empty_like(v)
        num_chunks = (tokens + chunk_size - 1) // chunk_size

        for chunk in range(num_chunks):
            start = chunk * chunk_size
            end = min(tokens, start + chunk_size)
            length = end - start
            state = h[:, chunk].float()
            if state_v_first:
                state = state.transpose(-2, -1)
            query_chunk = query[:, start:end]
            key_chunk = key[:, start:end]
            value_chunk = v[:, start:end].float()
            inter = torch.einsum(
                "bthk,bhkv->bthv", query_chunk, state
            )
            scores = torch.matmul(
                query_chunk.permute(0, 2, 1, 3),
                key_chunk.permute(0, 2, 1, 3).transpose(-1, -2),
            )

            if gate_mode == 1:
                gate = g[:, start:end].float().permute(0, 2, 1)
                inter = inter * torch.exp2(
                    gate.permute(0, 2, 1)[..., None]
                )
                scores = scores * torch.exp2(
                    gate[..., :, None] - gate[..., None, :]
                )
            elif gate_mode == 2:
                positions = torch.arange(
                    1,
                    length + 1,
                    device=q.device,
                    dtype=torch.float32,
                )
                gate = g_gamma.float()[None, :, None] * positions[None, None, :]
                inter = inter * torch.exp2(
                    gate.permute(0, 2, 1)[..., None]
                )
                scores = scores * torch.exp2(
                    gate[..., :, None] - gate[..., None, :]
                )
            elif gate_mode != 0:
                raise ValueError("gate_mode must be 0, 1, or 2")

            causal = torch.tril(
                torch.ones(
                    length,
                    length,
                    device=q.device,
                    dtype=torch.float32,
                )
            )
            scores = scores * causal
            intra = torch.matmul(
                scores, value_chunk.permute(0, 2, 1, 3)
            )
            result = (
                inter.permute(0, 2, 1, 3) + intra
            ) * scale
            output[:, start:end] = result.permute(0, 2, 1, 3).to(v.dtype)
        return output


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, scale=0.12):
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
    ) * 0.06
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
            _random_tensor(specs["q"], 42 + case_index * 6, 0.1),
            _random_tensor(specs["k"], 43 + case_index * 6, 0.1),
            _random_tensor(specs["v"], 44 + case_index * 6, 0.15),
            _random_tensor(specs["h"], 45 + case_index * 6, 0.04),
            _cumulative_gate(specs["g"], 46 + case_index * 6, chunk_size),
            _random_tensor(specs["g_gamma"], 47 + case_index * 6, 0.03),
            specs["scale"]["value"],
            chunk_size,
            specs["gate_mode"]["value"],
            specs["state_v_first"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
