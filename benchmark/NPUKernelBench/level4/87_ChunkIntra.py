import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k, v, g, beta, chunk_size, use_gate):
        torch.manual_seed(42)
        batch, tokens, heads, key_dim = k.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
        if value_heads % heads != 0:
            raise ValueError("value heads must be divisible by key heads")
        key = k.repeat_interleave(value_heads // heads, dim=2).float()
        w = torch.empty(
            batch,
            tokens,
            value_heads,
            key_dim,
            device=k.device,
            dtype=k.dtype,
        )
        u = torch.empty_like(v)
        inverse_output = torch.zeros(
            batch,
            tokens,
            value_heads,
            chunk_size,
            device=k.device,
            dtype=k.dtype,
        )

        for start in range(0, tokens, chunk_size):
            end = min(tokens, start + chunk_size)
            length = end - start
            key_chunk = key[:, start:end].permute(0, 2, 1, 3)
            beta_chunk = beta[:, start:end].float().permute(0, 2, 1)
            key_products = torch.matmul(
                key_chunk, key_chunk.transpose(-1, -2)
            )
            strict_lower = torch.tril(
                torch.ones(
                    length,
                    length,
                    device=k.device,
                    dtype=torch.float32,
                ),
                diagonal=-1,
            )
            if use_gate:
                gate = g[:, start:end].float().permute(0, 2, 1)
                decay = torch.exp2(
                    gate[..., :, None] - gate[..., None, :]
                )
            else:
                decay = 1.0
            lower = (
                key_products
                * decay
                * beta_chunk[..., :, None]
                * strict_lower
            )
            identity = torch.eye(
                length, device=k.device, dtype=torch.float32
            )
            inverse = identity.expand(
                batch, value_heads, length, length
            ).clone()
            for row in range(1, length):
                coefficients = -torch.matmul(
                    lower[..., row : row + 1, :row],
                    inverse[..., :row, :row],
                ).squeeze(-2)
                inverse[..., row, :row] = coefficients

            inverse_output[:, start:end, :, :length] = (
                inverse.permute(0, 2, 1, 3).to(k.dtype)
            )
            value_beta = (
                v[:, start:end].float()
                * beta[:, start:end].float()[..., None]
            ).permute(0, 2, 1, 3)
            key_beta = (
                key[:, start:end]
                * beta[:, start:end].float()[..., None]
            )
            if use_gate:
                key_beta = key_beta * torch.exp2(
                    g[:, start:end].float()
                )[..., None]
            key_beta = key_beta.permute(0, 2, 1, 3)
            u_chunk = torch.matmul(inverse, value_beta)
            w_chunk = torch.matmul(inverse, key_beta)
            u[:, start:end] = u_chunk.permute(0, 2, 1, 3).to(v.dtype)
            w[:, start:end] = w_chunk.permute(0, 2, 1, 3).to(k.dtype)
        return w, u, inverse_output


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, scale=0.1):
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
    ) * 0.05
    output = torch.empty_like(increments)
    tokens = increments.shape[1]
    for start in range(0, tokens, chunk_size):
        end = min(tokens, start + chunk_size)
        output[:, start:end] = increments[:, start:end].cumsum(dim=1)
    return output.npu()


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
        chunk_size = specs["chunk_size"]["value"]
        groups.append([
            _random_tensor(specs["k"], 42 + case_index * 4, 0.08),
            _random_tensor(specs["v"], 43 + case_index * 4, 0.14),
            _cumulative_gate(specs["g"], 44 + case_index * 4, chunk_size),
            _beta_tensor(specs["beta"], 45 + case_index * 4),
            chunk_size,
            specs["use_gate"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
