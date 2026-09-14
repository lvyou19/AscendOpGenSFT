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
        g,
        A,
        g_bias,
        head_k_dim,
        beta,
        threshold,
        lower_bound,
        use_bias,
        use_lower_bound,
    ):
        torch.manual_seed(42)
        heads = A.numel()
        if g.shape[-1] != heads * head_k_dim:
            raise ValueError("the flattened gate dimension is inconsistent")
        original_shape = g.shape[:-1]
        gate_input = g.float().reshape(-1, heads, head_k_dim)
        if use_bias:
            gate_input = gate_input + g_bias.float().reshape(
                heads, head_k_dim
            )
        rate = torch.exp(A.float())[:, None]
        if use_lower_bound:
            output = lower_bound * torch.sigmoid(rate * gate_input)
        else:
            output = -rate * F.softplus(
                gate_input, beta=beta, threshold=threshold
            )
        return output.reshape(*original_shape, heads, head_k_dim)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, scale=0.8):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    ) * scale
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
        groups.append([
            _random_tensor(specs["g"], 42 + case_index * 3, 0.9),
            _random_tensor(specs["A"], 43 + case_index * 3, 0.4),
            _random_tensor(specs["g_bias"], 44 + case_index * 3, 0.3),
            specs["head_k_dim"]["value"],
            specs["beta"]["value"],
            specs["threshold"]["value"],
            specs["lower_bound"]["value"],
            specs["use_bias"]["value"],
            specs["use_lower_bound"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
