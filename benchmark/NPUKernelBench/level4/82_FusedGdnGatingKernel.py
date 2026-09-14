import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A_log, a, b, dt_bias, beta, threshold):
        torch.manual_seed(42)
        x = a.float() + dt_bias.float()
        gate = -torch.exp(A_log.float()) * F.softplus(
            x, beta=beta, threshold=threshold
        )
        beta_output = torch.sigmoid(b.float()).to(dtype=b.dtype)
        return gate.unsqueeze(0), beta_output.unsqueeze(0)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, low=-2.0, high=2.0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
    tensor = tensor * (high - low) + low
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
            _random_tensor(specs["A_log"], 42 + case_index * 4, -1.5, 0.75),
            _random_tensor(specs["a"], 43 + case_index * 4, -4.0, 4.0),
            _random_tensor(specs["b"], 44 + case_index * 4, -5.0, 5.0),
            _random_tensor(specs["dt_bias"], 45 + case_index * 4, -2.0, 2.0),
            specs["beta"]["value"],
            specs["threshold"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
