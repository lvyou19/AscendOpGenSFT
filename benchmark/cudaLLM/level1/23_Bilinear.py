import os
import json
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        """Original __init__ parameters are supplied by _get_init_inputs()."""
        _init_vals = _get_init_inputs()
        super().__init__()
        self.bilinear = nn.Bilinear(_init_vals[0], _init_vals[1], _init_vals[2])

    def forward(self, x1, x2):
        return self.bilinear(x1, x2)
batch_size = 16
in1_features = 32
in2_features = 64
out_features = 10

def _get_init_inputs():
    return [in1_features, in2_features, out_features]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "23_Bilinear.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    def make_tensor(dtype_name, shape, rng=None):
        dtype = dtype_map[dtype_name]
        shape = tuple(shape)
        if dtype_name in ("int64", "int32", "int16", "int8", "uint8"):
            low, high = rng if rng is not None else [0, 9]
            return torch.randint(low, high + 1, shape, dtype=dtype)
        elif dtype_name == "bool":
            return (torch.rand(shape) > 0.5).to(dtype)
        elif dtype_name in ("complex64", "complex128"):
            return torch.randn(shape, dtype=dtype)
        else:
            return torch.randn(shape, dtype=dtype)

    for case in cases:
        inputs = case["inputs"]
        args = []
        for inp in inputs:
            if inp["type"] == "tensor":
                args.append(make_tensor(inp["dtype"], inp["shape"], inp.get("range")))
            elif inp["type"] == "tensor_list":
                args.append([
                    make_tensor(t["dtype"], t["shape"], t.get("range"))
                    for t in inp["tensors"]
                ])
            else:
                args.append(inp["value"])
        input_groups.append(args)
    return input_groups

def get_init_inputs():
    return []

