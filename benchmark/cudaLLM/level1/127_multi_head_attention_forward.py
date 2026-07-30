import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        """Original __init__ parameters are supplied by _get_init_inputs()."""
        _init_vals = _get_init_inputs()
        super().__init__()
        self.embed_dim = _init_vals[0]
        self.kdim = None if None is not None else _init_vals[0]
        self.vdim = None if None is not None else _init_vals[0]
        self.num_heads = _init_vals[1]
        self.qkv_same_embed_dim = self.kdim == _init_vals[0] and self.vdim == _init_vals[0]
        self.in_proj_weight = nn.Parameter(torch.randn(3 * _init_vals[0], _init_vals[0]))
        self.in_proj_bias = nn.Parameter(torch.randn(3 * _init_vals[0]))
        self.out_proj = nn.Linear(_init_vals[0], _init_vals[0])
        self.attn_drop_p = 0.1
        self.dropout_p = 0.1
        self.scaling = _init_vals[0] ** (-0.5)

    def forward(self, query, key, value):
        return F.multi_head_attention_forward(query=query, key=key, value=value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads, in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=self.dropout_p, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, training=self.training, key_padding_mask=None, need_weights=False, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=True)
batch_size = 16
seq_len = 50
embed_dim = 256
num_heads = 8

def _get_init_inputs():
    return [embed_dim, num_heads]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "127_multi_head_attention_forward.json")
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

