"""Auto-generated benchmark file for QuantizedBatchNorm.

Format mirrors /home/l00899543/Ascend_evaluation/AscendOpGenAgent/benchmarks/NPUKernelBench/level1
- get_input_groups(): list of [args...] per case
- get_init_inputs():  list of [init_args...] per case (same length, [] when none)
The Model class is the source's torch reference (NPU-only imports stripped).
prepare_inputs logic is embedded so each case is generated with the same
semantics as the original validation harness; the JSON file holds one CSV row
per line as the parameter dict.
"""

import sys as _sys
import types as _types

# Stub NPU / framework deps so import-time `import torch_npu` etc. succeeds
# even when this file is run on a host without those libraries installed.
for _n in (
    "torch_npu", "torch_npu.contrib", "torch_npu.contrib.transfer_to_npu",
    "kernel_gen_ops", "framework", "framework.utils", "framework.kernel_gen_config",
):
    if _n not in _sys.modules:
        _m = _types.ModuleType(_n); _m.__path__ = []
        _sys.modules[_n] = _m

import torch
import torch.nn as nn
import json
import os

# Patch torch.randn/rand to tolerate integer dtypes on CPU (some upstream
# prepare_inputs.py call torch.randn(shape, dtype=torch.int*) which only
# works on NPU).
_INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}
_orig_randn, _orig_rand = torch.randn, torch.rand


def _safe_make(_orig, _lo, _hi, *args, **kwargs):
    dtype = kwargs.get("dtype")
    if dtype in _INT_DTYPES:
        size = args
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            size = tuple(args[0])
        new_kwargs = dict(kwargs)
        if dtype in (torch.uint8, torch.bool):
            _lo, _hi = 0, (2 if dtype == torch.bool else 100)
        return torch.randint(_lo, _hi, tuple(int(x) for x in size), **new_kwargs)
    return _orig(*args, **kwargs)


torch.randn = lambda *a, **kw: _safe_make(_orig_randn, -100, 100, *a, **kw)
torch.rand = lambda *a, **kw: _safe_make(_orig_rand, 0, 100, *a, **kw)


# ---- Model (cleaned from source module.py) ----
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_features: int, eps: float):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, input_scale: torch.Tensor, input_zero_point: torch.Tensor, output_scale: torch.Tensor, output_zero_point: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_fp32 = (x.float() - input_zero_point.item()) * input_scale.item()
        normalized = (x_fp32 - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) / torch.sqrt(var.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.eps)
        if weight is not None:
            normalized = normalized * weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if bias is not None:
            normalized = normalized + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_fp32 = normalized / output_scale.item() + output_zero_point.item()
        y = y_fp32.round().clamp(-128, 127).to(x.dtype)
        return y

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the QuantizedBatchNorm operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[1, 1, 1, 1]'))
    num_features = param.get('num_features', 1)
    dtype_str = param.get('dtype', 'int8')
    if dtype_str == 'int8':
        input_dtype = torch.int8
        quant_min, quant_max = (-128, 127)
    elif dtype_str == 'uint8':
        input_dtype = torch.uint8
        quant_min, quant_max = (0, 255)
    else:
        input_dtype = torch.int32
        quant_min, quant_max = (-2147483648, 2147483647)
    x = torch.randint(quant_min, quant_max + 1, input_shape, device=device, dtype=input_dtype)
    mean = torch.rand(num_features, device=device, dtype=torch.float32)
    var = torch.rand(num_features, device=device, dtype=torch.float32) + 0.001
    input_scale = torch.tensor([0.1], device=device, dtype=torch.float32).contiguous()
    input_zero_point = torch.tensor([0], device=device, dtype=torch.int32).contiguous()
    output_scale = torch.tensor([0.1], device=device, dtype=torch.float32).contiguous()
    output_zero_point = torch.tensor([0], device=device, dtype=torch.int32).contiguous()
    affine = param.get('affine', True)
    weight = None
    bias = None
    if affine:
        weight = torch.rand(num_features, device=device, dtype=torch.float32)
        bias = torch.rand(num_features, device=device, dtype=torch.float32)
    return (x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters (num_features, eps) for the model.
    """
    num_features = param.get('num_features', 1)
    eps = float(param.get('epsilon', 1e-05))
    return [num_features, eps]


def _coerce(v):
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "":
        return s
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lstrip("-+").isdigit():
        try:
            return int(s)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        pass
    return s


def _load_cases():
    json_path = os.path.join(os.path.dirname(__file__), "QuantizedBatchNorm.json")
    with open(json_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _try_get_inputs(case):
    coerced = {k: _coerce(v) for k, v in case.items()}
    last_err = None
    for param in (coerced, case):
        try:
            return get_inputs(param, device=None), param
        except Exception as e:
            last_err = e
    raise last_err


def get_input_groups():
    groups = []
    for case in _load_cases():
        args, _ = _try_get_inputs(case)
        groups.append(list(args) if isinstance(args, (list, tuple)) else [args])
    return groups


def get_init_inputs():
    out = []
    for case in _load_cases():
        coerced = {k: _coerce(v) for k, v in case.items()}
        try:
            init = get_init_inputs_per_case(coerced, device=None)
        except Exception:
            try:
                init = get_init_inputs_per_case(case, device=None)
            except Exception:
                init = []
        out.append(list(init) if isinstance(init, (list, tuple)) else [init])
    return out
