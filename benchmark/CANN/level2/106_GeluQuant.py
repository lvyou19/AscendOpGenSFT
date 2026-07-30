"""Auto-generated benchmark file for GeluQuant.

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
from typing import List, Tuple
import torch
import torch.nn as nn

def gelu_compute_erf(input_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a GELU approximation using a polynomial approximation of the erf function.
    This implementation mirrors the provided numpy version for float32 precision.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed GELU approximation.
    """
    input_x = input_x.to(torch.float32)
    input_x_clamped_min = torch.max(input_x, torch.tensor(-13.25, dtype=torch.float32))
    x1 = torch.min(input_x_clamped_min, torch.tensor(5.75, dtype=torch.float32))
    x_pow = x1 * x1
    a1 = torch.tensor(-3.512339572e-09, dtype=torch.float32)
    a2 = torch.tensor(2.64526617e-07, dtype=torch.float32)
    a3 = torch.tensor(-7.929488134e-06, dtype=torch.float32)
    a4 = torch.tensor(0.000110612384, dtype=torch.float32)
    a5 = torch.tensor(6.518995814e-05, dtype=torch.float32)
    a6 = torch.tensor(-0.07266616915, dtype=torch.float32)
    a7 = torch.tensor(-1.595769883, dtype=torch.float32)
    y = x_pow * a1 + a2
    y = y * x_pow + a3
    y = y * x_pow + a4
    y = y * x_pow + a5
    y = y * x_pow + a6
    y = y * x_pow + a7
    y = y * x1
    y = torch.exp(y) + 1.0
    res = input_x / y
    return res

def tanh_parameter_compute(input_x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute the x + 0.044715*x^3 term for the tanh GELU approximation.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed value.
    """
    input_x = input_x.to(torch.float32)
    y = input_x * input_x
    y = y * input_x
    y = y * torch.tensor(0.044715, dtype=torch.float32)
    result = input_x + y
    return result

def gelu_compute_tanh(input_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a GELU approximation using the tanh formula:
    gelu(x) = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715*x^3)))
    This implementation mirrors the provided numpy version for float32 precision.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed GELU approximation.
    """
    input_x = input_x.to(torch.float32)
    tanh_parameter = tanh_parameter_compute(input_x)
    mul_0 = tanh_parameter * torch.tensor(-1.5957691, dtype=torch.float32)
    temp = torch.exp(mul_0) + 1.0
    res = input_x / temp
    return res

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, approximate: str='tanh', quant_mode: str='static') -> List[torch.Tensor]:
        x_f = x.float()
        scale = scale.float()
        offset = offset.float()
        if approximate == 'none':
            gelu = gelu_compute_erf(x)
        else:
            gelu = gelu_compute_tanh(x)
        if scale.dim() == 1:
            scale = scale.view(*[1] * (x.dim() - 1), -1)
        if offset is not None and offset.dim() == 1:
            offset = offset.view(*[1] * (x.dim() - 1), -1)
        if quant_mode == 'static':
            quant = torch.round(gelu * scale + offset).clamp(-128, 127).to(torch.int8)
            return [quant]
        else:
            mul_res = gelu * scale
            max_abs = torch.amax(mul_res.abs(), dim=-1, keepdim=True)
            tmp_out_scale = 127.0 / (max_abs + 1e-06)
            out_scale = 1.0 / tmp_out_scale
            tmp_out_scale = tmp_out_scale.expand_as(mul_res)
            quant = torch.round(mul_res * tmp_out_scale).clamp(-128, 127).to(torch.int8)
            return [quant, out_scale]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np
from typing import List, Tuple

def get_inputs(param, device=None):
    """
    返回：
        x         : Tensor         输入张量（不再是 List）
        scale     : Tensor(float)  量化 scale
        offset    : Tensor(float)  量化 offset
        quantMode : str            "static" or "dynamic"
    """
    input_shape = eval(param.get('input_shape', '[8, 2048]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    scale_value = float(param.get('scale_value', 1.0))
    offset_value = float(param.get('offset_value', 0.0))
    approximate = param.get('approximate', 'tanh')
    quant_mode = param.get('quant_mode', 'static')
    x = torch.rand(input_shape, device=device, dtype=dtype) * 2 - 1
    scale_tensor = torch.tensor(scale_value, dtype=dtype, device=device)
    offset_tensor = torch.tensor(offset_value, dtype=dtype, device=device)
    return (x, scale_tensor, offset_tensor, approximate, quant_mode)

def get_init_inputs_per_case(param, device=None) -> List:
    return []


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
    json_path = os.path.join(os.path.dirname(__file__), "GeluQuant.json")
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
