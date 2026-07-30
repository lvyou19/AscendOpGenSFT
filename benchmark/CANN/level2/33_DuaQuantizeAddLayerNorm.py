"""Auto-generated benchmark file for DuaQuantizeAddLayerNorm.

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
from typing import List, Optional
import torch
import torch.nn as nn
AXIS_MUL_MODE = -65535

def _layer_norm_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, axis: int=-1) -> torch.Tensor:
    dim = axis if axis >= 0 else x.dim() + axis
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True, unbiased=False) + eps
    rstd = torch.rsqrt(var)
    return (x - mean) * rstd * gamma + beta

def _quantize_per_tensor(x: torch.Tensor, scale: torch.Tensor, zero_point: Optional[torch.Tensor], div_mode: bool) -> torch.Tensor:
    """与 kernel 一致：div_mode 时 y=round(x/scale+zp)，mul 时 y=round(x*scale+zp)。"""
    if div_mode:
        q = x / scale
    else:
        q = x * scale
    if zero_point is not None:
        q = q + zero_point.float()
    return torch.round(q).clamp(-128, 127).to(torch.int8)

class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, bias: torch.Tensor, scales1: torch.Tensor, scales2: torch.Tensor, zero_points1: Optional[torch.Tensor]=None, zero_points2: Optional[torch.Tensor]=None, dtype_attr: int=0, axis: int=-1, epsilon: float=1e-05, additional_output: bool=False):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.bias = bias
        self.scales1 = scales1
        self.scales2 = scales2
        self.zero_points1 = zero_points1
        self.zero_points2 = zero_points2
        self.dtype_attr = dtype_attr
        self.axis = axis
        self.epsilon = epsilon
        self.additional_output = additional_output

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        dtype = x1.dtype
        x1_f = x1.float()
        x2_f = x2.float()
        x_sum = x1_f + x2_f + self.bias.float()
        x_norm = _layer_norm_ref(x_sum, self.gamma.float(), self.beta.float(), self.epsilon, self.axis)
        div_mode = self.axis != AXIS_MUL_MODE
        zp1 = self.zero_points1.float() if self.zero_points1 is not None else None
        zp2 = self.zero_points2.float() if self.zero_points2 is not None else None
        y1 = _quantize_per_tensor(x_norm, self.scales1.float(), zp1, div_mode)
        y2 = _quantize_per_tensor(x_norm, self.scales2.float(), zp2, div_mode)
        x_out = torch.round(x_sum).to(dtype)
        return [y1, y2, x_out]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
from typing import Optional

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[2, 256]'))
    dtype_str = param.get('dtype', 'bfloat16')
    dtype = getattr(torch, dtype_str)
    x1 = torch.rand(shape, device=device, dtype=dtype)
    x2 = torch.rand(shape, device=device, dtype=dtype)
    return (x1, x2)

def get_init_inputs_per_case(param, device=None):
    shape = eval(param.get('normalized_shape', '[256]'))
    dtype_str = param.get('dtype', 'bfloat16')
    dtype = getattr(torch, dtype_str)
    epsilon = float(param.get('epsilon', 1e-05))
    axis = int(param.get('axis', -1))
    dtype_attr = int(param.get('dtype_attr', 0))
    additional_output = bool(param.get('additional_output', False))
    gamma = torch.rand(shape, device=device, dtype=dtype)
    beta = torch.rand(shape, device=device, dtype=dtype)
    bias = torch.rand(shape, device=device, dtype=dtype)
    scales1 = torch.rand(shape, device=device, dtype=dtype) * 0.01 + 0.01
    scales2 = torch.rand(shape, device=device, dtype=dtype) * 0.01 + 0.01
    zero_points1 = torch.zeros(shape, device=device, dtype=dtype)
    zero_points2 = torch.zeros(shape, device=device, dtype=dtype)
    return [gamma, beta, bias, scales1, scales2, zero_points1, zero_points2, dtype_attr, axis, epsilon, additional_output]


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
    json_path = os.path.join(os.path.dirname(__file__), "DuaQuantizeAddLayerNorm.json")
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
