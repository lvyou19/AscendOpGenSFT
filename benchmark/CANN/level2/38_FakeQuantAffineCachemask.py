"""Auto-generated benchmark file for FakeQuantAffineCachemask.

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
"""
CPU golden：torch.fake_quantize_per_channel_affine（axis=0，与 tiling 中 headNum=dim0 一致）。
mask：round(x/scale)+zero_point（float 路径）落在 [quant_min, quant_max] 内（与核心里比较逻辑近似）。
"""
from typing import List, Tuple
import torch
import torch.nn as nn

def _mask_in_quant_range(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, qmin: int, qmax: int) -> torch.Tensor:
    """按通道广播后，判断 round(x/s + zp) 是否在整型量化网格范围内。"""
    if axis != 0:
        raise ValueError('golden 仅对齐本算子 tiling：headNum = x.shape[0]，axis 须为 0')
    c = x.shape[0]
    sc = scale.reshape(c, *[1] * (x.ndim - 1)).float()
    zp = zero_point.reshape(c, *[1] * (x.ndim - 1)).float()
    t = torch.round(x.float() / sc + zp)
    return (t >= qmin) & (t <= qmax)

def golden(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, quant_min: int, quant_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
    zp = zero_point.to(torch.int32)
    y = torch.fake_quantize_per_channel_affine(x, scale.float(), zp, axis, quant_min, quant_max)
    m = _mask_in_quant_range(x, scale, zero_point, axis, quant_min, quant_max)
    return (y, m)

class Model(nn.Module):

    def __init__(self, axis: int, quant_min: int, quant_max: int):
        super().__init__()
        self.axis = int(axis)
        self.quant_min = int(quant_min)
        self.quant_max = int(quant_max)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        y, m = golden(x, scale, zero_point, self.axis, self.quant_min, self.quant_max)
        return [y, m]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_init_inputs_per_case(param, device=None):
    axis = int(param.get('axis', 0))
    qmin = int(param.get('quant_min', 0))
    qmax = int(param.get('quant_max', 255))
    return (axis, qmin, qmax)

def get_inputs(param, device=None):
    x_shape = eval(param.get('x_shape', '[8,16]'), {'__builtins__': {}})
    x_dtype_str = str(param.get('x_dtype', 'float16')).lower()
    x_dtype = getattr(torch, x_dtype_str) if hasattr(torch, x_dtype_str) else torch.float16
    scale_shape = eval(param.get('scale_shape', '[8]'), {'__builtins__': {}})
    if len(scale_shape) != 1:
        raise ValueError('scale_shape must be 1D [N] to match tiling headNum on dim0')
    n = scale_shape[0]
    if x_shape[0] != n:
        raise ValueError('x_shape[0] must equal scale_shape[0] (per-channel on dim0)')
    qmin = int(param.get('quant_min', 0))
    qmax = int(param.get('quant_max', 255))
    x = (torch.rand(x_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(x_dtype)
    scale = (torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.5 + 0.05).to(x_dtype)
    zero_point = torch.randint(qmin, qmax + 1, scale_shape, device=device, dtype=torch.int32)
    return (x, scale, zero_point)


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
    json_path = os.path.join(os.path.dirname(__file__), "FakeQuantAffineCachemask.json")
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
