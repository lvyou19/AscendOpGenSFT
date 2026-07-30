"""Auto-generated benchmark file for GridSampler2DGrad.

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
from typing import List
import torch
import torch.nn as nn

class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad, x, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        input_dtype = x.dtype
        x_float = x.detach().float().requires_grad_(True)
        grid_float = grid.detach().float().requires_grad_(True)
        grad_float = grad.float()
        output = torch.nn.functional.grid_sample(x_float, grid_float, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)
        output.backward(grad_float)
        dx = x_float.grad.to(input_dtype)
        dgrid = grid_float.grad.to(input_dtype)
        return (dx, dgrid)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    x_shape = eval(param.get('x_shape', '[1, 3, 4, 4]'))
    grid_shape = eval(param.get('grid_shape', '[1, 2, 2, 2]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    interpolation_mode = param.get('interpolation_mode', 'bilinear')
    padding_mode = param.get('padding_mode', 'zeros')
    align_corners = param.get('align_corners', 'False')
    align_corners = align_corners == 'True'
    x = torch.randn(x_shape, device=device, dtype=dtype)
    grid = torch.rand(grid_shape, device=device, dtype=dtype) * 2 - 1
    N = grid_shape[0]
    C = x_shape[1]
    outH = grid_shape[1]
    outW = grid_shape[2]
    grad_shape = [N, C, outH, outW]
    grad = torch.randn(grad_shape, device=device, dtype=dtype)
    return [grad, x, grid, interpolation_mode, padding_mode, align_corners]

def get_init_inputs_per_case(param, device=None):
    """返回空列表。"""
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
    json_path = os.path.join(os.path.dirname(__file__), "GridSampler2DGrad.json")
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
