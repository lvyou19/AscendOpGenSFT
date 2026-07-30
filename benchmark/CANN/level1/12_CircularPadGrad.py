"""Auto-generated benchmark file for CircularPadGrad.

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
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """CPU 参考：与 ops ST executor 一致，对 circular pad 做反向（grad 对 unpadded 输入）。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, padding):
        pad = tuple((int(p) for p in padding))
        nd = grad_output.dim()
        if len(pad) == 4:
            h = grad_output.shape[nd - 2] - pad[2] - pad[3]
            w = grad_output.shape[nd - 1] - pad[0] - pad[1]
            self_shape = tuple(grad_output.shape[:-2]) + (h, w)
        else:
            d = grad_output.shape[nd - 3] - pad[4] - pad[5]
            h = grad_output.shape[nd - 2] - pad[2] - pad[3]
            w = grad_output.shape[nd - 1] - pad[0] - pad[1]
            self_shape = tuple(grad_output.shape[:-3]) + (d, h, w)
        grad_input = torch.zeros(self_shape, dtype=grad_output.dtype, device=grad_output.device, requires_grad=True)
        out = F.pad(grad_input, pad, mode='circular')
        out.backward(grad_output)
        return grad_input.grad

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import ast
import torch

def _padded_shape(self_shape, padding):
    """与 CircularPad 正向一致：由 unpadded(self) 形状与 padding 得到 grad_output 形状。"""
    s = list(self_shape)
    if len(padding) == 4:
        s[-2] += padding[2] + padding[3]
        s[-1] += padding[0] + padding[1]
    else:
        s[-3] += padding[4] + padding[5]
        s[-2] += padding[2] + padding[3]
        s[-1] += padding[0] + padding[1]
    return tuple(s)

def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get('input_shape', '[1,1,4,4]'))
    shape = tuple((int(x) for x in shape))
    padding = [int(x) for x in ast.literal_eval(param.get('padding', '[1,1,1,1]'))]
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    grad_shape = _padded_shape(shape, padding)
    grad_output = torch.randn(grad_shape, device=device, dtype=dtype)
    return (grad_output, padding)

def get_init_inputs_per_case(param, device=None):
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
    json_path = os.path.join(os.path.dirname(__file__), "CircularPadGrad.json")
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
