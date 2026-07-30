"""Auto-generated benchmark file for DynamicQuant.

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
CPU golden 与 AICore 对称动态量化路径一致（见 answer/0/op_kernel/dynamic_quant.h::Compute）：
Cast(x)->fp32；可选乘 smooth；Abs 后沿最后一维求 max；scale_fp = 127/max_abs；
输出 scale 张量为 max_abs/127（与内核 scaleLocal.SetValue(i, 1 / scale) 一致）；
量化：tempFp32 * (127/max_abs) 后 CAST_RINT 再落到 int8。
Golden 用 float32 + numpy.rint 逼近 CAST_RINT。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

def golden_dynamic_quant_int8(x: torch.Tensor, smooth: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    x_f = x.detach().float()
    if smooth is not None:
        x_f = x_f * smooth.detach().float()
    last = x_f.shape[-1]
    flat = x_f.reshape(-1, last)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    inv = 127.0 / max_abs
    qf = flat * inv
    q_np = np.rint(qf.cpu().numpy())
    q = torch.from_numpy(q_np).to(device=x.device, dtype=torch.float32).clamp(-128, 127).to(torch.int8)
    y = q.reshape_as(x)
    scale = (max_abs / 127.0).squeeze(-1).reshape(x.shape[:-1]).to(torch.float32)
    return (y, scale)

class Model(nn.Module):

    def forward(self, x: torch.Tensor, smooth: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y, scale = golden_dynamic_quant_int8(x, smooth)
        return [y, scale]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[8, 32]'), {'__builtins__': {}})
    dtype_str = param.get('x_dtype', 'float16')
    x_dtype = getattr(torch, dtype_str)
    x = (torch.rand(shape, device='cpu', dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)
    if _parse_bool(param.get('has_smooth', False)):
        last = shape[-1]
        smooth = (torch.rand((last,), device='cpu', dtype=torch.float32) * 0.6 + 0.2).to(device=device, dtype=x_dtype)
    else:
        smooth = None
    return (x, smooth)

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
    json_path = os.path.join(os.path.dirname(__file__), "DynamicQuant.json")
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
