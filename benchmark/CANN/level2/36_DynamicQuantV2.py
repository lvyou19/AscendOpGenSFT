"""Auto-generated benchmark file for DynamicQuantV2.

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
CPU golden 对齐 answer/0/op_kernel 同目录下 dynamic_quant.h::ComputAsymmetric（int8）：
NPU 上 offset 输出缓冲区非空 → InitParams 将 isAsymmetrical 置 true，走非对称路径。
scale = max((max-min)/255, eps)；offset = 127 - max/scale；逐行 x/scale+offset 后 CAST_RINT → int8。
输出 scale/offset 形状与 infershape 默认 pertoken 一致（x 去掉最后一维）。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

def golden_dynamic_quant_v2_int8_pertoken(x: torch.Tensor, smooth: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_f = x.detach().float()
    if smooth is not None:
        x_f = x_f * smooth.detach().float()
    last = x_f.shape[-1]
    flat = x_f.reshape(-1, last)
    max_v = flat.amax(dim=1, keepdim=True)
    min_v = flat.amin(dim=1, keepdim=True)
    eps = 1e-12
    scale = torch.maximum((max_v - min_v) / 255.0, torch.full_like(max_v, eps))
    offset = 127.0 - max_v / scale
    t = flat / scale + offset
    q_np = np.rint(t.cpu().numpy())
    q = torch.from_numpy(q_np).to(device=x.device).clamp(-128, 127).to(torch.int8)
    y = q.reshape_as(x)
    scale_out = scale.squeeze(-1).reshape(x.shape[:-1]).to(torch.float32)
    offset_out = offset.squeeze(-1).reshape(x.shape[:-1]).to(torch.float32)
    return (y, scale_out, offset_out)

class Model(nn.Module):

    def __init__(self, dst_type: int):
        super().__init__()
        self.dst_type = int(dst_type)

    def forward(self, x: torch.Tensor, smooth: Optional[torch.Tensor]=None, group_index: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        _ = self.dst_type
        _ = group_index
        y, s, o = golden_dynamic_quant_v2_int8_pertoken(x, smooth)
        return [y, s, o]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')

def get_init_inputs_per_case(param, device=None):
    dst_type = int(param.get('dst_type', 2))
    return (dst_type,)

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
    if _parse_bool(param.get('has_group_index', False)):
        group_index = torch.zeros(shape[0], device=device, dtype=torch.int32)
    else:
        group_index = None
    return (x, smooth, group_index)


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
    json_path = os.path.join(os.path.dirname(__file__), "DynamicQuantV2.json")
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
