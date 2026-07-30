"""Auto-generated benchmark file for GroupedMatmulSliceKPerTokenDequant.

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
import math

class Model(nn.Module):
    """
    实现add算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        results = []
        m = a.shape[0]
        n = b.shape[1]
        offset_a = 0
        offset_b = 0
        a = a.flatten()
        b = b.flatten()
        start_idx = 0
        for ind, end_idx in enumerate(group_list):
            k = end_idx - start_idx
            if k > 0:
                size_a = m * k
                size_b = k * n
                group_a_flat = a[offset_a:offset_a + size_a]
                group_b_flat = b[offset_b:offset_b + size_b]
                group_a = group_a_flat.view(k, m).transpose(0, 1)
                group_b = group_b_flat.view(k, n)
                result = torch.matmul(group_a.to(torch.int32), group_b.to(torch.int32))
                result = result.to(torch.float32) * scale[ind].unsqueeze(0).to(torch.float32) * per_token_scale[ind].unsqueeze(1).to(torch.float32)
                results.append(result.flatten())
                offset_a += size_a
                offset_b += size_b
                start_idx = end_idx
            else:
                results.append(torch.zeros([m, n], device=per_token_scale.device, dtype=per_token_scale.dtype).flatten())
        return torch.cat(results)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    m = param.get('m')
    k = param.get('k')
    n = param.get('n')
    groupCount = param.get('groupCount')
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    a = torch.randint(-16, 17, (m, k), device=device, dtype=torch.int8)
    b = torch.randint(-16, 17, (k, n), device=device, dtype=torch.int8)
    scale = torch.rand([groupCount, n], device=device, dtype=dtype)
    per_token_scale = torch.rand([groupCount, m], device=device, dtype=dtype)
    group_list = torch.randint(0, k + 1, (groupCount,), dtype=torch.int64, device=device)
    group_list, _ = torch.sort(group_list)
    return (a, b, scale, per_token_scale, group_list)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
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
    json_path = os.path.join(os.path.dirname(__file__), "GroupedMatmulSliceKPerTokenDequant.json")
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
