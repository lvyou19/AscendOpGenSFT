"""Auto-generated benchmark file for InplaceIndexAddWithSorted.

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

class Model(nn.Module):
    """
    实现InplaceIndexAddWithSorted算子功能的模型(torch标杆)。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_indices: torch.Tensor, pos: torch.Tensor, axis: int, alpha: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        实现InplaceIndexAddWithSorted算子功能。

        Args:
            var: 待更新的张量
            value: 更新值张量
            sorted_indices: 已排序的索引张量
            pos: 位置索引张量
            axis: 操作的维度
            alpha: 可选的缩放因子

        Returns:
            更新后的var张量
        """
        result = var.clone()
        result_dtype = result.dtype
        if result_dtype == torch.bfloat16:
            result = result.float()
        alpha_value = alpha.item() if alpha is not None else 1.0
        for i in range(len(sorted_indices)):
            idx = sorted_indices[i].item()
            p = pos[i].item()
            if axis == 0:
                result[idx] = result[idx] + alpha_value * value[p]
            else:
                raise NotImplementedError('Only axis=0 is supported')
        if result_dtype == torch.bfloat16:
            result = result.to(torch.bfloat16)
        return result

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    var_shape = eval(param.get('var_shape', '[10, 8]'))
    value_shape = eval(param.get('value_shape', '[5, 8]'))
    indices_num = int(param.get('indices_num', 5))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    axis = int(param.get('axis', 0))
    enable_alpha = int(param.get('enable_alpha', 0))
    if dtype in [torch.int32, torch.int16]:
        var = torch.randint(-10, 10, var_shape, device=device, dtype=dtype)
    else:
        var = torch.randn(var_shape, device=device, dtype=dtype)
    if dtype in [torch.int32, torch.int16]:
        value = torch.randint(-10, 10, value_shape, device=device, dtype=dtype)
    else:
        value = torch.randn(value_shape, device=device, dtype=dtype)
    max_idx = var_shape[axis]
    sorted_indices = torch.randint(0, max_idx, (indices_num,), device=device, dtype=torch.int32)
    sorted_indices, _ = torch.sort(sorted_indices)
    max_pos = value_shape[axis]
    pos = torch.randint(0, max_pos, (indices_num,), device=device, dtype=torch.int32)
    if enable_alpha == 1:
        alpha = torch.tensor([2.0], device=device, dtype=torch.float32)
    else:
        alpha = None
    return (var, value, sorted_indices, pos, axis, alpha)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for InplaceIndexAddWithSorted.

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
    json_path = os.path.join(os.path.dirname(__file__), "InplaceIndexAddWithSorted.json")
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
