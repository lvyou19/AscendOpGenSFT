"""Auto-generated benchmark file for GroupedBiasAddGrad.

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
from torch import Tensor
from typing import Optional
from typing import List

class Model(nn.Module):

    def __init__(self, grad_y: Tensor, group_idx: Tensor):
        super(Model, self).__init__()
        self.grad_y = grad_y
        self.group_idx = group_idx

    def forward(self, grad_y: Tensor, group_idx: Tensor) -> Tensor:
        """
        CPU 实现 grouped_bias_add_grad，与 golden 一致
        Args:
            grad_y: shape [C, H] 或 [G, C, H]，dtype: float32/float16/bfloat16
            group_idx: shape [G]，int32/int64，可选

        Returns:
            out: shape [G, H]，dtype 与 grad_y 一致
        """
        if group_idx is None:
            grad_y = grad_y.float()
            out = torch.sum(grad_y, dim=1)
            return out.to(grad_y.dtype)
        else:
            group_idx = group_idx.int()
            grad_y = grad_y.float()
            out = torch.tensor([]).to(grad_y.device)
            for i in range(len(group_idx)):
                start = group_idx[i - 1] if i > 0 else 0
                end = group_idx[i]
                chunk = grad_y[start:end, :]
                tmp = torch.sum(chunk, dim=0, keepdim=True)
                out = torch.cat((out, tmp), dim=0)
            return out.to(grad_y.dtype)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np
import random

def get_inputs(param, device=None):
    """
    生成输入张量，用于 forward 方法。

    Args:
        param (dict): 包含 'input_shape', 'dtype', 'with_group_idx', 'num_groups' 的字典
        device (torch.device): 设备，如 'cpu' 或 'npu'

    Returns:
        tuple: (grad_y, group_idx) 或 (grad_y, None)
    """
    input_shape = eval(param.get('input_shape', '[1, 1]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    with_group_idx = param.get('with_group_idx', 'true') == 1
    if with_group_idx:
        grad_y = torch.rand(input_shape, device=device, dtype=dtype)
        if dtype == torch.bfloat16:
            grad_y = grad_y.float()
        num_groups = int(param.get('num_groups', 4))
        indices = torch.randint(1, input_shape[0] // 2, (num_groups - 1,), dtype=torch.int32, device=device)
        indices = torch.sort(indices).values
        group_idx = torch.cat([indices, torch.tensor([input_shape[0]], dtype=torch.int32, device=device)])
        return (grad_y, group_idx)
    else:
        G = int(input_shape[0])
        C = int(input_shape[1])
        H = int(input_shape[2])
        grad_y = torch.rand((G, C, H), device=device, dtype=dtype)
        if dtype == torch.bfloat16:
            grad_y = grad_y.float()
        C_total = G * C
        group_idx = torch.tensor([C_total], dtype=torch.int32, device=device)
        return (grad_y, None)

def get_init_inputs_per_case(param, device=None):
    """
    提取初始化参数，用于模型构造。

    Args:
        param (dict): 来自 DataFrame 的一行

    Returns:
        tuple: (grad_y, group_idx)
    """
    input_shape = eval(param.get('input_shape', '[1, 1]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    with_group_idx = param.get('with_group_idx', 'true') == 1
    if with_group_idx:
        grad_y = torch.rand(input_shape, device=device, dtype=dtype)
        if dtype == torch.bfloat16:
            grad_y = grad_y.float()
        num_groups = int(param.get('num_groups', 4))
        indices = torch.randint(1, input_shape[0] // 2, (num_groups - 1,), dtype=torch.int32, device=device)
        indices = torch.sort(indices).values
        group_idx = torch.cat([indices, torch.tensor([input_shape[0]], dtype=torch.int32, device=device)])
        return (grad_y, group_idx)
    else:
        G = int(input_shape[0])
        C = int(input_shape[1])
        H = int(input_shape[2])
        grad_y = torch.rand((G, C, H), device=device, dtype=dtype)
        if dtype == torch.bfloat16:
            grad_y = grad_y.float()
        C_total = G * C
        group_idx = torch.tensor([C_total], dtype=torch.int32, device=device)
        return (grad_y, None)


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
    json_path = os.path.join(os.path.dirname(__file__), "GroupedBiasAddGrad.json")
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
