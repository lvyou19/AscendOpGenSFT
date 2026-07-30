"""Auto-generated benchmark file for ThreeInterpolateBackward.

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
    """使用 PyTorch 原生算子的参考实现（golden model）。

    ThreeInterpolateBackward: 反向传播算子。
    给定 grad_x (B, C, N), idx (B, N, 3), weight (B, N, 3)，
    计算 grad_y (B, C, M)，其中 M 由属性 m 指定。

    计算公式:
        grad_y[b, c, idx[b, n, k]] += weight[b, n, k] * grad_x[b, c, n]
        对所有 n in [0, N) 和 k in [0, 3)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_x, idx, weight, m):
        B, C, N = grad_x.shape
        grad_y = torch.zeros(B, C, m, dtype=grad_x.dtype, device=grad_x.device)
        for k in range(3):
            idx_k = idx[:, :, k].long().unsqueeze(1).expand(-1, C, -1)
            weight_k = weight[:, :, k].unsqueeze(1)
            contrib = grad_x * weight_k
            grad_y.scatter_add_(2, idx_k, contrib)
        return grad_y

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    B = int(param['B'])
    C = int(param['C'])
    N = int(param['N'])
    M = int(param['M'])
    grad_dtype = param['grad_dtype']
    idx_dtype = param['idx_dtype']
    grad_x = torch.randn(B, C, N, dtype=getattr(torch, grad_dtype), device=device)
    idx = torch.randint(0, M, (B, N, 3), dtype=getattr(torch, idx_dtype), device=device)
    weight = torch.randn(B, N, 3, dtype=getattr(torch, grad_dtype), device=device)
    m = M
    return (grad_x, idx, weight, m)

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
    json_path = os.path.join(os.path.dirname(__file__), "ThreeInterpolateBackward.json")
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
