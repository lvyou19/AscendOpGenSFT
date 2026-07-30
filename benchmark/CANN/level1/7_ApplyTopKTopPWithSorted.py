"""Auto-generated benchmark file for ApplyTopKTopPWithSorted.

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
import torch.nn.functional as F

class Model(nn.Module):
    """
    实现ApplyTopKTopPWithSorted算子功能的模型（PyTorch标杆）。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, sorted_value: torch.Tensor, sorted_indices: torch.Tensor, p: Optional[torch.Tensor]=None, k: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        实现ApplyTopKTopPWithSorted算子功能。

        Args:
            sorted_value: 已排序的概率值张量 [batch, vocab]
            sorted_indices: 对应的索引张量 [batch, vocab]
            p: Top-P阈值 [batch] (可选)
            k: Top-K数量 [batch] (可选)

        Returns:
            输出张量，满足top-k/top-p条件的保留原值，其余位置为-inf
        """
        if not k.dim() == 0:
            kth_idx = sorted_value.size(1) - k.long()
            kth_value = sorted_value.gather(1, kth_idx.unsqueeze(dim=1))
            top_k_mask = sorted_value < kth_value
            sorted_value.masked_fill_(top_k_mask, -float('inf'))
        if not p.dim() == 0:
            softmax_res = sorted_value.to(torch.float32).softmax(dim=-1)
            cumsum_res = softmax_res.cumsum(dim=-1)
            top_p_mask = cumsum_res <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False
            sorted_value.masked_fill_(top_p_mask, -float('inf'))
        out = torch.empty_like(sorted_value).scatter_(dim=-1, index=sorted_indices.long(), src=sorted_value)
        return out

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    batch_size = int(param.get('batch_size', 2))
    vocab_size = int(param.get('vocab_size', 100))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    mode = param.get('mode', 'topk_topp')
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    sorted_value, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    sorted_value = sorted_value.to(dtype)
    sorted_indices = sorted_indices.to(torch.int32)
    p = None
    k = None
    if mode == 'topk_topp':
        p_value = float(param.get('p_value', 0.9))
        k_value = int(param.get('k_value', 10))
        p = torch.full((batch_size,), p_value, device=device, dtype=dtype)
        k = torch.full((batch_size,), k_value, device=device, dtype=torch.int32)
    elif mode == 'topk':
        k_value = int(param.get('k_value', 10))
        p = torch.empty(())
        k = torch.full((batch_size,), k_value, device=device, dtype=torch.int32)
    elif mode == 'topp':
        p_value = float(param.get('p_value', 0.9))
        p = torch.full((batch_size,), p_value, device=device, dtype=dtype)
        k = torch.empty(())
    return (sorted_value, sorted_indices, p, k)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for ApplyTopKTopPWithSorted.

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
    json_path = os.path.join(os.path.dirname(__file__), "ApplyTopKTopPWithSorted.json")
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
