"""Auto-generated benchmark file for MoeTokenUnpermuteGrad.

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
from typing import Optional, List

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens: Tensor, unpermuted_tokens_grad: Tensor, sorted_indices: Tensor, probs: Optional[Tensor], padded_mode: bool) -> List[Tensor]:
        if probs is not None:
            topk = probs.size(1)
            H = unpermuted_tokens_grad.size(0)
            D = unpermuted_tokens_grad.size(1)
            N = sorted_indices.size(0)
            permuted_tokens_float = permuted_tokens.float()
            unpermuted_tokens = torch.zeros(N, D, dtype=torch.float32)
            unpermuted_tokens = permuted_tokens_float.index_select(0, sorted_indices)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            probs_grad = torch.sum(unpermuted_tokens_grad.unsqueeze(1) * unpermuted_tokens, -1, keepdim=False)
            permuted_tokens_grad = (unpermuted_tokens_grad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, D)
            permuted_tokens_grad1 = torch.zeros(N, D, dtype=torch.float32)
            permuted_tokens_grad1.index_add_(0, sorted_indices, permuted_tokens_grad.to(torch.float32))
            return (permuted_tokens_grad1.to(permuted_tokens.dtype), probs_grad.to(probs.dtype))
        else:
            topk = 1
            D = unpermuted_tokens_grad.size(1)
            N = sorted_indices.size(0)
            permuted_tokens_float = permuted_tokens.float()
            unpermuted_tokens = torch.zeros(N, D, dtype=torch.float32)
            unpermuted_tokens = permuted_tokens_float.index_select(0, sorted_indices)
            if permuted_tokens.size(-1) == 0:
                unpermuted_tokens = unpermuted_tokens.reshape(sorted_indices.shape[0] // topk, topk, permuted_tokens.size(-1))
            else:
                unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            permuted_tokens_grad1 = torch.zeros(N, D, dtype=torch.float32)
            permuted_tokens_grad1.index_copy_(0, sorted_indices.to(torch.int64), unpermuted_tokens_grad.to(torch.float32))
            return permuted_tokens_grad1.to(permuted_tokens.dtype)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np
import random

def get_inputs(param, device=None):
    """
    生成输入张量，用于 forward 方法。

    Args:
        param (dict): 包含 'tokens_num', 'topk', 'hidden_size', 'dtype', 'with_probs' 的字典
        device (torch.device): 设备，如 'cpu' 或 'npu'

    Returns:
        tuple: (permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs)
    """
    tokens_num = int(param.get('tokens_num', 4))
    topk = int(param.get('topk', 2))
    hidden_size = int(param.get('hidden_size', 16))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    with_probs = bool(param.get('with_probs', 1))
    if not with_probs:
        topk = 1
    padded_mode = bool(param.get('padded_mode', 0))
    permuted_shape = (tokens_num * topk, hidden_size)
    permuted_tokens = torch.rand(permuted_shape, device=device, dtype=dtype)
    unpermuted_shape = (tokens_num, hidden_size)
    unpermuted_tokens_grad = torch.rand(unpermuted_shape, device=device, dtype=dtype)
    total_tokens = tokens_num * topk
    sorted_indices_after = torch.randint(low=0, high=total_tokens, size=(tokens_num, topk)).to(torch.int32).to(device)

    def mPermute(tokens, indices, num_out_tokens=None, padded_mode=False, with_probs=True):
        if indices.dim() == 1 or not with_probs:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        oringin_sorted_indices = sorted_indices
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        sorted_indices = sorted_indices
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return (permuted_tokens, sorted_indices, oringin_sorted_indices)
    permuted_tokens, sorted_indices_tmp, oringin_sorted_indices = mPermute(permuted_tokens, sorted_indices_after)
    if with_probs:
        probs_shape = (tokens_num, topk)
        probs = torch.rand(probs_shape, device=device, dtype=dtype)
    else:
        probs = None
    return (permuted_tokens, unpermuted_tokens_grad, oringin_sorted_indices.to(torch.int32), probs, padded_mode)

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenUnpermuteGrad.json")
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
