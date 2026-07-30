"""Auto-generated benchmark file for MoeTokenUnpermuteWithEpGrad.

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
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

class Model(nn.Module):
    """PyTorch 原生算子参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, unpermuted_tokens_grad: torch.Tensor, sorted_indices: torch.Tensor, permuted_tokens: Optional[torch.Tensor], probs: Optional[torch.Tensor], padded_mode: bool, restore_shape: List[int], range_vals: List[int], topk_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_num = unpermuted_tokens_grad.shape[0]
        hidden_size = unpermuted_tokens_grad.shape[1]
        total_indices = sorted_indices.shape[0]
        start = range_vals[0] if range_vals[0] >= 0 else 0
        end = range_vals[1] if range_vals[1] >= 0 else total_indices
        permuted_tokens_grad = torch.zeros(total_indices, hidden_size, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        has_probs = probs is not None and probs.numel() > 0
        if not has_probs:
            for i in range(total_indices):
                idx = sorted_indices[i].item()
                if start <= idx < end:
                    token_idx = i // topk_num
                    permuted_tokens_grad[idx - start] = unpermuted_tokens_grad[token_idx]
            probs_grad = torch.zeros(tokens_num, topk_num, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        else:
            topk = probs.shape[1] if probs.dim() == 2 else topk_num
            probs_grad = torch.zeros_like(probs)
            for i in range(total_indices):
                idx = sorted_indices[i].item()
                token_idx = i // topk
                k_idx = i % topk
                if start <= idx < end:
                    permuted_tokens_grad[idx - start] = unpermuted_tokens_grad[token_idx] * probs[token_idx, k_idx]
            if permuted_tokens is not None and permuted_tokens.numel() > 0:
                for i in range(total_indices):
                    idx = sorted_indices[i].item()
                    token_idx = i // topk
                    k_idx = i % topk
                    if start <= idx < end:
                        prod = permuted_tokens[idx - start].float() * unpermuted_tokens_grad[token_idx].float()
                        probs_grad[token_idx, k_idx] = prod.sum().to(probs.dtype)
        return (permuted_tokens_grad, probs_grad)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np
from typing import List, Optional

def get_inputs(param, device=None):
    """根据 test_cases.csv 行生成输入张量。返回与 Model.forward() 签名匹配的元组。"""
    if device is None:
        device = torch.device('npu:0')
    tokens_num = int(param['tokens_num'])
    hidden_size = int(param['hidden_size'])
    topk_num = int(param['topk_num'])
    dtype_str = str(param['dtype'])
    has_probs = int(param['has_probs'])
    range_start = int(param['range_start'])
    range_end = int(param['range_end'])
    if dtype_str == 'float32' or dtype_str == 'torch.float32':
        dtype = torch.float32
    elif dtype_str == 'float16' or dtype_str == 'torch.float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16' or dtype_str == 'torch.bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    total_indices = tokens_num * topk_num
    sorted_indices = torch.randperm(total_indices)[:total_indices].int().to(device)
    unpermuted_tokens_grad = torch.randn(tokens_num, hidden_size, dtype=dtype, device=device)
    permuted_tokens = torch.randn(total_indices, hidden_size, dtype=dtype, device=device)
    if has_probs:
        probs = torch.randn(tokens_num, topk_num, dtype=dtype, device=device)
    else:
        probs = None
    padded_mode = False
    restore_shape = [1, 1]
    if range_start < 0 or range_end < 0:
        range_vals = [0, total_indices]
    else:
        range_vals = [range_start, range_end]
    return (unpermuted_tokens_grad, sorted_indices, permuted_tokens, probs, padded_mode, restore_shape, range_vals, topk_num)

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenUnpermuteWithEpGrad.json")
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
