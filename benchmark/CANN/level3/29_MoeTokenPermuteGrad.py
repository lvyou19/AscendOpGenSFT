"""Auto-generated benchmark file for MoeTokenPermuteGrad.

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

def permute(tokens, indices, num_out_tokens=None, padded_mode=False):
    if padded_mode:
        return (tokens.index_select(dim=0, index=indices.view(-1)), indices)
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)
    if num_out_tokens is not None and num_out_tokens != 0:
        sorted_indices = sorted_indices[:num_out_tokens]
    s_k = sorted_indices // topk
    permuted_tokens = tokens.index_select(0, s_k)
    return (permuted_tokens, sorted_indices1)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tokens, permuted_output_grad, indices, num_topk, padded_mode=False):
        tokens.requires_grad_(True)
        permuted_tokens, sorted_indices = permute(tokens, indices, num_topk, padded_mode)
        permuted_tokens.backward(permuted_output_grad)
        return tokens.grad

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def permute(tokens, indices, num_out_tokens=None, padded_mode=False):
    if padded_mode:
        return (tokens.index_select(dim=0, index=indices.view(-1)), indices)
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)
    if num_out_tokens is not None and num_out_tokens != 0:
        sorted_indices = sorted_indices[:num_out_tokens]
    s_k = sorted_indices // topk
    permuted_tokens = tokens.index_select(0, s_k)
    return (permuted_tokens, sorted_indices1)

def get_inputs(param, device=None):
    tokens_shape = eval(param.get('tokens_shape', [1, 7]))
    indices_shape = eval(param.get('indices_shape', [1]))
    num_out_tokens = int(param.get('num_out_tokens', 1548263338))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    padded_mode = bool(param.get('padded_mode', 0))
    total_tokens = indices_shape[0]
    if len(indices_shape) == 2:
        total_tokens = total_tokens * indices_shape[1]
    tokens = torch.rand(tokens_shape, device='cpu', dtype=dtype)
    indices = torch.randint(low=0, high=total_tokens, size=indices_shape).to(torch.int32).to('cpu')
    tmp, sorted_indices = permute(tokens, indices, num_out_tokens, padded_mode)
    permuted_output_grad = torch.rand(tmp.shape, device='cpu', dtype=dtype)
    return (tokens.to(device), permuted_output_grad.to(device), indices.to(torch.int32).to(device), num_out_tokens, padded_mode)

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenPermuteGrad.json")
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
