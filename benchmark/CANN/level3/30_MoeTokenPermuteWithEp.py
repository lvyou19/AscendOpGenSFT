"""Auto-generated benchmark file for MoeTokenPermuteWithEp.

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
import numpy as np

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, indices: torch.Tensor, probs: torch.Tensor, range_vals: list, num_token_out: int, pad_mode: bool):
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices.float(), stable=True)
        sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True)
        sorted_indices1 = sorted_indices1.to(torch.int32)
        if range_vals is not None:
            start = range_vals[0]
            end = range_vals[1]
            sorted_indices_sliced = sorted_indices[start:end]
        else:
            sorted_indices_sliced = sorted_indices
        permuted_tokens = x.index_select(0, sorted_indices_sliced // topk)
        if probs is not None:
            flatten_probs = probs.view(-1)
            permuted_probs = flatten_probs.index_select(0, sorted_indices_sliced)
        else:
            permuted_probs = torch.empty(0, device=x.device, dtype=probs.dtype) if probs is not None else None
        return [permuted_tokens, sorted_indices1, permuted_probs]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import random
torch.manual_seed(42)
random.seed(42)

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float32')
    if dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = getattr(torch, dtype_str)
    x_shape = eval(param.get('x_shape', '[3, 4]'))
    indices_shape = eval(param.get('indices_shape', '[3, 2]'))
    probs_shape = eval(param.get('probs_shape', '[3, 2]'))
    range_vals = eval(param.get('range', '[1, 5]'))
    num_token_out = int(param.get('num_token_out', '6'))
    pad_mode = bool(param.get('pad_mode', 'False'))
    x = torch.randn(x_shape, device=device, dtype=dtype)
    indices = torch.randint(0, x_shape[1], indices_shape, device=device, dtype=torch.int32)
    probs = torch.rand(probs_shape, device=device, dtype=dtype)
    return (x, indices, probs, range_vals, num_token_out, pad_mode)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeTokenPermuteWithEp.
    
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
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenPermuteWithEp.json")
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
