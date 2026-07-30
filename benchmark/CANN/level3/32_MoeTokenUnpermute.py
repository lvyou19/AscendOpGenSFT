"""Auto-generated benchmark file for MoeTokenUnpermute.

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

class Model(torch.nn.Module):
    """参考模型：与 api_desc 一致 — sorted_indices[k] 为从 permuted_tokens 取行的下标（gather），再按 topK 聚合。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, probs=None, padded_mode=False):
        rows, hidden = permuted_tokens.shape
        if probs is not None:
            tokens_num, topk = probs.shape
        else:
            tokens_num = rows
            topk = 1
        if rows != tokens_num * topk:
            raise ValueError(f'permuted_tokens dim0 ({rows}) must equal tokens_num*topk ({tokens_num}*{topk})')
        if sorted_indices.numel() != rows:
            raise ValueError('sorted_indices length must match permuted_tokens rows')
        pt = permuted_tokens.float()
        idx = sorted_indices.to(torch.int64).clamp(0, rows - 1)
        gathered = pt.index_select(0, idx.reshape(-1))
        if probs is not None:
            gathered = gathered * probs.float().reshape(-1, 1)
        out = gathered.view(tokens_num, topk, hidden).sum(dim=1)
        return out.to(permuted_tokens.dtype)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import math
import torch

def _cell_str(param, key, default=''):
    v = param.get(key, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return str(v).strip()

def get_inputs(param, device=None):
    permuted_tokens_shape = eval(param.get('permuted_tokens_shape', '[1,512]'))
    probs_shape = _cell_str(param, 'probs_shape', '')
    restore_shape = eval(param.get('restore_shape', '[16,16]'))
    padded_mode = bool(param.get('padded_mode', 0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    rows, hidden = (permuted_tokens_shape[0], permuted_tokens_shape[1])
    use_probs = bool(int(param.get('has_probs', 1)))
    permuted_tokens = torch.randn(permuted_tokens_shape, dtype=dtype)
    if use_probs and probs_shape not in ('', 'none', 'None'):
        probs_shape = eval(probs_shape)
        tokens_num, topk = (probs_shape[0], probs_shape[1])
        if rows != tokens_num * topk:
            raise ValueError(f'Shape mismatch: permuted_tokens dim0 {rows} != tokens_num*topk ({tokens_num}*{topk})')
        probs = torch.rand(probs_shape, dtype=dtype)
    else:
        tokens_num, topk = (rows, 1)
        probs = None
    nidx = tokens_num * topk
    sorted_indices = torch.randperm(nidx, dtype=torch.int32)
    return (permuted_tokens.to(device), sorted_indices.to(device), probs.to(device) if probs is not None else None, padded_mode)

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenUnpermute.json")
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
