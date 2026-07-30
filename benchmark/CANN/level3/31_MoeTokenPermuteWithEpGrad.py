"""Auto-generated benchmark file for MoeTokenPermuteWithEpGrad.

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

    def __init__(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        super(Model, self).__init__()
        self.topk_num = topk_num
        self.range_vals = range_vals
        self.padded_mode = padded_mode

    def forward(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        hidden_size = unpermuted_tokens_grad.shape[1]
        num_tokens = sorted_indices.shape[0]
        token_grad_out = torch.zeros(num_tokens, hidden_size, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        sorted_indices_long = sorted_indices.to(torch.int64)
        if range_vals is not None and len(range_vals) == 2:
            start = range_vals[0]
            end = range_vals[1]
            mask = (sorted_indices_long >= start) & (sorted_indices_long < end)
            valid_indices = sorted_indices_long[mask] - start
            token_grad_out[mask] = unpermuted_tokens_grad[valid_indices]
        token_grad_out = token_grad_out.reshape(-1, topk_num, hidden_size)
        token_grad_out = token_grad_out.sum(dim=1)
        if permuted_probs_output_grad is not None:
            probs_grad_out = torch.zeros(num_tokens, topk_num, dtype=permuted_probs_output_grad.dtype, device=permuted_probs_output_grad.device)
            if range_vals is not None and len(range_vals) == 2:
                start = range_vals[0]
                end = range_vals[1]
                mask = (sorted_indices_long >= start) & (sorted_indices_long < end)
                valid_indices = sorted_indices_long[mask] - start
                probs_grad_out[mask] = permuted_probs_output_grad.view(-1, 1).expand(-1, topk_num)[valid_indices]
            if range_vals is not None and len(range_vals) == 2:
                probs_grad_out = probs_grad_out[range_vals[0]:range_vals[1]]
            probs_grad_out = probs_grad_out.reshape(-1, topk_num)
        else:
            probs_grad_out = None
        return [token_grad_out, probs_grad_out]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np
import json

def get_init_inputs_per_case(row, device=None):
    np.random.seed(2023)
    torch.manual_seed(2023)
    case_id = row['case_id']
    dtype = row['dtype']
    unpermuted_tokens_grad_shape = eval(row['unpermuted_tokens_grad_shape'])
    sorted_indices_shape = eval(row['sorted_indices_shape'])
    permuted_probs_output_grad_shape = eval(row['permuted_probs_output_grad_shape'])
    topk_num = row['topk_num']
    range_vals = eval(row['range'])
    padded_mode = row['padded_mode'] == 1
    if dtype == 'float32':
        torch_dtype = torch.float32
        np_dtype = np.float32
    elif dtype == 'float16':
        torch_dtype = torch.float16
        np_dtype = np.float16
    elif dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
        np_dtype = np.float32
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    unpermuted_tokens_grad = torch.tensor(np.random.uniform(-1.0, 1.0, unpermuted_tokens_grad_shape), dtype=torch_dtype)
    num_tokens = sorted_indices_shape[0]
    sorted_indices = torch.randint(0, num_tokens, sorted_indices_shape, dtype=torch.int32)
    if permuted_probs_output_grad_shape is not None and permuted_probs_output_grad_shape != []:
        permuted_probs_output_grad = torch.tensor(np.random.uniform(-1.0, 1.0, permuted_probs_output_grad_shape), dtype=torch_dtype)
    else:
        permuted_probs_output_grad = None
    if device is not None:
        unpermuted_tokens_grad = unpermuted_tokens_grad.to(device)
        sorted_indices = sorted_indices.to(device)
        if permuted_probs_output_grad is not None:
            permuted_probs_output_grad = permuted_probs_output_grad.to(device)
    return [unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode]

def get_inputs(row, device=None):
    return get_init_inputs_per_case(row, device=device)


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
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenPermuteWithEpGrad.json")
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
