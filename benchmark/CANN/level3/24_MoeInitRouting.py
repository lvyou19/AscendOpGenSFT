"""Auto-generated benchmark file for MoeInitRouting.

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

    def forward(self, x, row_idx, expert_idx, active_num):
        x_np = x.to(torch.float32).numpy()
        expert_idx = expert_idx.numpy()
        row_idx = row_idx.numpy()
        num_rows = x_np.shape[0]
        k = expert_idx.shape[-1]
        sort_expert_idx = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind='stable')
        expanded_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1, kind='stable')
        expanded_dst_to_src_row = np.take_along_axis(row_idx.reshape((-1,)), sort_expert_idx, axis=-1)
        expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape, dtype=np.int32)
        expanded_row_idx[expanded_dst_to_src_row] = np.arange(expanded_dst_to_src_row.shape[-1], dtype=np.int32)
        active_num = min(active_num, num_rows) * k
        expanded_x = x_np[expanded_dst_to_src_row[:active_num] % num_rows, :]
        return (torch.from_numpy(expanded_x).to(x.dtype), torch.from_numpy(expanded_row_idx.astype(np.int32)), torch.from_numpy(expanded_expert_idx))

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    x_shape = eval(param.get('x_shape', [1, 9]))
    row_idx_shape = eval(param.get('row_idx_shape', [1, 1]))
    expert_idx_shape = eval(param.get('expert_idx_shape', [1, 1]))
    active_num = int(param.get('active_num', 209))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x = torch.randn(x_shape, dtype=dtype).to(device)
    row_idx = torch.arange(expert_idx_shape[0] * expert_idx_shape[1]).reshape(expert_idx_shape).to(torch.int32).to(device)
    expert_idx = torch.randint(1, 2, size=expert_idx_shape, dtype=torch.int32).to(device)
    return (x, row_idx, expert_idx, active_num)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeInitRoutingV3.

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeInitRouting.json")
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
