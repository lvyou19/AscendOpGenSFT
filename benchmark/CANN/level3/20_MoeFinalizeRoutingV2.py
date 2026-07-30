"""Auto-generated benchmark file for MoeFinalizeRoutingV2.

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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, expandedX: torch.Tensor, expandedRowIdx: torch.Tensor, x1Optional: torch.Tensor, x2Optional: torch.Tensor, biasOptional: torch.Tensor, scalesOptional: torch.Tensor, expertIdxOptional: torch.Tensor, dropPadMode: int) -> torch.Tensor:
        if len(expandedX.shape) == 2:
            num_rows = expertIdxOptional.shape[0]
            k = expertIdxOptional.shape[1]
            hidden_size = expandedX.shape[-1]
            output = torch.zeros(num_rows, hidden_size, device=expandedX.device, dtype=expandedX.dtype)
            has_bias = biasOptional.numel() > 0
            for i in range(num_rows):
                temp_sum = torch.zeros(hidden_size, device=expandedX.device, dtype=expandedX.dtype)
                for j in range(k):
                    temp_sum += scalesOptional[i, j] * expandedX[expandedRowIdx[i + j * num_rows]]
                    if has_bias:
                        expert_id = expertIdxOptional[i, j].item()
                        temp_sum += scalesOptional[i, j] * biasOptional[expert_id]
                output[i] = x1Optional[i] + x2Optional[i] + temp_sum
            return output

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import random
torch.manual_seed(42)
random.seed(42)

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    batch_size = int(param.get('batch_size', 16))
    num_experts = int(param.get('num_experts', 8))
    tokens_per_expert = int(param.get('tokens_per_expert', 16))
    hidden_size = int(param.get('hidden_size', 256))
    k = int(param.get('k', 2))
    use_bias = bool(param.get('use_bias', True))
    expanded_x_shape = (batch_size * k, hidden_size)
    expandedX = torch.randn(expanded_x_shape, device=device, dtype=dtype)
    expandedRowIdx = torch.randint(0, batch_size, (batch_size * k,), device=device, dtype=torch.int32)
    x1Optional = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    x2Optional = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    if use_bias:
        biasOptional = torch.randn(num_experts, hidden_size, device=device, dtype=dtype)
    else:
        biasOptional = torch.zeros((0,), device=device, dtype=dtype)
    scalesOptional = torch.randn(batch_size, k, device=device, dtype=dtype)
    expertIdxOptional = torch.randint(0, num_experts, (batch_size, k), device=device, dtype=torch.int32)
    dropPadMode = int(param.get('dropPadMode', 0))
    return (expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional, expertIdxOptional, dropPadMode)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeFinalizeRoutingV2.

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeFinalizeRoutingV2.json")
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
