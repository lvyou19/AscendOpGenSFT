"""Auto-generated benchmark file for MoeComputeExpertTokens.

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
import torch.nn as nn

class Model(nn.Module):
    """
    CPU golden model for MoeComputeExpertTokens operator.
    
    Function: Calculate total token count before each expert based on sorted expert indices.
    """

    def __init__(self):
        """
        Initialize model.
        """
        super(Model, self).__init__()

    def forward(self, sorted_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
        """
        CPU golden implementation: Calculate total token count before each expert.
        
        Args:
            sorted_experts: Sorted expert indices, shape [num_tokens], dtype int32.
            num_experts: Number of experts.
        
        Returns:
            total_rows_before_expert: Total token count before each expert, 
                                      shape [num_experts], dtype int32.
        """
        sorted_experts_np = sorted_experts.cpu().numpy()
        num_experts = int(num_experts)
        arr_length = sorted_experts_np.shape[-1]
        res = np.arange(num_experts)
        for i in range(num_experts):
            target = i
            low = 0
            high = arr_length - 1
            target_location = -1
            while low <= high:
                mid = (low + high) // 2
                if sorted_experts_np[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
                    target_location = mid
            res[i] = target_location + 1
        res = res.astype(np.int32)
        return torch.from_numpy(res)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import random

def get_inputs(param, device=None):
    torch.manual_seed(42)
    random.seed(42)
    '\n    Generate input tensors for MoeComputeExpertTokens operator.\n    \n    输入:\n        sorted_experts: 已排序的专家索引，形状为 [num_tokens]，数据类型 int32\n        num_experts: 专家数量，标量整数\n    \n    输出:\n        total_rows_before_expert: 每个专家之前的总 token 数，形状为 [num_experts]，数据类型 int32\n    '
    shape = eval(param.get('input_shape', '[2048]'))
    dtype_str = param.get('dtype', 'int32')
    num_experts = int(param.get('num_experts', '8'))
    dtype = getattr(torch, dtype_str)
    sorted_experts = torch.randint(0, num_experts, shape, device=device, dtype=dtype)
    sorted_experts, _ = torch.sort(sorted_experts)
    return (sorted_experts, num_experts)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeComputeExpertTokens.
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
    json_path = os.path.join(os.path.dirname(__file__), "MoeComputeExpertTokens.json")
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
