"""Auto-generated benchmark file for LinearIndexV2.

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
from typing import List
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    实现LinearIndexV2算子功能的模型 - 将多维索引转换为线性索引。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, indices_list: List[torch.Tensor], stride: torch.Tensor, value_size: torch.Tensor) -> torch.Tensor:
        """
        实现LinearIndexV2算子功能。
        
        计算公式: output += (indices[i] % value_size[i]) * stride[i]

        Args:
            indices_list: 索引张量列表
            stride: 步长张量 [dim_num]
            value_size: 维度大小张量 [dim_num]

        Returns:
            线性索引张量 (int32)
        """
        output = torch.zeros_like(indices_list[0], dtype=torch.int32)
        for i, indices in enumerate(indices_list):
            indices_int64 = indices.to(torch.int64)
            value_size_val = value_size[i].item()
            stride_val = stride[i].item()
            quotient = torch.div(indices_int64, value_size_val, rounding_mode='floor')
            remainder = indices_int64 - quotient * value_size_val
            contribution = (remainder * stride_val).to(torch.int32)
            output = output + contribution
        return output

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    num_dims = int(param.get('num_dims', 2))
    index_shape_str = param.get('index_shape', '[10]')
    index_shape = eval(index_shape_str)
    dtype_str = param.get('dtype', 'int32')
    dtype = getattr(torch, dtype_str)
    stride_list = eval(param.get('stride', '[1]'))
    value_size_list = eval(param.get('value_size', '[10]'))
    indices_list = []
    for i in range(num_dims):
        max_val = value_size_list[i] * 2
        if dtype in [torch.int32, torch.int64]:
            indices = torch.randint(0, max_val, index_shape, device=device, dtype=dtype)
        else:
            indices = torch.randint(0, max_val, index_shape, device=device, dtype=torch.int32).to(dtype)
        indices_list.append(indices)
    stride = torch.tensor(stride_list, device=device, dtype=torch.int32)
    value_size = torch.tensor(value_size_list, device=device, dtype=torch.int32)
    return (indices_list, stride, value_size)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for LinearIndexV2.

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
    json_path = os.path.join(os.path.dirname(__file__), "LinearIndexV2.json")
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
