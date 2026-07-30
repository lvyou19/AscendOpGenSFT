"""Auto-generated benchmark file for StridedSliceAssignV2.

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
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var_ref: torch.Tensor, input_value: torch.Tensor, begin: torch.Tensor, end: torch.Tensor, strides: torch.Tensor, axes_optional: torch.Tensor) -> torch.Tensor:
        """
        Performs a strided slice assignment on var_ref, with all slice parameters
        (begin, end, strides, axes_optional) passed as explicit torch.Tensor inputs.

        Args:
            var_ref (torch.Tensor): The reference tensor to be modified.
            input_value (torch.Tensor): The tensor whose values will be assigned.
            begin (torch.Tensor): Tensor containing start indices for slicing (int64, 1D).
            end (torch.Tensor): Tensor containing end indices for slicing (int64, 1D).
            strides (torch.Tensor): Tensor containing step sizes for slicing (int64, 1D).
            axes_optional (torch.Tensor): Optional tensor specifying the axes along which
                                        to slice (int64, 1D). If empty, slicing occurs
                                        along sequential dimensions.

        Returns:
            torch.Tensor: The modified var_ref tensor.
        """
        output_var_ref = var_ref.clone()
        begin_list = begin.tolist()
        end_list = end.tolist()
        strides_list = strides.tolist()
        axes_list = axes_optional.tolist()
        num_dims = output_var_ref.dim()
        slices: List[slice] = [slice(None)] * num_dims
        if not axes_list:
            for i in range(len(begin_list)):
                if i < num_dims:
                    s_begin = begin_list[i] if i < len(begin_list) else 0
                    s_end = end_list[i] if i < len(end_list) else output_var_ref.shape[i]
                    s_stride = strides_list[i] if i < len(strides_list) else 1
                    slices[i] = slice(s_begin, s_end, s_stride)
        else:
            for i, axis_idx in enumerate(axes_list):
                if axis_idx >= 0 and axis_idx < num_dims:
                    s_begin = begin_list[i] if i < len(begin_list) else 0
                    s_end = end_list[i] if i < len(end_list) else output_var_ref.shape[axis_idx]
                    s_stride = strides_list[i] if i < len(strides_list) else 1
                    slices[axis_idx] = slice(s_begin, s_end, s_stride)
                else:
                    raise IndexError(f'Axis index {axis_idx} out of bounds for tensor with {num_dims} dimensions.')
        output_var_ref[tuple(slices)] = input_value
        return output_var_ref

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 StridedSliceAssignV2 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含所有输入张量 (var_ref, input_value, begin, end, strides, axes_optional)
    """
    var_ref_shape = eval(param.get('var_ref_shape', '[10, 10]'))
    input_value_shape = eval(param.get('input_value_shape', '[5, 10]'))
    begin_val = eval(param.get('begin_val', '[0]'))
    end_val = eval(param.get('end_val', '[5]'))
    strides_val = eval(param.get('strides_val', '[1]'))
    axes_optional_val = eval(param.get('axes_optional_val', '[0]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    var_ref = torch.rand(var_ref_shape, device=device, dtype=dtype)
    input_value = torch.rand(input_value_shape, device=device, dtype=dtype)
    begin = torch.tensor(begin_val, device=device, dtype=torch.int64)
    end = torch.tensor(end_val, device=device, dtype=torch.int64)
    strides = torch.tensor(strides_val, device=device, dtype=torch.int64)
    axes_optional = torch.tensor(axes_optional_val, device=device, dtype=torch.int64)
    return (var_ref, input_value, begin, end, strides, axes_optional)

def get_init_inputs_per_case(param, device=None):
    """
    StridedSliceAssignV2 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
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
    json_path = os.path.join(os.path.dirname(__file__), "StridedSliceAssignV2.json")
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
